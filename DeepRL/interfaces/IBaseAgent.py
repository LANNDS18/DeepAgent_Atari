import os
import gym
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from abc import ABC
from collections import deque
from datetime import timedelta, datetime
from time import perf_counter, sleep
from termcolor import colored
from DeepRL.utils.common import write_from_dict


class BaseAgent(ABC):
    """
        Base class for various types of dqn agents.
    """

    def __init__(
            self,
            env,
            model,
            buffer,
            mean_reward_step=100,
            gamma=0.99,
            model_path=None,
            log_history=False,
            plateau_reduce_factor=0.9,
            plateau_reduce_patience=10,
            early_stop_patience=3,
            divergence_monitoring_steps=None,
            quiet=False,
            epsilon=0.1
    ):
        self.env = env
        self.model = model
        self.buffer = buffer

        self.total_rewards = deque(maxlen=mean_reward_step)

        self.gamma = gamma

        self.plateau_reduce_factor = plateau_reduce_factor
        self.plateau_reduce_patience = plateau_reduce_patience
        self.early_stop_patience = early_stop_patience
        self.divergence_monitoring_steps = divergence_monitoring_steps
        self.quiet = quiet

        self.state = self.env.reset()
        self.n_actions = self.env.action_space.n
        self.input_shape = self.env.observation_space.shape

        self.best_reward = 0
        self.mean_reward = 0
        self.episode_reward = 0
        self.plateau_count = 0
        self.early_stop_count = 0

        self.steps = 0
        self.max_steps = None
        self.last_reset_step = 0
        self.terminal_episodes = 0

        self.training_start_time = None
        self.last_reset_time = None
        self.frame_speed = 0

        self.done = False
        self.target_reward = None

        self.batch_size = self.buffer.batch_size

        self.reset_env()
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

        self.epsilon = epsilon

        self.model_path = model_path
        self.log_history = log_history

        if self.model_path and self.log_history:
            self.history_dict_path = self.model_path + 'history_check_point.json'
            self.train_log_dir = self.model_path + '/log' + datetime.now().strftime("%Y%m%d-%H%M%S")

    def display_message(self, *args, **kwargs):
        """
        Display messages to the console.
        Args:
            *args: args passed to print()
            **kwargs: kwargs passed to print()
        """
        if not self.quiet:
            print(*args, **kwargs)

    def reset_env(self):
        self.state = self.env.reset()

    def display_learning_state(self):
        """
        Display progress metrics to the console when environments complete a full episode.
        """
        display_titles = (
            'time',
            'steps',
            'terminal_episodes',
            'speed',
            'mean reward',
            'best reward',
        )

        display_values = (
            timedelta(seconds=perf_counter() - self.training_start_time),
            self.steps,
            self.terminal_episodes,
            f'{round(self.frame_speed)} steps/s',
            self.mean_reward,
            self.best_reward,
        )
        display = (
            f'{title}: {value}'
            for title, value in zip(display_titles, display_values)
        )
        self.display_message(', '.join(display))

    def check_training_state(self):
        """
        Update progress metrics which consist of last reset step and time used
        for calculation of fps, and update mean and best reward. The model is
        saved if there is a checkpoint path specified.
        """
        if self.episode_reward > self.best_reward:
            self.plateau_count = 0
            self.early_stop_count = 0
            self.display_message(
                f'Best reward updated: {colored(str(self.best_reward), "red")} -> '
                f'{colored(str(self.episode_reward), "green")}'
            )
            self.best_reward = max(self.episode_reward, self.best_reward)
            if self.model_path:
                self.model.save_weights(self.model_path)

        if (
                self.divergence_monitoring_steps
                and self.steps >= self.divergence_monitoring_steps
                and self.mean_reward <= self.best_reward
        ):
            self.plateau_count += 1

        if self.plateau_count >= self.plateau_reduce_patience:
            current_lr = self.model.optimizer.learning_rate
            new_lr = current_lr * self.plateau_reduce_factor
            self.display_message(
                f'Learning rate reduced {current_lr.numpy()} ' f'-> {new_lr.numpy()}'
            )
            current_lr.assign(new_lr)
            self.plateau_count = 0
            self.early_stop_count += 1
        self.frame_speed = (self.steps - self.last_reset_step) / (
                perf_counter() - self.last_reset_time
        )
        self.last_reset_step = self.steps
        self.mean_reward = np.around(
            np.mean(self.total_rewards), 5
        )

    def fill_buffer(self):
        """
        Fill replay buffer up to its initial size.
        """
        total_size = self.buffer.initial_size
        buffer = self.buffer
        state = self.state
        while buffer.current_size < buffer.initial_size:
            action = self.env.action_space.sample()
            new_state, reward, done, _ = self.env.step(action)
            buffer.append(state, action, reward, done, new_state)
            state = new_state
            if done:
                state = self.env.reset()
            filled = buffer.current_size
            self.display_message(
                f'\rFilling experience replay buffer => '
                f'{filled}/{total_size}',
                end='',
            )
        self.display_message('')
        self.reset_env()

    def record_tensorboard(self):
        train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        with train_summary_writer.as_default():
            tf.summary.scalar('episode reward', self.episode_reward, step=self.steps)
            tf.summary.scalar('mean reward', self.mean_reward, step=self.steps)
            tf.summary.scalar('loss', self.train_loss.result(), step=self.steps)
            tf.summary.scalar('epsilon', self.epsilon, step=self.steps)

    def check_episodes(self):
        """
        Check environment done counts to display progress and update metrics.
        """
        if self.done:
            if self.model_path and self.log_history:
                self.update_history()
                self.record_tensorboard()
            self.train_loss.reset_states()
            self.check_training_state()
            self.last_reset_time = perf_counter()
            self.display_learning_state()
            self.done = False
            self.episode_reward = 0

    def check_finish_training(self):
        """
        Check whether a target reward or maximum number of steps is reached.
        Returns:
            bool
        """
        if self.early_stop_count >= self.early_stop_patience:
            self.display_message(f'Early stopping')
            return True
        if self.target_reward and self.mean_reward >= self.target_reward:
            self.display_message(f'Mean Reward achieved in {self.steps} steps')
            return True
        if self.max_steps and self.steps >= self.max_steps:
            self.display_message(f'Maximum steps exceeded')
            return True
        return False

    def update_history(self):
        """
        Write 1 episode stats to .parquet history checkpoint.
        """
        data = {
            'mean_reward': [self.mean_reward],
            'best_reward': [self.best_reward],
            'episode_reward': [self.episode_reward],
            'step': [self.steps],
            'time': [perf_counter() - self.training_start_time],
        }
        write_from_dict(data, path=self.history_dict_path)

    def step_env(self, action, store_in_buffers=False):
        """
        Step environment in self.env_name, update metrics (if any done terminal_episodes)
            and return / store results.
        Args:
            action: An iterable of action to execute by environments.
            store_in_buffers: If True, each observation is saved separately in respective buffer.
        """
        observations = []
        state = self.state
        new_state, reward, done, _ = self.env.step(action)
        self.state = new_state
        self.done = done
        self.episode_reward += reward
        observation = state, action, reward, done, new_state
        if store_in_buffers:
            self.buffer.append(*observation)
        if done:
            self.total_rewards.append(self.episode_reward)
            self.terminal_episodes += 1
            self.state = self.env.reset()
        self.steps += 1
        return observations

    def load_history_from_path(self):
        """
        Load previous training session metadata and update agent metrics to go from there.
        """
        if os.path.exists(self.history_dict_path):
            previous_history = pd.read_json(self.history_dict_path).to_dict()
            self.mean_reward = previous_history['mean_reward']
            self.best_reward = previous_history['best_reward']
            history_start_steps = previous_history['step'][0]
            history_start_time = previous_history['time'][0]
            self.training_start_time = perf_counter() - history_start_time
            self.last_reset_step = self.steps = int(history_start_steps)
            self.total_rewards.append(previous_history['episode_reward'][0])
            self.terminal_episodes = previous_history.shape[0]

    def init_training(self, target_reward, max_steps):
        """
        Initialize training start time & model (self.model / self.target_model)
        Args:
            target_reward: Total reward per game value that whenever achieved,
                the training will stop.
            max_steps: Maximum time steps, if exceeded, the training will stop.
        """
        if self.model_path and self.log_history:
            self.load_history_from_path()
        self.target_reward = target_reward
        self.max_steps = max_steps
        self.training_start_time = perf_counter()
        self.last_reset_time = perf_counter()

    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.env_name, batching and gradient updates.
        """
        raise NotImplementedError(
            f'train_step() should be implemented by {self.__class__.__name__} subclasses'
        )

    def model_predict(self, x_pred, model, training=True):
        """
        Get single model outputs.
        Args:
            x_pred: Inputs as tensors / numpy arrays that are expected
                by the given model.
            model: A tf.keras.Model
            training: whether training or not
        Returns:
            q value list for single model
        """
        if isinstance(model, tf.keras.models.Model):
            return model(x_pred, training=training)
        else:
            raise AttributeError('model should be a tf.keras.Model')

    def learn(
            self,
            target_reward=None,
            max_steps=None,
    ):
        """
        Common training loop shared by subclasses, monitors training status
        and progress, performs all training steps, updates metrics, and logs progress.
        Args:
            target_reward: Target reward, if achieved, the training will stop
            max_steps: Maximum number of steps, if reached the training will stop.
        """
        self.init_training(target_reward, max_steps)
        while True:
            self.check_episodes()
            if self.check_finish_training():
                break
            self.at_step_start()
            self.train_step()
            self.at_step_end()

    def at_step_start(self):
        pass

    def at_step_end(self):
        pass

    def play(
            self,
            video_dir=None,
            render=False,
            frame_dir=None,
            frame_delay=0.0,
            max_steps=None,
            action_idx=0,
            frame_frequency=1,
    ):
        """
        Play and display a game.
        Args:
            video_dir: Path to directory to save the resulting game video.
            render: If True, the game will be displayed.
            frame_dir: Path to directory to save game frames.
            frame_delay: Delay between rendered frames.
            max_steps: Maximum environment steps.
            action_idx: Index of action output by self.model
            frame_frequency: If frame_dir is specified, save frames every n frames.
        """
        self.reset_env()
        total_reward = 0
        env_in_use = self.env
        if video_dir:
            env_in_use = gym.wrappers.Monitor(env_in_use, video_dir)
            env_in_use.reset()
        steps = 0
        for dir_name in (video_dir, frame_dir):
            os.makedirs(dir_name or '.', exist_ok=True)
        while True:
            if max_steps and steps >= max_steps:
                self.display_message(f'Maximum steps {max_steps} exceeded')
                break
            if render:
                env_in_use.render()
                sleep(frame_delay)
            if frame_dir and steps % frame_frequency == 0:
                frame = cv2.cvtColor(
                    env_in_use.render(mode='rgb_array'), cv2.COLOR_BGR2RGB
                )
                cv2.imwrite(os.path.join(frame_dir, f'{steps:05d}.jpg'), frame)
            action = self.model_predict(
                self.state, self.model, False
            )[action_idx].numpy()
            self.state, reward, done, _ = env_in_use.step(action)
            total_reward += reward
            if done:
                self.display_message(f'Total reward: {total_reward}')
                break
            steps += 1
