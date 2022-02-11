import os
import random
import gym
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb

from abc import ABC
from collections import deque
from datetime import timedelta
from pathlib import Path
from time import perf_counter, sleep
from termcolor import colored
from DeepRL.Common import write_from_dict


class OffPolicyAgent(ABC):
    """
        Base class for various types of dqn agents.
    """

    def __init__(
            self,
            env,
            model,
            buffer,
            mean_reward_step=100,
            n_steps=1,
            gamma=0.99,
            display_precision=2,
            seed=None,
            model_path=None,
            history_path=None,
            plateau_reduce_factor=0.9,
            plateau_reduce_patience=10,
            early_stop_patience=3,
            divergence_monitoring_steps=None,
            quiet=False
    ):
        self.env = env
        self.model = model
        self.buffer = buffer

        self.n_steps = n_steps
        self.total_rewards = deque(maxlen=mean_reward_step)

        self.gamma = gamma

        self.display_precision = display_precision
        self.seed = seed

        self.model_path = model_path
        self.history_path = history_path

        self.plateau_reduce_factor = plateau_reduce_factor
        self.plateau_reduce_patience = plateau_reduce_patience
        self.early_stop_patience = early_stop_patience
        self.divergence_monitoring_steps = divergence_monitoring_steps
        self.quiet = quiet

        self.state = self.env.reset()
        self.n_actions = self.env.action_space.n
        self.input_shape = self.env.observation_space.shape

        self.best_reward = -float('inf')
        self.mean_reward = -float('inf')
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
        self.done_count = 0  # consider deleting this later
        self.target_reward = None

        self.is_img = len(self.state.shape) >= 2

        self.batch_size = self.buffer.batch_size

        self.reset_env()
        if seed:
            self.set_seeds(seed)

    def display_message(self, *args, **kwargs):
        """
        Display messages to the console.
        Args:
            *args: args passed to print()
            **kwargs: kwargs passed to print()
        """
        if not self.quiet:
            print(*args, **kwargs)

    def set_seeds(self, seed):
        """
        Set random seeds for numpy, tensorflow, random, gym
        Args:
            seed: int, random seed.
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        os.environ['PYTHONHASHSEED'] = f'{seed}'
        random.seed(seed)

    def reset_env(self):
        """
        Reset all environments in self.env and update self.state
        """
        self.state = self.env.reset()

    def save_best_model(self):
        """
        Save model weights if current reward > best reward.
        """
        if self.mean_reward > self.best_reward:
            self.plateau_count = 0
            self.early_stop_count = 0
            self.display_message(
                f'Best reward updated: {colored(str(self.best_reward), "red")} -> '
                f'{colored(str(self.mean_reward), "green")}'
            )
            if self.model_path:
                self.model.save_weights(self.model_path)
        self.best_reward = max(self.mean_reward, self.best_reward)

    def display_metrics(self):
        """
        Display progress metrics to the console when environments complete a full episode each.
        Metrics consist of:
            - time: Time since training started.
            - steps: Time steps so far.
            - terminal_episodes: Finished terminal_episodes / episodes that resulted in a terminal state.
            - speed: Frame speed/s
            - mean reward: Mean game total reward.
            - best reward: Highest total episode score obtained.
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

    def update_metrics(self):
        """
        Update progress metrics which consist of last reset step and time used
        for calculation of fps, and update mean and best reward. The model is
        saved if there is a checkpoint path specified.

        Returns:
            None
        """
        self.save_best_model()
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
            np.mean(self.total_rewards), self.display_precision
        )

    def fill_buffer(self):
        """
        Fill replay buffer up to its initial size.

        Returns:
            None
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
                f'\rFilling replay buffer | '
                f'{filled}/{total_size}',
                end='',
            )
        self.display_message('')
        self.reset_env()

    def check_episodes(self):
        """
        Check environment done counts to display progress and update metrics.

        Returns:
            None
        """
        if self.done_count >= 1:
            self.update_metrics()
            self.last_reset_time = perf_counter()
            self.display_metrics()
            self.done_count = 0

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
            self.display_message(f'Reward achieved in {self.steps} steps')
            return True
        if self.max_steps and self.steps >= self.max_steps:
            self.display_message(f'Maximum steps exceeded')
            return True
        return False

    def update_history(self, episode_reward):
        """
        Write 1 episode stats to .parquet history checkpoint.
        Args:
            episode_reward: int, a finished episode reward
        Returns:
            None
        """
        data = {
            'mean_reward': [self.mean_reward],
            'best_reward': [self.best_reward],
            'episode_reward': [episode_reward],
            'step': [self.steps],
            'time': [perf_counter() - self.training_start_time],
        }
        write_from_dict(data, self.history_path)

    def step_env(self, action, store_in_buffers=False):
        """
        Step environment in self.env, update metrics (if any done terminal_episodes)
            and return / store results.
        Args:
            action: An iterable of action to execute by environments.
            store_in_buffers: If True, each observation is saved separately in respective buffer.
        """
        state = self.state
        new_state, reward, done, _ = self.env.step(action)
        self.state = new_state
        self.done = done
        self.episode_reward += reward
        observation = state, action, reward, done, new_state
        if store_in_buffers and hasattr(self, 'buffer'):
            self.buffer.append(*observation)
        if done:
            if self.history_path:
                self.update_history(self.episode_reward)
            self.done_count += 1
            self.total_rewards.append(self.episode_reward)
            self.terminal_episodes += 1
            self.episode_reward = 0
            self.state = self.env.reset()
            self.steps += 1

    def load_history_from_path(self):
        """
        Load previous training session metadata and update agent metrics
            to go from there.
        """
        previous_history = pd.read_parquet(self.history_path)
        last_row = previous_history.loc[previous_history['time'].idxmax()]
        self.mean_reward = last_row['mean_reward']
        self.best_reward = previous_history['best_reward'].max()
        history_start_steps = last_row['step']
        history_start_time = last_row['time']
        self.training_start_time = perf_counter() - history_start_time
        self.last_reset_step = self.steps = int(history_start_steps)
        self.total_rewards.append(last_row['episode_reward'])
        self.terminal_episodes = previous_history.shape[0]

    def init_training(self, target_reward, max_steps, monitor_session):
        """
        Initialize training start time, wandb session & model (self.model / self.target_model)
        Args:
            target_reward: Total reward per game value that whenever achieved,
                the training will stop.
            max_steps: Maximum time steps, if exceeded, the training will stop.
            monitor_session: Wandb session name.
        """
        self.target_reward = target_reward
        self.max_steps = max_steps
        if monitor_session:
            wandb.init(name=monitor_session)
        self.training_start_time = perf_counter()
        self.last_reset_time = perf_counter()
        if self.history_path and Path(self.history_path).exists():
            self.load_history_from_path()

    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.env, batching and gradient updates.
        """
        raise NotImplementedError(
            f'train_step() should be implemented by {self.__class__.__name__} subclasses'
        )

    def get_model_outputs(self, inputs, model, training=True):
        """
        Get single model outputs.
        Args:
            inputs: Inputs as tensors / numpy arrays that are expected
                by the given model.
            model: A tf.keras.Model
            training: whether training or not
        Returns:
            Outputs single model
        """
        if isinstance(model, tf.keras.models.Model):
            inputs = tf.reshape(-1, self.input_shape[0], self.input_shape[2], self.batch_size)
            return model(inputs, training=training)
        else:
            raise AttributeError('model should be a tf.keras.Model')

    def fit(
            self,
            target_reward=None,
            max_steps=None,
            monitor_session=None,
    ):
        """
        Common training loop shared by subclasses, monitors training status
        and progress, performs all training steps, updates metrics, and logs progress.
        Args:
            target_reward: Target reward, if achieved, the training will stop
            max_steps: Maximum number of steps, if reached the training will stop.
            monitor_session: Session name to use for monitoring the training with wandb.

        self.init_training(target_reward, max_steps, monitor_session)
        while True:
            self.check_episodes()
            if self.check_finish_training():
                break
            self.at_step_start()
            self.train_step()
            self.at_step_end()
        """
        raise NotImplementedError('Should implement in sub class')



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

        Returns:
            None
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
            action = self.get_model_outputs(
                self.state, self.model, False
            )[action_idx].numpy()
            self.state, reward, done, _ = env_in_use.step(action)
            total_reward += reward
            if done:
                self.display_message(f'Total reward: {total_reward}')
                break
            steps += 1

    '''delete this function later'''

    @staticmethod
    def concat_step_batches(*args):
        """
        Concatenate n-step batches.
        Args:
            *args: A list of numpy arrays which will be concatenated separately.

        Returns:
            A list of concatenated numpy arrays.
        """
        concatenated = []
        for arg in args:
            if len(arg.shape) == 1:
                arg = np.expand_dims(arg, -1)
            concatenated.append(arg.swapaxes(0, 1).reshape(-1, *arg.shape[2:]))
        return concatenated

    def at_step_start(self):
        pass

    def at_step_end(self):
        pass
