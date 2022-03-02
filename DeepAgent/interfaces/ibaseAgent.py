import copy
import os
import gym
import numpy as np
import pandas as pd
import tensorflow as tf

from abc import ABC
from pathlib import Path
from collections import deque
from datetime import timedelta, datetime
from time import perf_counter, sleep
from termcolor import colored
from DeepAgent.utils.common import write_from_dict


class BaseAgent(ABC):
    """
        Base class for various types of dqn agents.
    """

    def __init__(
            self,
            env,
            model,
            buffer,
            agent_id,
            replay_start_size=20000,
            mean_reward_step=100,
            gamma=0.99,
            frame_stack=4,
            optimizer=None,
            model_update_freq=4,
            target_sync_freq=1000,
            saving_model=False,
            log_history=False,
            quiet=False,
    ):
        self.env = env
        self.game_id = env.id
        self.agent_id = agent_id
        self.n_actions = self.env.action_space.n
        self.frame_stack = frame_stack
        self.input_shape = self.env.observation_space.shape

        self.buffer = buffer
        self.replay_start_size = replay_start_size
        self.mean_reward_buffer = deque(maxlen=mean_reward_step)
        self.mean_reward_step = mean_reward_step

        self.gamma = gamma
        self.epsilon = 0

        self.state = self.env.reset()
        self.done = False

        self.best_mean_reward = -float('inf')
        self.mean_reward = -float('inf')
        self.episode_reward = 0

        self.total_step = 0
        self.episode = 0
        self.max_steps = None
        self.last_reset_step = 0

        self.training_start_time = None
        self.last_reset_time = None
        self.frame_speed = 0

        self.model_update_freq = model_update_freq
        self.target_sync_freq = target_sync_freq

        self.model = copy.deepcopy(model)
        self.target_model = copy.deepcopy(model)

        self.optimizer = model.optimizer if optimizer is None else optimizer

        self.loss = tf.keras.losses.Huber()
        self.loss_metric = tf.keras.metrics.Mean('loss_metric', dtype=tf.float32)
        self.q_metric = tf.keras.metrics.Mean(name="Q_value")

        self.quiet = quiet

        self.saving_model = saving_model
        self.log_history = log_history

        if self.saving_model:
            self.saving_path = f'./models/{self.agent_id}'
            self.check_saving_path()
            self.history_dict_path = self.saving_path + '/history_check_point.json'
        if self.log_history:
            self.train_log_dir = './log/' + agent_id + '_' + self.game_id + datetime.now().strftime("%Y%m%d-%H%M%S")
            self.summary_writer = tf.summary.create_file_writer(self.train_log_dir)

        self.reset_env()

    def display_message(self, *args, **kwargs):
        """
        Display messages to the console.
        Args:
            *args: args passed to print()
            **kwargs: kwargs passed to print()
        """
        if not self.quiet:
            print(*args, **kwargs)

    def check_saving_path(self):
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

    def reset_env(self):
        """
        Reset env with no return
        """
        self.state = self.env.reset()

    def save_model(self):
        """
        Save model weight to checkpoint
        """
        if self.saving_model:
            self.display_message('Saving Weights...')
            self.model.save_weights(self.saving_path + '/main/')
            self.target_model.save_weights(self.saving_path + '/target/')
            self.display_message(f'Successfully saving to {self.saving_path}')

    def load_model(self):
        """
        Load model weight from saving_path
        """
        self.display_message('Loading Weights...')
        self.model.load_weights(self.saving_path + '/main/')
        self.target_model.load_weights(self.saving_path + '/target/')
        self.display_message(f'Loaded from {self.saving_path}')

    def display_learning_state(self):
        """
        Display progress metrics to the console when environments complete a full episode.
        """
        display_titles = (
            'time',
            'total_step',
            'episode',
            'speed',
            'mean reward',
            f'best moving avg ({self.mean_reward_step}) reward',
        )

        display_values = (
            timedelta(seconds=perf_counter() - self.training_start_time),
            self.total_step,
            self.episode,
            f'{round(self.frame_speed)} step/s',
            self.mean_reward,
            self.best_mean_reward,
        )
        display = (
            f'{title}: {value}'
            for title, value in zip(display_titles, display_values)
        )
        self.display_message(', '.join(display))

    def reset_episode_parameters(self):
        """
        Reset the state, episode reward, done
        """
        self.state = self.env.reset()
        self.episode_reward = 0.0
        self.done = False

        self.last_reset_time = perf_counter()
        self.last_reset_step = self.total_step

    def update_training_parameters(self):
        """
        Update progress metrics which consist of last reset step and time used
        for calculation of fps, and update mean and best reward. The model is
        saved if there is a checkpoint path specified.
        """
        self.episode += 1

        self.mean_reward_buffer.append(self.episode_reward)

        self.mean_reward = np.around(
            np.mean(self.mean_reward_buffer), 5
        )

        self.frame_speed = (self.total_step - self.last_reset_step) / (
                perf_counter() - self.last_reset_time
        )

        if self.mean_reward > self.best_mean_reward:
            self.display_message(
                f'Best Moving Average Reward Updated: {colored(str(self.best_mean_reward), "red")} -> '
                f'{colored(str(self.mean_reward), "green")}'
            )
            self.best_mean_reward = self.mean_reward
            if self.saving_model:
                self.update_history()

    def fill_buffer(self, fill_size=20000, load=False):
        """
        Fill replay buffer up to its initial size.
        """
        total_size = self.buffer.size
        buffer = self.buffer
        state = self.env.reset()
        while buffer.current_size < fill_size:
            if not load:
                action = self.env.action_space.sample()
            else:
                state = tf.expand_dims(state, axis=0)
                action = np.argmax(self.model(state))
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
        self.state = state
        self.display_message('')
        self.reset_env()

    def record_tensorboard(self):
        step = self.episode
        with self.summary_writer.as_default():
            tf.summary.scalar('Average Reward (100 Episode Moving Average)', self.mean_reward, step=step)
            tf.summary.scalar('Average Q', self.q_metric.result(), step=step)
            tf.summary.scalar('Episode Reward', self.episode_reward, step=step)
            tf.summary.scalar('Loss', self.loss_metric.result(), step=step)
            tf.summary.scalar('Epsilon', self.epsilon, step=step)
            tf.summary.scalar("Total Frames", self.total_step, step=step)
        self.loss_metric.reset_states()
        self.q_metric.reset_states()

    def check_episodes(self):
        """
        Check environment done counts to display progress and update metrics.
        """
        if self.done:
            if self.log_history:
                self.record_tensorboard()
            self.update_training_parameters()
            self.display_learning_state()
            self.reset_episode_parameters()

    def check_finish_training(self):
        """
        Check whether a target reward or maximum number of total_step is reached.
        Returns:
            bool
        """
        if self.max_steps and self.total_step >= self.max_steps:
            self.display_message(f'Maximum total_step exceeded')
            self.saving_path = self.saving_path + '/end'
            os.makedirs(self.saving_path, exist_ok=True)
            self.history_dict_path = self.saving_path + '/history_check_point.json'
            self.update_history()
            return True
        return False

    def update_history(self, model=True):
        """
        Write 1 episode stats to checkpoint and write model when it crosses interval.
        """
        data = {
            'mean_reward': [self.mean_reward],
            'best_mean_reward': [self.best_mean_reward],
            'episode_reward': [self.episode_reward],
            'step': [self.total_step],
            'time': [perf_counter() - self.training_start_time],
            'episode': [self.episode]
        }
        write_from_dict(data, path=self.history_dict_path)
        if model:
            self.save_model()

    def load_history_from_path(self):
        """
        Load previous training session metadata and update agent metrics to go from there.
        """
        # Load model from checkpoint
        self.load_model()
        # Load training data from json
        previous_history = pd.read_json(self.history_dict_path).to_dict()
        self.mean_reward = previous_history['mean_reward'][0]
        self.best_mean_reward = previous_history['best_mean_reward'][0]
        history_start_steps = previous_history['step'][0]
        history_start_time = previous_history['time'][0]
        self.training_start_time = perf_counter() - history_start_time
        self.last_reset_step = self.total_step = int(history_start_steps)
        self.episode = previous_history['episode'][0]
        for i in range(self.episode):
            self.mean_reward_buffer.append(self.mean_reward)

    def init_training(self, max_steps):
        """
        Initialize training start time & model (self.model / self.target_model)
        Args:
            max_steps: Maximum time total_step, if exceeded, the training will stop.
        """

        self.max_steps = max_steps

        self.state = self.env.reset()
        self.episode_reward = 0.0
        self.done = False
        self.last_reset_time = perf_counter()
        self.training_start_time = perf_counter()
        load = False
        if self.saving_model and Path(self.history_dict_path).is_file():
            self.display_message(f'Load history from {self.history_dict_path}')
            load = True
            self.load_history_from_path()
        else:
            self.total_step = 0
            self.episode = 1
        self.fill_buffer(fill_size=self.replay_start_size, load=load)

    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.env_name, batching and gradient updates.
        """
        raise NotImplementedError(
            f'train_step() should be implemented by {self.__class__.__name__} subclasses'
        )

    def learn(
            self,
            max_steps,
    ):
        """
        Common training loop shared by subclasses, monitors training status
        and progress, performs all training total_step, updates metrics, and logs progress.
        Args:
             max_steps: Maximum number of total_step, if reached the training will stop.
        """
        self.init_training(max_steps)
        while True:
            self.check_episodes()
            if self.check_finish_training():
                break
            while not self.done:
                self.at_step_start()
                self.train_step()
                self.at_step_end()
            raise NotImplementedError(f'The train step of learn() should '
                                      f'be implemented by {self.__class__.__name__} subclasses')

    def at_step_start(self):
        pass

    def at_step_end(self):
        pass

    def play(
            self,
            test_env,
            saving_path,
            render=False,
            video_dir=None,
            frame_delay=0.0,
            max_episode=100,
    ):
        """
        Play and display a test_env.
        Args:
            test_env: The env for testing the agent
            saving_path: The path for loading the model
            video_dir: Path to directory to save the resulting test_env video.
            render: If True, the test_env will be displayed.
            frame_delay: Delay between rendered frames.
            max_episode: Maximum environment episode.
        """
        self.saving_path = saving_path
        self.load_model()
        episode = 0
        steps = 0
        episode_reward = 0
        total_reward = []

        env = test_env
        state = env.reset()

        if video_dir:
            env = gym.wrappers.Monitor(env, video_dir)
            env.reset()
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)

        while True:
            if render:
                env.render()
                sleep(frame_delay)

            # Greedy choose
            state = tf.expand_dims(state, axis=0)
            action = np.argmax(self.model(state))
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                total_reward.append(episode_reward)
                self.display_message(f'Episode: {episode}, Episode Reward: {episode_reward}')

                episode += 1
                episode_reward = 0

                state = env.reset()

                if max_episode and episode >= max_episode:
                    self.display_message(f'Maximum total_step {max_episode} exceeded')
                    env.close()
                    break
            steps += 1
        env.close()
        env.__exit__()
        return total_reward
