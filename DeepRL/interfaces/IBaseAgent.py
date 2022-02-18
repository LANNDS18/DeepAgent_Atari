import os
from pathlib import Path

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
from tensorflow.keras.optimizers import Adam

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
            frame_stack=4,
            optimizer=None,
            model_update_freq=4,
            target_sync_freq=1000,
            model_save_interval=2000,
            model_path=None,
            log_history=False,
            quiet=False,
    ):
        self.game_id = env.id
        self.env = env
        self.n_actions = self.env.action_space.n
        self.frame_stack = frame_stack
        self.input_shape = self.env.observation_space.shape

        self.buffer = buffer
        self.mean_reward_buffer = deque(maxlen=mean_reward_step)

        self.gamma = gamma
        self.epsilon = 0

        self.state = self.env.reset()
        self.done = False

        self.best_mean_reward = 0
        self.mean_reward = 0
        self.episode_reward = 0

        self.total_step = 0
        self.episode = 0
        self.max_steps = None
        self.last_reset_step = 0

        self.training_start_time = None
        self.last_reset_time = None
        self.frame_speed = 0

        self.model_save_interval = model_save_interval
        self.model_update_freq = model_update_freq
        self.target_sync_freq = target_sync_freq

        self.model = model(n_actions=self.n_actions,
                           frame_stack=frame_stack,
                           input_shape=self.input_shape)

        self.target_model = model(n_actions=self.n_actions,
                                  frame_stack=self.frame_stack,
                                  input_shape=self.input_shape)

        self.optimizer = Adam(learning_rate=1e-4, epsilon=1e-6) if optimizer is None else optimizer

        self.loss = tf.keras.losses.Huber()
        self.loss_metric = tf.keras.metrics.Mean('loss_metric', dtype=tf.float32)
        self.q_metric = tf.keras.metrics.Mean(name="Q_value")

        self.quiet = quiet

        self.model_path = model_path
        self.log_history = log_history

        if self.model_path and self.log_history:
            self.history_dict_path = self.model_path + 'history_check_point.json'
            self.train_log_dir = self.model_path + '/log/' + datetime.now().strftime("%Y%m%d-%H%M%S")

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

    def reset_env(self):
        """
        Reset env with no return
        """
        self.state = self.env.reset()

    def save_model(self, model):
        """
        Save model weight to checkpoint
        Args:
             model:
                keras model
        """
        if self.model_path:
            model.save_weights(self.model_path)

    def load_model(self, model):
        """
        Load model weight from model_path
        Args:
            model: the initial model
        """
        if self.model_path:
            check_point = tf.train.latest_checkpoint(self.model_path)
            model.load_weights(check_point)
        return model

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
            'best moving avg (100) reward',
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

    def update_training_parameters(self):
        """
        Update progress metrics which consist of last reset step and time used
        for calculation of fps, and update mean and best reward. The model is
        saved if there is a checkpoint path specified.
        """
        self.mean_reward_buffer.append(self.episode_reward)
        self.mean_reward = np.around(
            np.mean(self.mean_reward_buffer), 5
        )

        if self.mean_reward > self.best_mean_reward:
            self.display_message(
                f'Best Moving Average Reward Updated: {colored(str(self.best_mean_reward), "red")} -> '
                f'{colored(str(self.mean_reward), "green")}'
            )
            self.best_mean_reward = self.mean_reward
            self.save_model(self.model)

        self.state = self.env.reset()
        self.episode_reward = 0.0
        self.done = False
        self.episode += 1

        self.frame_speed = (self.total_step - self.last_reset_step) / (
                perf_counter() - self.last_reset_time
        )
        self.last_reset_time = perf_counter()
        self.last_reset_step = self.total_step

    def fill_buffer(self):
        """
        Fill replay buffer up to its initial size.
        """
        total_size = self.buffer.initial_size
        buffer = self.buffer
        state = self.env.reset()
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
        self.state = state
        self.display_message('')
        self.reset_env()

    def record_tensorboard(self):
        train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        with train_summary_writer.as_default():
            tf.summary.scalar('Moving Average Reward (100 Episode)', self.mean_reward, step=self.episode)
            tf.summary.scalar('Average Q', self.q_metric.result(), step=self.episode)
            tf.summary.scalar('Episode Reward', self.episode_reward, step=self.episode)
            tf.summary.scalar('Loss', self.loss_metric.result(), step=self.episode)
            tf.summary.scalar('Epsilon', self.epsilon, step=self.episode)
            tf.summary.scalar("Total Frames", self.total_step, step=self.episode)
        self.loss_metric.reset_states()
        self.q_metric.reset_states()

    def check_episodes(self):
        """
        Check environment done counts to display progress and update metrics.
        """
        if self.done:
            if self.model_path and self.log_history:
                self.update_history()
                self.record_tensorboard()
            self.update_training_parameters()
            self.display_learning_state()

    def check_finish_training(self):
        """
        Check whether a target reward or maximum number of total_step is reached.
        Returns:
            bool
        """
        if self.max_steps and self.total_step >= self.max_steps:
            self.display_message(f'Maximum total_step exceeded')
            return True
        return False

    def update_history(self):
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

        if self.episode % self.model_update_freq == 0:
            self.save_model(self.model)

    def step_env(self, action):
        """
        Step environment in self.env_name, update metrics (if any done episode)
            and return / store results.
        Args:
            action: An iterable of action to execute by environments.
        """
        observations = []
        state = self.state
        new_state, reward, done, _ = self.env.step(action)
        self.state = new_state
        self.done = done
        self.episode_reward += reward
        observation = state, action, reward, done, new_state
        self.buffer.append(*observation)
        if done:
            self.mean_reward_buffer.append(self.episode_reward)
            self.episode += 1
            self.state = self.env.reset()
        self.total_step += 1
        return observations

    def load_history_from_path(self):
        """
        Load previous training session metadata and update agent metrics to go from there.
        """
        if Path(self.history_dict_path).is_file():
            # Load model from checkpoint
            self.model = self.load_model(self.model_path)
            # Load training data from json
            previous_history = pd.read_json(self.history_dict_path).to_dict()
            self.mean_reward = previous_history['mean_reward'][0]
            self.best_mean_reward = previous_history['best_mean_reward'][0]
            history_start_steps = previous_history['step'][0]
            history_start_time = previous_history['time'][0]
            self.training_start_time = perf_counter() - history_start_time
            self.last_reset_step = self.total_step = int(history_start_steps)
            self.mean_reward_buffer.append(previous_history['episode_reward'][0])
            self.episode = previous_history['episode'][0]

    def init_training(self, max_steps):
        """
        Initialize training start time & model (self.model / self.target_model)
        Args:
            max_steps: Maximum time total_step, if exceeded, the training will stop.
        """
        if self.model_path and self.log_history:
            self.load_history_from_path()
        self.max_steps = max_steps
        self.training_start_time = perf_counter()
        self.last_reset_time = perf_counter()
        self.total_step = 0
        self.episode = 0
        self.state = self.env.reset()
        self.episode_reward = 0.0
        self.done = False

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
            video_dir=None,
            render=False,
            frame_dir=None,
            frame_delay=0.0,
            max_steps=None,
            frame_frequency=1,
    ):
        """
        Play and display a game.
        Args:
            video_dir: Path to directory to save the resulting game video.
            render: If True, the game will be displayed.
            frame_dir: Path to directory to save game frames.
            frame_delay: Delay between rendered frames.
            max_steps: Maximum environment total_step.
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
                self.display_message(f'Maximum total_step {max_steps} exceeded')
                break
            if render:
                env_in_use.render()
                sleep(frame_delay)
            if frame_dir and steps % frame_frequency == 0:
                frame = cv2.cvtColor(
                    env_in_use.render(mode='rgb_array'), cv2.COLOR_BGR2RGB
                )
                cv2.imwrite(os.path.join(frame_dir, f'{steps:05d}.jpg'), frame)
            action = np.argmax(self.model(self.state))
            self.state, reward, done, _ = env_in_use.step(action)
            total_reward += reward
            if done:
                self.display_message(f'Total reward: {total_reward}')
                break
            steps += 1
