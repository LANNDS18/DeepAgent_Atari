import os
import gym
import cv2
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
            mean_reward_step=100,
            gamma=0.99,
            frame_stack=4,
            optimizer=None,
            model_update_freq=4,
            target_sync_freq=1000,
            saving_model=True,
            log_history=True,
            model_save_interval=1000,
            quiet=False,
    ):
        self.env = env
        self.game_id = env.id
        self.agent_id = agent_id
        self.n_actions = self.env.action_space.n
        self.frame_stack = frame_stack
        self.input_shape = self.env.observation_space.shape

        self.buffer = buffer
        self.mean_reward_buffer = deque(maxlen=mean_reward_step)
        self.mean_reward_step = mean_reward_step

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

        self.model = model

        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = self.model.optimizer if optimizer is None else optimizer

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
            self.train_log_dir = './log/' + agent_id + '/' + self.game_id + datetime.now().strftime("%Y%m%d-%H%M%S")

        self.loss_all = []
        self.q_all = []
        self.mean_reward_all = []
        self.episode_reward_all = []

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
            print('Saving Weights...')
            self.model.save_weights(self.saving_path + '/main/')
            self.target_model.save_weights(self.saving_path + '/target/')

    def load_model(self):
        """
        Load model weight from saving_path
        """
        if self.saving_model:
            print('Loading Weights...')
            self.model.load_weights(self.saving_path + '/main/')
            self.target_model.load_weights(self.saving_path + '/target/')

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

        self.mean_reward_all.append(self.mean_reward)
        self.episode_reward_all.append(self.episode_reward)

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
            self.update_history()

        if self.saving_model and self.episode % self.model_save_interval == 0:
            self.update_history()

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
        step = self.episode
        train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        with train_summary_writer.as_default():
            tf.summary.scalar('Moving Average Reward (100 Episode)', self.mean_reward, step=step)
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
            np.save(self.saving_path + '/loss', self.loss_all)
            np.save(self.saving_path + '/average q value', self.q_all)
            np.save(self.saving_path + '/moving average reward (100 episode)', self.mean_reward_all)
            np.save(self.saving_path + '/episode reward', self.episode_reward_all)
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
        self.save_model()

    def load_history_from_path(self):
        """
        Load previous training session metadata and update agent metrics to go from there.
        """
        if Path(self.history_dict_path).is_file():
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

        if Path(self.history_dict_path).is_file():
            self.load_history_from_path()
        else:
            self.total_step = 0
            self.episode = 1
            self.training_start_time = perf_counter()

        self.model.compile(self.optimizer)

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
