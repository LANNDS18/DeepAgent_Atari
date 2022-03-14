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


class OffPolicy(ABC):
    """
        Base class for various types of dqn agents.
    """

    def __init__(
            self,
            env,
            policy_network,
            target_network,
            buffer,
            agent_id,
            buffer_fill_size=1000,
            mean_reward_step=100,
            gamma=0.99,
            frame_stack=4,
            model_update_freq=4,
            target_sync_freq=10000,
            saving_model=False,
            log_history=False,
            validation_freq=10,
            quiet=False,
    ):
        self.env = env
        self.game_id = env.id
        self.agent_id = agent_id
        self.n_actions = self.env.action_space.n
        self.frame_stack = frame_stack
        self.input_shape = self.env.observation_space.shape

        self.buffer = buffer
        self.buffer_fill_size = buffer_fill_size
        self.real_mean_reward_buffer = deque(maxlen=mean_reward_step)
        self.mean_reward_step = mean_reward_step

        self.gamma = gamma
        self.n_step = buffer.n_step if buffer.n_step else 0
        self.epsilon = None

        self.state = self.env.reset()
        self.done = False

        self.real_best_mean_reward = -float('inf')
        self.real_mean_reward = -float('inf')
        self.real_episode_score = 0

        self.total_step = 0
        self.episode = 0

        self.max_steps = None
        self.target_reward = None

        self.last_reset_step = 0
        self.training_start_time = None
        self.last_reset_time = None
        self.frame_speed = 0

        self.model_update_freq = model_update_freq
        self.target_sync_freq = target_sync_freq

        self.policy_network = policy_network
        self.target_network = target_network

        self.loss_metric = tf.keras.metrics.Mean('loss_metric', dtype=tf.float32)
        self.q_metric = tf.keras.metrics.Mean(name="Q_value")

        self.quiet = quiet

        self.saving_model = saving_model
        self.log_history = log_history

        if self.saving_model:
            self.saving_path = f'./models/{self.agent_id}' + '_' + self.game_id
            self.check_and_create_path(self.saving_path)
            self.history_dict_file = '/history_check_point.json'
        if self.log_history:
            self.train_log_dir = './log/' + agent_id + '_' + self.game_id + '_' + datetime.now().strftime(
                "%Y%m%d-%H%M%S")
            self.summary_writer = tf.summary.create_file_writer(self.train_log_dir)

        self.validation_freq = validation_freq
        self.validation_score = -float('inf')
        self.max_validation_score = -float('inf')

        self.reset_env()

    def fill_buffer(self, load=False):
        """
        Fill replay buffer up to its initial size.
        """
        episode = 0
        total_size = self.buffer.size
        state = self.env.reset()
        while self.buffer.current_size < self.buffer_fill_size:
            if not load:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(self.policy_network.predict(tf.expand_dims(state, axis=0)))
            new_state, reward, done, _ = self.env.step(action)
            self.buffer.append(state, action, reward, done, new_state)
            state = new_state
            if self.env.was_real_done:
                state = self.env.reset()
                episode += 1
            filled = self.buffer.current_size
            self.display_message(
                f'\rFilling experience replay buffer => '
                f'{filled}/{total_size}',
                end='',
            )

        self.state = state
        self.display_message('')
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

    @staticmethod
    def check_and_create_path(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def reset_env(self):
        """
        Reset env with no return
        """
        self.state = self.env.reset()

    def save_model(self, path):
        """
        Save policy_network weight to checkpoint
        """
        if self.saving_model:
            self.display_message('Saving Weights...')
            self.policy_network.save(path + '/main/')
            self.target_network.save(path + '/target/')
            self.display_message(f'Successfully saving to {path}')

    def load_model(self, path):
        """
        Load policy_network weight from saving_path
        """
        self.display_message('Loading Weights...')
        self.policy_network.load(path + '/main/')
        self.target_network.load(path + '/target/')
        self.display_message(f'Loaded from {path}')

    def sync_target_model(self):
        """Synchronize weights of target network by those of main network."""
        self.target_network.model.set_weights(self.policy_network.model.get_weights())

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
            'epsilon',
        )

        display_values = (
            timedelta(seconds=perf_counter() - self.training_start_time),
            self.total_step,
            self.episode,
            f'{round(self.frame_speed)} step/s',
            self.real_mean_reward,
            self.real_best_mean_reward,
            self.epsilon,
        )
        display = (
            f'{title}: {colored(str(value), "blue")}'
            for title, value in zip(display_titles, display_values)
        )
        self.display_message(', '.join(display))

    def check_finish_training(self):
        """
        Check whether a target reward or maximum number of total_step is reached.
        Returns:
            bool
        """
        finish = False
        if self.max_steps and self.total_step >= self.max_steps:
            self.display_message(f'Maximum total_step exceeded')
            finish = True

        if self.target_reward is not None and self.real_mean_reward >= self.target_reward:
            self.display_message(f'Reach Target reward{self.target_reward}, early stop')
            finish = True

        if finish:
            saving_path = self.saving_path + '/end'
            self.check_and_create_path(saving_path)
            self.update_history(model_path=saving_path)
            return True
        return False

    def record_tensorboard(self):
        step = self.episode
        with self.summary_writer.as_default():
            tf.summary.scalar('Average Reward (100 Episode Moving Average)', self.real_mean_reward, step=step)
            tf.summary.scalar('Average Q', self.q_metric.result(), step=step)
            tf.summary.scalar('Episode Reward', self.real_episode_score, step=step)
            tf.summary.scalar('Loss', self.loss_metric.result(), step=step)
            tf.summary.scalar('Epsilon', self.epsilon, step=step)
            tf.summary.scalar("Total Frames", self.total_step, step=step)
            tf.summary.scalar('Validation Score', self.validation_score, step=step)
        self.loss_metric.reset_states()
        self.q_metric.reset_states()

    def update_history(self, model_path):
        """
        Write 1 episode stats to checkpoint and write policy_network when it crosses interval.
        """
        data = {
            'mean_reward': [self.real_mean_reward],
            'best_mean_reward': [self.real_best_mean_reward],
            'episode_reward': [self.real_episode_score],
            'step': [self.total_step],
            'time': [perf_counter() - self.training_start_time],
            'episode': [self.episode],
            'best_validation': [self.max_validation_score],
        }
        write_from_dict(data, path=model_path + '/history_check_point.json')
        self.save_model(model_path)

    def load_history_from_path(self, path):
        """
        Load previous training session metadata and update agent metrics to go from there.
        """
        # Load policy_network from checkpoint
        self.load_model(path)
        # Load training data from json
        previous_history = pd.read_json(path + self.history_dict_file).to_dict()
        self.real_mean_reward = previous_history['mean_reward'][0]
        self.real_best_mean_reward = previous_history['best_mean_reward'][0]
        history_start_steps = previous_history['step'][0]
        history_start_time = previous_history['time'][0]
        self.episode = previous_history['episode'][0]
        self.max_validation_score = previous_history['best_validation'][0]

        self.total_step = history_start_steps
        self.training_start_time = perf_counter() - history_start_time
        self.last_reset_step = self.total_step = int(history_start_steps)

        for i in range(self.mean_reward_step):
            self.real_mean_reward_buffer.append(self.real_mean_reward)

    def reset_episode_parameters(self):
        """
        Reset the state, episode reward, done
        """
        self.reset_env()
        self.done = False

        self.last_reset_time = perf_counter()
        self.last_reset_step = self.total_step

    def update_training_parameters(self):
        """
        Update progress metrics which consist of last reset step and time used
        for calculation of fps, and update mean and best reward. The policy_network is
        saved if there is a checkpoint path specified.
        """
        self.episode += 1
        self.real_episode_score = self.env.episode_returns
        self.real_mean_reward_buffer.append(self.real_episode_score)

        self.real_mean_reward = np.around(
            np.mean(self.real_mean_reward_buffer), 5
        )

        self.frame_speed = (self.total_step - self.last_reset_step) / (
                perf_counter() - self.last_reset_time
        )

        if self.real_mean_reward > self.real_best_mean_reward and self.episode >= self.mean_reward_step / 2:
            self.display_message(
                f'Best Moving Average Reward Updated: {colored(str(self.real_best_mean_reward), "red")} -> '
                f'{colored(str(self.real_mean_reward), "green")}'
            )
            self.real_best_mean_reward = self.real_mean_reward
            if self.saving_model:
                path = self.saving_path + '/best'
                self.check_and_create_path(path)
                self.update_history(model_path=path)

    def validation(self, epsilon=0, validation_episode=4.0, max_step=8000):
        if self.episode % self.validation_freq != 0 or self.episode <= self.mean_reward_step:
            return
        self.display_message(f'Validation model with {validation_episode} episodes...')
        total_reward = 0.0
        for i in range(validation_episode):
            self.reset_env()
            done = False
            step = 0
            while not done and step < max_step:
                action = self.get_action(tf.constant(self.state), tf.constant(epsilon, tf.float32))
                self.env.step(action)
                done = self.env.was_real_done
                step += 1
            total_reward += self.env.episode_returns

        self.validation_score = total_reward / validation_episode
        if self.validation_score > self.max_validation_score:
            self.display_message(
                f'Best Validation score Updated: {colored(str(self.max_validation_score), "magenta")} -> '
                f'{colored(str(self.validation_score), "yellow")}'
            )
            self.max_validation_score = self.validation_score
            saving_path = self.saving_path + '/valid'
            self.check_and_create_path(saving_path)
            self.update_history(model_path=saving_path)

    def check_episodes(self):
        """
        Check environment done counts to display progress and update metrics.
        """
        if self.done:
            self.update_training_parameters()
            self.validation()
            if self.log_history:
                self.record_tensorboard()
            self.display_learning_state()
            self.reset_episode_parameters()

    def init_training(self, max_steps):
        """
        Initialize training start time & policy_network (self.policy_network / self.target_network)
        Args:
            max_steps: Maximum time total_step, if exceeded, the training will stop.
        """

        self.max_steps = max_steps
        self.env.true_reset()
        self.reset_env()
        self.real_episode_score = 0.0
        self.done = False
        self.last_reset_time = perf_counter()
        self.training_start_time = perf_counter()

        load = False

        path = self.saving_path + '/best'
        if self.saving_model and Path(path + self.history_dict_file).is_file():
            self.display_message(f'Load history from {path + self.history_dict_file}')
            load = True
            self.load_history_from_path(path)
        else:
            self.total_step = 0
        self.fill_buffer(load=load)

    def get_action(self, state, epsilon):
        raise NotImplementedError(
            f'get_action() should be implemented by {self.__class__.__name__} subclasses'
        )

    def at_step_start(self):
        self.total_step += 1
        raise NotImplementedError(
            f'at_step_start() should be implemented by {self.__class__.__name__} subclasses'
        )

    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.env_name, batching and gradient updates.
        """
        raise NotImplementedError(
            f'train_step() should be implemented by {self.__class__.__name__} subclasses'
        )

    def at_step_end(self, render=False):
        if render:
            self.env.render()

    def learn(self, max_steps, target_reward=None, ):
        """
        Common training loop shared by subclasses, monitors training status
        and progress, performs all training total_step, updates metrics, and logs progress.
        Args:
             max_steps: Maximum number of total_step, if reached the training will stop.
             target_reward: The target moving average reward, if reached the training will stop, if null will be ignored
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
            raise NotImplementedError(f'The learn(**kwargs) should '
                                      f'be implemented by {self.__class__.__name__} subclasses')

    def play(
            self,
            model_load_path,
            render=False,
            video_dir=None,
            frame_delay=0.0,
            max_episode=100,
            epsilon=0.0001
    ):
        """
        Play and display a test_env.
        Args:
            model_load_path: The path for loading the policy_network
            video_dir: Path to directory to save the resulting test_env video.
            render: If True, the test_env will be displayed.
            frame_delay: Delay between rendered frames.
            max_episode: Maximum environment episode.
            epsilon: The rate that agent would choose a random action
        Return:
            total_reward: List of reward for each episode
        """
        self.saving_path = model_load_path
        path = self.saving_path + '/valid'
        self.load_model(path)
        episode = 0
        steps = 0
        episode_reward = 0
        total_reward = []

        env = self.env
        state = env.reset()

        if video_dir:
            env = gym.wrappers.Monitor(env, video_dir, force=True)
            env.reset()
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)

        while True:
            if render:
                env.render()
                sleep(frame_delay)

            # Greedy choose
            state = tf.expand_dims(state, axis=0)
            if tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32) < epsilon:
                action = tf.random.uniform((), minval=0, maxval=self.n_actions, dtype=tf.int32)
            else:
                action = self.policy_network.get_optimal_actions(tf.cast(state, tf.float32))

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


class EpsDecayAgent(ABC):

    def __init__(self, eps_schedule=None, ):
        if eps_schedule is None:
            self.eps_schedule = [[1.0, 0.1, 1000000], [0.1, 0.001, 5000000]]
        else:
            self.eps_schedule = eps_schedule
        self.epsilon = None
        self.eps_schedule = np.array(self.eps_schedule)
        self.eps_schedule[:, 2] = np.cumsum(self.eps_schedule[:, 2])
        self.eps_lag = 0

    def update_epsilon(self, total_step):
        if total_step > self.eps_schedule[0, 2] and self.eps_schedule.shape[0] > 1:
            self.eps_schedule = np.delete(self.eps_schedule, 0, 0)
            self.eps_lag = total_step
        max_eps, min_eps, eps_steps = self.eps_schedule[0]
        epsilon = max_eps - min(1, (total_step - self.eps_lag) / (eps_steps - self.eps_lag)) * (
                max_eps - min_eps)
        self.epsilon = np.round(epsilon, 5)
        return epsilon

    def learn(self, **kwargs):
        raise NotImplementedError(f'The learn(self, **kwargs) should '
                                  f'be implemented by {self.__class__.__name__} subclasses')
