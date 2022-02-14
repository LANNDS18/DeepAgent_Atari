import random
import abc
import tensorflow as tf
from scipy.stats import truncnorm


class Policy:
    @abc.abstractmethod
    def act(self, **kwargs):
        return NotImplementedError()

    def update_epsilon(self, current_step):
        pass


# greedy choosing the action
class Greedy(Policy):
    def act(self, q_values, n_actions):
        return tf.argmax(q_values)


# Epsilon greedy method
class EpsGreedy(Policy):

    def __init__(self, eps):
        self.eps = eps

    def act(self, q_value, n_actions):
        # when random number greater than ep using greedy else random。
        if random.random() > self.eps:
            return tf.argmax(q_value)
        return random.randrange(n_actions)


# Gaussian random + greedy choosing
class GaussianEpsGreedy(Policy):

    def __init__(self, eps_mean, eps_std):
        self.eps_mean = eps_mean
        self.eps_std = eps_std

    def act(self, q_value, n_actions):
        # Construct gaussian distribution to choose
        eps = truncnorm.rvs((0 - self.eps_mean) / self.eps_std, (1 - self.eps_mean) / self.eps_std)
        if random.random() > eps:
            return tf.argmax(q_value)
        return random.randrange(n_actions)


class EpsDecay(Policy):

    def __init__(self, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay_steps=150000):
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

    def act(self, q_value, n_actions):
        # when random number greater than ep using greedy else random。
        if random.random() > self.epsilon:
            return tf.argmax(q_value)
        return random.randrange(n_actions)

    def update_epsilon(self, current_step):
        """
        Decrement epsilon which aims to gradually reduce randomization.
        """
        self.epsilon = max(
            self.epsilon_end, self.epsilon_start - current_step / self.epsilon_decay_steps
        )
