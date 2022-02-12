import random
import abc
import tensorflow as tf
from scipy.stats import truncnorm


class Policy:
    @abc.abstractmethod
    def act(self, **kwargs):
        return NotImplementedError()


# greedy choosing the action
class Greedy(Policy):
    def act(self, q_values):
        return tf.argmax(q_values)


# Epsilon greedy method
class EpsGreedy(Policy):

    def __init__(self, eps):
        self.eps = eps

    def act(self, q_value):
        # when random number greater than ep using greedy else random。
        if random.random() > self.eps:
            return tf.argmax(q_value)
        return random.randrange(len(q_value))


# Gaussian random + greedy choosing
class GaussianEpsGreedy(Policy):

    def __init__(self, eps_mean, eps_std):
        self.eps_mean = eps_mean
        self.eps_std = eps_std

    def act(self, q_value):
        # Construct gaussian distribution to choose
        eps = truncnorm.rvs((0 - self.eps_mean) / self.eps_std, (1 - self.eps_mean) / self.eps_std)
        if random.random() > eps:
            return tf.argmax(q_value)
        return random.randrange(len(q_value))

