from collections import deque
from DeepRL.interfaces.IBaseBuffer import IBaseBuffer, Transition

import tensorflow as tf
import numpy as np


class ExperienceReplay(IBaseBuffer):
    """
    This class manages buffer of agent.
    """

    def __init__(self, size, **kwargs):
        super().__init__(size, **kwargs)
        self.size = int(size)
        self._buffer = deque(maxlen=size)
        self.current_size = 0

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, i):
        return self._buffer[i]

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.size)

    def append(self, *args):
        transition = Transition(*args)
        self._buffer.append(transition)
        self.current_size = len(self._buffer)

    def get_sample_indices(self):
        indices = []
        while len(indices) < self.batch_size:
            index = np.random.randint(low=0, high=self.size, dtype=np.int32)
            indices.append(index)
        return indices

    def get_sample(self, indices):
        states, actions, rewards, dones, new_states = [], [], [], [], []

        for index in indices:
            item = self._buffer[index]
            states.append(tf.constant(item.state, tf.float32))
            actions.append(tf.constant(item.action, tf.int32))
            rewards.append(tf.constant(item.reward, tf.float32))
            dones.append(tf.constant(item.done, tf.bool))
            new_states.append(tf.constant(item.new_state, tf.float32))

        return tf.stack(states, axis=0), tf.stack(actions, axis=0), tf.stack(rewards, axis=0), tf.stack(dones, axis=0), tf.stack(new_states, axis=0)


class PrioritizedExperienceReplay(IBaseBuffer):
    """
    This Prioritized Experience Replay Memory class
    """

    def __init__(self, size, prob_alpha=0.6, **kwargs):
        """
        Initialize replay buffer.
        Args:
            size: Buffer maximum size.
            **kwargs: kwargs passed to BaseBuffer.
            prob_alpha: The probability of being assigned the priority
        """
        super(PrioritizedExperienceReplay, self).__init__(size, **kwargs)
        self.main_buffer = []
        self.index_buffer = []
        self.priorities = np.array([])
        self.size = size
        self.epsilon = 1e-3
        self.prob_alpha = prob_alpha

    def append(self, *args):
        """
        Append experience and auto-allocate to temp buffer / main buffer(self)
        Args:
            *args: Items to store
        """
        raise NotImplementedError()
        if len(self.main_buffer) < self.size:
            self.main_buffer.append(args)
            self.priorities = np.append(
                self.priorities,
                self.epsilon
                if self.priorities.size == 0
                else self.priorities.max())

        else:
            idx = np.argmin(self.priorities)
            self.main_buffer[idx] = args
            self.priorities[idx] = self.priorities.max()

        self.current_size = len(self.main_buffer)

    def get_sample(self, indices):
        """
        Sample from stored experience based on priorities.
        Returns:
            Same number of args passed to append, having self.batch_size as first shape.
        """
        raise NotImplementedError()
        probs = self.priorities ** self.prob_alpha
        probs /= probs.sum()
        self.index_buffer = np.random.choice(len(self.main_buffer),
                                             self.batch_size,
                                             p=probs,
                                             replace=False)

        traces = [self.main_buffer[idx] for idx in self.index_buffer]
        return [np.array(item) for item in zip(*traces)]

    def update_priorities(self, abs_errors):
        """
        Update priorities for chosen samples
        Args:
            abs_errors: abs of Y and Y_Predict
        """
        raise NotImplementedError()
        self.priorities[self.index_buffer] = abs_errors + self.epsilon

    def __len__(self):
        return self.current_size
