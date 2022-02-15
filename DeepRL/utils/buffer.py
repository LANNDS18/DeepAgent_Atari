from collections import deque
from DeepRL.interfaces.IBaseBuffer import IBaseBuffer
import random
import numpy as np


class ExperienceReplay(IBaseBuffer):
    """
    deque-based replay buffer that holds state transitions
    """

    def __init__(self, size, **kwargs):
        """
        Initialize replay buffer.
        Args:
            size: Buffer maximum size.
            **kwargs: kwargs passed to BaseBuffer.
        """
        super(ExperienceReplay, self).__init__(size, **kwargs)
        self.main_buffer = deque(maxlen=size)
        self.temp_buffer = []

    def append(self, *args):
        """
        Append experience and auto-allocate to temp buffer / main buffer(self)
        Args:
            *args: Items to store
        """
        self.main_buffer.append(args)
        self.current_size = len(self.main_buffer)

    def get_sample(self):
        """
        Sample from stored experience.
        Returns:
            Same number of args passed to append, having self.batch_size as
            first shape.
        """
        memories = random.sample(self.main_buffer, self.batch_size)
        return [np.array(item) for item in zip(*memories)]

    def __len__(self):
        return self.current_size


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

    def get_sample(self):
        """
        Sample from stored experience based on priorities.
        Returns:
            Same number of args passed to append, having self.batch_size as first shape.
        """
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
        self.priorities[self.index_buffer] = abs_errors + self.epsilon

    def __len__(self):
        return self.current_size
