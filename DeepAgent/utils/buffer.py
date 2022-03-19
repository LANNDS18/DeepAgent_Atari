import tensorflow as tf
import numpy as np

from collections import deque
from DeepAgent.interfaces.ibaseBuffer import BaseBuffer, Transition


class ExperienceReplay(BaseBuffer):
    """
    This class manages buffer of agent.
    """

    def __init__(self, size, **kwargs):
        super(ExperienceReplay, self).__init__(size, **kwargs)
        self._buffer = deque(maxlen=size)

    def append(self, *args):
        transition = Transition(*args)
        self._buffer.append(transition)
        self.current_size = len(self._buffer)

    def get_sample_indices(self):
        assert self.current_size > self.n_step
        indices = []
        while len(indices) < self.batch_size:
            index = np.random.randint(low=0, high=self.current_size - self.n_step, dtype=np.int32)
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

        return (tf.stack(states, axis=0), tf.stack(actions, axis=0), tf.stack(rewards, axis=0),
                tf.stack(dones, axis=0), tf.stack(new_states, axis=0))

    def get_n_step_sample(self, indices, gamma=0.99):
        n_step_rewards, n_step_dones, n_step_next_states = [], [], []

        for index in indices:
            item = self._buffer[index]
            total_reward = item.reward
            next_state = item.new_state
            next_done = item.done

            for i in range(self.n_step):
                next_item = self._buffer[index + i]
                if next_done is True:
                    break
                total_reward += (gamma ** i) * next_item.reward
                next_done = next_item.done
                next_state = next_item.new_state

            n_step_rewards.append(tf.constant(total_reward, tf.float32))
            n_step_dones.append(tf.constant(next_done, tf.bool))
            n_step_next_states.append(tf.constant(next_state, tf.float32))

        return tf.stack(n_step_rewards, axis=0), tf.stack(n_step_dones, axis=0), tf.stack(n_step_next_states, axis=0)

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, i):
        return self._buffer[i]

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.size)


class PrioritizedExperienceReplay(ExperienceReplay):
    """
    This Prioritized Experience Replay Memory class
    """

    def __init__(self, size, prob_alpha=0.6, epsilon=1e-3, **kwargs):
        """
        Initialize replay buffer.
        Args:
            size: Buffer maximum size.
            **kwargs: kwargs passed to BaseBuffer.
            prob_alpha: The probability of being assigned the priority
            epsilon: The small priority for new transition appending into buffer
        """
        super(PrioritizedExperienceReplay, self).__init__(size, **kwargs)
        self.size = int(size)
        self._buffer = []
        self.priorities = np.array([])
        self.prob_alpha = prob_alpha
        self.epsilon = epsilon

    def append(self, *args):
        """
        Append experience and auto-allocate to temp buffer / main buffer(self)
        Args:
            *args: Items to store
        """
        transition = Transition(*args)
        if self.current_size < self.size:
            self._buffer.append(transition)
            self.priorities = np.append(
                self.priorities,
                self.epsilon if self.priorities.size == 0 else np.amax(self.priorities))
        else:
            idx = np.argmin(self.priorities)
            self._buffer[idx] = transition
            self.priorities[idx] = np.amax(self.priorities)
        self.current_size = len(self._buffer)

    def get_sample_indices(self):
        probs = np.copy(self.priorities[:-self.n_step])
        probs /= probs.sum()
        indices = np.random.choice(self.current_size - self.n_step,
                                   self.batch_size,
                                   p=probs,
                                   replace=False)
        return indices

    def update_priorities(self, indices, errors):
        """
        Update priorities for chosen samples
        Args:
            errors: abs of Y and Y_Predict
            indices: The index of the element
        """
        self.priorities[indices] = errors ** self.prob_alpha + self.epsilon

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, i):
        return self._buffer[i]

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.size)
