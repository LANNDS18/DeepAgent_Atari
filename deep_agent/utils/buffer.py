from collections import deque
from deep_agent.interfaces.IBaseBuffer import IBaseBuffer
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
        Returns:
            None
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
        if self.batch_size > 1:
            return [np.array(item) for item in zip(*memories)]
        return memories[0]

    def __len__(self):
        return self.current_size


class PrioritizedExperienceReplay(IBaseBuffer):
    """
    初始化参数，包括capacity是存储器的存储容量,steps存储记录的步数,exclude_boundaries是标记是否存储边界值,prob_alpha 概率系数,
    traces存储数组以及trances_index索引数组,priorities优先级数组
    """

    def __init__(self, size, **kwargs):
        """
        Initialize replay buffer.
        Args:
            size: Buffer maximum size.
            **kwargs: kwargs passed to BaseBuffer.
        """
        super(PrioritizedExperienceReplay, self).__init__(size, **kwargs)
        self.main_buffer = deque(maxlen=size)
        self.temp_buffer = []

    def append(self, *args):
        """
        Append experience and auto-allocate to temp buffer / main buffer(self)
        Args:
            *args: Items to store
        Returns:
            None
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

        # 将优先级数组进行归一化，将优先级转换为概率分布，然后使用随机选择函数依照优先级概率分布确定取出记录的索引，最后依据索引从存储器中取出记录并拆解后返回。
        probs = self.priorities ** self.prob_alpha

        probs /= probs.sum()

        self.traces_indexes = np.random.choice(len(self.traces), self.batch_size, p=probs, replace=False)

        traces = [self.traces[idx] for idx in self.traces_indexes]

        return unpack(traces)

        if self.batch_size > 1:
            return [np.array(item) for item in zip(*memories)]
        return memories[0]

    def __len__(self):
        return self.current_size

    def __init__(self, capacity, size, steps=1, exclude_boundaries=False, prob_alpha=0.6, eps=1e-3):

        super().__init__(size)
        self.traces = []

        self.priorities = np.array([])

        self.buffer = []

        self.capacity = capacity

        self.steps = steps
        self.exclude_boundaries = exclude_boundaries

        self.prob_alpha = prob_alpha

        self.traces_indexes = []

        self.eps = eps

    def append(self, transition):

        self.buffer.append(transition)

        # 达到一定的记录步数之后，且没有达到最大存储规模，将存储buffer存入到trances中去，并新增优先级到priorities中去。

        if len(self.buffer) < self.steps:
            return

        if len(self.traces) < self.capacity:

            self.traces.append(tuple(self.buffer))
            # 如果优先级数组内为空，则使用默认值EPS，否则取priorities的最大值
            self.priorities = np.append(self.priorities,
                                        self.eps if self.priorities.size == 0 else self.priorities.max())
        else:
            # 如果存储器达到最大存储规模上线了，则使用当前记录替换掉存储器中优先级最小的记录,并将原本记录的优先级更新成最高优先级。
            idx = np.argmin(self.priorities)

            self.traces[idx] = tuple(self.buffer)

            self.priorities[idx] = self.priorities.max()

        # 如果不包含边界数据且记录中的下一个状态不存在，则将buffer清空

        if self.exclude_boundaries and transition.next_state is None:
            self.buffer = []

            return
        self.buffer = self.buffer[1:]

    def last_traces_idxs(self):
        return self.traces_indexes.copy()

    # 定义优先级更新函数，为了防止优先级为0，因此需要加上EPS
    def update_priorities(self, trace_idxs, new_priorities):

        self.priorities[trace_idxs] = new_priorities + self.eps

