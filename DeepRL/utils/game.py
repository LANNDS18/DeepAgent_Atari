import cv2
import gym
from DeepRL.utils.common import process_frame
from collections import deque


class GameEnv(gym.Wrapper):
    """Wrapper for the environment provided by Gym"""

    def __init__(self, env_name, output_shape=(84, 84), frame_stack=1):
        super().__init__(env_name)
        self.env = gym.make(env_name)
        self.output_shape = output_shape
        self.frame_stack = frame_stack
        self.frames = deque([], maxlen=frame_stack)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.frame_stack):
            frame = process_frame(ob, shape=self.output_shape)
            self.frames.append(frame)
        return cv2.merge(self.frames)

    def step(self, action):
        """Performs an action and observes the result
        Arguments:
            action: An integer describe action the agent chose
        Returns:
            next_state: The processed new frame as a result of that action
            reward: The reward for taking that action
            done: Whether the game has ended
            info: other information
        """
        next_state, reward, done, info = self.env.step(action)
        next_state = process_frame(next_state)
        self.frames.append(next_state)
        return cv2.merge(self.frames), reward, done, info

