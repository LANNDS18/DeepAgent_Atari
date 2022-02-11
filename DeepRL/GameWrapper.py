import gym
from DeepRL.Common import process_frame


class GameWrapper(gym.Wrapper):
    """Wrapper for the environment provided by Gym"""

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    @staticmethod
    def process_reward(reward):
        if reward > 0:
            reward = 1
        if reward == 0:
            reward = -0.0005
        if reward < 0:
            reward = -1
        return reward

    def reset(self):
        return process_frame(self.env.reset())

    def step(self, action, render_mode='human'):
        """Performs an action and observes the result
        Arguments:
            action: An integer describe action the agent chose
            render_mode: None doesn't render anything, 'human' renders the screen in a new window
        Returns:
            next_state: The processed new frame as a result of that action
            reward: The reward for taking that action
            done: Whether the game has ended
            info: other information
        """
        next_state, reward, done, info = self.env.step(action)
        next_state = process_frame(next_state)
        if render_mode == 'human':
            self.env.render()

        return next_state, reward, done, info
