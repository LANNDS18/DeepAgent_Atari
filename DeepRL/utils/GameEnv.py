import gym
from DeepRL.utils.Common import process_frame


class GameEnv(gym.Wrapper):
    """Wrapper for the environment provided by Gym"""

    def __init__(self, env_name, output_shape=(84, 84)):
        super().__init__(env_name)
        self.env = gym.make(env_name)
        self.output_shape = output_shape

    def reset(self):
        return process_frame(self.env.reset(), shape=self.output_shape)

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
        return next_state, reward, done, info
