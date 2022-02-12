from DeepRL.agents.DQNAgent import DQNAgent
from DeepRL.networks.DQNNetwork import build_q_network
from DeepRL.utils.ReplayBuffer import ExperienceReplay
from DeepRL.utils.GameWrapper import GameWrapper

import gym


def train_dqn():

    buffer_size = 10000
    env = gym.make('DemonAttack-v0')
    game = GameWrapper(env)
    model = build_q_network(env.action_space.n)
    buffer = ExperienceReplay(size=buffer_size)
    agent = DQNAgent(game, model=model, buffer=buffer)

    agent.fill_buffer()
    agent.fit(target_reward=10000, max_steps=100)


if __name__ == '__main__':
    train_dqn()
