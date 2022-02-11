from DeepRL.DQN.DQNAgent import DQNAgent
from DeepRL.DQN.DQNNetwork import build_q_network
from DeepRL.ReplayBuffer import ExperienceReplay
from DeepRL.GameWrapper import GameWrapper

import gym


def train_dqn():

    buffer_size = 100000
    env = gym.make('DemonAttack-v0')
    game = GameWrapper(env)
    model = build_q_network(env.action_space.n)  # create_dqn_model(32)
    buffer = ExperienceReplay(size=buffer_size)
    agent = DQNAgent(game, model=model, buffer=buffer)

    agent.fill_buffer()
    agent.fit(target_reward=10000, max_steps=19999999)


if __name__ == '__main__':
    train_dqn()
