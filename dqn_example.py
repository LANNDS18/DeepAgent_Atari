from DeepRL.agents.dqn import DQNAgent
from DeepRL.networks.dqn import build_dqn_network
from DeepRL.utils.buffer import ExperienceReplay
from DeepRL.utils.game import GameEnv
from config import *


def train_dqn():
    game = GameEnv('DemonAttack-v0')
    model = build_dqn_network(game.action_space.n, learning_rate=LEARNING_RATE)
    buffer = ExperienceReplay(size=BUFFER_SIZE, batch_size=BATCH_SIZE)
    agent = DQNAgent(game, model=model, buffer=buffer)
    agent.fill_buffer()
    agent.learn(target_reward=1000)


if __name__ == '__main__':
    train_dqn()
