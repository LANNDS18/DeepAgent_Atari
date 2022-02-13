from DeepRL.agents.DQNAgent import DQNAgent
from DeepRL.networks.DQNNetwork import build_q_network
from DeepRL.utils.ReplayBuffer import ExperienceReplay
from DeepRL.utils.GameEnv import GameEnv


def train_dqn():
    buffer_size = 10000
    game = GameEnv('DemonAttack-v0')
    model = build_q_network(game.action_space.n)
    buffer = ExperienceReplay(size=buffer_size)
    agent = DQNAgent(game, model=model, buffer=buffer)
    agent.fill_buffer()
    agent.learn(target_reward=1000)


if __name__ == '__main__':
    train_dqn()
