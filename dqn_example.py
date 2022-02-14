from DeepRL.networks.dueling import build_dueling_network
from DeepRL.utils.buffer import ExperienceReplay
from DeepRL.utils.game import GameEnv
from DeepRL.agents.double_dqn import DoubleDQNAgent
from config import *


def train_dqn():

    game = GameEnv(ENV_NAME, output_shape=IMAGE_SHAPE)
    model = build_dueling_network(game.action_space.n, learning_rate=LEARNING_RATE, input_shape=IMAGE_SHAPE)
    buffer = ExperienceReplay(size=BUFFER_SIZE, batch_size=BATCH_SIZE)

    agent = DoubleDQNAgent(env=game,
                           model=model,
                           buffer=buffer,
                           gamma=GAMMA,
                           model_path=DDDQN_PATH,
                           log_history=True
                           )
    agent.fill_buffer()
    agent.learn(max_steps=10000)


if __name__ == '__main__':
    train_dqn()
