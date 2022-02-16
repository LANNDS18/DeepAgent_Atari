from DeepRL.networks.cnn import build_dqn_network
from DeepRL.utils.buffer import ExperienceReplay
from DeepRL.utils.game import GameEnv
from DeepRL.agents.dqn import DQNAgent
from config import *


def train_dqn():
    game = GameEnv(ENV_NAME,
                   output_shape=IMAGE_SHAPE,
                   frame_stack=FRAME_STACK)

    model = build_dqn_network(n_actions=game.action_space.n,
                              input_shape=IMAGE_SHAPE,
                              frame_stack=FRAME_STACK)

    buffer = ExperienceReplay(size=BUFFER_SIZE, batch_size=BATCH_SIZE)

    agent = DQNAgent(
        env=game,
        model=model,
        buffer=buffer,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        model_path=DQN_PATH,
        log_history=True,
    )

    agent.fill_buffer()
    agent.learn(max_steps=TRAINING_STEP, target_reward=TARGET_MEAN_REWARD)


if __name__ == '__main__':
    train_dqn()
