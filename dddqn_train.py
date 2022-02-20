from DeepAgent.networks.dueling import build_dueling_network
from DeepAgent.utils.buffer import ExperienceReplay
from DeepAgent.utils.game import GameEnv
from DeepAgent.agents.double_dqn import DoubleDQNAgent
from config import *


def train_dddqn():
    game = GameEnv(ENV_NAME,
                   output_shape=IMAGE_SHAPE,
                   frame_stack=FRAME_STACK)

    model = build_dueling_network(game.action_space.n,
                                  learning_rate=LEARNING_RATE,
                                  input_shape=IMAGE_SHAPE,
                                  frame_stack=FRAME_STACK)

    buffer = ExperienceReplay(size=BUFFER_SIZE, batch_size=BATCH_SIZE)

    agent = DoubleDQNAgent(
        env=game,
        model=model,
        buffer=buffer,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        model_path=DDDQN_PATH,
        log_history=True,
    )

    agent.fill_buffer()
    agent.learn(max_steps=TRAINING_STEP, target_reward=TARGET_MEAN_REWARD)


if __name__ == '__main__':
    train_dddqn()
