from DeepAgent.networks.cnn import build_dqn_network
from DeepAgent.agents.DQN import DQNAgent
from DeepAgent.utils.buffer import ExperienceReplay
from DeepAgent.utils.game import GameEnv
from config import *


def train_dqn():
    game = GameEnv(env_name=ENV_NAME,
                   output_shape=IMAGE_SHAPE,
                   frame_stack=FRAME_STACK,
                   train=True)

    buffer = ExperienceReplay(size=BUFFER_SIZE,
                              batch_size=BATCH_SIZE)

    dqn_network = build_dqn_network(
        n_actions=game.action_space.n,
        learning_rate=LEARNING_RATE,
        input_shape=IMAGE_SHAPE,
        frame_stack=FRAME_STACK
    )

    agent = DQNAgent(
        agent_id='DQN',
        env=game,
        model=dqn_network,
        buffer=buffer,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        saving_model=True,
        log_history=True,
        model_save_interval=MODEL_SAVE_INTERVAL
    )

    agent.fill_buffer(fill_size=FILL_BUFFER_SIZE)
    agent.learn(max_steps=TRAINING_STEP)


if __name__ == '__main__':
    train_dqn()
