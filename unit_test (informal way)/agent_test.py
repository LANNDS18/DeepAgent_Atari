from pathlib import Path
from time import perf_counter

import pandas as pd

from DeepAgent.networks.dueling import build_dueling_network
from DeepAgent.utils.buffer import ExperienceReplay, PrioritizedExperienceReplay
from DeepAgent.utils.game import GameEnv
from DeepAgent.agents.DoubleDQN import DoubleDQNAgent
from DeepAgent.agents.DoublePER import D3NPERAgent
from config import *


def test_d3qn_with_batch_1():
    game = GameEnv(ENV_NAME, output_shape=IMAGE_SHAPE)

    model = build_dueling_network(game.action_space.n,
                                  learning_rate=LEARNING_RATE,
                                  input_shape=IMAGE_SHAPE)

    buffer = ExperienceReplay(size=10, batch_size=1)

    agent = DoubleDQNAgent(env=game,
                           model=model,
                           buffer=buffer,
                           gamma=GAMMA,
                           epsilon_start=EPSILON_START,
                           epsilon_end=EPSILON_END,
                           epsilon_decay_steps=EPSILON_DECAY_STEPS, )

    agent.fill_buffer()
    agent.learn(max_steps=100)


def test_loading():
    total_rewards = []
    path = DDDQN_PATH + 'history_check_point.json'
    my_file = Path(path)
    if Path(my_file).is_file():
        previous_history = pd.read_json(my_file)
        mean_reward = previous_history['mean_reward'][0]
        best_reward = previous_history['best_mean_reward'][0]
        history_start_steps = previous_history['step'][0]
        history_start_time = previous_history['time'][0]
        training_start_time = perf_counter() - history_start_time
        last_reset_step = int(history_start_steps)
        total_rewards.append(previous_history['episode_reward'][0])
        terminal_episodes = previous_history['episode'][0]
        print(mean_reward, best_reward, training_start_time, last_reset_step, terminal_episodes)


def test_per_d3n_with_wrong_buffer():
    game = GameEnv(env_name=ENV_NAME,
                   output_shape=IMAGE_SHAPE,
                   frame_stack=FRAME_STACK)

    buffer = ExperienceReplay(size=BUFFER_SIZE,
                              batch_size=BATCH_SIZE)

    dqn_network = build_dueling_network(
        n_actions=game.action_space.n,
        learning_rate=LEARNING_RATE,
        input_shape=IMAGE_SHAPE,
        frame_stack=FRAME_STACK
    )

    agent = D3NPERAgent(
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

    agent.fill_buffer()
    agent.learn(max_steps=TRAINING_STEP)


def test_per_d3n():
    game = GameEnv(env_name=ENV_NAME,
                   output_shape=IMAGE_SHAPE,
                   frame_stack=FRAME_STACK)

    buffer = PrioritizedExperienceReplay(size=20000,
                                         batch_size=32)

    dqn_network = build_dueling_network(
        n_actions=game.action_space.n,
        learning_rate=LEARNING_RATE,
        input_shape=IMAGE_SHAPE,
        frame_stack=FRAME_STACK
    )

    agent = D3NPERAgent(
        agent_id='D3N_PER_test',
        env=game,
        model=dqn_network,
        buffer=buffer,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        saving_model=False,
        log_history=False,
        model_save_interval=MODEL_SAVE_INTERVAL
    )

    agent.fill_buffer()
    agent.learn(max_steps=TRAINING_STEP)


if __name__ == '__main__':
    test_per_d3n()
