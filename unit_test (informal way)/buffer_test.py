from DeepAgent.utils.buffer import PrioritizedExperienceReplay, ExperienceReplay
from DeepAgent.utils.game import GameEnv
from config import *


def test_fill_buffer(buffer, env):
    state = env.reset()
    while buffer.current_size < buffer.initial_size:
        action = env.action_space.sample()
        new_state, reward, done, _ = env.step(action)
        buffer.append(state, action, reward, done, new_state)
        state = new_state
        if done:
            state = env.reset()
    # fill one more
    action = env.action_space.sample()
    new_state, reward, done, _ = env.step(action)
    buffer.append(state, action, reward, done, new_state)
    env.reset()
    return buffer


def test_per_buffer():
    buffer = PrioritizedExperienceReplay(size=2, batch_size=1)
    game = GameEnv(ENV_NAME, output_shape=IMAGE_SHAPE)
    buffer = test_fill_buffer(buffer, game)
    a_id = buffer.get_sample_indices()
    a = buffer.get_sample(a_id)
    print(a)

def test_buffer_three_batch_size():
    buffer = ExperienceReplay(size=8, batch_size=3)
    game = GameEnv(ENV_NAME, output_shape=IMAGE_SHAPE)
    buffer = test_fill_buffer(buffer, game)
    a_id = buffer.get_sample_indices()
    a = buffer.get_sample(a_id)
    print(a)

def test_buffer_one_batch_size():
    buffer = ExperienceReplay(size=8, batch_size=1)
    game = GameEnv(ENV_NAME, output_shape=IMAGE_SHAPE)
    buffer = test_fill_buffer(buffer, game)
    a_id = buffer.get_sample_indices()
    a = buffer.get_sample(a_id)
    print(a)


if __name__ == '__main__':
    test_buffer_three_batch_size()