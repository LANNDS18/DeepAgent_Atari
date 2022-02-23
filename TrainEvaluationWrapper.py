from config import *
from DeepAgent.interfaces import IBaseAgent, IBaseBuffer
from gym import Wrapper

import tensorflow as tf


def trainWrapper(env, buffer, network, agent, train_id):
    _env = env(env_name=ENV_NAME,
               output_shape=IMAGE_SHAPE,
               frame_stack=FRAME_STACK,
               train=True)

    _buffer = buffer(size=BUFFER_SIZE,
                     batch_size=BATCH_SIZE)

    _network = network().build(
        n_actions=_env.action_space.n,
        learning_rate=LEARNING_RATE,
        input_shape=IMAGE_SHAPE,
        frame_stack=FRAME_STACK
    )

    _agent = agent(
        agent_id=train_id,
        env=_env,
        model=_network,
        buffer=_buffer,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        saving_model=SAVING_MODEL,
        log_history=LOG_HISTORY,
        model_save_interval=MODEL_SAVE_INTERVAL
    )

    assert isinstance(_agent, IBaseAgent.BaseAgent)
    assert isinstance(_network, tf.keras.Model)
    assert isinstance(_buffer, IBaseBuffer.BaseBuffer)
    assert isinstance(_env, Wrapper)

    _agent.fill_buffer(fill_size=FILL_BUFFER_SIZE)
    return _agent


def testWrapper(agent, env, network, buffer, test_id):
    _env = env(env_name=ENV_NAME, output_shape=IMAGE_SHAPE, frame_stack=FRAME_STACK, train=False)
    _buffer = buffer(size=TEST_BUFFER_SIZE, batch_size=TEST_BATCH_SIZE)
    _network = network().build(
        n_actions=_env.action_space.n,
        learning_rate=LEARNING_RATE,
        input_shape=IMAGE_SHAPE,
        frame_stack=FRAME_STACK
    )
    _agent = agent(
        agent_id=test_id,
        env=_env,
        model=_network,
        buffer=_buffer,
    )
    assert isinstance(_agent, IBaseAgent.BaseAgent)
    assert isinstance(_network, tf.keras.Model)
    assert isinstance(_buffer, IBaseBuffer.BaseBuffer)
    assert isinstance(_env, Wrapper)
    return _agent, _env
