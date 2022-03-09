from config import *
from DeepAgent.interfaces import ibaseAgent, ibaseBuffer, ibasePolicy
from gym import Wrapper

import tensorflow as tf

if GPU:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def trainWrapper(env, buffer, network, agent, train_id):
    _env = env(env_name=ENV_NAME,
               output_shape=IMAGE_SHAPE,
               frame_stack=FRAME_STACK,
               train=True,
               crop=CROP)

    _buffer = buffer(size=BUFFER_SIZE,
                     batch_size=BATCH_SIZE)

    _policy = network(conv_layers=CONV_LAYERS,
                      dense_layers=None,
                      input_shape=IMAGE_SHAPE,
                      frame_stack=FRAME_STACK,
                      n_actions=_env.action_space.n,
                      optimizer=OPTIMIZER,
                      lr_schedule=LEARNING_RATE,
                      one_step_weight=ONE_STEP_WEIGHT,
                      l2_weight=0.0)

    _target = network(conv_layers=CONV_LAYERS,
                      dense_layers=None,
                      input_shape=IMAGE_SHAPE,
                      frame_stack=FRAME_STACK,
                      n_actions=_env.action_space.n,
                      optimizer=OPTIMIZER,
                      lr_schedule=LEARNING_RATE,
                      one_step_weight=ONE_STEP_WEIGHT,
                      l2_weight=0.0)

    _agent = agent(
        agent_id=train_id,
        env=_env,
        policy_network=_policy,
        target_network=_target,
        buffer=_buffer,
        gamma=GAMMA,
        warm_up_episode=WARM_UP_EPISODE,
        eps_schedule=EPS_SCHEDULE,
        target_sync_freq=TARGET_SYNC_FREQ,
        saving_model=SAVING_MODEL,
        log_history=LOG_HISTORY,
    )

    assert isinstance(_agent, ibaseAgent.OffPolicy)
    assert isinstance(_policy, ibasePolicy.BaseNNPolicy)
    assert isinstance(_buffer, ibaseBuffer.BaseBuffer)
    assert isinstance(_env, Wrapper)

    return _agent


def testWrapper(agent, env, network, buffer, test_id):
    _env = env(env_name=ENV_NAME,
               output_shape=IMAGE_SHAPE,
               frame_stack=FRAME_STACK,
               crop=CROP,
               train=False)
    _buffer = buffer(size=TEST_BUFFER_SIZE, batch_size=TEST_BATCH_SIZE)

    _policy = network(conv_layers=CONV_LAYERS,
                      dense_layers=None,
                      input_shape=IMAGE_SHAPE,
                      frame_stack=FRAME_STACK,
                      n_actions=_env.action_space.n,
                      optimizer=OPTIMIZER,
                      lr_schedule=LEARNING_RATE,
                      one_step_weight=1.0,
                      l2_weight=0.0)

    _target = network(conv_layers=CONV_LAYERS,
                      dense_layers=None,
                      input_shape=IMAGE_SHAPE,
                      frame_stack=FRAME_STACK,
                      n_actions=_env.action_space.n,
                      optimizer=OPTIMIZER,
                      lr_schedule=LEARNING_RATE,
                      one_step_weight=1.0,
                      l2_weight=0.0)

    _agent = agent(
        agent_id=test_id,
        env=_env,
        policy_network=_policy,
        target_network=_target,
        buffer=_buffer,
    )
    assert isinstance(_agent, ibaseAgent.OffPolicy)
    assert isinstance(_policy, ibasePolicy.BaseNNPolicy)
    assert isinstance(_buffer, ibaseBuffer.BaseBuffer)
    assert isinstance(_env, Wrapper)
    return _agent, _env
