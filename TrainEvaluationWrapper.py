from DeepAgent.interfaces import ibaseAgent, ibaseBuffer, ibasePolicy
from gym import Wrapper
import tensorflow as tf


def use_gpu(use=False):
    if use:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def trainWrapper(config, env, buffer, network, agent, train_id):
    use_gpu(config.USE_GPU)

    _env = env(env_name=config.ENV_NAME,
               output_shape=config.IMAGE_SHAPE,
               frame_stack=config.FRAME_STACK,
               train=True,
               crop=config.CROP)

    _buffer = buffer(size=config.BUFFER_SIZE,
                     batch_size=config.BATCH_SIZE)

    _policy = network(conv_layers=config.CONV_LAYERS,
                      dense_layers=None,
                      input_shape=config.IMAGE_SHAPE,
                      frame_stack=config.FRAME_STACK,
                      n_actions=_env.action_space.n,
                      optimizer=config.OPTIMIZER,
                      lr_schedule=config.LEARNING_RATE,
                      one_step_weight=config.ONE_STEP_WEIGHT,
                      n_step_weight=config.N_STEP_WEIGHT,
                      l2_weight=0.0)

    _target = network(conv_layers=config.CONV_LAYERS,
                      dense_layers=None,
                      input_shape=config.IMAGE_SHAPE,
                      frame_stack=config.FRAME_STACK,
                      n_actions=_env.action_space.n,
                      optimizer=config.OPTIMIZER,
                      lr_schedule=config.LEARNING_RATE,
                      one_step_weight=config.ONE_STEP_WEIGHT,
                      n_step_weight=config.N_STEP_WEIGHT,
                      l2_weight=0.0)

    _agent = agent(
        agent_id=train_id,
        env=_env,
        policy_network=_policy,
        target_network=_target,
        buffer=_buffer,
        gamma=config.GAMMA,
        n_step=config.N_STEP,
        warm_up_episode=config.WARM_UP_EPISODE,
        eps_schedule=config.EPS_SCHEDULE,
        target_sync_freq=config.TARGET_SYNC_FREQ,
        saving_model=config.SAVING_MODEL,
        log_history=config.LOG_HISTORY,
    )

    assert isinstance(_agent, ibaseAgent.OffPolicy)
    assert isinstance(_policy, ibasePolicy.BaseNNPolicy)
    assert isinstance(_buffer, ibaseBuffer.BaseBuffer)
    assert isinstance(_env, Wrapper)

    return _agent


def testWrapper(config, agent, env, network, buffer, test_id):
    _env = env(env_name=config.ENV_NAME,
               output_shape=config.IMAGE_SHAPE,
               frame_stack=config.FRAME_STACK,
               crop=config.CROP,
               train=False)
    _buffer = buffer(size=config.TEST_BUFFER_SIZE, batch_size=config.TEST_BATCH_SIZE)

    _policy = network(conv_layers=config.CONV_LAYERS,
                      dense_layers=None,
                      input_shape=config.IMAGE_SHAPE,
                      frame_stack=config.FRAME_STACK,
                      n_actions=_env.action_space.n,
                      optimizer=config.OPTIMIZER,
                      lr_schedule=config.LEARNING_RATE,
                      one_step_weight=1.0,
                      l2_weight=0.0)

    _target = network(conv_layers=config.CONV_LAYERS,
                      dense_layers=None,
                      input_shape=config.IMAGE_SHAPE,
                      frame_stack=config.FRAME_STACK,
                      n_actions=_env.action_space.n,
                      optimizer=config.OPTIMIZER,
                      lr_schedule=config.LEARNING_RATE,
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
