import tensorflow as tf


def use_gpu(use=False):
    if use:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def TrainWrapper(config, env, buffer, policy, agent, train_id):
    """
    The wrapper can be used to read and pass configuration to specific environment, buffer, policy, and agent.

    Return:
        _agent: an OffPolicy agent that has been init, and call learn() to train the NN of this agent.
    """
    use_gpu(config.USE_GPU)

    _env = env(
        env_name=config.ENV_NAME,
        output_shape=config.IMAGE_SHAPE,
        frame_stack=config.FRAME_STACK,
        train=True,
        crop=config.CROP,
        reward_processor=config.REWARD_PROCESSOR
    )

    _buffer = buffer(
        size=config.BUFFER_SIZE,
        batch_size=config.BATCH_SIZE,
        n_step=config.N_STEP,
    )

    _main = policy(
        conv_layers=config.CONV_LAYERS,
        dense_layers=None,
        input_shape=config.IMAGE_SHAPE,
        frame_stack=config.FRAME_STACK,
        n_actions=_env.action_space.n,
        optimizer=config.OPTIMIZER,
        lr_schedule=config.LEARNING_RATE,
        one_step_weight=config.ONE_STEP_WEIGHT,
        n_step_weight=config.N_STEP_WEIGHT,
        l2_weight=0.0
    )

    _target = policy(
        conv_layers=config.CONV_LAYERS,
        dense_layers=None,
        input_shape=config.IMAGE_SHAPE,
        frame_stack=config.FRAME_STACK,
        n_actions=_env.action_space.n,
        optimizer=config.OPTIMIZER,
        lr_schedule=config.LEARNING_RATE,
        one_step_weight=config.ONE_STEP_WEIGHT,
        n_step_weight=config.N_STEP_WEIGHT,
        l2_weight=0.0,
        quiet=True,
    )

    _agent = agent(
        agent_id=train_id,
        env=_env,
        policy_network=_main,
        target_network=_target,
        buffer=_buffer,
        gamma=config.GAMMA,
        buffer_fill_size=config.BUFFER_FILL_SIZE,
        eps_schedule=config.EPS_SCHEDULE,
        target_sync_freq=config.TARGET_SYNC_FREQ,
        saving_model=config.SAVING_MODEL,
        log_history=config.LOG_HISTORY,
    )
    return _agent


def EnvTestWrapper(config, env):
    _env = env(
        env_name=config.ENV_NAME,
        output_shape=config.IMAGE_SHAPE,
        frame_stack=config.FRAME_STACK,
        crop=config.CROP,
        train=False
    )
    return _env


def PolicyTestWrapper(config, policy, _env):
    _policy = policy(
        conv_layers=config.CONV_LAYERS,
        dense_layers=None,
        input_shape=config.IMAGE_SHAPE,
        frame_stack=config.FRAME_STACK,
        n_actions=_env.action_space.n,
        optimizer=config.OPTIMIZER,
        lr_schedule=config.LEARNING_RATE,
        one_step_weight=1.0,
        l2_weight=0.0
    )
    return _policy


def VisualizationWrapper(config, env, policy):
    use_gpu(config.USE_GPU)

    _env = EnvTestWrapper(config, env)
    _policy = PolicyTestWrapper(config, policy, _env)
    return _env, _policy


def TestWrapper(config, agent, env, policy, buffer):
    use_gpu(config.USE_GPU)

    _env = EnvTestWrapper(config, env)
    _policy = PolicyTestWrapper(config, policy, _env)

    _buffer = buffer(
        size=config.TEST_BUFFER_SIZE,
        batch_size=config.TEST_BATCH_SIZE,
    )

    _agent = agent(
        env=_env,
        policy_network=_policy,
        target_network=_policy,
        buffer=_buffer,
    )
    return _agent
