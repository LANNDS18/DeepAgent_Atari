import tensorflow as tf
from abc import ABC


class BaseConfig(ABC):
    """
    The base config can be used by train_evaluation_wrapper in DeepAgent.utils, please override the attributes in
    implementation.
    """

    USE_GPU = False

    '''Training parameters'''
    MAX_STEP = 1e7
    RENDER = False
    TARGET_REWARD = None

    '''Env Parameters'''
    FRAME_STACK = 4
    IMAGE_SHAPE = (84, 84)
    ENV_NAME = None
    CROP = None
    REWARD_PROCESSOR = lambda x, y, z: x

    '''
    Learning rate decay policy
    format: [[start_lr, update_lr, total_update], [..., ..., ...]]
    total_update = total_step / update_freq
    5e5 * 4 = 2e6
    '''
    LEARNING_RATE = [[5e-4, 2.5e-4, 5e5], [2.5e-4, 1e-4, 1e6]]

    """
    Agent Parameters
    """
    GAMMA = 0.99
    N_STEP = 10
    ONE_STEP_WEIGHT = 0.5
    N_STEP_WEIGHT = 0.5
    EPS_SCHEDULE = [[1, 0.2, 1e5],
                    [0.2, 0.1, 1e6]]

    TARGET_SYNC_FREQ = 10000
    SAVING_MODEL = False
    LOG_HISTORY = False

    '''
    Network Parameters
    '''
    CONV_LAYERS = {
        'filters': [32, 64, 64],
        'kernel_sizes': [8, 4, 3],
        'strides': [4, 2, 1],
        'paddings': ['valid' for _ in range(3)],
        'activations': ['relu' for _ in range(3)],
        'initializers': [tf.initializers.VarianceScaling(scale=2.) for _ in range(3)],
        'names': ['conv_%i' % i for i in range(1, 4)]
    }

    OPTIMIZER = tf.keras.optimizers.Adam

    """
    Buffer Parameters
    """
    BUFFER_SIZE = 600000
    BATCH_SIZE = 32
    BUFFER_FILL_SIZE = 50000

    '''
    Evaluation Parameters
    '''
    TEST_BUFFER_SIZE = 10
    TEST_BATCH_SIZE = 1
    TEST_MAX_EPISODE = 10
    VIDEO_DIR = None
