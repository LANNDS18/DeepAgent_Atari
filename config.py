from abc import ABC
from keras.optimizer_v2.adam import Adam
from keras.initializers.initializers_v2 import VarianceScaling
from DeepAgent.utils.common import pong_crop, demon_attack_crop


class BaseConfig(ABC):
    USE_GPU = False

    '''Training parameters'''
    MAX_STEP = 1e7
    TARGET_REWARD = None

    '''Env Parameters'''
    FRAME_STACK = 4
    IMAGE_SHAPE = (84, 84)
    ENV_NAME = None
    CROP = None

    '''Learning rate decay policy'''
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
    SAVING_MODEL = True
    LOG_HISTORY = True

    '''
    Network Parameters
    '''
    CONV_LAYERS = {
        'filters': [32, 64, 64],
        'kernel_sizes': [8, 4, 3],
        'strides': [4, 2, 1],
        'paddings': ['valid' for _ in range(3)],
        'activations': ['relu' for _ in range(3)],
        'initializers': [VarianceScaling(scale=2.0) for _ in range(3)],
        'names': ['conv_%i' % i for i in range(1, 4)]
    }

    OPTIMIZER = Adam

    """
    Buffer Parameters
    """
    BUFFER_SIZE = 300000
    BATCH_SIZE = 32
    WARM_UP_EPISODE = 100

    '''
    Evaluation Parameters
    '''
    TEST_BUFFER_SIZE = 10
    TEST_BATCH_SIZE = 1
    TEST_MAX_EPISODE = 10


class PongConfig(BaseConfig):
    USE_GPU = False

    TARGET_REWARD = 20

    ENV_NAME = 'PongNoFrameskip-v4'
    CROP = pong_crop

    LEARNING_RATE = [[5e-4, 2.5e-4, 5e5], [2.5e-4, 1e-4, 1e6]]

    GAMMA = 0.99
    N_STEP = 10
    ONE_STEP_WEIGHT = 0.5
    N_STEP_WEIGHT = 0.5
    EPS_SCHEDULE = [[1, 0.2, 1e5],
                    [0.2, 0.1, 1e6],
                    [0.1, 0.01, 2e6]]

    TARGET_SYNC_FREQ = 10000


class DemonAttackConfig(BaseConfig):
    USE_GPU = False

    ENV_NAME = 'DemonAttackNoFrameskip-v4'
    CROP = demon_attack_crop

    LEARNING_RATE = [[5e-4, 2.5e-4, 1e6], [2.5e-4, 1e-4, 2e6]]

    GAMMA = 0.99
    N_STEP = 10
    ONE_STEP_WEIGHT = 0.5
    N_STEP_WEIGHT = 0.5
    EPS_SCHEDULE = [[1, 0.2, 1e5],
                    [0.2, 0.1, 1e6],
                    [0.1, 0.01, 2e6]]

    TARGET_SYNC_FREQ = 10000


