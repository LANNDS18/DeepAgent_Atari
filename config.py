from keras.optimizer_v2.adam import Adam
from keras.initializers.initializers_v2 import VarianceScaling

GPU = False

"""The parameter for training models"""

'''Env Parameters'''
ENV_NAME = 'PongNoFrameskip-v4'
FRAME_STACK = 4
IMAGE_SHAPE = (84, 84)
pong_crop = lambda x: x[25:-5, :]
demon_attack_crop = lambda x: x[20:-20, :]
CROP = pong_crop

'''
Network Parameters
'''
CONV_LAYERS = {
    'filters': [32, 64, 64],
    'kernel_sizes': [8, 4, 3],
    'strides': [4, 2, 1, 1],
    'paddings': ['valid' for _ in range(3)],
    'activations': ['relu' for _ in range(3)],
    'initializers': [VarianceScaling(scale=2.0) for _ in range(3)],
    'names': ['conv_%i' % i for i in range(1, 5)]
}

LEARNING_RATE = [[5e-4, 2.5e-4, 5e5], [2.5e-4, 1e-4, 1e6]]
OPTIMIZER = Adam
ONE_STEP_WEIGHT = 1

"""
Buffer Parameters
"""
BUFFER_SIZE = 300000
BATCH_SIZE = 32
WARM_UP_EPISODE = 5

"""
Agent Parameters
"""
GAMMA = 0.99
EPS_SCHEDULE = [[1, 0.2, 1e5],
                [0.2, 0.1, 1e6],
                [0.1, 0.01, 2e6]]

TARGET_SYNC_FREQ = 10000
SAVING_MODEL = True
LOG_HISTORY = True

'''
Evaluation Parameters
'''
TEST_BUFFER_SIZE = 10
TEST_BATCH_SIZE = 1
TEST_MAX_EPISODE = 10
