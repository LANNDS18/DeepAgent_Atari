from keras.optimizer_v2.adam import Adam
from keras.initializers.initializers_v2 import VarianceScaling

"""The parameter for training models"""

'''Env Parameters'''
ENV_NAME = 'DemonAttackNoFrameskip-v4'
FRAME_STACK = 4
IMAGE_SHAPE = (84, 84)
CROP = lambda x: x[20:-20, :]

'''
Network Parameters
'''
CONV_LAYERS = {
    'filters': [32, 64, 64, 1024],
    'kernel_sizes': [8, 4, 3, 7],
    'strides': [4, 2, 1, 1],
    'paddings': ['valid' for _ in range(4)],
    'activations': ['relu' for _ in range(4)],
    'initializers': [VarianceScaling(scale=2.0) for _ in range(4)],
    'names': ['conv_%i' % i for i in range(1, 5)]
}

LEARNING_RATE = [[5e-4, 2.5e-4, 1e5], [2.5e-4, 1e-4, 5e5]]
OPTIMIZER = Adam
ONE_STEP_WEIGHT = 0.5

"""
Buffer Parameters
"""
BUFFER_SIZE = 300000
BATCH_SIZE = 32
WARM_UP_EPISODE = 50

"""
Agent Parameters
"""
GAMMA = 0.99
EPSILON_START = 0.9
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = int(2e6)
TARGET_SYNC_FREQ = 10000
SAVING_MODEL = True
LOG_HISTORY = True

'''
Evaluation Parameters
'''
TEST_BUFFER_SIZE = 10
TEST_BATCH_SIZE = 1
TEST_MAX_EPISODE = 10
