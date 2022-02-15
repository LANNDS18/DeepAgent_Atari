"""The parameter for training models"""

'''Env Parameters'''
ENV_NAME = 'DemonAttack-v0'
IMAGE_SHAPE = (84, 84)
FRAME_STACK = 4

'''Network Parameters '''
LEARNING_RATE = 1e-4

"""Buffer Parameters"""
BUFFER_SIZE = 200000
BATCH_SIZE = 64

'''Agent Parameters'''
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY_STEPS = 100000

'''Learning Parameters'''
TRAINING_STEP = 10000000
TARGET_MEAN_REWARD = 2000

'''Model Paths'''
DQN_PATH = 'models/dqn/'
DDDQN_PATH = 'models/dddqn/'
