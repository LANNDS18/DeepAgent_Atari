"""The parameter for training models"""

'''Env Parameters'''
ENV_NAME = 'DemonAttack-v0'
IMAGE_SHAPE = (84, 84)
TRAINING_STEP = int(1e9)

'''Network Parameters '''
LEARNING_RATE = 1e-3

"""Buffer Parameters"""
BUFFER_SIZE = 200000
BATCH_SIZE = 64

'''Agent Parameters'''
GAMMA = 0.95
EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 150000

'''Model Paths'''
DQN_PATH = 'models/dqn/'
DDDQN_PATH = 'models/dddqn/'
