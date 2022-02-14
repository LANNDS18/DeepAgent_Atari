"""The parameter for training models"""

'''Common Parameters'''
ENV_NAME = 'DemonAttack-v0'
IMAGE_SHAPE = (84, 84)

'''Network Parameters '''
LEARNING_RATE = 1e-3

"""Buffer Parameters"""
BUFFER_SIZE = 10000
BATCH_SIZE = 32

'''Agent Parameters'''
GAMMA = 0.95

'''Model Paths'''
DQN_PATH = 'models/dqn/'
DDDQN_PATH = 'models/dddqn/'
