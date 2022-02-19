"""The parameter for training models"""

'''Env Parameters'''
ENV_NAME = 'DemonAttackNoFrameskip-v0'
IMAGE_SHAPE = (84, 84)
FRAME_STACK = 4

'''Network Parameters '''
LEARNING_RATE = 3e-4

"""Buffer Parameters"""
BUFFER_SIZE = 80000
BATCH_SIZE = 32

'''Agent Parameters'''
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY_STEPS = int(1e6)
MODEL_SAVE_INTERVAL = 5000

'''Learning Parameters'''
TRAINING_STEP = int(1e7)

'''Model Paths'''
DQN_PATH = 'models/dqn/'
DDDQN_PATH = 'models/dddqn/'
