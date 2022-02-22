"""The parameter for training models"""

'''Env Parameters'''
ENV_NAME = 'DemonAttackNoFrameskip-v0'
IMAGE_SHAPE = (84, 84)
FRAME_STACK = 4

'''Network Parameters '''
LEARNING_RATE = 5e-4

"""Buffer Parameters"""
BUFFER_SIZE = 80000
BATCH_SIZE = 32
FILL_BUFFER_SIZE = 200

'''Agent Parameters'''
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = int(1e6)
MODEL_SAVE_INTERVAL = 500
SAVING_MODEL = True
LOG_HISTORY = True

'''Learning Parameters'''
TRAINING_STEP = int(2e6)

'''Evaluation Parameters'''
TEST_BUFFER_SIZE = 10
TEST_BATCH_SIZE = 1
TEST_MAX_EPISODE = 10
