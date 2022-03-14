from DeepAgent.interfaces.ibaseConfig import BaseConfig
from DeepAgent.utils.common import pong_crop, demon_attack_crop


class PongConfig(BaseConfig):
    USE_GPU = False
    RENDER = True

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
    SAVING_MODEL = True
    LOG_HISTORY = True

    MODEL_LOAD_PATH = './models/DQN_PongNoFrameskip-v4'
    VIDEO_DIR = './video/PongDQN'


class DemonAttackConfig(BaseConfig):
    USE_GPU = False

    RENDER = False
    TARGET_REWARD = 8000

    ENV_NAME = 'DemonAttackNoFrameskip-v4'
    CROP = demon_attack_crop

    LEARNING_RATE = [[3e-4, 2.5e-4, 5e5], [2.5e-4, 2e-4, 1e6]]

    GAMMA = 0.99
    N_STEP = 10
    ONE_STEP_WEIGHT = 0.5
    N_STEP_WEIGHT = 0.5
    EPS_SCHEDULE = [[1, 0.1, 1e6],
                    [0.1, 0.01, 2e6],
                    [0.01, 0.001, 5e6]]

    TARGET_SYNC_FREQ = 10000
    SAVING_MODEL = True
    LOG_HISTORY = True

    MODEL_LOAD_PATH = './models/DDDQN_DemonAttackNoFrameskip-v4'
    VIDEO_DIR = './video/DemonAttackDQN'
