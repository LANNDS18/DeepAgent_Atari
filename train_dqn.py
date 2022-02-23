from DeepAgent.networks.dqn import DQNNetwork
from DeepAgent.agents.DQN import DQNAgent
from DeepAgent.utils.buffer import ExperienceReplay
from DeepAgent.utils.game import GameEnv
from TrainEvaluationWrapper import trainWrapper
from config import *

if __name__ == '__main__':
    agent = trainWrapper(GameEnv, ExperienceReplay, DQNNetwork, DQNAgent, 'DQN_v2')
    agent.learn(max_steps=TRAINING_STEP)
