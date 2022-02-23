from DeepAgent.networks.dueling import DuelingNetwork
from DeepAgent.agents.DoubleDQN import DoubleDQNAgent
from DeepAgent.utils.buffer import ExperienceReplay
from DeepAgent.utils.game import GameEnv
from TrainEvaluationWrapper import trainWrapper
from config import *

if __name__ == '__main__':
    agent = trainWrapper(
        env=GameEnv,
        buffer=ExperienceReplay,
        network=DuelingNetwork,
        agent=DoubleDQNAgent,
        train_id='DDDQN_v1')

    agent.learn(max_steps=TRAINING_STEP)
