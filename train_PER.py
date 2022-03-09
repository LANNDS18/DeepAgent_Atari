from DeepAgent.policy.dueling import DuelingNetwork
from DeepAgent.utils.buffer import PrioritizedExperienceReplay
from DeepAgent.utils.game import GameEnv
from DeepAgent.agents.d3nPER import D3NPERAgent
from TrainEvaluationWrapper import trainWrapper
from config import *

if __name__ == '__main__':
    agent = trainWrapper(
        env=GameEnv,
        buffer=PrioritizedExperienceReplay,
        network=DuelingNetwork,
        agent=D3NPERAgent,
        train_id='D3N_PER_v1')

    agent.learn(max_steps=int(10e6))
