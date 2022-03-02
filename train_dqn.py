from DeepAgent.networks.dqn import DQNNetwork
from DeepAgent.agents.baseDQN import DQNAgent
from DeepAgent.utils.buffer import ExperienceReplay
from DeepAgent.utils.game import GameEnv
from TrainEvaluationWrapper import trainWrapper

if __name__ == '__main__':
    agent = trainWrapper(GameEnv,
                         ExperienceReplay,
                         DQNNetwork,
                         DQNAgent,
                         'DQN_v3')

    agent.learn(max_steps=int(10e6))
