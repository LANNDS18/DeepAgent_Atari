from DeepAgent.policy.cnnPolicy import CNN
from DeepAgent.agents.baseDQN import DQNAgent
from DeepAgent.utils.buffer import ExperienceReplay
from DeepAgent.utils.game import GameEnv
from TrainEvaluationWrapper import trainWrapper

if __name__ == '__main__':
    agent = trainWrapper(GameEnv,
                         ExperienceReplay,
                         CNN,
                         DQNAgent,
                         'DQN_v1')

    agent.learn(max_steps=int(1e7))
