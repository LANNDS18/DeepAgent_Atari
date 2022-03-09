from DeepAgent.policy.cnnPolicy import CNN
from DeepAgent.agents.baseDQN import DQNAgent
from DeepAgent.utils.buffer import ExperienceReplay
from DeepAgent.utils.game import GameEnv
from TrainEvaluationWrapper import trainWrapper
from config import PongConfig, DemonAttackConfig

if __name__ == '__main__':
    agent = trainWrapper(
        config=PongConfig,
        env=GameEnv,
        buffer=ExperienceReplay,
        network=CNN,
        agent=DQNAgent,
        train_id='DQN')

    agent.learn(max_steps=int(1e7), target_reward=20)
