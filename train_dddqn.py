from DeepAgent.policy.duelingPolicy import Dueling
from DeepAgent.agents.doubleDQN import DoubleDQNAgent
from DeepAgent.utils.buffer import ExperienceReplay
from DeepAgent.utils.game import GameEnv
from TrainEvaluationWrapper import trainWrapper
from config import PongConfig, DemonAttackConfig

if __name__ == '__main__':
    agent = trainWrapper(
        config=PongConfig,
        env=GameEnv,
        buffer=ExperienceReplay,
        network=Dueling,
        agent=DoubleDQNAgent,
        train_id='DDDQN')

    agent.learn(max_steps=1e7, target_reward=20)
