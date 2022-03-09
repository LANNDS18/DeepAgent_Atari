from DeepAgent.policy.duelingPolicy import Dueling
from DeepAgent.utils.buffer import PrioritizedExperienceReplay
from DeepAgent.utils.game import GameEnv
from DeepAgent.agents.d3nPER import D3NPERAgent
from TrainEvaluationWrapper import trainWrapper
from config import PongConfig, DemonAttackConfig

if __name__ == '__main__':
    agent = trainWrapper(
        config=PongConfig,
        env=GameEnv,
        buffer=PrioritizedExperienceReplay,
        network=Dueling,
        agent=D3NPERAgent,
        train_id='D3N_PER')

    agent.learn(max_steps=int(1e7), target_reward=20)
