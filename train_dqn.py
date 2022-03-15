from DeepAgent.policy import CNNPolicy
from DeepAgent.agents import DQNAgent
from DeepAgent.utils import ExperienceReplay, GameEnv, trainWrapper

from atari_config import DemonAttackConfig

if __name__ == '__main__':
    _config = DemonAttackConfig
    agent = trainWrapper(
        config=_config,
        env=GameEnv,
        buffer=ExperienceReplay,
        policy=CNNPolicy,
        agent=DQNAgent,
        train_id='DQN')

    agent.learn(
        max_steps=_config.MAX_STEP,
        target_reward=_config.TARGET_REWARD,
        render=_config.RENDER,
    )
