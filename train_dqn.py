from DeepAgent.networks import DQNNetwork
from DeepAgent.agents import DQNAgent
from DeepAgent.utils import ExperienceReplay, GameEnv, TrainWrapper

from atari_config import EnduroConfig, DemonAttackConfig

if __name__ == '__main__':
    _config = DemonAttackConfig
    agent = TrainWrapper(
        config=_config,
        env=GameEnv,
        buffer=ExperienceReplay,
        policy=DQNNetwork,
        agent=DQNAgent,
        train_id='DQN')

    agent.learn(
        max_steps=_config.MAX_STEP,
        target_reward=_config.TARGET_REWARD,
        render=_config.RENDER,
    )
