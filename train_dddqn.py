from DeepAgent.networks import DuelingNetwork
from DeepAgent.agents import DoubleDQNAgent
from DeepAgent.utils import ExperienceReplay, GameEnv, TrainWrapper

from atari_config import DemonAttackConfig

if __name__ == '__main__':
    _config = DemonAttackConfig
    agent = TrainWrapper(
        config=_config,
        env=GameEnv,
        buffer=ExperienceReplay,
        policy=DuelingNetwork,
        agent=DoubleDQNAgent,
        train_id='DDDQN')

    agent.learn(
        max_steps=_config.MAX_STEP,
        target_reward=_config.TARGET_REWARD,
        render=_config.RENDER,
    )
