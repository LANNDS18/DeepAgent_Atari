from DeepAgent.policy import DuelingPolicy
from DeepAgent.agents import DoubleDQNAgent
from DeepAgent.utils import ExperienceReplay, GameEnv, TrainWrapper

from atari_config import EnduroConfig

if __name__ == '__main__':
    _config = EnduroConfig
    agent = TrainWrapper(
        config=_config,
        env=GameEnv,
        buffer=ExperienceReplay,
        policy=DuelingPolicy,
        agent=DoubleDQNAgent,
        train_id='DDDQN')

    agent.learn(
        max_steps=_config.MAX_STEP,
        target_reward=_config.TARGET_REWARD,
        render=_config.RENDER,
    )
