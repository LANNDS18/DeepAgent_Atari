from DeepAgent.policy.duelingPolicy import Dueling
from DeepAgent.agents.doubleDQN import DoubleDQNAgent
from DeepAgent.utils.buffer import ExperienceReplay
from DeepAgent.utils.game import GameEnv
from DeepAgent.utils.train_evaluation_wrapper import trainWrapper

from atari_config import DemonAttackConfig

if __name__ == '__main__':
    _config = DemonAttackConfig
    agent = trainWrapper(
        config=_config,
        env=GameEnv,
        buffer=ExperienceReplay,
        policy=Dueling,
        agent=DoubleDQNAgent,
        train_id='DDDQN')

    agent.learn(
        max_steps=_config.MAX_STEP,
        target_reward=_config.TARGET_REWARD,
        render=_config.RENDER,
    )
