from DeepAgent.policy.cnnPolicy import CNN
from DeepAgent.agents.baseDQN import DQNAgent
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
        policy=CNN,
        agent=DQNAgent,
        train_id='DQN')

    agent.learn(
        max_steps=_config.MAX_STEP,
        target_reward=_config.TARGET_REWARD,
        render=_config.RENDER,
    )
