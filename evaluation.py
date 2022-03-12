from DeepAgent.agents.baseDQN import DQNAgent
from DeepAgent.utils.game import GameEnv
from DeepAgent.utils.buffer import ExperienceReplay
from DeepAgent.utils.train_evaluation_wrapper import testWrapper
from DeepAgent.policy.cnnPolicy import CNN

from atari_config import PongConfig, DemonAttackConfig


config = DemonAttackConfig

test_dqn_agent = testWrapper(
    config=config,
    agent=DQNAgent,
    env=GameEnv,
    policy=CNN,
    buffer=ExperienceReplay,
    test_id=config.ENV_NAME
)

test_dqn_agent.play(
    model_load_path=config.MODEL_LOAD_PATH,
    render=True,
    video_dir=config.VIDEO_DIR,
    max_episode=DemonAttackConfig.TEST_MAX_EPISODE
)



