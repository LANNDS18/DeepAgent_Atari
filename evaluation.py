from DeepAgent.agents import DQNAgent, DoubleDQNAgent, D3NPERAgent
from DeepAgent.utils import GameEnv, ExperienceReplay, testWrapper
from DeepAgent.policy import CNNPolicy, DuelingPolicy

from atari_config import PongConfig, DemonAttackConfig


config = PongConfig

test_dqn_agent = testWrapper(
    config=config,
    agent=DQNAgent,
    env=GameEnv,
    policy=DuelingPolicy,
    buffer=ExperienceReplay,
    test_id=config.ENV_NAME
)

test_double_agent = testWrapper(
    config=config,
    agent=DQNAgent,
    env=GameEnv,
    policy=DuelingPolicy,
    buffer=ExperienceReplay,
    test_id=config.ENV_NAME
)


test_dqn_agent.play(
    model_load_path=config.MODEL_LOAD_PATH,
    render=True,
    video_dir=config.VIDEO_DIR,
    max_episode=DemonAttackConfig.TEST_MAX_EPISODE
)



