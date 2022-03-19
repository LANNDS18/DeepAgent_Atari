from DeepAgent.agents import DQNAgent, DoubleDQNAgent
from DeepAgent.utils import GameEnv, ExperienceReplay
from DeepAgent.networks import DQNNetwork, DuelingNetwork

from DeepAgent.utils import TestWrapper
from atari_config import DemonAttackConfig, PongConfig, EnduroConfig


config = PongConfig

test_dqn_agent = TestWrapper(
    config=config,
    agent=DQNAgent,
    env=GameEnv,
    policy=DQNNetwork,
    buffer=ExperienceReplay,
)

test_double_agent = TestWrapper(
    config=config,
    agent=DoubleDQNAgent,
    env=GameEnv,
    policy=DuelingNetwork,
    buffer=ExperienceReplay,
)


test_double_agent.play(
    model_load_path='./models/DDDQN_PongNoFrameskip-v4/best/',
    render=True,
    video_dir=config.VIDEO_DIR,
    max_episode=config.TEST_MAX_EPISODE
)



