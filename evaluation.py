from DeepAgent.agents import DQNAgent, DoubleDQNAgent
from DeepAgent.utils import GameEnv, ExperienceReplay
from DeepAgent.policy import CNNPolicy, DuelingPolicy

from DeepAgent.utils.dqn_train_evaluation_wrapper import testWrapper
from atari_config import DemonAttackConfig


config = DemonAttackConfig

test_dqn_agent = testWrapper(
    config=config,
    agent=DQNAgent,
    env=GameEnv,
    policy=CNNPolicy,
    buffer=ExperienceReplay,
)

test_double_agent = testWrapper(
    config=config,
    agent=DoubleDQNAgent,
    env=GameEnv,
    policy=DuelingPolicy,
    buffer=ExperienceReplay,
)


test_double_agent.play(
    model_load_path='./models/DDDQN_DemonAttackNoFrameskip-v4/valid',
    render=True,
    video_dir=config.VIDEO_DIR,
    max_episode=config.TEST_MAX_EPISODE
)



