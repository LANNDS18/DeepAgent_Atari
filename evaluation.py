from DeepAgent.agents.baseDQN import DQNAgent
from DeepAgent.utils.game import GameEnv
from DeepAgent.utils.buffer import ExperienceReplay
from DeepAgent.policy.cnnPolicy import CNN

from config import PongConfig
from TrainEvaluationWrapper import testWrapper


test_dqn_pong = testWrapper(
    config=PongConfig,
    agent=DQNAgent,
    env=GameEnv,
    policy=CNN,
    buffer=ExperienceReplay,
    test_id='PongDQN'
)

test_dqn_pong.play(
    model_load_path='./models/DQN_PongNoFrameskip-v4',
    render=True,
    video_dir='./video',
    max_episode=PongConfig.TEST_MAX_EPISODE
)



