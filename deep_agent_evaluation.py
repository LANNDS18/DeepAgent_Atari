from DeepAgent.agents import DQNAgent, DoubleDQNAgent, D3NPERAgent
from DeepAgent.utils import GameEnv, ExperienceReplay, PrioritizedExperienceReplay
from DeepAgent.networks import DQNNetwork, DuelingNetwork, DuelingResNet
from DeepAgent.utils import TestWrapper
from atari_config import DemonAttackConfig, PongConfig, EnduroConfig

AGENT = 0
CONFIG = DemonAttackConfig

agent_configs = {
    0: [ExperienceReplay, DQNNetwork, DQNAgent],
    1: [ExperienceReplay, DuelingNetwork, DoubleDQNAgent],
    2: [PrioritizedExperienceReplay, DuelingNetwork, D3NPERAgent],
    3: [ExperienceReplay, DuelingResNet, DoubleDQNAgent],
}

Loading_Path = {
    PongConfig:
        ['./models/DQN_PongNoFrameskip-v4/best/', './models/DDDQN_PongNoFrameskip-v4/best/', None],
    DemonAttackConfig:
        ['./models/DQN_DemonAttackNoFrameskip-v4/best/', './models/DDDQN_DemonAttackNoFrameskip-v4/best/', None],
    EnduroConfig: []
}


if __name__ == "__main__":
    test_agent = TestWrapper(
        config=CONFIG,
        env=GameEnv,
        buffer=agent_configs[AGENT][0],
        policy=agent_configs[AGENT][1],
        agent=agent_configs[AGENT][2],
    )

    test_agent.play(
        model_load_path=Loading_Path[CONFIG][AGENT],
        render=True,
        video_dir=CONFIG.VIDEO_DIR,
        max_episode=CONFIG.TEST_MAX_EPISODE
    )



