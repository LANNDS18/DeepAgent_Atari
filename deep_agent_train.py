from DeepAgent.agents import DQNAgent, DoubleDQNAgent, D3NPERAgent
from DeepAgent.utils import GameEnv, ExperienceReplay, PrioritizedExperienceReplay
from DeepAgent.networks import DQNNetwork, DuelingNetwork, DuelingResNet
from DeepAgent.utils import TrainWrapper
from atari_config import DemonAttackConfig, PongConfig, EnduroConfig

AGENT = 'RES_DDDQN'
CONFIG = DemonAttackConfig

agent_configs = {
    'DQN': [ExperienceReplay, DQNNetwork, DQNAgent],
    'DDDQN': [ExperienceReplay, DuelingNetwork, DoubleDQNAgent],
    'D3N_PER': [PrioritizedExperienceReplay, DuelingNetwork, D3NPERAgent],
    'ResNet_DQN': [ExperienceReplay, DuelingResNet, DoubleDQNAgent],
}

if __name__ == '__main__':
    _config = CONFIG
    agent = TrainWrapper(
        config=_config,
        env=GameEnv,
        buffer=agent_configs[AGENT][0],
        policy=agent_configs[AGENT][1],
        agent=agent_configs[AGENT][2],
        train_id=AGENT
    )

    agent.learn(
        max_steps=_config.MAX_STEP,
        target_reward=_config.TARGET_REWARD,
        render=_config.RENDER,
    )
