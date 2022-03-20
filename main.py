from DeepAgent.networks import DQNNetwork, DuelingNetwork, DuelingResNet
from DeepAgent.agents import DQNAgent, DoubleDQNAgent, D3NPERAgent
from DeepAgent.utils import ExperienceReplay, PrioritizedExperienceReplay
from atari_config import PongConfig, DemonAttackConfig, EnduroConfig

from DeepAgent.dqnclient import DQNClient

agent_configs = {
    'DQN': [ExperienceReplay, DQNNetwork, DQNAgent],
    'DDDDQN': [ExperienceReplay, DuelingNetwork, DoubleDQNAgent],
    'D3n_PER': [PrioritizedExperienceReplay, DuelingNetwork, D3NPERAgent],
    'ResNet_DQN': [ExperienceReplay, DuelingResNet, DoubleDQNAgent],
}

game_configs = {
    'PongConfig': PongConfig,
    'DemonAttackConfig': DemonAttackConfig,
    'EnduroConfig': EnduroConfig
}

if __name__ == "__main__":
    client = DQNClient(agent_configs, game_configs)
    args = client.args_parse()
    print(args)
    client.process_args(args)
