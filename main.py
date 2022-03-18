from DeepAgent.policy import CNNPolicy, DuelingPolicy
from DeepAgent.agents import DQNAgent, DoubleDQNAgent, D3NPERAgent
from DeepAgent.utils import ExperienceReplay, PrioritizedExperienceReplay
from atari_config import PongConfig, DemonAttackConfig, EnduroConfig

from DeepAgent.client import Client

agent_configs = {
    'DQN': [ExperienceReplay, CNNPolicy, DQNAgent],
    'DDDDQN': [ExperienceReplay, DuelingPolicy, DoubleDQNAgent],
    'D3n_PER': [PrioritizedExperienceReplay, DuelingPolicy, D3NPERAgent,]
}

game_configs = {
    'PongConfig': PongConfig,
    'DemonAttackConfig': DemonAttackConfig,
    'EnduroConfig': EnduroConfig
}

if __name__ == "__main__":
    client = Client(agent_configs, game_configs)
    args = client.args_parse()
    print(args)
    client.process_args(args)

