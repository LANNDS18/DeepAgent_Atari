import pyglet
from DeepAgent.networks import DuelingNetwork, DQNNetwork, DuelingResNet
from DeepAgent.utils import VisualizationWrapper, GameEnv
from atari_config import EnduroConfig, PongConfig, DemonAttackConfig
from DeepAgent.visualization import DeepAgent_Vis

NETWORK = 0
CONFIG = DemonAttackConfig

RLNetworks = {
    0: DQNNetwork,
    1: DuelingNetwork,
    2: DuelingResNet,
}

Loading_Path = {
    PongConfig:
        ['./models/DQN_PongNoFrameskip-v4/best/', './models/DDDQN_PongNoFrameskip-v4/best/', None],
    DemonAttackConfig:
        ['./models/DQN_DemonAttackNoFrameskip-v4/best/', './models/DDDQN_DemonAttackNoFrameskip-v4/best/', None],
    EnduroConfig: []
}


if __name__ == "__main__":
    nn = RLNetworks[NETWORK]
    path = Loading_Path[CONFIG][NETWORK]

    env, agent_policy = VisualizationWrapper(CONFIG, GameEnv, nn)
    agent_policy.load(path)

    window = DeepAgent_Vis('DoubleDuelingDQN', agent_policy, env)

    pyglet.clock.schedule_interval(window.update, window.frame_rate)
    pyglet.app.run()
