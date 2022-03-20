import pyglet

from DeepAgent.networks import DuelingNetwork
from DeepAgent.utils import VisualizationWrapper, GameEnv
from atari_config import EnduroConfig, PongConfig
from DeepAgent.visualization import DeepAgent_Vis

if __name__ == "__main__":
    config = PongConfig
    env, agent_policy = VisualizationWrapper(config, GameEnv, DuelingNetwork)
    path = './models/DDDQN_PongNoFrameskip-v4/best/'
    agent_policy.load(path)
    print("The environment has the following {} actions: {}".format(env.env.action_space.n,
                                                                    env.env.unwrapped.get_action_meanings()))

    window = DeepAgent_Vis('DoubleDuelingDQN', agent_policy, env)

    pyglet.clock.schedule_interval(window.update, window.frame_rate)
    pyglet.app.run()
