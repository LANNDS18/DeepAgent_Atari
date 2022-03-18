import pyglet

from DeepAgent.policy import DuelingPolicy
from DeepAgent.utils import VisualizationWrapper, GameEnv
from atari_config import EnduroConfig
from DeepAgent.visualization import DeepAgent_Vis

if __name__ == "__main__":
    config = EnduroConfig
    env, agent_policy = VisualizationWrapper(config, GameEnv, DuelingPolicy)
    path = './models/DDDQN_EnduroNoFrameskip-v4/best/main/'
    agent_policy.load(path)
    print("The environment has the following {} actions: {}".format(env.env.action_space.n,
                                                                    env.env.unwrapped.get_action_meanings()))

    window = DeepAgent_Vis('DoubleDuelingDQN', agent_policy, env)

    pyglet.clock.schedule_interval(window.update, window.frame_rate)
    pyglet.app.run()
