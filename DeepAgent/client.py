from DeepAgent.utils import GameEnv, TrainWrapper, VisualizationWrapper, TestWrapper
from DeepAgent.visualization import DeepAgent_Vis
from datetime import datetime

import argparse
import pyglet


class DQNClient:
    def __init__(
            self,
            agent_configs,
            game_configs,
    ):
        assert isinstance(agent_configs, dict), \
            'agent_configs should be a dict in {"agent_name": [buffer, policy, agent], ...} format'
        assert isinstance(game_configs, dict), \
            'game_configs should be a dict in {"config_id": config, ...} format'

        self.agent_configs = agent_configs
        self.agent_config_id = agent_configs.keys()

        self.game_configs = game_configs
        self.game_config_id = game_configs.keys()

    def args_parse(self):
        parser = argparse.ArgumentParser(description="DeepAgent Client")

        parser.add_argument('--config',
                            default="PongConfig",
                            choices=self.game_config_id,
                            help='a config inherit base config from atari_config.py')

        parser.add_argument("-a", "--agent",
                            type=str,
                            default='DQN',
                            choices=self.agent_config_id,
                            help="the deep reinforcement learning agent would be used")

        parser.add_argument('--train_id',
                            type=str,
                            default=f'Deep_rl_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                            help="the id to specify a deep agent")

        parser.add_argument("-r", "--render",
                            action='store_true',
                            help="render the game to the screen")

        parser.add_argument("--train", action='store_true')
        parser.add_argument("--test", action='store_true')
        parser.add_argument("-v", "--visualization", action='store_true')

        parser.add_argument("--load_dir",
                            help="a checkpoint using to load the pre-trained model if testing or visualization")

        args = parser.parse_args()
        return args

    def process_args(self, args):
        _config = self.game_configs[args.config]

        if args.agent not in self.agent_config_id:
            print('Please specify the correct agent')
            return

        buffer_fn, policy_fn, agent_fn = self.agent_configs[args.agent]

        if args.train:
            agent = TrainWrapper(_config, GameEnv, buffer_fn, policy_fn, agent_fn, train_id=args.train_id)
            agent.learn(
                max_steps=_config.MAX_STEP,
                target_reward=_config.TARGET_REWARD,
                render=args.render,
            )

        elif args.test or args.visualization:
            if args.load_dir is None:
                print('Please specify the loading dir for model')
                return

            elif args.visualization:
                env, agent_policy = VisualizationWrapper(_config, GameEnv, policy_fn)
                agent_policy.load(args.load_dir)
                window = DeepAgent_Vis(args.agent, agent_policy, env)

                pyglet.clock.schedule_interval(window.update, window.frame_rate)
                pyglet.app.run()

            elif args.test:
                test_agent = TestWrapper(
                    config=_config,
                    agent=agent_fn,
                    env=GameEnv,
                    policy=policy_fn,
                    buffer=buffer_fn,
                )

                test_agent.play(
                    model_load_path=args.load_dir,
                    render=args.render,
                    video_dir=_config.VIDEO_DIR,
                    max_episode=_config.TEST_MAX_EPISODE
                )
