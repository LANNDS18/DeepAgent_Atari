from DeepAgent.policy.duelingPolicy import Dueling
from DeepAgent.utils.buffer import PrioritizedExperienceReplay
from DeepAgent.utils.game import GameEnv
from DeepAgent.agents.d3nPER import D3NPERAgent
from DeepAgent.utils.train_evaluation_wrapper import trainWrapper
from atari_config import PongConfig

if __name__ == '__main__':
    _config = PongConfig
    agent = trainWrapper(
        config=_config,
        env=GameEnv,
        buffer=PrioritizedExperienceReplay,
        policy=Dueling,
        agent=D3NPERAgent,
        train_id='D3N_PER')

    agent.learn(
        max_steps=_config.MAX_STEP,
        target_reward=_config.TARGET_REWARD,
        render=_config.RENDER,
    )
