from DeepAgent.policy.duelingPolicy import Dueling
from DeepAgent.agents.baseDQN import DQNAgent
from DeepAgent.utils.buffer import ExperienceReplay
from DeepAgent.utils.game import GameEnv
from TrainEvaluationWrapper import trainWrapper

if __name__ == '__main__':
    agent = trainWrapper(
        env=GameEnv,
        buffer=ExperienceReplay,
        network=Dueling,
        agent=DQNAgent,
        train_id='DDDQN_v1')

    agent.learn(max_steps=1e7)
