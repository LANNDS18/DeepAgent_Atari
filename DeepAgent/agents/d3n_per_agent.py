import tensorflow as tf
from DeepAgent.agents.double_dqn import DoubleDQNAgent
from tensorflow.keras.losses import MSE


class D3NPERAgent(DoubleDQNAgent):

    def __init__(
            self,
            env,
            model,
            buffer,
            agent_id='PER_D3N',
            **kwargs,
    ):
        """
        Initialize networks agent.
        Args:
            env: A gym environment.
            model: tf.keras.models.Model that is expected to be compiled
                with an optimizer before training starts.
            buffer: A prioritized experience replay buffer objects
            **kwargs: kwargs passed to super classes.
        """
        super(D3NPERAgent, self).__init__(env, model, buffer, agent_id, **kwargs)
        self.tds_error = 0
