import tensorflow as tf
from DeepRL.agents.dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):

    def __init__(
            self,
            env,
            model,
            buffer,
            **kwargs,
    ):
        """
        Initialize networks agent.
        Args:
            env: A gym environment.
            model: tf.keras.models.Model that is expected to be compiled
                with an optimizer before training starts.
            buffer: A buffer objects
            **kwargs: kwargs passed to super classes.
        """
        super(DoubleDQNAgent, self).__init__(env, model, buffer, **kwargs)

    def get_targets(self, state, action, reward, done, new_states):
        """
        Get targets for gradient update.
        Args:
            state: size = (batch size * self.input_shape)
            action: size =  buffer batch size
            reward: size =  buffer batch size
            done: size =  buffer batch size
            new_states: size = (total buffer batch size, *self.input_shape)
        Returns:
            Target values: size = (total buffer batch size, *self.input_shape)
        """
        q_states = self.model_predict(state, self.model)[1]
        # Double Q
        new_state_actions = self.model_predict(new_states, self.model)[0]
        new_state_q_values = self.model_predict(new_states, self.target_model)[1]
        a = self.get_action_indices(self.batch_indices, new_state_actions)
        new_state_values = tf.gather_nd(new_state_q_values, a)

        new_state_values = tf.where(
            tf.cast(done, tf.bool),
            tf.constant(0, new_state_values.dtype),
            new_state_values,
        )

        target_values = tf.identity(q_states)

        target_value_update = \
            new_state_values * self.gamma + tf.cast(reward, tf.float32)

        indices = self.get_action_indices(self.batch_indices, action)
        target_values = tf.tensor_scatter_nd_update(
            target_values, indices, target_value_update
        )
        return target_values
