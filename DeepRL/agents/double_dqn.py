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

    @tf.function
    def update_main_model(self, states, actions, rewards, dones, new_states):
        """Update main q network by experience replay method.
        Args:
            states (tf.float32): Batch of states.
            actions (tf.int32): Batch of actions.
            rewards (tf.float32): Batch of rewards.
            dones (tf.bool): Batch or terminal status.
            new_states (tf.float32): Batch of next states.
        Returns:
            loss (tf.float32): Huber loss of temporal difference.
        """

        with tf.GradientTape() as tape:
            q_online = self.model(new_states)
            action_q_online = tf.math.argmax(q_online, axis=1)

            q_target = self.target_model(new_states)
            double_q = tf.reduce_sum(q_target * tf.one_hot(action_q_online, self.env.get_action_space_size(), 1.0, 0.0),
                                     axis=1)
            # Double Q Equation #
            expected_q = rewards + self.gamma * double_q * (
                    1.0 - tf.cast(dones, tf.float32))
            main_q = tf.reduce_sum(self.model(states) * tf.one_hot(actions, self.env.get_action_space_size(), 1.0, 0.0),
                                   axis=1)
            loss = self.loss(tf.stop_gradient(expected_q), main_q)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(grad, 10) for grad in gradients]
        self.optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_variables))

        self.loss_metric.update_state(loss)
        self.q_metric.update_state(main_q)

        return loss
