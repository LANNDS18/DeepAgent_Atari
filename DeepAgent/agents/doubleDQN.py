import tensorflow as tf
from DeepAgent.agents.dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):

    def __init__(
            self,
            env,
            policy_network,
            target_network,
            buffer,
            agent_id='DDDQN',
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
        super(DoubleDQNAgent, self).__init__(env, policy_network, target_network, buffer, agent_id, **kwargs)

    @tf.function
    def get_target(self, rewards, dones, next_states, n_step_rewards, n_step_dones, n_step_next, ):
        """
        get target q for both single step and n_step

        Args:
            rewards (tf.float32): Batch of rewards.
            dones (tf.bool): Batch of terminal status.
            next_states (tf.float32): Batch of next states.

            n_step_rewards (tf.float32): Batch of after n_step rewards,
            n_step_dones (tf.bool): Batch of terminal status after n_step.
            n_step_next (tf.float32): Batch of after n_step states.
        """
        action_online = tf.math.argmax(self.policy_network.predict(next_states), axis=1)
        double_q = tf.reduce_sum(self.target_network.predict(next_states)
                                 * tf.one_hot(action_online, self.n_actions, 1.0, 0.0), axis=1)

        target_q = rewards + self.gamma * double_q * (1.0 - tf.cast(dones, tf.float32))

        n_step_action_online = tf.math.argmax(self.target_network.predict(n_step_next), axis=1)
        n_step_double_q = tf.reduce_sum(self.target_network.predict(n_step_next)
                                        * tf.one_hot(n_step_action_online, self.n_actions, 1.0, 0.0), axis=1)

        n_target_q = n_step_rewards + self.gamma * n_step_double_q * (1.0 - tf.cast(n_step_dones, tf.float32))

        return target_q, n_target_q
