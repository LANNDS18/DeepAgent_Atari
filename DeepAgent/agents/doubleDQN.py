import tensorflow as tf
from DeepAgent.agents.dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):

    def __init__(
            self,
            env,
            policy_network,
            target_network,
            buffer,
            agent_id='Double DQN',
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
        q_online = self.policy_network.predict(next_states)
        action_q_online = tf.math.argmax(q_online, axis=1)
        q_target = self.target_network.predict(next_states)
        double_q = tf.reduce_sum(q_target * tf.one_hot(action_q_online, self.n_actions, 1.0, 0.0), axis=1)

        n_step_q_online = self.target_network.predict(n_step_next)
        n_step_action_q_online = tf.math.argmax(n_step_q_online, axis=1)
        n_step_q_target = self.target_network.predict(n_step_next)
        n_step_double_q = tf.reduce_sum(n_step_q_target * tf.one_hot(n_step_action_q_online, self.n_actions, 1.0, 0.0),
                                        axis=1)

        return double_q, n_step_double_q
