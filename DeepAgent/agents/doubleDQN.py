import tensorflow as tf
from DeepAgent.agents.baseDQN import DQNAgent


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
    def update_gradient(self, states, actions, rewards, dones, new_states, batch_weights=1):
        """Update main q network by experience replay method.
        Args:
            states (tf.float32): Batch of states.
            actions (tf.int32): Batch of actions.
            rewards (tf.float32): Batch of rewards.
            dones (tf.bool): Batch or terminal status.
            new_states (tf.float32): Batch of next states.
            batch_weights(tf.float32): weights of this batch
        Returns:
            loss (tf.float32): Huber loss of temporal difference.
        """
        q_online = self.policy_network.predict(new_states)
        action_q_online = tf.math.argmax(q_online, axis=1)

        q_target = self.target_network.predict(new_states)
        double_q = tf.reduce_sum(
            q_target * tf.one_hot(action_q_online, self.n_actions, 1.0, 0.0),
            axis=1)

        self.policy_network.update_lr()

        with tf.GradientTape() as tape:
            tape.watch(self.policy_network.model.trainable_weights)

            target_q = rewards + self.gamma * double_q * (
                    1.0 - tf.cast(dones, tf.float32))
            main_q = tf.reduce_sum(self.policy_network.model(states) * tf.one_hot(actions, self.n_actions, 1.0, 0.0),
                                   axis=1)

            losses = self.policy_network.loss_function(main_q, target_q) * self.policy_network.one_step_weight
            if self.policy_network.l2_weight > 0:
                losses += self.policy_network.l2_weight * tf.reduce_sum(
                    [tf.reduce_sum(tf.square(layer_weights))
                     for layer_weights in self.policy_network.model.trainable_weights])
            loss = tf.reduce_mean(losses * batch_weights)

        self.policy_network.optimizer.minimize(loss, self.policy_network.model.trainable_variables, tape=tape)

        self.loss_metric.update_state(loss)
        self.q_metric.update_state(main_q)

        return main_q, target_q

