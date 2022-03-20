import tensorflow as tf

from DeepAgent.agents.doubleDQN import DoubleDQNAgent
from DeepAgent.utils.buffer import PrioritizedExperienceReplay


class D3NPERAgent(DoubleDQNAgent):

    def __init__(
            self,
            env,
            policy_network,
            target_network,
            buffer,
            agent_id='D3N_PER',
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
        super(D3NPERAgent, self).__init__(env, policy_network, target_network, buffer, agent_id, **kwargs)
        assert (isinstance(self.buffer, PrioritizedExperienceReplay)), \
            'The buffer should be a PrioritizedExperienceReplay buffer.'

    @tf.function
    def get_target(self, rewards, dones, next_states):
        """
        get target q for both single step and n_step

        Args:
            rewards (tf.float32): Batch of rewards.
            dones (tf.bool): Batch of terminal status.
            next_states (tf.float32): Batch of next states.
        """
        action_online = tf.math.argmax(self.policy_network.predict(next_states), axis=1)
        double_q = tf.reduce_sum(self.target_network.predict(next_states)
                                 * tf.one_hot(action_online, self.n_actions, 1.0, 0.0), axis=1)

        target_q = rewards + self.gamma * double_q * (1.0 - tf.cast(dones, tf.float32))

        return target_q

    @tf.function
    def update_gradient(self, target_q, states, actions, batch_weights=1):

        """
        Update main q network by experience replay method.

        Args:
            target_q (tf.float32): Target Q value for barch.

            states (tf.float32): Batch of states.
            actions (tf.int32): Batch of actions.

            batch_weights(tf.float32): weights of this batch.
        """

        self.policy_network.update_lr()
        with tf.GradientTape() as tape:
            tape.watch(self.policy_network.model.trainable_weights)
            main_q = tf.reduce_sum(
                self.policy_network.model(states) * tf.one_hot(actions, self.n_actions, 1.0, 0.0),
                axis=1)

            losses = self.policy_network.loss_function(main_q, target_q)

            if self.policy_network.l2_weight > 0:
                losses += self.policy_network.l2_weight * tf.reduce_sum(
                    [tf.reduce_sum(tf.square(layer_weights))
                     for layer_weights in self.policy_network.model.trainable_weights])

            loss = tf.reduce_mean(losses * batch_weights)

        self.policy_network.optimizer.minimize(loss, self.policy_network.model.trainable_variables, tape=tape)

        self.loss_metric.update_state(loss)
        self.q_metric.update_state(main_q)

        return main_q, loss

    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.env_name, batching and gradient updates.
        """
        action = self.get_action(tf.constant(self.state), tf.constant(self.epsilon, tf.float32))

        next_state, reward, done, info = self.env.step(action)
        self.buffer.append(self.state, action, reward, done, next_state)

        self.state = next_state
        self.done = self.env.was_real_done

        if self.total_step % self.model_update_freq == 0:
            indices = self.buffer.get_sample_indices()
            states, actions, rewards, dones, next_states = self.buffer.get_sample(indices)

            target_q = self.get_target(rewards, dones, next_states)
            main_q, loss = self.update_gradient(target_q, states, actions)

            error = main_q - target_q
            is_small_error = tf.abs(error) < 1
            squared_loss = tf.square(error) / 2
            linear_loss = tf.abs(error) - 0.5
            error = tf.where(is_small_error, squared_loss, linear_loss)

            self.buffer.update_priorities(indices, error)
