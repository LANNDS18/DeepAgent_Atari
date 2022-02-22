import tensorflow as tf
from DeepAgent.agents.DQN import DQNAgent
from DeepAgent.utils.buffer import PrioritizedExperienceReplay


class D3NPERAgent(DQNAgent):

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
        assert (isinstance(self.buffer, PrioritizedExperienceReplay)), \
            'The buffer should be a PrioritizedExperienceReplay buffer.'

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
        q_online = self.model(new_states)
        action_q_online = tf.math.argmax(q_online, axis=1)

        q_target = self.target_model(new_states)
        double_q = tf.reduce_sum(q_target * tf.one_hot(action_q_online, self.n_actions, 1.0, 0.0), axis=1)

        with tf.GradientTape() as tape:
            # Double Q Equation #
            target_q = rewards + self.gamma * double_q * (
                    1.0 - tf.cast(dones, tf.float32))
            main_q = tf.reduce_sum(self.model(states) * tf.one_hot(actions, self.n_actions, 1.0, 0.0), axis=1)
            loss = self.loss(tf.stop_gradient(target_q), main_q)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(grad, 10) for grad in gradients]
        self.optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_variables))

        self.loss_metric.update_state(loss)
        self.q_metric.update_state(main_q)

        return main_q, target_q

    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.env_name, batching and gradient updates.
        """
        action = self.get_action(tf.constant(self.state), tf.constant(self.epsilon, tf.float32))

        next_state, reward, done, info = self.env.step(action)
        self.buffer.append(self.state, action, reward, done, next_state)

        self.episode_reward += reward
        self.state = next_state
        self.done = done

        if self.total_step % self.model_update_freq == 0:
            indices = self.buffer.get_sample_indices()
            states, actions, rewards, dones, next_states = self.buffer.get_sample(indices)

            main_q, target_q = self.update_main_model(states, actions, rewards, dones, next_states)
            abs_error = abs(main_q - target_q)
            self.buffer.update_priorities(indices, abs_error)
