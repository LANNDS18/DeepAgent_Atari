import numpy as np
import tensorflow as tf

from DeepAgent.interfaces.IBaseAgent import BaseAgent


class DQNAgent(BaseAgent):

    def __init__(
            self,
            env,
            model,
            buffer,
            agent_id='DQN',
            epsilon_start=1.0,
            epsilon_end=0.02,
            epsilon_decay_steps=150000,
            **kwargs,
    ):
        """
        Initialize networks agent.
        Args:
            env: A gym environment.
            model: tf.keras.models.Model that is expected to be compiled
                with an optimizer before training starts.
            buffer: A buffer objects
            epsilon_start: Starting epsilon value which is used to control random exploration.
                It should be decremented and adjusted according to implementation needs.
            epsilon_end: End epsilon value which is the minimum exploration rate.
            epsilon_decay_steps: Number of total_step for epsilon to reach `epsilon_end`
                from `epsilon_start`,
            target_sync_steps: Steps to sync target model after each.
            **kwargs: kwargs passed to super classes.
        """
        super(DQNAgent, self).__init__(env, model, buffer, agent_id, **kwargs)
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

    def update_epsilon(self):
        """
        Decrement epsilon which aims to gradually reduce randomization.
        """
        self.epsilon = max(
            self.epsilon_end, self.epsilon_start - self.total_step / self.epsilon_decay_steps
        )

    @tf.function
    def get_action(self, state, epsilon):
        """Get action by ε-greedy method.

        Args:
            state (np.uint8): recent self.history_length frames. (Default: (84, 84, 4))
            epsilon (int): Exploration rate for deciding random or optimal action.

        Returns:
            action (tf.int32): Action index
        """
        recent_state = tf.expand_dims(state, axis=0)
        if tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32) < epsilon:
            action = tf.random.uniform((), minval=0, maxval=self.n_actions, dtype=tf.int32)
        else:
            q_value = self.model(tf.cast(recent_state, tf.float32))
            action = tf.cast(tf.squeeze(tf.math.argmax(q_value, axis=1)), dtype=tf.int32)
        return action

    @tf.function
    def sync_target_model(self):
        """Synchronize weights of target network by those of main network."""
        main_vars = self.model.trainable_variables
        target_vars = self.target_model.trainable_variables
        for main_var, target_var in zip(main_vars, target_vars):
            target_var.assign(main_var)

    def at_step_start(self):
        """
        Execute total_step that will run before self.train_step() which decays epsilon.
        """
        self.update_epsilon()

    @tf.function
    def update_main_model(self, states, actions, rewards, dones, new_states):
        """Update main q network by experience replay method.

        Args:
            states (tf.float32): Batch of states.
            actions (tf.int32): Batch of actions.
            rewards (tf.float32): Batch of rewards.
            new_states (tf.float32): Batch of next states.
            dones (tf.bool): Batch or terminal status.

        Returns:
            loss (tf.float32): Huber loss of temporal difference.
        """

        with tf.GradientTape() as tape:
            next_state_q = self.target_model(new_states)
            next_state_max_q = tf.math.reduce_max(next_state_q, axis=1)
            expected_q = rewards + self.gamma * next_state_max_q * (
                    1.0 - tf.cast(dones, tf.float32))
            main_q = tf.reduce_sum(
                self.model(states) * tf.one_hot(actions, self.n_actions, 1.0, 0.0),
                axis=1)
            loss = self.loss(tf.stop_gradient(expected_q), main_q)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(grad, 10) for grad in gradients]
        self.optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_variables))
        self.loss_metric.update_state(loss)
        self.q_metric.update_state(main_q)

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

            self.update_main_model(states, actions, rewards, dones, next_states)

    def at_step_end(self):
        if self.total_step % self.target_sync_freq == 0:
            self.sync_target_model()
        self.total_step += 1
        self.env.render()

    def learn(
            self,
            max_steps,
    ):
        """
        Args:
            max_steps: Maximum number of total_step, if reached the training will stop.
        """
        self.init_training(max_steps)
        while True:
            self.check_episodes()
            if self.check_finish_training():
                break
            while not self.done:
                self.at_step_start()
                self.train_step()
                self.at_step_end()
