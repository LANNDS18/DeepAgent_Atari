import numpy as np
import tensorflow as tf

from DeepAgent.interfaces.ibaseAgent import OffPolicy, EpsDecayAgent


class DQNAgent(OffPolicy, EpsDecayAgent):

    def __init__(
            self,
            env,
            policy_network,
            target_network,
            buffer,
            agent_id='DQN',
            eps_schedule=None,
            **kwargs,
    ):
        """
        Initialize DQN agent.
        Args:
            env: A gym environment.
            model: tf.keras.models.Model that is expected to be compiled
                with an optimizer before training starts.
            buffer: A buffer objects
            **kwargs: kwargs passed to super classes.
        """
        OffPolicy.__init__(self,
                           env=env,
                           policy_network=policy_network,
                           target_network=target_network,
                           buffer=buffer,
                           agent_id=agent_id,
                           **kwargs)

        EpsDecayAgent.__init__(self, eps_schedule=eps_schedule)

    @tf.function
    def get_action(self, state, epsilon):
        """Get action by ε-greedy method.

        Args:
            state (np.uint8): recent self.history_length frames. (Default: (84, 84, 4))
            epsilon (int): Exploration rate for deciding random or optimal action.

        Returns:
            action (tf.int32): Action index
        """
        state = tf.expand_dims(state, axis=0)
        if tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32) < epsilon:
            action = tf.random.uniform((), minval=0, maxval=self.n_actions, dtype=tf.int32)
        else:
            action = self.policy_network.get_optimal_actions(tf.cast(state, tf.float32))
        return action

    def at_step_start(self):
        """
        Execute total_step that will run before self.train_step() which decays epsilon.
        """
        self.total_step += 1
        self.update_epsilon(total_step=self.total_step)

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

        next_state_q = self.target_network.predict(next_states)
        next_state_max_q = tf.math.reduce_max(next_state_q, axis=1)
        target_q = rewards + self.gamma * next_state_max_q * (1.0 - tf.cast(dones, tf.float32))

        n_step_next_q = self.target_network.predict(n_step_next)
        n_step_max_q = tf.reduce_max(n_step_next_q, axis=1)
        n_target_q = n_step_rewards + self.gamma * n_step_max_q * (1.0 - tf.cast(n_step_dones, tf.float32))
        return target_q, n_target_q

    @tf.function
    def update_gradient(self, target_q, n_step_target_q, states, actions, batch_weights=1):

        """
        Update main q network by experience replay method.

        Args:
            target_q (tf.float32): Target Q value for barch.
            n_step_target_q (tf.int32): Target Q value after n_step.

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

            losses = self.policy_network.loss_function(main_q, target_q) * self.policy_network.one_step_weight
            losses += self.policy_network.loss_function(main_q, n_step_target_q) * self.policy_network.n_step_weight

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
            n_step_rewards, n_step_dones, n_step_next = self.buffer.get_n_step_sample(indices, gamma=self.gamma)
            target_q, n_step_target_q = self.get_target(rewards, dones, next_states,
                                                        n_step_rewards, n_step_dones, n_step_next)

            self.update_gradient(target_q, n_step_target_q, states, actions)

    def at_step_end(self, render=False):
        if self.total_step % self.target_sync_freq == 0:
            self.display_message("Synchronizing target policy...")
            self.sync_target_model()
        super().at_step_end(render=render)

    def learn(
            self,
            max_steps,
            target_reward=None,
            render=False,
    ):
        """
        Args:
            max_steps: Maximum number of total_step, if reached the training will stop.
            target_reward: Target_reward: The target moving average reward, if reached the training will stop,
            if null will be ignored
            render: display the env of training
        """
        self.init_training(max_steps)
        while True:
            self.check_episodes()
            if self.check_finish_training():
                break
            while not self.done:
                self.at_step_start()
                self.train_step()
                self.at_step_end(render)
