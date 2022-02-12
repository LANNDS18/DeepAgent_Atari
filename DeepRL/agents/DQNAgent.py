import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import MSE
from DeepRL.interfaces.IOffPolicyAgent import OffPolicyAgent


class DQNAgent(OffPolicyAgent):

    def __init__(
            self,
            env,
            model,
            buffer,
            epsilon_start=1.0,
            epsilon_end=0.02,
            epsilon_decay_steps=150000,
            target_sync_steps=1000,
            **kwargs,
    ):
        """
        Initialize networks agent.
        Args:
            env: A list of gym environments.
            model: tf.keras.model.Model that is expected to be compiled
                with an optimizer before training starts.
            buffer: A list of replay buffer objects whose length should match
                `env`s'.
            epsilon_start: Starting epsilon value which is used to control random exploration.
                It should be decremented and adjusted according to implementation needs.
            epsilon_end: End epsilon value which is the minimum exploration rate.
            epsilon_decay_steps: Number of steps for epsilon to reach `epsilon_end`
                from `epsilon_start`,
            target_sync_steps: Steps to sync target model after each.
            **kwargs: kwargs passed to super classes.
        """
        super(DQNAgent, self).__init__(env, model, buffer, **kwargs)
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        self.epsilon_start = self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_sync_steps = target_sync_steps
        self.batch_indices = tf.range(self.batch_size, dtype=tf.int64)[:, tf.newaxis]
        self.batch_dtypes = ['uint8', 'int64', 'float64', 'bool', 'uint8']

    @staticmethod
    def get_action_indices(batch_indices, actions):
        """
        Get indices that will be passed to tf.gather_nd()
        Args:
            batch_indices: tf.range() result of the same shape as the batch size.
            actions: Action tensor of same shape as the batch size.

        Returns:
            Indices as a tensor.
        """
        return tf.concat((batch_indices, tf.cast(actions[:, tf.newaxis], tf.int64)), -1)

    @tf.function
    def model_predict(self, inputs, model, training=True):
        """
        Get model outputs (action)
        Args:
            inputs: Inputs as tensors / numpy arrays that are expected
                by the given model.
            model: A tf.keras.Model or a list of tf.keras.Model(s)
            training:
        Returns:
            Outputs that is expected from the given model.
        """
        # q_values = model.predict(inputs)
        q_values = super(DQNAgent, self).model_predict(inputs, model, training=training)
        return tf.argmax(q_values, axis=1), q_values

    def update_epsilon(self):
        """
        Decrement epsilon which aims to gradually reduce randomization.
        """
        self.epsilon = max(
            self.epsilon_end, self.epsilon_start - self.steps / self.epsilon_decay_steps
        )

    def sync_target_model(self):
        """
        Sync target model weights with main's

        Returns:
            None
        """
        if self.steps % self.target_sync_steps == 0:
            self.target_model.set_weights(self.model.get_weights())

    def get_action(self):
        """
        Generate action following an epsilon-greedy policy.

        Returns:
            A random action or Q argmax.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        state = np.array([self.state])
        action = self.model_predict(state, self.model)[0].numpy().tolist()[0]
        return action

    def get_targets(self, state, action, reward, done, new_states):
        """
        Get targets for gradient update.
        Args:
            state: A tensor of shape (total buffer batch size, *self.input_shape)
            action: A tensor of shape (total buffer batch size)
            reward: A tensor of shape (total buffer batch size)
            done: A tensor of shape (s total buffer batch size)
            new_states: A tensor of shape (total buffer batch size, *self.input_shape)

        Returns:
            Target values, a tensor of shape (total buffer batch size, self.n_actions)
        """
        q_states = self.model_predict(state, self.model)[1]

        new_state_values = tf.reduce_max(
            self.model_predict(new_states, self.target_model)[1], axis=1
        )

        new_state_values = tf.where(
            tf.cast(done, tf.bool),
            tf.constant(0, new_state_values.dtype),
            new_state_values,
        )

        target_values = tf.identity(q_states)

        target_value_update = new_state_values * self.gamma + tf.cast(reward, tf.float32)
        indices = self.get_action_indices(self.batch_indices, action)
        target_values = tf.tensor_scatter_nd_update(
            target_values, indices, target_value_update
        )
        return target_values

    def update_gradients(self, x, y):
        """
        Train on a given batch.
        Args:
            x: States tensor
            y: Targets tensor

        Below not compatible with Apple Silicon


        """
        with tf.GradientTape() as tape:
            y_pred = self.model_predict(x, self.model)[1]
            loss = MSE(y, y_pred)
        self.model.optimizer.minimize(loss, self.model.trainable_variables, tape=tape)

    def at_step_start(self):
        """
        Execute steps that will run before self.train_step() which decays epsilon.
        """
        self.update_epsilon()

    # @tf.function
    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.env, batching and gradient updates.
        """

        action = tf.numpy_function(self.get_action, [], tf.int64)
        tf.numpy_function(self.step_env, [action, True], [])
        training_batch = tf.numpy_function(
            self.buffer.get_sample,
            [],
            self.batch_dtypes,
        )
        '''
        action = self.get_action()
        self.step_env(action=action, store_in_buffers=True)
        training_batch = self.buffer.get_sample()
        '''

        targets = self.get_targets(*training_batch)
        self.update_gradients(training_batch[0], targets)

    def at_step_end(self):
        """
        Execute steps that will run after self.train_step() which
        updates target model.
        """
        self.sync_target_model()

    def fit(
            self,
            target_reward=None,
            max_steps=None,
            monitor_session=None,
    ):
        """
        Common training loop shared by subclasses, monitors training status
        and progress, performs all training steps, updates metrics, and logs progress.
        Args:
            target_reward: Target reward, if achieved, the training will stop
            max_steps: Maximum number of steps, if reached the training will stop.
            monitor_session: Session name to use for monitoring the training with wandb.
        """
        self.init_training(target_reward, max_steps, monitor_session)
        while True:
            self.check_episodes()
            if self.check_finish_training():
                break
            self.at_step_start()
            self.train_step()
            self.at_step_end()
