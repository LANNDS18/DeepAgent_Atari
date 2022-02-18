
import tensorflow as tf
from tensorflow.keras.losses import MSE

self = 1

class DQNUnused:
    def update_gradients(self, x, y):
        """
        Train on a given batch.
        Args:
            x: States tensor
            y: Targets tensor
        """
        with tf.GradientTape() as tape:
            y_pred = self.model_predict(x, self.model)[1]
            loss = MSE(y, y_pred)
        self.model.optimizer.minimize(loss, self.model.trainable_variables, tape=tape)
        self.train_loss(loss)

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

        new_state_values = tf.reduce_max(
            self.model_predict(new_states, self.target_model)[1], axis=1
        )

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

        next_state_q = self.target_model(new_states)
        next_state_max_q = tf.reduce_max(
            self.model_predict(new_states, self.target_model)[1], axis=1
        )
        next_state_max_q = tf.math.reduce_max(next_state_q, axis=1)
        expected_q = tf.cast(reward, tf.float32) + self.gamma * next_state_max_q * (
                1.0 - tf.cast(done, tf.float32))
        return expected_q

    def update_gradients(self, x, y):
        """
        Train on a given batch.
        Args:
            x: States tensor
            y: Targets tensor
        """
        with tf.GradientTape() as tape:
            y_pred = self.model_predict(x, self.model)[1]
            loss = MSE(y, y_pred)
        self.model.optimizer.minimize(loss, self.model.trainable_variables, tape=tape)
        self.train_loss(loss)