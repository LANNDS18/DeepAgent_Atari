import tensorflow as tf
from DeepRL.agents.double_dqn import DoubleDQNAgent
from tensorflow.keras.losses import MSE


class D3NPERAgent(DoubleDQNAgent):

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
            buffer: A prioritized experience replay buffer objects
            **kwargs: kwargs passed to super classes.
        """
        super(D3NPERAgent, self).__init__(env, model, buffer, **kwargs)
        self.tds_error = 0

    def update_gradients_PER(self, x, y, action):
        """
        Train on a given batch.
        Args:
            x: States tensor
            y: Targets tensor
            action: action from self.get_action
        """
        with tf.GradientTape() as tape:
            y_pred = self.model_predict(x, self.model)[1]
            loss = MSE(y, y_pred)
        self.model.optimizer.minimize(loss, self.model.trainable_variables, tape=tape)
        self.train_loss(loss)
        self.tds_error = tf.abs(y[:, action] - y_pred[:, action])

    @tf.function
    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.env_name, batching and gradient updates.
        """
        action = tf.numpy_function(self.get_action, [], tf.int64)
        tf.numpy_function(self.step_env, [action], [])
        training_batch = tf.numpy_function(
            self.buffer.get_sample,
            [],
            self.batch_dtypes,
        )
        targets = self.get_targets(*training_batch)
        self.update_gradients_PER(training_batch[0], targets, action)
        tf.numpy_function(self.buffer.update_priorities, [self.tds_error], [])
