from abc import ABC

import numpy as np
import tensorflow as tf


class BaseNNPolicy(ABC):

    def __init__(self,
                 conv_layers=None,
                 dense_layers=None,
                 input_shape=(84, 84),
                 frame_stack=4,
                 n_actions=6,
                 optimizer=tf.keras.optimizers.Adam,
                 lr_schedule=None,
                 loss_function=tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE),
                 one_step_weight=1.0,
                 n_step_weight=1.0,
                 l2_weight=0.0,
                 quiet=False,
                 ):

        conv_layers = {
            'filters': [32, 64, 64],
            'kernel_sizes': [8, 4, 3],
            'strides': [4, 2, 1],
            'paddings': ['valid' for _ in range(3)],
            'activations': ['relu' for _ in range(3)],
            'initializers': [tf.initializers.VarianceScaling(scale=2.0) for _ in range(3)],
            'names': ['conv_%i' % i for i in range(1, 4)]
        } if conv_layers is None else conv_layers

        dense_layers = {
            'units': [512],
            'activations': ['relu'],
            'initializers': [tf.initializers.VarianceScaling(scale=2.0)],
            'names': ['dense_1']
        } if dense_layers is None else dense_layers

        if lr_schedule is None:
            lr_schedule = [[0.00025, 0.00005, 500000], [0.00005, 0.00001, 1000000]]

        # define training parameters
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.frame_stack = frame_stack

        self.conv_layers = conv_layers
        self.dense_layers = dense_layers

        self.optimizer = optimizer(learning_rate=lr_schedule[0][0])
        self.loss_function = loss_function
        self.one_step_weight = one_step_weight
        self.n_step_weight = n_step_weight
        self.l2_weight = l2_weight

        self.lr_lag = 0
        self.lr_schedule = np.array(lr_schedule)
        self.lr_schedule[:, 2] = np.cumsum(self.lr_schedule[:, 2])
        self.update_counter = 0

        self.model = self.build()
        if not quiet:
            self.model.summary()

    def build(self):
        """
        Build the keras model from self.dense_layers and self.conv_layers
        """
        raise NotImplementedError

    def load(self, path):
        self.model.load_weights(path)

    def save(self, path):
        self.model.save_weights(path)

    @tf.function
    def predict(self, states):
        """Perform a forward pass through the network, (Predict Q values)"""
        predictions = self.model(states, training=False)
        return predictions

    @tf.function
    def get_optimal_actions(self, states):
        """Get the optimal actions for some states corresponding
           to the current policy defined by the network parameters"""
        q_value = self.model(states, training=False)
        action = tf.cast(tf.squeeze(tf.math.argmax(q_value, axis=1)), dtype=tf.int32)
        return action

    def _get_current_lr(self):
        if self.update_counter > self.lr_schedule[0, 2] and self.lr_schedule.shape[0] > 1:
            self.lr_schedule = np.delete(self.lr_schedule, 0, 0)
            self.lr_lag = self.update_counter
        max_lr, min_lr, lr_steps = self.lr_schedule[0]
        lr = max_lr - min(1, (self.update_counter - self.lr_lag) / (lr_steps - self.lr_lag)) * (max_lr - min_lr)
        return lr

    def update_lr(self):
        self.update_counter += 1
        self.optimizer._set_hyper('learning_rate', self._get_current_lr())

    def get_last_conv2d_name(self):
        last_conv_layer_name = list(filter(lambda x: isinstance(x, tf.keras.layers.Conv2D), self.model.layers))[-1].name
        return last_conv_layer_name
