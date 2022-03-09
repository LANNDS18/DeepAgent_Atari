import tensorflow as tf
import numpy as np

from keras.models import Model
from keras.losses import Huber
from keras.optimizers import rmsprop_v2
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from keras.initializers.initializers_v2 import VarianceScaling

from DeepAgent.interfaces.ibasePolicy import BaseNNPolicy


class DeepQNetwork(BaseNNPolicy):

    def __init__(self,
                 conv_layers=None,
                 dense_layers=None,
                 input_shape=(84, 84),
                 frame_stack=4,
                 n_actions=6,
                 optimizer=rmsprop_v2.RMSprop,
                 lr_schedule=None,
                 loss_function=Huber(reduction=tf.keras.losses.Reduction.NONE),
                 one_step_weight=1.0,
                 l2_weight=0.0):

        super(DeepQNetwork, self).__init__(conv_layers,
                                           dense_layers,
                                           input_shape,
                                           frame_stack,
                                           n_actions,
                                           optimizer,
                                           lr_schedule,
                                           loss_function,
                                           one_step_weight,
                                           l2_weight)

    def build(self):

        model_input = Input(shape=(self.input_shape[0], self.input_shape[1], self.frame_stack))
        scale = Lambda(lambda p: p / 255.0)(model_input)

        conv_layers = []

        for layer_id in tf.range(len(self.conv_layers['filters'])):
            if layer_id == 0:
                conv_input = scale
            else:
                conv_input = conv_layers[-1]

            conv_layers.append(Conv2D(filters=self.conv_layers['filters'][layer_id],
                                      kernel_size=self.conv_layers['kernel_sizes'][layer_id],
                                      strides=self.conv_layers['strides'][layer_id],
                                      padding=self.conv_layers['paddings'][layer_id],
                                      activation=self.conv_layers['activations'][layer_id],
                                      kernel_initializer=self.conv_layers['initializers'][layer_id],
                                      name=self.conv_layers['names'][layer_id],
                                      use_bias=False
                                      )(conv_input))

        flatten = Flatten()(conv_layers[-1])
        dense_layers = []

        for layer_id in tf.range(len(self.dense_layers['units'])):
            if layer_id == 0:
                dense_input = flatten
            else:
                dense_input = dense_layers[-1]

            dense_layers.append(Dense(units=self.dense_layers['units'][layer_id],
                                      activation=self.dense_layers['activations'][layer_id],
                                      kernel_initializer=self.dense_layers['initializers'][layer_id],
                                      name=self.dense_layers['names'][layer_id]
                                      )(dense_input))

        out_layer = Dense(units=self.n_actions,
                          kernel_initializer=VarianceScaling(scale=2.0),
                          name='out_layer'
                          )(dense_layers[-1])

        model = Model(inputs=[model_input], outputs=[out_layer])
        model.summary()
        return model

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
