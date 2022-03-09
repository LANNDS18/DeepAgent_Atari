import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from keras.initializers.initializers_v2 import VarianceScaling
from DeepAgent.interfaces.ibasePolicy import BaseNNPolicy


class Dueling(BaseNNPolicy):

    def __init__(self, **kwargs):

        super(Dueling, self).__init__(**kwargs)

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

        value_stream, advantage_stream = tf.split(conv_layers[-1], 2, 3)

        value_layer = Dense(units=1,
                            kernel_initializer=VarianceScaling(scale=2.0),
                            name='value_layer'
                            )(Flatten()(value_stream))

        advantage_layer = Dense(units=self.n_actions,
                                kernel_initializer=VarianceScaling(scale=2.0),
                                name='advantage_layer'
                                )(Flatten()(advantage_stream))

        out_layer = value_layer + tf.math.subtract(advantage_layer,
                                                   tf.reduce_mean(advantage_layer, axis=1,
                                                                  keepdims=True))

        model = Model(inputs=[model_input], outputs=[out_layer])
        model.summary()
        return model
