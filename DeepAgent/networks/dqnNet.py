import tensorflow as tf

from DeepAgent.interfaces.ibaseNetwork import BaseNetwork


class DQNNetwork(BaseNetwork):

    def __init__(self, **kwargs):

        super(DQNNetwork, self).__init__(**kwargs)

    def build(self):

        model_input = tf.keras.layers.Input(shape=(self.input_shape[0], self.input_shape[1], self.frame_stack))
        scale = tf.keras.layers.Lambda(lambda p: p / 255.0)(model_input)

        conv_layers = []

        for layer_id in tf.range(len(self.conv_layers['filters'])):
            if layer_id == 0:
                conv_input = scale
            else:
                conv_input = conv_layers[-1]

            conv_layers.append(tf.keras.layers.Conv2D(filters=self.conv_layers['filters'][layer_id],
                                                      kernel_size=self.conv_layers['kernel_sizes'][layer_id],
                                                      strides=self.conv_layers['strides'][layer_id],
                                                      padding=self.conv_layers['paddings'][layer_id],
                                                      activation=self.conv_layers['activations'][layer_id],
                                                      kernel_initializer=self.conv_layers['initializers'][layer_id],
                                                      name=self.conv_layers['names'][layer_id],
                                                      use_bias=False
                                                      )(conv_input))

        flatten = tf.keras.layers.Flatten()(conv_layers[-1])
        dense_layers = []

        for layer_id in tf.range(len(self.dense_layers['units'])):
            if layer_id == 0:
                dense_input = flatten
            else:
                dense_input = dense_layers[-1]

            dense_layers.append(tf.keras.layers.Dense(units=self.dense_layers['units'][layer_id],
                                                      activation=self.dense_layers['activations'][layer_id],
                                                      kernel_initializer=self.dense_layers['initializers'][layer_id],
                                                      name=self.dense_layers['names'][layer_id]
                                                      )(dense_input))

        out_layer = tf.keras.layers.Dense(units=self.n_actions,
                                          kernel_initializer=tf.initializers.VarianceScaling(scale=2.0),
                                          name='out_layer'
                                          )(dense_layers[-1])

        model = tf.keras.models.Model(inputs=[model_input], outputs=[out_layer])
        return model
