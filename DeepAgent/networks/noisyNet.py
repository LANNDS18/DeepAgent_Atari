import numpy as np
import tensorflow as tf

from DeepAgent.interfaces.ibaseNetwork import BaseNetwork


class NoisyNet(BaseNetwork):

    def __init__(self, dense_layers=None, **kwargs):

        super(NoisyNet, self).__init__(dense_layers=dense_layers, **kwargs)
        if dense_layers is None:
            self.dense_layers = None
        self.build()

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

        if self.dense_layers is None:
            value_stream, advantage_stream = tf.split(conv_layers[-1], 2, 3)
            value_stream = tf.keras.layers.Flatten()(value_stream)
            advantage_stream = tf.keras.layers.Flatten()(advantage_stream)
        else:
            dense_layers = []

            for layer_id in tf.range(len(self.dense_layers['units'])):
                if layer_id == 0:
                    dense_input = tf.keras.layers.Flatten()(conv_layers[-1])
                else:
                    dense_input = dense_layers[-1]

                dense_layers.append(NoisyDense(
                    units=self.dense_layers['units'][layer_id],
                    name=self.dense_layers['names'][layer_id]
                )(dense_input))
                dense_layers.append(tf.keras.activations.get(
                    self.dense_layers['activations'][layer_id]
                )(dense_layers[-1]))

            value_stream = dense_layers[-1]
            advantage_stream = dense_layers[-1]

        value_layer_2 = NoisyDense(units=1, name='value_layer')(value_stream)

        advantage_layer_2 = NoisyDense(units=self.n_actions, name='advantage_layer')(advantage_stream)

        out_layer = value_layer_2 + tf.math.subtract(advantage_layer_2,
                                                     tf.reduce_mean(advantage_layer_2, axis=1,
                                                                    keepdims=True))

        model = tf.keras.models.Model(inputs=[model_input], outputs=[out_layer])
        self.model = model
        super().build()


class NoisyDense(tf.keras.layers.Layer):

    def __init__(self, units, std_init=0.5, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)

        self.units = units

        self.std_init = std_init

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.reset_noise(input_dim)

        mu_range = 1 / np.sqrt(input_dim)
        mu_initializer = tf.random_uniform_initializer(-mu_range, mu_range)
        sigma_initializer = tf.constant_initializer(self.std_init / np.sqrt(self.units))

        self.weight_mu = tf.Variable(initial_value=mu_initializer(shape=(input_dim, self.units), dtype='float32'),
                                     trainable=True)

        self.weight_sigma = tf.Variable(initial_value=sigma_initializer(shape=(input_dim, self.units), dtype='float32'),
                                        trainable=True)

        self.bias_mu = tf.Variable(initial_value=mu_initializer(shape=(self.units,), dtype='float32'),
                                   trainable=True)

        self.bias_sigma = tf.Variable(initial_value=sigma_initializer(shape=(self.units,), dtype='float32'),
                                      trainable=True)
        self.built = True

    def call(self, inputs):
        self.kernel = self.weight_mu + self.weight_sigma * self.weights_eps
        self.bias = self.bias_mu + self.bias_sigma * self.bias_eps
        return tf.matmul(inputs, self.kernel) + self.bias

    @staticmethod
    def _scale_noise(dim):
        noise = tf.random.normal([dim])
        return tf.sign(noise) * tf.sqrt(tf.abs(noise))

    def reset_noise(self, input_shape):
        eps_in = self._scale_noise(input_shape)
        eps_out = self._scale_noise(self.units)
        self.weights_eps = tf.multiply(tf.expand_dims(eps_in, 1), eps_out)
        self.bias_eps = eps_out

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)