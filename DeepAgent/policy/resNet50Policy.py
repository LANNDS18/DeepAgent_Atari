import tensorflow as tf

from DeepAgent.interfaces.ibasePolicy import BaseNNPolicy


class ResNet50Policy(BaseNNPolicy):

    def __init__(self, **kwargs):

        super(ResNet50Policy, self).__init__(**kwargs)

    def build(self):
        model_input = tf.keras.layers.Input(shape=(self.input_shape[0], self.input_shape[1], self.frame_stack))
        scale = tf.keras.layers.Lambda(lambda p: p / 255.0)(model_input)
        pre_input = tf.keras.applications.mobilenet.preprocess_input(scale)

        x = tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_tensor=pre_input)(pre_input)

        dense_layers = []

        for layer_id in tf.range(len(self.dense_layers['units'])):
            if layer_id == 0:
                dense_input = tf.keras.layers.Flatten()(x)
            else:
                dense_input = dense_layers[-1]

            dense_layers.append(tf.keras.layers.Dense(units=self.dense_layers['units'][layer_id],
                                                      activation=self.dense_layers['activations'][layer_id],
                                                      kernel_initializer=self.dense_layers['initializers'][layer_id],
                                                      name=self.dense_layers['names'][layer_id]
                                                      )(dense_input))

        value_layer = tf.keras.layers.Dense(units=1,
                                            kernel_initializer=tf.initializers.VarianceScaling(scale=2.0),
                                            name='value_layer'
                                            )(dense_layers[-1])

        advantage_layer = tf.keras.layers.Dense(units=self.n_actions,
                                                kernel_initializer=tf.initializers.VarianceScaling(scale=2.0),
                                                name='advantage_layer'
                                                )(dense_layers[-1])

        out_layer = value_layer + tf.math.subtract(advantage_layer,
                                                   tf.reduce_mean(advantage_layer, axis=1,
                                                                  keepdims=True))

        model = tf.keras.models.Model(inputs=[model_input], outputs=[out_layer])
        return model
