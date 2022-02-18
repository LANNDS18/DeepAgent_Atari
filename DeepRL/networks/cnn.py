import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda


class DQNNetwork(Model):
    """
    Class for DQN model architecture.
    """

    def __init__(self, n_actions, frame_stack=4, input_shape=(84, 84)):
        super(DQNNetwork, self).__init__()
        self.normalize = Lambda(lambda x: x / 255.0)
        self.conv1 = Conv2D(filters=32, kernel_size=8, strides=4,
                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu",
                            input_shape=(None, input_shape[0], input_shape[1], frame_stack))
        self.conv2 = Conv2D(filters=64, kernel_size=4, strides=2,
                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")
        self.conv3 = Conv2D(filters=64, kernel_size=3, strides=1,
                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")
        self.flatten = Flatten()
        self.dense1 = Dense(512, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation='relu')
        self.dense2 = Dense(n_actions, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
                            activation="linear")

    @tf.function
    def call(self, x):
        normalized = self.normalize(x)
        l1 = self.conv1(normalized)
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        l4 = self.flatten(l3)
        l5 = self.dense1(l4)
        output = self.dense2(l5)
        return output
