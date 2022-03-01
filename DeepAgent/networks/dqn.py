from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from DeepAgent.interfaces.ibaseNN import BaseNN


class DQNNetwork(BaseNN):
    def __init__(self):
        super().__init__()

    def build(self, n_actions, learning_rate=0.00001, input_shape=(84, 84), frame_stack=4):
        """Builds a dueling networks as a Keras model
        Arguments:
            n_actions: Number of possible action the agent can take
            learning_rate: Learning rate
            input_shape: Shape of the preprocessed frame the model sees
            frame_stack: The length of the stack of frames
        Returns:
            A compiled Keras model
        """
        model_input = Input(shape=(input_shape[0], input_shape[1], frame_stack))
        x = Lambda(lambda p: p / 255.0)(model_input)
        x = Conv2D(filters=32, kernel_size=8, strides=4,
                   kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
        x = Conv2D(filters=64, kernel_size=4, strides=2,
                   kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
        x = Conv2D(filters=64, kernel_size=3, strides=1,
                   kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)

        x = Flatten()(x)
        x = Dense(512, kernel_initializer=VarianceScaling(scale=2.), activation='relu')(x)

        output = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.0), activation="linear")(x)
        model = Model(model_input, output)
        model.compile(RMSprop(learning_rate=learning_rate, epsilon=1e-6))
        model.summary()

        return model
