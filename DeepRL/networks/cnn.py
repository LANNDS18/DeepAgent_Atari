from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_dqn_network(n_actions, learning_rate=0.00001, input_shape=(84, 84), frame_stack=4):
    """Builds a cnn networks as a Keras model
    Arguments:
        n_actions: Number of possible action the agent can take
        learning_rate: Learning rate
        input_shape: Shape of the preprocessed frame the model sees
        frame_stack: length of frame stack
    Returns:
        A compiled Keras model
    """
    model_input = Input(shape=(input_shape[0], input_shape[1], frame_stack))
    x = Lambda(lambda p: p / 255.0)(model_input)
    x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
        x)
    x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
        x)
    x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
        x)
    x = Flatten()(x)
    x = Dense(512, kernel_initializer=VarianceScaling(scale=2.))(x)
    out = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(x)

    model = Model(model_input, out)
    model.compile(Adam(learning_rate=learning_rate))
    model.summary()

    return model
