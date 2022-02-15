import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Add, Lambda, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_dueling_network(n_actions, learning_rate=0.00001, input_shape=(84, 84), frame_stack=4):
    """Builds a dueling networks as a Keras model
    Arguments:
        n_actions: Number of possible action the agent can take
        learning_rate: Learning rate
        input_shape: Shape of the preprocessed frame the model sees
        frame_stack: The length of the stack of frames
    Returns:
        A compiled Keras model
    """
    input = Input(shape=(input_shape[0], input_shape[1], frame_stack))
    x = input  # normalize by 255
    x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
        x)
    x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
        x)
    x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
        x)
    x = Flatten()(x)
    x = Dense(128, kernel_initializer=VarianceScaling(scale=2.))(x)
    x = Dense(64, kernel_initializer=VarianceScaling(scale=2.))(x)

    value_output = Dense(1)(x)
    advantage_output = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(x)

    reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))
    output = Add()([value_output, Subtract()([advantage_output, reduce_mean(advantage_output)])])

    model = Model(input, output)
    model.compile(Adam(learning_rate))
    model.summary()

    return model
