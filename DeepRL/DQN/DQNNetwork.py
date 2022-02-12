import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input, Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# Define how to calculate loss
def dqn_loss(data, y_pred):
    """
    data: action with corresponding reward
    y_pred: predict y value
    """
    # Extract data and target action
    action_batch, target_q_values = data[:, 0], data[:, 1]
    # Construct an action sequence, it based on action index to gather the y_pred abd corresponding q_value
    # initialise this sequence
    seq = tf.cast(tf.range(0, tf.shape(action_batch)[0]), tf.int32)
    action_index = tf.transpose(tf.stack([seq, tf.cast(action_batch, tf.int32)]))
    q_values = tf.gather_nd(y_pred, action_index)
    return tf.keras.losses.mse(q_values, target_q_values)


def create_dqn_model(optimizer=None, input_shape=(84, 84), n_action=6):
    """
    Argument:
        data: action with corresponding reward
        y_pred: predict y value
    Return:
        A compiled NN model for agent to learn
    """
    # Initial optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) if optimizer is None else optimizer
    # Build a sequential model
    network = Sequential()
    network.add(Input(shape=(input_shape[0], input_shape[1], 1)))
    network.add((Conv2D(filters=32, kernel_size=(8, 8), kernel_initializer=VarianceScaling(scale=2.),
                        activation='relu', use_bias=False)))
    network.add(Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.),
                       activation='relu', use_bias=False))
    network.add(Flatten())  # Flat the convolution layer
    network.add(Dense(32, activation='relu'))
    network.add(Dense(16, activation='relu'))
    network.optimizer = optimizer
    network.add(Dense(n_action, activation='linear'))
    network.compile(optimizer=optimizer, loss=dqn_loss)
    return network


def build_q_network(n_actions, learning_rate=0.00001, input_shape=(84, 84)):
    """Builds a dueling DQN as a Keras model
    Arguments:
        n_actions: Number of possible action the agent can take
        learning_rate: Learning rate
        input_shape: Shape of the preprocessed frame the model sees
    Returns:
        A compiled Keras model
    """
    '''
    model = Sequential(
        [
            Input(shape=(input_shape[0], input_shape[1], 1)),
            Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu',
                   use_bias=False),
            Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu',
                   use_bias=False),
            Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu',
                   use_bias=False),
            Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu',
                   use_bias=False),
            Flatten(),
            Dense(64, kernel_initializer=VarianceScaling(scale=2.)),
            Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))
        ]
    )
    '''
    model_input = Input(shape=(input_shape[0], input_shape[1], 1))
    x = model_input  # normalize by 255
    x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
        x)
    x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
        x)
    x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
        x)
    x = Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu',
               use_bias=False)(x)

    # Split into value and advantage streams
    val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)  # custom splitting layer

    val_stream = Flatten()(val_stream)
    val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)

    adv_stream = Flatten()(adv_stream)
    adv = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)

    # Combine streams into Q-Values
    reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))  # custom layer for reduce mean

    q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])

    # Build model
    model = Model(model_input, q_vals)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())
    model.summary()

    return model
