class BaseNN:
    def __init__(self):
        pass
    """
    The base class for networks
    """
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
        raise NotImplementedError(f'The build should be implement in by {self.__class__.__name__} subclasses')