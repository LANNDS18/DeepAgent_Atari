class IBaseBuffer:
    """
    Base class for replay buffer.
    """

    def __init__(self, size, initial_size=None, batch_size=32):
        """
        Initialize replay buffer.
        Args:
            size: Buffer maximum size.
            initial_size: Buffer initial size to be filled before training starts.
                To be used by caller.
            batch_size: Size of the batch that should be used in get_sample() implementation.
        """
        assert (
            initial_size is None or initial_size > 0
        ), f'Buffer initial size should be > 0, got {initial_size}'
        assert size > 0, f'Buffer size should be > 0,  got {size}'
        assert batch_size > 0, f'Buffer batch size should be > 0, got {batch_size}'
        assert (
            batch_size <= size
        ), f'Buffer batch size `{batch_size}` should be <= size `{size}`'
        if initial_size:
            assert size >= initial_size, 'Buffer initial size exceeds max size'
        self.size = size
        self.initial_size = initial_size or size
        self.batch_size = batch_size
        self.current_size = 0

    def append(self, *args):
        """
        Add experience to buffer.
        Args:
            *args: Items to store, types are implementation specific.
        """
        raise NotImplementedError(
            f'append() should be implemented by {self.__class__.__name__} subclasses'
        )

    def get_sample(self):
        """
        Sample from stored experience.
        Returns:
            Sample as numpy array.
        """
        raise NotImplementedError(
            f'get_sample() should be implemented by {self.__class__.__name__} subclasses'
        )

