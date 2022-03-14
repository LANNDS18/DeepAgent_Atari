import cv2
import numpy as np


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize buffer usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the policy_network.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


def process_frame(frame, shape=(84, 84), crop=None):
    """
    Preprocesses a 210x160x3 frame to 84x84x1 grayscale
    Arguments:
        shape: The output shape
        crop: The top and bottom boundary for crop
        frame: The frame to process.  Must have values ranging from 0-255
    Returns:
        The processed frame
    """
    frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if crop:
        frame = crop(frame)
    frame = cv2.resize(frame, shape)
    frame = frame.reshape(*shape, -1)
    return frame

