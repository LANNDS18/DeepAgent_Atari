import cv2
import numpy as np
import json


def write_from_dict(_dict, path):
    """
    Args:
        _dict: Dictionary of label: [scalar]
        path: Path to .json file.
    """
    with open(path, 'w') as fp:
        json.dump(_dict, fp)


def process_frame(frame, shape=(84, 84)):
    """
    Preprocesses a 210x160x3 frame to 84x84x1 grayscale
    Arguments:
        shape:
        frame: The frame to process.  Must have values ranging from 0-255
    Returns:
        The processed frame
    """
    frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, shape)
    frame = frame.reshape(*shape, -1)
    return frame
