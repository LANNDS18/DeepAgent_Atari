import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_from_dict(_dict, path):
    """
    Write to .parquet given a dict
    Args:
        _dict: Dictionary of label: [scalar]
        path: Path to .parquet file.

    Returns:
        None
    """
    table = pa.Table.from_pydict(_dict)
    pq.write_to_dataset(table, root_path=path, compression='gzip')


# This function can resize to any shape, but was built to resize to 84x84
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
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    frame = frame.reshape(*shape, -1)
    # cv2.imshow('DemonAttack', frame)
    return frame


