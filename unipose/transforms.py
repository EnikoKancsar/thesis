# import collections
# import cv2
# import numbers
# import numpy as np
# import random
import torch

def to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.

    h , w , c -> c, h, w

    See ``ToTensor`` for more details.

    Args:
        pic (numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """

    img = torch.from_numpy(pic.transpose((2, 0, 1)))

    return img.float()
