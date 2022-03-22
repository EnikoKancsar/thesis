import torch

def to_tensor(pic):
    """Convert a numpy.ndarray to tensor.

    h , w , c -> c, h, w
    :param pic (numpy.ndarray): Image to be converted to tensor.

    :returns: Tensor: Converted image.
    """

    img = torch.from_numpy(pic.transpose((2, 0, 1)))

    return img.float()
