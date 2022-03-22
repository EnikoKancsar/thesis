import torch


def normalize(tensor, mean, std):
    """Normalize a torch.tensor

    :param tensor (torch.tensor): tensor to be normalized
    :param mean: (list): the mean of BGR
    :param std: (list): the std of BGR
    """
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def to_tensor(pic):
    """Convert a numpy.ndarray to tensor.

    h , w , c -> c, h, w
    :param pic (numpy.ndarray): Image to be converted to tensor.

    :returns: Tensor: Converted image.
    """

    img = torch.from_numpy(pic.transpose((2, 0, 1)))

    return img.float()
