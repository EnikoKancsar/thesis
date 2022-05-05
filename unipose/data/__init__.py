import numpy as np
import torch


def gaussian_kernel(size_w, size_h, center_x, center_y, sigma):
    """Gaussian Kernel (or Radial Basic Function (RBF) kernel)
    It is most widely used.
    Each kernel entry is a dissimilarity measure through using the square of
    Euclidean distance between two data points in a negative exponential.
    The sigma parameter contained in the entry is the Parzen window width for
    RBF kernel.

    source: Han 2011, Sigma Tuning of Gaussian Kernels
    https://www.cs.rpi.edu/~szymansk/papers/han.10.pdf

    numpy.mgrid()
    array([
    x= [[ 0,  0,  0, ...,  0,  0,  0],
        [ 1,  1,  1, ...,  1,  1,  1],
        [ 2,  2,  2, ...,  2,  2,  2],
        ...,
        [43, 43, 43, ..., 43, 43, 43],
        [44, 44, 44, ..., 44, 44, 44],
        [45, 45, 45, ..., 45, 45, 45]],

    y= [[ 0,  1,  2, ..., 43, 44, 45],
        [ 0,  1,  2, ..., 43, 44, 45],
        [ 0,  1,  2, ..., 43, 44, 45],
        ...,
        [ 0,  1,  2, ..., 43, 44, 45],
        [ 0,  1,  2, ..., 43, 44, 45],
        [ 0,  1,  2, ..., 43, 44, 45]]
    ])
    """
    grid_y, grid_x = np.mgrid[0:size_h, 0:size_w]
    D2 = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


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
