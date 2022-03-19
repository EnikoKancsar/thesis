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


# def resize(img, kpt, center, ratio):
#     """Resize the ``numpy.ndarray`` and points as ratio.

#     Args:
#         img    (numpy.ndarray):   Image to be resized.
#         kpt    (list):            Keypoints to be resized.
#         center (list):            Center points to be resized.
#         ratio  (tuple or number): the ratio to resize.

#     Returns:
#         numpy.ndarray: Resized image.
#         lists:         Resized keypoints.
#         lists:         Resized center points.
#     """

#     if not (isinstance(ratio, numbers.Number) or (isinstance(ratio, collections.Iterable) and len(ratio) == 2)):
#         raise TypeError('Got inappropriate ratio arg: {}'.format(ratio))
    
#     h, w, _ = img.shape
#     if w < 64:
#         img = cv2.copyMakeBorder(img, 0, 0, 0, 64 - w, cv2.BORDER_CONSTANT, value=(128, 128, 128))
#         w = 64
    
#     if isinstance(ratio, numbers.Number):
#         num = len(kpt)
#         for i in range(num):
#             kpt[i][0] *= ratio
#             kpt[i][1] *= ratio
#         center[0] *= ratio
#         center[1] *= ratio
#         return cv2.resize(img, (0, 0), fx=ratio, fy=ratio), kpt, center
#     else:

#         num = len(kpt)
#         for i in range(num):
#             kpt[i][0] *= ratio[0]
#             kpt[i][1] *= ratio[1]
#         center[0] *= ratio[0]
#         center[1] *= ratio[1]
#         # for i in range(len(center)):
#             # center[i][0] *= ratio[0]
#             # center[i][1] *= ratio[1]

#     return np.ascontiguousarray(cv2.resize(img,(int(img.shape[0]*ratio[0]),int(img.shape[1]*ratio[1])),interpolation=cv2.INTER_CUBIC)), kpt, center


# class TestResized(object):
#     """Resize the given numpy.ndarray to the size for test.

#     Args:
#         size: the size to resize.
#     """

#     def __init__(self, size):
#         assert (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2))
#         if isinstance(size, int):
#             self.size = (size, size)
#         else:
#             self.size = size

#     @staticmethod
#     def get_params(img, output_size):

#         height, width, _ = img.shape
        
#         return (output_size[0] * 1.0 / height, output_size[1] * 1.0 / width)

#     def __call__(self, img, kpt, center):
#         """
#         Args:
#             img     (numpy.ndarray): Image to be resized.
#             kpt     (list):          keypoints to be resized.
#             center: (list):          center points to be resized.

#         Returns:
#             numpy.ndarray: Randomly resize image.
#             list:          Randomly resize keypoints.
#             list:          Randomly resize center points.
#         """

#         ratio = self.get_params(img, self.size)

#         return resize(img, kpt, center, ratio)



class Compose(object):
    """Composes several transforms together.

    Example:
        >>> transforms.Compose([
        >>>      transforms.RandomResized(),
        >>>      transforms.RandomRotate(40),
        >>>      transforms.RandomCrop(368),
        >>>      transforms.RandomHorizontalFlip(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, kpt, center, scale=None):

        for t in self.transforms:
            if isinstance(t, RandomResized):
                img, kpt, center = t(img, kpt, center, scale)
            else:
                img, kpt, center = t(img, kpt, center)

        return img, kpt, center