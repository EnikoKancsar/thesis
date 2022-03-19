from collections import namedtuple
import os

import numpy as np
from torch import LongTensor
from torch import nn
from torch import prod
from torch.utils.data import DataLoader

from unipose.data.mpii import MPII
# import unipose.transforms


def gaussian_kernel(size_w, size_h, center_x, center_y, sigma):
    """Gaussian Kernel (or Radial Basic Function (RBF) kernel)
    It is most widely used.
    Each kernel entry is a dissimilarity measure through using the square of
    Euclidean distance between two data points in a negative exponential.
    The sigma parameter contained in the entry is the Parzen window width for
    RBF kernel.

    source: Han 2011, Sigma Tuning of Gaussian Kernels
    https://www.cs.rpi.edu/~szymansk/papers/han.10.pdf

    """
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    summary = []

    ModuleDetails = namedtuple(
        "Layer",
        ["name", "input_size", "output_size", "num_params", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if (class_name.find("Conv") != -1
                    or class_name.find("BatchNorm") != -1
                    or class_name.find("Linear") != -1):
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    prod(LongTensor(list(module.weight.data.size())))
                    * prod(LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (prod(LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_params=params,
                    multiply_adds=flops))

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_params
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_params,
                ' ' * (space_len - len(str(layer.num_params))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(
        flops_sum/(1024**3)) + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details


def getDataloader(dataset, train_dir, val_dir, test_dir, sigma, stride,
                  workers, batch_size):
    """ torch.utils.data.Dataloader
    
    :param dataset: 
    :param batch_size (int, optional, default=1)
        how many samples per batch to load
    :param shuffle: (bool, optional, default=False)
        True: have the data reshuffled at every epoch
    :param num_workers: (int, optional, default=0)
        how many subprocesses to use for data loading.
        0: the data will be loaded in the main process
    :param pin_memory: (bool, optional, default=False)
        True: the data loader will copy Tensors into CUDA pinned memory
              before returning them
    """

    if dataset == 'MPII':
        train_loader = DataLoader(
            MPII(train_dir, sigma, "Train", stride
                #  transforms.Compose([transforms.TestResized(368),])
                 ),
            batch_size=batch_size, shuffle=True, num_workers=workers,
            pin_memory=True)

        val_loader = DataLoader(
            MPII(val_dir, sigma, "Val",
                #  transforms.Compose([transforms.TestResized(368),])
                 ),
            batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

        test_loader = DataLoader(
            MPII(test_dir, sigma, "Val",
                #  transforms.Compose([transforms.TestResized(368),])
                 ),
            batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    return train_loader, val_loader, test_loader
