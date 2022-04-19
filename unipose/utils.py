from collections import namedtuple
import os

import cv2  # image analysis
import numpy as np
from torch import LongTensor
from torch import nn
from torch import prod
from torch.utils.data import DataLoader

from thesis.unipose.data.mpii import MPII


def adjust_learning_rate(optimizer, iters, base_lr, gamma, step_size,
                         policy='step', multiple=[1]):
    if policy == 'fixed':
        lr = base_lr
    elif policy == 'step':
        lr = base_lr * (gamma ** (iters // step_size))

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    return lr


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
            if isinstance(output, (list, tuple)):
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
        details += "{} : {} layers\n".format(layer, layer_instances[layer])

    return details


def getDataloader(dataset, sigma, stride, workers, batch_size):
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
            MPII(sigma, is_train=True, stride=stride),
            batch_size=batch_size, shuffle=True, num_workers=workers,
            pin_memory=True)

        val_loader = DataLoader(
            MPII(sigma, is_train=False, stride=stride),
            batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

        test_loader = DataLoader(
            MPII(sigma, is_train=False, stride=stride),
            batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_kpts(maps, img_h = 368.0, img_w = 368.0):

    # maps (1,15,46,46)
    maps = maps.clone().cpu().data.numpy()
    map_6 = maps[0]

    kpts = []
    for m in map_6[1:]:
        h, w = np.unravel_index(m.argmax(), m.shape)
        x = int(w * img_w / m.shape[1])
        y = int(h * img_h / m.shape[0])
        kpts.append([x,y])
    return kpts



def draw_paint(im, kpts, mapNumber, epoch, model_arch, dataset):

           #       RED           GREEN           RED          YELLOW          YELLOW          PINK          GREEN
    colors = [[000,000,255], [000,255,000], [000,000,255], [255,255,000], [255,255,000], [255,000,255], [000,255,000],\
              [255,000,000], [255,255,000], [255,000,255], [000,255,000], [000,255,000], [000,000,255], [255,255,000], [255,000,000]]
           #       BLUE          YELLOW          PINK          GREEN          GREEN           RED          YELLOW           BLUE

    if dataset == "LSP":
        limbSeq = [[13, 12], [12, 9], [12, 8], [9, 10], [8, 7], [10,11], [7, 6], [12, 3],\
                    [12, 2], [ 2, 1], [ 1, 0], [ 3, 4], [4,  5], [15,16], [16,18], [17,18], [15,17]]
        kpts[15][0] = kpts[15][0]  - 25
        kpts[15][1] = kpts[15][1]  - 50
        kpts[16][0] = kpts[16][0]  - 25
        kpts[16][1] = kpts[16][1]  + 50
        kpts[17][0] = kpts[17][0] + 25
        kpts[17][1] = kpts[17][1] - 50
        kpts[18][0] = kpts[18][0] + 25
        kpts[18][1] = kpts[18][1] + 50

    elif dataset == "MPII":
                #    HEAD    R.SLDR  R.BICEP  R.FRARM   L.SLDR  L.BICEP  L.FRARM   TORSO    L.HIP   L.THIGH   L.CALF   R.HIP   R.THIGH   R.CALF  EXT.HEAD
        limbSeq = [[ 8, 9], [ 7,12], [12,11], [11,10], [ 7,13], [13,14], [14,15], [ 7, 6], [ 6, 2], [ 2, 1], [ 1, 0], [ 6, 3], [ 3, 4], [ 4, 5], [ 7, 8]]

    # im = cv2.resize(cv2.imread(img_path),(368,368))
    # draw points
    for k in kpts:
        x = k[0]
        y = k[1]
        cv2.circle(im, (x, y), radius=3, thickness=-1, color=(0, 0, 255))

    # draw lines
    for i in range(len(limbSeq)):
        cur_im = im.copy()
        limb = limbSeq[i]
        [Y0, X0] = kpts[limb[0]]
        [Y1, X1] = kpts[limb[1]]
        # mX = np.mean([X0, X1])
        # mY = np.mean([Y0, Y1])
        # length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
        # angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
        # polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
        # cv2.fillConvexPoly(cur_im, polygon, colors[i])
        # if X0!=0 and Y0!=0 and X1!=0 and Y1!=0:
        #     im = cv2.addWeighted(im, 0.4, cur_im, 0.6, 0)

        if X0!=0 and Y0!=0 and X1!=0 and Y1!=0:
            if i<len(limbSeq)-4:
                cv2.line(cur_im, (Y0,X0), (Y1,X1), colors[i], 5)
            else:
                cv2.line(cur_im, (Y0,X0), (Y1,X1), [0,0,255], 5)

        im = cv2.addWeighted(im, 0.2, cur_im, 0.8, 0)

    cv2.imwrite('samples/WASPpose/Pose/'+str(mapNumber)+'.png', im)


def printAccuracies(mAP, AP, mPCKh, PCKh, mPCK, PCK, dataset):
    if dataset == "MPII":
        print("\nmAP:   %.2f%%" % (mAP*100))
        print("APs:   %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%"\
            % (AP[0]*100,AP[1]*100,AP[2]*100,AP[3]*100,AP[4]*100,AP[5]*100,AP[6]*100,AP[7]*100,AP[8]*100,AP[9]*100,PCKh[10]*100,\
                AP[11]*100,AP[12]*100,AP[13]*100,AP[14]*100,AP[15]*100,AP[16]*100))

        print("mPCK:  %.2f%%" % (mPCK*100))
        print("PCKs:  %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%"\
            % (PCK[0]*100,PCK[1]*100,PCK[2]*100,PCK[3]*100,PCK[4]*100,PCK[5]*100,PCK[6]*100,PCK[7]*100,PCK[8]*100,PCK[9]*100,PCK[10]*100,\
                PCK[11]*100,PCK[12]*100,PCK[13]*100,PCK[14]*100,PCK[15]*100,PCK[16]*100))

        print("mPCKh: %.2f%%" % (mPCKh*100))
        print("PCKhs: %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%"\
            % (PCKh[0]*100,PCKh[1]*100,PCKh[2]*100,PCKh[3]*100,PCKh[4]*100,PCKh[5]*100,PCKh[6]*100,PCKh[7]*100,PCKh[8]*100,PCKh[9]*100,\
                PCKh[10]*100,PCKh[11]*100,PCKh[12]*100,PCKh[13]*100,PCKh[14]*100,PCKh[15]*100,PCKh[16]*100))

        with open('./output.txt', 'a') as output_file:
            output_file.write("\nmAP:   %.2f%%" % (mAP*100))
            output_file.write("APs:   %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%"\
            % (AP[0]*100,AP[1]*100,AP[2]*100,AP[3]*100,AP[4]*100,AP[5]*100,AP[6]*100,AP[7]*100,AP[8]*100,AP[9]*100,PCKh[10]*100,\
                AP[11]*100,AP[12]*100,AP[13]*100,AP[14]*100,AP[15]*100,AP[16]*100))

            output_file.write("mPCK:  %.2f%%" % (mPCK*100))
            output_file.write("PCKs:  %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%"\
            % (PCK[0]*100,PCK[1]*100,PCK[2]*100,PCK[3]*100,PCK[4]*100,PCK[5]*100,PCK[6]*100,PCK[7]*100,PCK[8]*100,PCK[9]*100,PCK[10]*100,\
                PCK[11]*100,PCK[12]*100,PCK[13]*100,PCK[14]*100,PCK[15]*100,PCK[16]*100))

            output_file.write("mPCKh: %.2f%%" % (mPCKh*100))
            output_file.write("PCKhs: %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%, %2.2f%%"\
            % (PCKh[0]*100,PCKh[1]*100,PCKh[2]*100,PCKh[3]*100,PCKh[4]*100,PCKh[5]*100,PCKh[6]*100,PCKh[7]*100,PCKh[8]*100,PCKh[9]*100,\
                PCKh[10]*100,PCKh[11]*100,PCKh[12]*100,PCKh[13]*100,PCKh[14]*100,PCKh[15]*100,PCKh[16]*100))
