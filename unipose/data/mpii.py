# -*-coding:UTF-8-*-
import configparser
import json
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data

from thesis.unipose import transforms


CONF = configparser.ConfigParser()
CONF.read('./conf.ini')


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


class MPII(data.Dataset):
    def __init__(self, sigma, is_train, stride=8):
        self.width       = 368
        self.height      = 368
        self.is_train    = is_train
        self.sigma       = sigma
        self.stride      = stride

        self.videosFolders = {}
        self.labelFiles    = {}
        self.full_img_List = {}
        self.numPeople     = []

        self.images_dir, anno_file = (
            (CONF.get("MPII", "DIR_IMAGES_TRAIN"),
             CONF.get("MPII", "ANNOTATIONS_TRAIN"))
            if self.is_train is True
            else (CONF.get("MPII", "DIR_IMAGES_VAL"),
                  CONF.get("MPII", "ANNOTATIONS_VAL"))
        )

        with open(anno_file) as anno_file:
            self.annotations = json.load(anno_file)

    def __getitem__(self, index):
        scale_factor = 0.25
        annotation = self.annotations[index]
        
        img_path = os.path.join(self.images_dir, annotation['image_name'])

        joints = annotation['list_of_people'][0]['joints']
        points = []
        for joint_id, joint_value in joints.items():
            point = [
                joint_value["x"], joint_value["y"], joint_value["is_visible"]
            ]
            if point[0]==-1.0:
                point[2] = 0.00
            points.append(point)
        points = torch.Tensor(points)

        objpos = annotation['list_of_people'][0]['objpos']
        center = torch.Tensor([objpos['x'], objpos['y']])
        
        scale  = annotation['list_of_people'][0]['scale']

        if center[0] != -1:
            center[1] = center[1] + 15*scale
            scale     = scale*1.25

        # Single Person
        nParts = points.size(0)
        img    = cv2.imread(img_path)

        kpt = points

        if img.shape[0] != 368 or img.shape[1] != 368:
            kpt[:,0] = kpt[:,0] * (368/img.shape[1])
            kpt[:,1] = kpt[:,1] * (368/img.shape[0])
            img = cv2.resize(img, (368,368))
        height, width, _ = img.shape

        heatmap = np.zeros(
            (int(height/self.stride), int(width/self.stride), int(len(kpt)+1)),
            dtype=np.float32)
        for i in range(len(kpt)):
            # resize from 368 to 46
            x = int(kpt[i][0]) * 1.0 / self.stride
            y = int(kpt[i][1]) * 1.0 / self.stride
            heat_map = gaussian_kernel(
                size_h=int(height/self.stride), 
                size_w=int(width/self.stride),
                center_x=x, center_y=y, sigma=self.sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[:, :, i + 1] = heat_map

        # for background
        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)

        centermap = np.zeros(
            (int(height/self.stride), int(width/self.stride), 1),
            dtype=np.float32)
        center_map = gaussian_kernel(
            size_h=int(height/self.stride),
            size_w=int(width/self.stride),
            center_x=int(center[0]/self.stride),
            center_y=int(center[1]/self.stride),
            sigma=3)
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        centermap[:, :, 0] = center_map

        orig_img = cv2.imread(img_path)
        img = transforms.normalize(transforms.to_tensor(img),
                                   [128.0, 128.0, 128.0],
                                   [256.0, 256.0, 256.0])
        heatmap   = transforms.to_tensor(heatmap)
        centermap = transforms.to_tensor(centermap)

        return img, heatmap, centermap, img_path

    def __len__(self):
        return len(self.annotations)
