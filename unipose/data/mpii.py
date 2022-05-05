# -*-coding:UTF-8-*-
import configparser
import json
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data

from thesis.unipose.data import gaussian_kernel
from thesis.unipose.data import normalize
from thesis.unipose.data import to_tensor


CONF = configparser.ConfigParser()
CONF.read('./conf.ini')


class MPII(data.Dataset):
    def __init__(self, sigma, is_train, stride=8):
        self.width    = 368
        self.height   = 368
        self.is_train = is_train
        self.sigma    = sigma
        self.stride   = stride

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
        annotation = self.annotations[index]
        
        img_path = os.path.join(self.images_dir, annotation['image_name'])

        joints_of_first_person = annotation['list_of_people'][0]['joints']
        joints = []
        for joint_coordinates in joints_of_first_person.values():
            point = [
                joint_coordinates["x"], joint_coordinates["y"],
                joint_coordinates["is_visible"]
            ]
            if point[0]==-1.0:
                point[2] = 0.00
            joints.append(point)
        joints = torch.Tensor(joints)

        # Single Person
        img = cv2.imread(img_path)

        # img.shape = height, width, channels AND NOT width, height order
        if img.shape[0] != self.height or img.shape[1] != self.width:
            joints[:,0] = joints[:,0] * (self.width/img.shape[1])
            joints[:,1] = joints[:,1] * (self.height/img.shape[0])
            img = cv2.resize(img, (self.width, self.height))
        height, width, _ = img.shape

        heatmap = np.zeros(
            (int(height/self.stride), int(width/self.stride), int(len(joints)+1)),
            dtype=np.float32)
        for joint in range(len(joints)):
            # resize from 368 to 46
            x = int(joints[joint][0]) * 1.0 / self.stride
            y = int(joints[joint][1]) * 1.0 / self.stride
            heatmap_of_joint = gaussian_kernel(
                size_h=int(height/self.stride), size_w=int(width/self.stride),
                center_x=x, center_y=y,
                sigma=self.sigma)
            heatmap_of_joint[heatmap_of_joint > 1] = 1
            heatmap_of_joint[heatmap_of_joint < 0.0099] = 0
            heatmap[:, :, joint + 1] = heatmap_of_joint

        # for background - what does it do?
        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)

        # i don't understand what normalize does exactly or why it is important
        normalized_img = normalize(
            to_tensor(img), [128.0, 128.0, 128.0], [256.0, 256.0, 256.0])
        # heatmap.shape -> 46, 46, 17
        heatmap = to_tensor(heatmap)

        return normalized_img, heatmap

    def __len__(self):
        return len(self.annotations)
