from __future__ import print_function
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
import random

import torch
from torch.utils import data
from torchvision import transforms

from .util import Equirec2Cube


def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
            # ld_r_u = line.strip().split(" ")
            # ld = []
            # ld.append(ld_r_u[0])
            # ld.append(ld_r_u[3])
            # # ld.append(ld_r_u[6])
            # rgb_depth_list.append(ld)
            # r = []
            # r.append(ld_r_u[1])
            # r.append(ld_r_u[4])
            # # r.append(ld_r_u[7])
            # rgb_depth_list.append(r)
            # u = []
            # u.append(ld_r_u[2])
            # u.append(ld_r_u[5])
            # # u.append(ld_r_u[8])
            # rgb_depth_list.append(u)
    return rgb_depth_list


def recover_filename(file_name):

    splits = file_name.split('.')
    rot_ang = splits[0].split('_')[-1]
    file_name = splits[0][:-len(rot_ang)] + "0." + splits[-2] + "." + splits[-1]

    return file_name, int(rot_ang)


class ThreeD60(data.Dataset):
    """The 3D60 Dataset"""

    def __init__(self, root_dir, list_file, height=256, width=512, disable_color_augmentation=False,
                 disable_LR_filp_augmentation=False, disable_yaw_rotation_augmentation=False, is_training=False):
        """
        Args:
            root_dir (string): Directory of the 3D60 Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir
        self.rgb_depth_list = read_list(list_file)
        self.w = width
        self.h = height
        self.is_training = is_training

        # self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)

        self.color_augmentation = not disable_color_augmentation
        self.LR_filp_augmentation = not disable_LR_filp_augmentation
        self.yaw_rotation_augmentation = not disable_yaw_rotation_augmentation
        self.transform = transforms.ToPILImage()
        self.max_depth_meters = 8.0#10.0
        try:
            self.brightness = [0.8, 1.2]
            self.contrast = [0.8, 1.2]
            self.saturation = [0.8, 1.2]
            self.hue = [-0.1, 0.1]
            self.color_aug= transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)#.get_params
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
            self.color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)#.get_params

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}

        rgb_name, rot_ang = recover_filename(os.path.join(self.root_dir, self.rgb_depth_list[idx][0]))

        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h))

        depth_name, _ = recover_filename(os.path.join(self.root_dir, self.rgb_depth_list[idx][1]))
        gt_depth = cv2.imread(depth_name, cv2.IMREAD_ANYDEPTH)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth[gt_depth>self.max_depth_meters] = self.max_depth_meters + 1

        if self.is_training and self.yaw_rotation_augmentation:
            # random rotation
            roll_idx = random.randint(0, self.w//4) + (self.w*rot_ang)//360
        else:
            roll_idx = (self.w * rot_ang) // 360

        rgb = np.roll(rgb, roll_idx, 1)
        gt_depth = np.roll(gt_depth, roll_idx, 1)

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)


        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = self.transform(rgb)
            aug_rgb = self.color_aug(aug_rgb)
            aug_rgb = np.asarray(aug_rgb)
        else:
            aug_rgb = rgb

        #cube_rgb, cube_gt_depth = self.e2c.run(rgb, gt_depth[..., np.newaxis])
        # cube_rgb = self.e2c.run(rgb)
        # cube_aug_rgb = self.e2c.run(aug_rgb)

        rgb = self.to_tensor(rgb.copy())
        # cube_rgb = self.to_tensor(cube_rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())
        # cube_aug_rgb = self.to_tensor(cube_aug_rgb.copy())

        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(aug_rgb)

        # inputs["cube_rgb"] = cube_rgb
        # inputs["normalized_cube_rgb"] = self.normalize(cube_aug_rgb)

        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        inputs["val_mask"] = ((inputs["gt_depth"] > 0.1) & (inputs["gt_depth"] <= self.max_depth_meters)
                                & ~torch.isnan(inputs["gt_depth"]))

        """
        cube_gt_depth = torch.from_numpy(np.expand_dims(cube_gt_depth[..., 0], axis=0))
        inputs["cube_gt_depth"] = cube_gt_depth
        inputs["cube_val_mask"] = ((cube_gt_depth > 0) & (cube_gt_depth <= self.max_depth_meters)
                                   & ~torch.isnan(cube_gt_depth))
        """
        return inputs



