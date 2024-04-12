"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.05
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for the KITTI dataset

# Refer: https://github.com/ghimiredhikura/Complex-YOLOv3
"""

import config.kitti_config as cnf
from data_process import transformation, kitti_bev_utils, kitti_data_utils
from utils.evaluation_utils import rescale_boxes
import sys
import os
import random

import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision
import torch.nn.functional as F
import cv2
from scipy.spatial.transform import Rotation as R
from PIL import Image

sys.path.append('../')


normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))


class KittiDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', lidar_transforms=None, aug_transforms=None, multiscale=False,
                 num_samples=None, mosaic=False, random_padding=False):
        self.dataset_dir = dataset_dir
        assert mode in ['train', 'val',
                        'test'], 'Invalid mode: {}'.format(mode)
        self.mode = mode
        self.is_train = (self.mode == 'train')
        self.is_test = (self.mode == 'test')
        sub_folder = 'object\\' + ('testing' if self.is_test else 'training')

        self.multiscale = multiscale
        self.lidar_transforms = lidar_transforms
        self.aug_transforms = aug_transforms
        self.data_aug_conf = {
            'resize_lim': (0.193, 0.225),
            'final_dim': (128, 352),
            'rot_lim': (-5.4, 5.4),
            'H': 375, 'W': 1242,    # 370 x 1224
            'rand_flip': True,
            'bot_pct_lim': (0.0, 0.22),
            'cams': ['P2', 'P3'],
            'Ncams': 2,
        }
        self.img_size = cnf.BEV_WIDTH
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.mosaic = mosaic
        self.random_padding = random_padding
        self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]

        self.lidar_dir = os.path.join(self.dataset_dir, sub_folder, "velodyne")
        self.image_dir = os.path.join(self.dataset_dir, sub_folder, "image_2")
        self.calib_dir = os.path.join(self.dataset_dir, sub_folder, "calib")
        self.label_dir = os.path.join(self.dataset_dir, sub_folder, "label_2")
        split_txt_path = os.path.join(
            self.dataset_dir, 'ImageSets', '{}.txt'.format(mode))
        self.image_idx_list = [x.strip()
                               for x in open(split_txt_path).readlines()]

        if self.is_test:
            self.sample_id_list = [int(sample_id)
                                   for sample_id in self.image_idx_list]
        else:
            self.sample_id_list = self.remove_invalid_idx(self.image_idx_list)

        if num_samples is not None:
            self.sample_id_list = self.sample_id_list[:num_samples]
        self.num_samples = len(self.sample_id_list)

    def __getitem__(self, index):
        if self.is_test:
            return self.load_img_only(index)
        else:
            if self.mosaic:
                img_files, rgb_map, targets = self.load_mosaic(index)

                return img_files[0], rgb_map, targets
            else:
                return self.load_img_with_targets(index)

    def load_img_only(self, index):
        """Load only image for the testing phase"""

        sample_id = int(self.sample_id_list[index])
        lidarData = self.get_lidar(sample_id)
        b = kitti_bev_utils.removePoints(lidarData, cnf.boundary)

        rgb_map = kitti_bev_utils.makeBVFeature(
            b, cnf.DISCRETIZATION, cnf.boundary)
        img_file = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))

        return self.get_image_data(index), rgb_map

    def load_img_with_targets(self, index):
        """Load images and targets for the training and validation phase"""

        sample_id = int(self.sample_id_list[index])
        lidarData = self.get_lidar(sample_id)
        objects = self.get_label(sample_id)
        calib = self.get_calib(sample_id)

        labels, noObjectLabels = kitti_bev_utils.read_labels_for_bevbox(
            objects)

        if not noObjectLabels:
            labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
                                                               calib.P)  # convert rect cam to velo cord

        if self.lidar_transforms is not None:
            lidarData, labels[:, 1:] = self.lidar_transforms(
                lidarData, labels[:, 1:])

        b = kitti_bev_utils.removePoints(lidarData, cnf.boundary)
        rgb_map = kitti_bev_utils.makeBVFeature(
            b, cnf.DISCRETIZATION, cnf.boundary)
        target = kitti_bev_utils.build_yolo_target(labels)

        # on image space: targets are formatted as (box_idx, class, x, y, w, l, im, re)
        n_target = len(target)
        targets = torch.zeros((n_target, 8))
        if n_target > 0:
            targets[:, 1:] = torch.from_numpy(target)

        rgb_map = torch.from_numpy(rgb_map).float()

        if self.aug_transforms is not None:
            rgb_map, targets = self.aug_transforms(rgb_map, targets)
        # binimg = self.get_binimg(targets) # TODO fill binimg
        return self.get_image_data(index), rgb_map, targets

    def get_image_data(self, index):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        sample_id = int(self.sample_id_list[index])
        calib = self.get_calib(sample_id)
        v2c = calib.V2C
        rot = v2c[:3, :3]
        r = R.from_matrix(rot.tolist())
        # The value is in scalar-last (x, y, z, w) format.
        quat = r.as_quat()
        # Coordinate system orientation as quaternion: w, x, y, z. (4)
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])
        # Coordinate system origin in meters: x, y, z. (3)
        tran = v2c[:3, 3]
        for cam in ["P2", "P3"]:  # kitti cams.
            img_file = os.path.join(
                self.image_dir, '{:06d}.png'.format(sample_id))
            img = Image.open(img_file)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # Intrinsic camera calibration. Empty for sensors that are not cameras. (3x3)
            intrin = getattr(calib, cam)    # 3x4
            intrin = intrin[:, :3]           # 3x3

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = transformation.img_transform(
                img, post_rot, post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(torch.from_numpy(intrin))
            rots.append(torch.from_numpy(rot))
            trans.append(torch.from_numpy(tran))
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_binimg(self, targets):
        # sample_id = int(self.sample_id_list[index])
        # objects = self.get_label(sample_id)
        # calib = self.get_calib(sample_id)
        binimg = np.zeros((608, 608, 3))

        # Rescale target
        targets[:, 2:6] *= cnf.BEV_WIDTH
        # Get yaw angle
        targets[:, 6] = torch.atan2(targets[:, 6], targets[:, 7])

        for cls_id, x, y, w, l, yaw in targets[:, 1:7].numpy():
            # Draw rotated box
            # TODO fill box with cv2.fillPoly()
            kitti_bev_utils.drawRotatedBox(
                binimg, x, y, w, l, yaw, cnf.colors[int(cls_id)])

        # Draw camera(rgb image)
        # for obj in objects:
        #     if obj.type == 'DontCare': continue
        #     # cv2.rectangle(img2, (int(obj.xmin),int(obj.ymin)),
        #     #    (int(obj.xmax),int(obj.ymax)), (0,255,0), 2)
        #     box3d_pts_2d, box3d_pts_3d = kitti_data_utils.compute_box_3d(obj, calib.P)
        #     if box3d_pts_2d is not None:
        #         target = kitti_data_utils.draw_projected_box3d(target, box3d_pts_2d, cnf.colors[obj.cls_id])

        return binimg

        # TODO
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            box = Box(inst['translation'], inst['size'],
                      Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0)

    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def load_mosaic(self, index):
        """loads images in a mosaic
        Refer: https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py
        """

        targets_s4 = []
        img_file_s4 = []
        if self.random_padding:
            yc, xc = [int(random.uniform(-x, 2 * self.img_size + x))
                      for x in self.mosaic_border]  # mosaic center
        else:
            yc, xc = [self.img_size, self.img_size]  # mosaic center

        # 3 additional image indices
        indices = [index] + \
            [random.randint(0, self.num_samples - 1) for _ in range(3)]
        for i, index in enumerate(indices):
            img_file, img, targets = self.load_img_with_targets(index)
            img_file_s4.append(img_file)

            c, h, w = img.size()  # (3, 608, 608), torch tensor

            # place img in img4
            if i == 0:  # top left
                img_s4 = torch.full(
                    (c, self.img_size * 2, self.img_size * 2), 0.5, dtype=torch.float)
                # xmin, ymin, xmax, ymax (large image)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                # xmin, ymin, xmax, ymax (small image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(
                    yc - h, 0), min(xc + w, self.img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(
                    xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - \
                    (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(
                    xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img_s4[:, y1a:y2a, x1a:x2a] = img[:, y1b:y2b,
                                              x1b:x2b]  # img_s4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # on image space: targets are formatted as (box_idx, class, x, y, w, l, sin(yaw), cos(yaw))
            if targets.size(0) > 0:
                targets[:, 2] = (targets[:, 2] * w + padw) / \
                    (2 * self.img_size)
                targets[:, 3] = (targets[:, 3] * h + padh) / \
                    (2 * self.img_size)
                targets[:, 4] = targets[:, 4] * w / (2 * self.img_size)
                targets[:, 5] = targets[:, 5] * h / (2 * self.img_size)

            targets_s4.append(targets)
        if len(targets_s4) > 0:
            targets_s4 = torch.cat(targets_s4, 0)
            torch.clamp(targets_s4[:, 2:4], min=0., max=(
                1. - 0.5 / self.img_size), out=targets_s4[:, 2:4])

        return img_file_s4, img_s4, targets_s4

    def __len__(self):
        return len(self.sample_id_list)

    def remove_invalid_idx(self, image_idx_list):
        """Discard samples which don't have current training class objects, which will not be used for training."""

        sample_id_list = []
        for sample_id in image_idx_list:
            sample_id = int(sample_id)
            objects = self.get_label(sample_id)
            calib = self.get_calib(sample_id)
            labels, noObjectLabels = kitti_bev_utils.read_labels_for_bevbox(
                objects)
            if not noObjectLabels:
                labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
                                                                   calib.P)  # convert rect cam to velo cord

            valid_list = []
            for i in range(labels.shape[0]):
                if int(labels[i, 0]) in cnf.CLASS_NAME_TO_ID.values():
                    if self.check_point_cloud_range(labels[i, 1:4]):
                        valid_list.append(labels[i, 0])

            if len(valid_list) > 0:
                sample_id_list.append(sample_id)

        return sample_id_list

    def check_point_cloud_range(self, xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range = [cnf.boundary["minX"], cnf.boundary["maxX"]]
        y_range = [cnf.boundary["minY"], cnf.boundary["maxY"]]
        z_range = [cnf.boundary["minZ"], cnf.boundary["maxZ"]]

        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    def collate_fn(self, batch):
        cams, lids, targets = list(zip(*batch)) # tuple, torch.Size([3, 608, 608]), torch.Size([10, 8])
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if (self.batch_count % 10 == 0) and self.multiscale and (not self.mosaic):
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))
        # Resize lidars to input shape
        lids = torch.stack(lids)
        if self.img_size != cnf.BEV_WIDTH:
            lids = F.interpolate(lids, size=self.img_size,
                                 mode="bilinear", align_corners=True)
        # Unpack cameras
        cams = [torch.stack(cam) for cam in zip(*cams)]
        self.batch_count += 1

        return cams, lids, targets

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '{:06d}.png'.format(idx))
        # assert os.path.isfile(img_file)
        # (H, W, C) -> (H, W, 3) OpenCV reads in BGR mode
        return cv2.imread(img_file)

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '{:06d}.bin'.format(idx))
        # assert os.path.isfile(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '{:06d}.txt'.format(idx))
        # assert os.path.isfile(calib_file)
        return kitti_data_utils.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '{:06d}.txt'.format(idx))
        # assert os.path.isfile(label_file)
        return kitti_data_utils.read_label(label_file)

    def set_label(self, idx, preds):
        if not self.is_test:
            raise Exception("You can set label only in test mode")
        if not os.path.exists(self.label_dir):
            os.makedirs(self.label_dir)
        label_file = os.path.join(self.label_dir, '{:06d}.txt'.format(idx))
        with open(label_file, 'w') as file:
            file.writelines([pred.to_kitti_format()+'\n' for pred in preds])
