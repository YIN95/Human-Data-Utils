# -*- coding:utf-8 -*-
"""
Author:
    Wenjie Yin, yinw@kth.se
"""

import os
import re
import quaternion
import torch

import numpy as np
import pandas as pd
import PIL.Image as Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

actionList = {'acting1', 'acting2', 'acting3',
              'freestyle1', 'freestyle2', 'freestyle3',
              'rom1', 'rom2', 'rom3',
              'walking1', 'walking2', 'walking3'}

gt = {'Ori', 'Pos'}


def get_transform(mode='default'):
    if mode in ['default']:
        return transforms.Compose([
            transforms.ToTensor(),
        ])


class TotalCapture(Dataset):
    def __init__(self, data_path, mode='train'):
        super(TotalCapture, self).__init__()

        self.data_path = data_path
        self.image_transform = get_transform('default')
        '''
        # The train and test partition are performed wrt to the subjects
        # and sequences, the training is perfomed on subjects 1,2 and 3
        # of the following seqeunces:
        #     ROM 1,2,3
        #     Walking 1,3
        #     Freestyle 1,2
        #     Acting 1,2

        # test set is perfomed on sbjects 1,2,3,4 and 5,
        # on the following seqeunces：
        #     Walking 2
        #     Freestyle 3
        #     Acting 3
        '''
        self.training_subjects = ['S1', 'S2', 'S3']
        self.testing_subjects = ['S1', 'S2', 'S3', 'S4', 'S5']

        self.training_actions = ['acting1', 'acting2',
                                 'freestyle1', 'freestyle2',
                                 'walking1', 'walking3',
                                 'rom1', 'rom2', 'rom3']
        self.testing_actions = ['acting3', 'freestyle3', 'walking2']

        if mode == 'train':
            self.subjects = self.training_subjects
            self.actions = self.training_actions
        elif mode == 'test':
            self.subjects = self.testing_subjects
            self.actions = self.testing_actions
        elif mode == 'debug':
            self.subjects = ['S1']
            self.actions = ['acting2']

        self.data_dict = {}
        self._length = 0

        for sub in tqdm(self.subjects,
                        total=len(self.subjects),
                        desc='Scanning Data'):
            self.data_dict[sub] = {}
            for act in tqdm(self.actions,
                            total=len(self.actions),
                            desc='Subject - {}'.format(sub)):
                self.data_dict[sub][act] = {}
                self.data_dict[sub][act]['len_imu'] = self.get_len_imu(
                    sub, act)
                self.data_dict[sub][act]['len_video'] = self.get_len_video(
                    sub, act)
                self.data_dict[sub][act]['len'] = min(
                    self.data_dict[sub][act]['len_imu'],
                    self.data_dict[sub][act]['len_video'])

                self.data_dict[sub][act]['imu'] = self.init_imu(sub, act)

                self._length += self.data_dict[sub][act]['len']

    def get_len_imu(self, sub, act):
        imu_path = os.path.join(self.data_path, sub, 'imu')
        sub_lower = str.lower(sub)
        imu_sensors_path = os.path.join(
            imu_path, '{0}_{1}_Xsens.sensors'.format(sub_lower, act))
        with open(imu_sensors_path, mode='r', newline='\n') as f:
            first_line = re.compile(' |\t').split(f.readline())

        return int(first_line[1])

    def get_len_video(self, sub, act):
        return 999999
        images_path = os.path.join(self.data_path, sub, 'images', act)
        camera_1 = os.listdir(images_path)[0]
        images_camera_1_path = os.path.join(images_path, camera_1)

        return len(os.listdir(images_camera_1_path))

    def get_len_data(self, sub, act):
        return min(self.get_len_imu(sub, act), self.get_len_video(sub, act))

    def init_imu(self, sub, act):
        imu_path = os.path.join(self.data_path, sub, 'imu')
        sub_lower = str.lower(sub)

        imu_bone_path = os.path.join(
            imu_path, '{0}_{1}_calib_imu_bone.txt'.format(sub_lower, act))
        imu_ref_path = os.path.join(
            imu_path, '{0}_{1}_calib_imu_ref.txt'.format(sub_lower, act))
        imu_sensors_path = os.path.join(
            imu_path, '{0}_{1}_Xsens.sensors'.format(sub_lower, act))

        # load calib_imu_bone, then convert xyzw to wxyz
        imu_bone = pd.read_csv(
            imu_bone_path,
            sep=' |\t',
            names=list(range(5)),
            engine='python'
        )
        imu_bone = imu_bone.iloc[1:, 1:].values.astype(np.float32)  # xyzw
        imu_bone = np.tile(imu_bone, (1, 2))[:, 3:7]  # wxyz

        imu_ref = pd.read_csv(
            imu_ref_path,
            sep=' |\t',
            names=list(range(5)),
            engine='python'
        )
        imu_ref = imu_ref.iloc[1:, 1:].values.astype(np.float32)  # xyzw
        imu_ref = np.tile(imu_ref, (1, 2))[:, 3:7]  # wxyz

        imu_sensors = pd.read_csv(
            imu_sensors_path,
            sep=' |\t',
            names=list(range(8)),
            engine='python'
        )
        imu_sensors = imu_sensors.iloc[1:, 1:].values.astype(
            np.float32).reshape(-1, 14, 7)[:, 1:14, :4]     # wxyz

        imu_sensors_length = imu_sensors.shape[0]       # length of imu data
        imu_sensors_num = imu_sensors.shape[1]      # the number of sensors

        # sensors data to quaternion
        imu_sensors = imu_sensors.reshape(-1, 4)
        imu_sensors = quaternion.from_float_array(imu_sensors)

        # bone
        # inverse
        imu_bone = -imu_bone
        imu_bone[:, 0] = -imu_bone[:, 0]
        imu_bone = quaternion.from_float_array(imu_bone)
        imu_bone = np.tile(imu_bone, imu_sensors_length)

        # reference
        imu_ref = quaternion.from_float_array(imu_ref)
        imu_ref = np.tile(imu_ref, imu_sensors_length)

        # R^g_bi = R_ig Â· R_i Â· (R_ib)^âˆ’1
        imu_sensors = imu_ref*imu_sensors*imu_bone
        imu_sensors = quaternion.as_float_array(imu_sensors).astype(np.float32)

        imu_sensors = imu_sensors.reshape(
            imu_sensors_length, imu_sensors_num, 4)

        return imu_sensors

    def get_imgs(self, sub, act, index):
        _images_path = os.path.join(self.data_path, sub, 'images', act)
        _cameras = os.listdir(_images_path)
        imgs = []
        for _camera in _cameras:
            _path = os.path.join(_images_path, _camera, str(index)+'.jpg')
            _img = Image.open(_path)
            _img = self.image_transform(_img)
            # _img
            imgs.append(_img)
        return imgs

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        _sum = 0
        _next_sum = 0
        for sub in self.subjects:
            for act in self.actions:
                _sum = _next_sum
                _next_sum = _sum + self.data_dict[sub][act]['len']
                if index < _next_sum:
                    sub_index = index-_sum
                    _imu = self.data_dict[sub][act]['imu'][sub_index]
                    # _imgs = self.get_imgs(sub, act, sub_index)
                    return torch.tensor(_imu).float()

if __name__ == '__main__':
    data_path = '/media/ywj/Data/totalcapture/totalcapture'
    tp_data = TotalCapture(data_path, mode='debug')

    tp_data_loader = DataLoader(tp_data, batch_size=1, shuffle=False)
    for i in tp_data_loader:
        print(i)
