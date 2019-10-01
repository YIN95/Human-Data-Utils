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

gt = {'Ori', 'Pos'}


def get_transform(mode='default'):
    if mode in ['default']:
        return transforms.Compose([
            transforms.ToTensor(),
        ])


class TotalCapture(Dataset):
    def __init__(self, data_path, cues='all', mode='train'):
        super(TotalCapture, self).__init__()

        self.data_path = data_path
        self.image_transform = get_transform('default')
        self.cues = cues
        '''
        The train and test partition are performed wrt to the subjects
        and sequences, the training is perfomed on subjects 1,2 and 3
        of the following seqeunces:
            ROM 1,2,3
            Walking 1,3
            Freestyle 1,2
            Acting 1,2

        test set is perfomed on sbjects 1,2,3,4 and 5,
        on the following seqeunces：
            Walking 2
            Freestyle 3
            Acting 3
        '''
        self.training_subjects = ['S1', 'S2', 'S3']
        self.testing_subjects = ['S1', 'S2', 'S3', 'S4', 'S5']
        self.training_actions = ['acting1', 'acting2',
                                 'freestyle1', 'freestyle2',
                                 'walking1', 'walking3',
                                 'rom1', 'rom2', 'rom3']
        self.testing_actions = ['acting3', 'freestyle3', 'walking2']

        self.c_training_subjects = ['S1', 'S2', 'S3']
        self.c_testing_subjects = ['S4', 'S5']
        self.c_training_actions = ['acting1', 'acting2', 'acting3',
                                   'freestyle1', 'freestyle2', 'freestyle3'
                                   'walking1', 'walking2', 'walking3',
                                   'rom1', 'rom2', 'rom3']
        self.c_testing_actions = ['acting3', 'freestyle1',
                                  'rom3', 'freestyle3', 'walking2']

        # The IMU data is provided by 13 sensors on key body parts
        self._imu_joint = ['Head', 'Sternum', 'Pelvis',
                           'L_UpArm', 'R_UpArm', 'L_LowArm', 'R_LowArm',
                           'L_UpLeg', 'R_UpLeg', 'L_LowLeg', 'R_LowLeg',
                           'L_Foot', 'R_Foot']
        self.imu_joint = ['Head', 'Spine3', 'Hips',
                          'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm',
                          'LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg',
                          'LeftFoot', 'RightFoot']
        # The automatically labelled ground truth utilises the optical
        # marker based Vicon system, it calculates 21 3D world joint
        # positions and angles.
        self.vicon_joint = ['Hips',
                            'Spine', 'Spine1', 'Spine2', 'Spine3',
                            'Neck', 'Head',
                            'RightShoulder', 'RightArm',
                            'RightForeArm', 'RightHand',
                            'LeftShoulder', 'LeftArm',
                            'LeftForeArm', 'LeftHand',
                            'RightUpLeg', 'RightLeg', 'RightFoot',
                            'LeftUpLeg', 'LeftLeg', 'LeftFoot']

        if mode == 'train':
            self.subjects = self.training_subjects
            self.actions = self.training_actions
        elif mode == 'test':
            self.subjects = self.testing_subjects
            self.actions = self.testing_actions
        if mode == 'c-train':
            self.subjects = self.c_training_subjects
            self.actions = self.c_training_actions
        elif mode == 'c-test':
            self.subjects = self.c_testing_subjects
            self.actions = self.c_testing_actions
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
                self.data_dict[sub][act]['vicon'] = self.init_vicon(sub, act)

                self.data_dict[sub][act]['imu_mean'] = \
                    np.mean(self.data_dict[sub][act]['imu'], axis=(0, 1))
                self.data_dict[sub][act]['imu_std'] = \
                    np.std(self.data_dict[sub][act]['imu'], axis=(0, 1))
                self.data_dict[sub][act]['vicon_mean'] = \
                    np.mean(self.data_dict[sub][act]['vicon'], axis=(0, 1))
                self.data_dict[sub][act]['vicon_std'] = \
                    np.std(self.data_dict[sub][act]['vicon'], axis=(0, 1))
                self._length += self.data_dict[sub][act]['len']

        self.data_dict['imu_mean'] = self.get_mean('imu')
        self.data_dict['vicon_mean'] = self.get_mean('vicon')
        self.data_dict['imu_std'] = self.get_std('imu')
        self.data_dict['vicon_std'] = self.get_std('vicon')

        self.normalization()

    def normalization(self):

        self.data_dict['imu_mean'] = self.get_mean('imu')
        self.data_dict['vicon_mean'] = self.get_mean('vicon')
        self.data_dict['imu_std'] = self.get_std('imu')
        self.data_dict['vicon_std'] = self.get_std('vicon')

        for sub in self.subjects:
            for act in self.actions:
                self.data_dict[sub][act]['imu'] -= \
                    self.data_dict['imu_mean']
                self.data_dict[sub][act]['vicon'] -= \
                    self.data_dict['vicon_mean']

        self.data_dict['imu_max'] = self.get_max('imu')
        self.data_dict['vicon_max'] = self.get_max('vicon')

        for sub in self.subjects:
            for act in self.actions:
                self.data_dict[sub][act]['imu'] /= \
                    self.data_dict['imu_max']
                self.data_dict[sub][act]['vicon'] /= \
                    self.data_dict['vicon_max']

    def get_mean(self, cue):
        temp = np.zeros((7, ))
        for sub in self.subjects:
            for act in self.actions:
                temp += self.data_dict[sub][act][cue+'_mean'] * \
                    self.data_dict[sub][act]['len']
        temp /= self._length
        return temp

    def get_max(self, cue):
        cue_max = np.zeros((7, ))
        for sub in self.subjects:
            for act in self.actions:
                cue_max_temp = np.max(self.data_dict[sub][act][cue],
                                      axis=(0, 1))
                cue_max = np.array([cue_max, cue_max_temp])
                cue_max = np.max(cue_max, axis=0)

        return cue_max

    def get_std(self, cue):
        temp = np.zeros((7, ))
        for sub in self.subjects:
            for act in self.actions:
                temp += self.data_dict[sub][act]['len'] * \
                    (pow(self.data_dict[sub][act][cue+'_mean'], 2) +
                     pow(self.data_dict[sub][act][cue+'_std'], 2))
        temp /= self._length
        temp -= pow(self.data_dict[cue+'_mean'], 2)
        std = np.sqrt(temp)
        return std

    def get_len_imu(self, sub, act):
        imu_path = os.path.join(self.data_path, sub, 'imu')
        sub_lower = str.lower(sub)
        imu_sensors_path = os.path.join(
            imu_path, '{0}_{1}_Xsens.sensors'.format(sub_lower, act))
        with open(imu_sensors_path, mode='r', newline='\n') as f:
            first_line = re.compile(' |\t').split(f.readline())

        return int(first_line[1])

    def get_len_video(self, sub, act):
        images_path = os.path.join(self.data_path, sub, 'images', act)
        camera_1 = os.listdir(images_path)[0]
        images_camera_1_path = os.path.join(images_path, camera_1)

        return len(os.listdir(images_camera_1_path))

    def get_len_vicon(self, sub, act):
        # vicon_path = os.path.join(self.data_path, sub, 'vicon', act)
        return self.get_len_imu(sub, act)

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
        imu_quaternions = imu_sensors.iloc[1:, 1:].values.astype(
            np.float32).reshape(-1, 14, 7)[:, 1:14, :4]     # wxyz

        # length of imu data
        imu_sensors_length = imu_quaternions.shape[0]
        imu_sensors_num = imu_quaternions.shape[1]      # the number of sensors

        # sensors data to quaternion
        imu_quaternions = imu_quaternions.reshape(-1, 4)
        imu_quaternions = quaternion.from_float_array(imu_quaternions)

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
        imu_quaternions = imu_ref*imu_quaternions*imu_bone
        imu_quaternions = quaternion.as_float_array(
            imu_quaternions).astype(np.float32)

        imu_quaternions = imu_quaternions.reshape(
            imu_sensors_length, imu_sensors_num, 4)

        imu_acceleration = imu_sensors.iloc[1:, 1:].values.astype(
            np.float32).reshape(-1, 14, 7)[:, 1:14, 4:]

        imu_sensors_data = np.concatenate(
            (imu_quaternions, imu_acceleration), axis=2)

        return imu_sensors_data

    def init_vicon(self, sub, act):
        vicon_path = os.path.join(self.data_path, sub, 'vicon', act)
        pos_path = os.path.join(vicon_path, 'gt_skel_gbl_pos.txt')
        ori_path = os.path.join(vicon_path, 'gt_skel_gbl_ori.txt')
        pos = pd.read_csv(
            pos_path,
            sep=' |\t',
            names=list(range(63)),
            engine='python'
        )
        ori = pd.read_csv(
            ori_path,
            sep=' |\t',
            names=list(range(84)),
            engine='python'
        )

        pos = pos.iloc[1:, :].values.astype(np.float32).reshape(-1, 21, 3)
        ori = ori.iloc[1:, :].values.astype(np.float32).reshape(-1, 21, 4)

        vicon = np.concatenate((pos, ori), axis=2)
        return vicon

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

                    if self.cues == 'all':
                        _imu = self.data_dict[sub][act]['imu'][sub_index]
                        _imgs = self.get_imgs(sub, act, sub_index)
                        _vicon = self.data_dict[sub][act]['vicon'][sub_index]
                        return torch.tensor(_imu).float(), _imgs, \
                            torch.tensor(_vicon).float()
                    elif self.cues == 'imu':
                        _imu = self.data_dict[sub][act]['imu'][sub_index]
                        return torch.tensor(_imu).float()
                    elif self.cues == 'images':
                        _imgs = self.get_imgs(sub, act, sub_index)
                        return _imgs
                    elif self.cues == 'vicon':
                        _vicon = self.data_dict[sub][act]['vicon'][sub_index]
                        return torch.tensor(_vicon).float()
                    elif self.cues == 'vicon-imu':
                        _vicon = self.data_dict[sub][act]['vicon'][sub_index]
                        _imu = self.data_dict[sub][act]['imu'][sub_index]
                        return torch.tensor(_vicon).float(), \
                            torch.tensor(_imu).float()
                    elif self.cues == 'vicon-imu-ori':
                        _vicon = self.data_dict[
                            sub][act]['vicon'][sub_index][:, 3:]
                        _imu = self.data_dict[
                            sub][act]['imu'][sub_index][:, :4]
                        return torch.tensor(_vicon).float(), \
                            torch.tensor(_imu).float()


if __name__ == '__main__':
    data_path = '/media/ywj/Data/totalcapture/totalcapture'
    tp_data = TotalCapture(data_path, cues='all', mode='test')

    tp_data_loader = DataLoader(tp_data, batch_size=1, shuffle=False)
    for i in tp_data_loader:
        print(i)
