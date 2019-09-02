# -*- coding:utf-8 -*-
"""
Author:
    Wenjie Yin, yinw@kth.se
"""

import os
import quaternion

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

actionList = {'acting1', 'acting2', 'acting3',
              'freestyle1', 'freestyle2', 'freestyle3',
              'rom1', 'rom2', 'rom3',
              'walking1', 'walking2', 'walking3'}

gt = {'Ori', 'Pos'}


class TotalCapture(Dataset):
    def __init__(self, data_path, data_transform):
        self.data_path = data_path
        self.data_transform = data_transform
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

        # for sub in self.training_subjects:
        #     # self.data_dict[sub] = {}
        #     for act in self.training_actions:
        #         pass
        self.init_imu(self.training_subjects[0], self.training_actions[0])

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
        imu_bone = quaternion.from_float_array(imu_bone)
        imu_bone = np.tile(imu_bone, imu_sensors_length)

        # reference
        imu_ref = quaternion.from_float_array(imu_ref)
        imu_ref = np.tile(imu_ref, imu_sensors_length)

        # R^g_bi = R_ig Â· R_i Â· (R_ib)^âˆ’1
        imu_sensors = imu_ref*imu_sensors/imu_bone
        imu_sensors = quaternion.as_float_array(imu_sensors).astype(np.float32)

        imu_sensors = imu_sensors.reshape(
            imu_sensors_length, imu_sensors_num, 4)

        return imu_sensors

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


if __name__ == '__main__':
    data_path = '/media/ywj/Data/totalcapture/totalcapture'
    data_transform = 'default'
    tp_data = TotalCapture(data_path, data_transform)

    pass
