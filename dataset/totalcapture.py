# -*- coding:utf-8 -*-
"""
Author:
    Wenjie Yin, yinw@kth.se
"""

import os
import torch

import numpy as np

from torch.utils.data import Dataset, DataLoader

actionList = {'acting1', 'acting2', 'acting3',
              'freestyle1', 'freestyle2', 'freestyle3',
              'rom1', 'rom2', 'rom3',
              'walking1', 'walking2', 'walking3'}

gt = {'Ori', 'Pos'}

class TotalCapture(Dataset):
    def __init__(self, root_path, data_transform):
        self.root_path = root_path
        self.data_transform = data_transform

        '''
            The train and test partition are performed wrt to the subjects and sequences, 
        the training is perfomed on subjects 1,2 and 3 of the following seqeunces:
            ROM 1,2,3
            Walking 1,3
            Freestyle 1,2
            Acting 1,2
            
        test set is perfomed on sbjects 1,2,3,4 and 5, on the following seqeunces
            Walking 2
            Freestyle 3
            Acting 3

        '''
        self.training_subjects = ['S1', 'S2', 'S3']
        self.testing_subjects = ['S1', 'S2', 'S3', 'S4', 'S5']

        


        
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
    




if __name__ == '__main__':
