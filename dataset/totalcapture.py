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
