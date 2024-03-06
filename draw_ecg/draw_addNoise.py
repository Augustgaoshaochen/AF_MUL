# -*- coding: utf-8 -*-

# Author : chenpeng
# Time : 2023/5/21 15:21
from draw_ecg import plot_sample
from MlultModal.tool.CPSC_dataloader.load_data_cpsc2021 import load_icentia11k_data,load_data
import numpy as np

def get_data():
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_data(
        '/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL_100HZ/', 1000,1)
    for i in train_data:
        for j in i:
            plot_sample(j)



get_data()