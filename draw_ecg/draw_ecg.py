# -*- coding: utf-8 -*-

# Author : chenpeng
# Time : 2022/11/18 20:28
# 导入库
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_sample(signal):
    try:
        # Plot
        fig = plt.figure(figsize=(30, 6), dpi=100)
        x = np.arange(0, 1000, 100)
        x_labels = np.arange(0, 10)
        fsize = 20
        plt.plot(signal, color='green')
        plt.title("Non-Atrial Fibrillation: ", fontsize=fsize)
        plt.xticks(x, x_labels)
        plt.xlabel('time (s)', fontsize=16)
        plt.ylabel('value (mV)', fontsize=16)
        fig.tight_layout()
        plt.show()
        plt.close()
        return True

    except Exception as e:
        print(e)
        return False


def plot_sample_2(signal, signal_noise, noise_label=0, save_dir=None, datename=None):    # 多道联心电图可视化
    label = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    try:
        # Plot
        signal = signal[:, :1000]
        fig = plt.figure(figsize=(30, 36), dpi=100)
        x = np.arange(0, 1000, 100)
        x_labels = np.arange(0, 10)
        fsize = 20
        for i in range(len(signal)):
            plt.subplot(12, 1, idx[i])
            plt.plot(signal[i], color='green', label=str(i))
            plt.title("Atrial Fibrillation: " + label[i], fontsize=fsize)
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=fsize)
            plt.ylabel('value (mV)', fontsize=fsize)
        # for i in range(len(signal_noise)):
        #     plt.subplot(12, 1, i+7)
        #     plt.plot(signal_noise[i], color='green', label=str(i))
        #     plt.title("Atrial Fibrillation: " + label[i], fontsize=fsize)
        #     plt.xticks(x, x_labels)
        #     plt.xlabel('time (s)', fontsize=fsize)
        #     plt.ylabel('value (mV)', fontsize=fsize)
        fig.tight_layout()
        if save_dir is not None and datename is not None:
            filename = f"{datename}_{noise_label}.svg"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, format='svg')
        # if save_dir is not None and datename is not None:

        plt.show()
        plt.close()
        return True

    except Exception as e:
        print(e)
        return False


def plot_sample_3(signal, signal_noise, save_dir=None, datename=None):
    LEAD = ["I", "II"]
    Noise_Label = ['Gussian', 'Uniform', 'Exponential', 'Rayleign', 'Gamma']
    try:
        # Plot
        signal = signal[:, :1000]
        fig = plt.figure(figsize=(30, 12), dpi=100)
        x = np.arange(0, 1000, 100)
        x_labels = np.arange(0, 10)

        idx = [1, 2, 3, 4, 5, 6]
        for i in range(len(signal)):
            plt.subplot(6, 1, idx[i])
            plt.plot(signal[i], color='green', label=str(i))
            plt.title(LEAD[i] + ": ", fontsize=16)
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=16)
            plt.ylabel('value (mV)', fontsize=16)
        for i in range(len(signal_noise)):
            plt.subplot(6, 1, idx[i + 2])
            plt.plot(signal_noise[i], color='green', label=str(i))
            plt.title(LEAD[i] + "_" + Noise_Label[0] + ": ", fontsize=16)
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=16)
            plt.ylabel('value (mV)', fontsize=16)

        fig.tight_layout()

        plt.savefig("Plot_Hist/ptbxl_{}".format(datename) + '.svg', bbox_inches='tight')
        plt.show()
        plt.close()
        return True

    except Exception as e:
        print(e)
        return False


if __name__ == '__main__':
    # from MlultModal.tool.CPSC_dataloader.load_data_cpsc2021 import load_icentia11k_data,load_data
    # train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_icentia11k_data(
    #         '/data/icentia11k/', 1)
    # train_data = np.expand_dims(train_data, axis=1)
    # val_normal_data = np.expand_dims(val_normal_data, axis=1)
    # val_abnormal_data = np.expand_dims(val_abnormal_data, axis=1)
    # test_normal_data = np.expand_dims(test_normal_data, axis=1)
    # test_abnormal_data = np.expand_dims(test_abnormal_data, axis=1)
    # # train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_data(
    # #     '/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL_100HZ/', 1000, 1)
    # for i in range(10):
    #     plot_sample_2(train_data[i+100],test_abnormal_data[i+100],datename=i)
    # import torch
    import torch.nn.functional as F
    # import torch.nn as nn
    #
    # input_data = torch.randn(1, 3, 64, 64)  # 输入数据的形状为 [1, 3, 256, 256]
    #
    # conv = nn.Conv2d(32, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2), bias=False)
    #
    #
    # # 调整输入数据的形状为 [1, 3, 128, 128]
    # resized_input = F.interpolate(input_data, size=(128, 128), mode='bilinear', align_corners=False)
    #
    # print(conv(input_data))
    # print(resized_input.shape)  # 输出调整后的输入形状
    import pickle as pkl

    data = pkl.load(open('/home/hanhan/workSpace/ECG/My/utils_2023/data_processing/ptbxl_data/AF_data.pickle', "rb"))
    data1 = pkl.load(open('/home/hanhan/workSpace/ECG/My/utils_2023/data_processing/ptbxl_data/NORM_data.pickle', "rb"))
    data = np.transpose(data, (0, 2, 1))
    data1 = np.transpose(data1, (0, 2, 1))
    save_dir = '/home/gaoshaochen/Python/AF_Mul/experiments/MlultModal/draw_ECG/Plot_Hist'
    datename = 'AF'

    for i in range(data1.shape[0]):
        plot_sample_2(data[i], data[i], save_dir=save_dir, datename=f"{datename}_{i}")


