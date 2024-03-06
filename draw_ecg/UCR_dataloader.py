
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as data
#from dataloader.transformer import r_plot, paa, rescale
#import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
#import pywt
# import librosa
import scipy
#import yaml
import matplotlib.pyplot as plt
import pickle
from scipy.signal import savgol_filter, wiener

#from pykalman import KalmanFilter
import math
from lib_for_SLMR.utils import noise_mask

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer


#*****************ECG Func   start*****************
def getFloderK(data,folder,label):
    normal_cnt = data.shape[0]
    folder_num = int(normal_cnt / 5)
    folder_idx = folder * folder_num

    folder_data = data[folder_idx:folder_idx + folder_num]

    remain_data = np.concatenate([data[:folder_idx], data[folder_idx + folder_num:]])
    if label==0:
        folder_data_y = np.zeros((folder_data.shape[0], 1))
        remain_data_y=np.zeros((remain_data.shape[0], 1))
    elif label==1:
        folder_data_y = np.ones((folder_data.shape[0], 1))
        remain_data_y = np.ones((remain_data.shape[0], 1))
    else:
        raise Exception("label should be 0 or 1, get:{}".format(label))
    return folder_data,folder_data_y,remain_data,remain_data_y

def getPercent(data_x,data_y,percent,seed):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,test_size=percent,random_state=seed)
    return train_x, test_x, train_y, test_y

def get_full_data(dataloader):

    full_data_x=[]
    full_data_y=[]
    for batch_data in dataloader:
        batch_x,batch_y=batch_data[0],batch_data[1]
        batch_x=batch_x.numpy()
        batch_y=batch_y.numpy()

        # print(batch_x.shape)
        # assert False
        for i in range(batch_x.shape[0]):
            full_data_x.append(batch_x[i,0,:])
            full_data_y.append(batch_y[i])

    full_data_x=np.array(full_data_x)
    full_data_y=np.array(full_data_y)
    assert full_data_x.shape[0]==full_data_y.shape[0]
    print("full data size:{}".format(full_data_x.shape))
    return full_data_x,full_data_y


def data_aug(train_x,train_y,times=2):
    res_train_x=[]
    res_train_y=[]
    for idx in range(train_x.shape[0]):
        x=train_x[idx]
        y=train_y[idx]
        res_train_x.append(x)
        res_train_y.append(y)

        for i in range(times):
            x_aug=aug_ts(x)
            res_train_x.append(x_aug)
            res_train_y.append(y)

    res_train_x=np.array(res_train_x)
    res_train_y=np.array(res_train_y)

    return res_train_x,res_train_y

def aug_ts(x):
    left_ticks_index = np.arange(0, 140)
    right_ticks_index = np.arange(140, 319)
    np.random.shuffle(left_ticks_index)
    np.random.shuffle(right_ticks_index)
    left_up_ticks = left_ticks_index[:7]
    right_up_ticks = right_ticks_index[:7]
    left_down_ticks = left_ticks_index[7:14]
    right_down_ticks = right_ticks_index[7:14]

    x_1 = np.zeros_like(x)
    j = 0
    for i in range(x.shape[1]):
        if i in left_down_ticks or i in right_down_ticks:
            continue
        elif i in left_up_ticks or i in right_up_ticks:
            x_1[:, j] =x[:,i]
            j += 1
            x_1[:, j] = (x[:, i] + x[:, i + 1]) / 2
            j += 1
        else:
            x_1[:, j] = x[:, i]
            j += 1
    return x_1
#*****************ECG Func  end*****************

#**********************MEL strat ******************************
#params = yaml.load(open('./Mel_Para.yaml'))
extract_params={
  'audio_len_s': 2,
  'eps': 2.220446049250313e-16,
  'fmax': 22050,
  'fmin': 0,
  'fs': 44100,
  'hop_length_samples': 82,    # 影响第三维
  'load_mode': 'varup',
  'log': True,
  'mono': True,
  'n_fft': 1024,
  'n_mels': 96,
  'normalize_audio': True,
  'patch_hop': 50,
  'patch_len': 100,
  'spectrogram_type': True,
  'win_length_samples': 564,
  'spectrogram_type': 'power',
    'audio_len_samples':88200
}




def wavelet_preprocessing_set(X, waveletLevel=1, waveletFilter='haar'):  ##(1221,26,512)
    '''
    :param X: (sample_num, feature_num, sequence_length)
    :param waveletLevel:
    :param waveletFilter:
    :return: result (sample_num, extended_sequence_length, feature_num)
    '''

    if len(X.shape)==2:
        X = np.expand_dims(X, 1)  # (292,1,140)

    N = X.shape[0]  #1121
    feature_dim = X.shape[1]   #26
    length = X.shape[2]   #512

    signal_length = []
    stats = []
    extened_X = []
    # extened_X.append(X)  #（1221，512，26）

    for i in range(N):# for each sample
        for j in range(feature_dim): # for each dim
            wavelet_list = pywt.wavedec(X[i][j], waveletFilter, level=waveletLevel)  # X(1221,26,512)    512-->(32,32,64,128,256)
            # 多尺度一维小波分解。  返回信号X在N层的小波分解
            if i == 0 and j == 0:
                for l in range(waveletLevel):
                    current_length = len(wavelet_list[waveletLevel - l])   # 256 128 64 32 32
                    signal_length.append(current_length)
                    extened_X.append(np.zeros((N, feature_dim, current_length)))
            for l in range(waveletLevel):
                extened_X[l][i, j, :] = wavelet_list[waveletLevel-l]    # (1221,512,26)-->(1221,256,26) .....(,32,)

    result = None
    first = True
    for mat in extened_X:
        mat_mean = mat.mean()
        mat_std = mat.std()
        mat = (mat-mat_mean) / mat_std
        stats.append((mat_mean, mat_std))
        if first:
            result = mat
            first = False
        else:
            result = np.concatenate((result, mat), -1)  #512+256+128+64+32=992

    return result, signal_length

#
# def mel_spectrogram_precessing_set(audio, params_extract=None):
#     """
#
#     :param audio:
#     :param params_extract:
#     :return:
#     """
#     audio=np.array(audio)
#     # make sure rows are channels and columns the samples
#     Mel_Matrix=[]
#     for idx in range(audio.shape[0]):
#
#         audio_idx = audio[idx].reshape([1, -1])
#         window = scipy.signal.hamming(params_extract.get('win_length_samples'), sym=False)  #window len 1764
#
#         mel_basis = librosa.filters.mel(sr=params_extract.get('fs'),     #梅尔滤波器 将能量谱转换为梅尔频率
#                                         n_fft=params_extract.get('n_fft'),
#                                         n_mels=params_extract.get('n_mels'),
#                                         fmin=params_extract.get('fmin'),
#                                         fmax=params_extract.get('fmax'),
#                                         htk=False,
#                                         norm=None)
#
#         # init mel_spectrogram expressed as features: row x col = frames x mel_bands = 0 x mel_bands (to concatenate axis=0)
#         feature_matrix = np.empty((0, params_extract.get('n_mels')))
#         for channel in range(0, audio_idx.shape[0]):
#             spectrogram = get_spectrogram(       #梅尔谱图
#                 y=audio_idx[channel, :],
#                 n_fft=params_extract.get('n_fft'),
#                 win_length_samples=params_extract.get('win_length_samples'),
#                 hop_length_samples=params_extract.get('hop_length_samples'),
#                 spectrogram_type=params_extract.get('spectrogram_type') if 'spectrogram_type' in params_extract else 'magnitude',
#                 center=True,
#                 window=window,
#                 params_extract=params_extract
#             )
#
#             mel_spectrogram = np.dot(mel_basis, spectrogram)   #梅尔频率点乘梅尔谱图=梅尔频谱图
#             mel_spectrogram = mel_spectrogram.T
#
#             if params_extract.get('log'):
#                 mel_spectrogram = np.log10(mel_spectrogram + params_extract.get('eps'))
#
#             feature_matrix = np.append(feature_matrix, mel_spectrogram, axis=0)
#
#             Mel_Matrix.append(feature_matrix)
#     Mel_Matrix=np.expand_dims(Mel_Matrix,1)
#
#     #print()
#
#     return Mel_Matrix,Mel_Matrix.shape[1:]
#
#
# def get_spectrogram(y,
#                     n_fft=1024,
#                     win_length_samples=0.04,
#                     hop_length_samples=0.02,
#                     window=scipy.signal.hamming(512, sym=False),
#                     center=True,
#                     spectrogram_type='magnitude',
#                     params_extract=None):
#
#     if spectrogram_type == 'power':
#         return np.abs(librosa.stft(y + params_extract.get('eps'),        #STFT  短时傅里叶变换  时频信号转换
#                                    n_fft=n_fft,
#                                    win_length=win_length_samples,
#                                    hop_length=hop_length_samples,
#                                    center=center,
#                                    window=window)) ** 2

#**********************MEL end******************************

def RP_preprocessing_set(X_train):
    ##################
    # Down-sample
    ##################
    signal_dim = X_train.shape[-1]
    if signal_dim > 500:
        down_scale = X_train.shape[-1] // 128
    else:
        down_scale = X_train.shape[-1] // 32

    (size_H, size_W) = (X_train.shape[-1] // down_scale, X_train.shape[-1] // down_scale)
    print('[INFO] Raw Size: {}'.format((X_train.shape[-1], X_train.shape[-1])))
    print('[INFO] Downsample Size: {}'.format((size_H, size_W)))

    X_train_ds = paa(X_train, down_scale)

    # 2.1.RP image
    X_train_rp = np.empty(shape=(len(X_train_ds), size_H, size_W), dtype=np.float32)

    for i in range(len(X_train_ds)):
        X_train_rp[i, :, :] = r_plot(X_train_ds[i, :])

    X_train_rp = np.expand_dims(X_train_rp, 1)
    return X_train_rp, X_train_rp.shape[1:]
import  time


#
# class EM_FK():
#
#     def __init__(self,A,C,Q,R,B,D,m0,P0,random_state):
#
#         self.A = A  #transition_matrix
#         self.C = C  #observation_matrix
#         self.Q = Q  #transition_covariance
#         self.R = R  #observation_covariance
#         self.B = B  #transition_offset
#         self.D = D  #observation_offset
#         self.m = m0 #initial_state_mean
#         self.p = P0  #initial_state_covariance
#         self.random_state = random_state
#         self.ft = KalmanFilter(self.A, self.C, self.Q, self.R, self.B, self.D, self.m, self.p, self.random_state)
#
#         T = 0.01
#         A_init = np.array([[1, T, 0.5 * T * T], [0, 1, T], [0, 0, 1]])
#         B_init = [0, 0, 0]
#         C_init = [1, 0, 0]
#         D_init = [0]
#         Q_init = 0.02 * np.eye(3)
#         R_init = np.eye(1)
#         m0_init = [0, 0, 1]
#         P0_init = 5 * np.eye(3)
#         random_state_init = np.random.RandomState(0)
#
#         self.ft_init = KalmanFilter(A_init, C_init, Q_init, R_init, B_init, D_init, m0_init, P0_init, random_state_init)
#
#     def filter(self, x):
#
#         filtered_state_estimater, nf_cov = self.ft.filter(x)
#         smoothed_state_estimater, ns_cov = self.ft.smooth(x)
#
#         pred_state = np.squeeze(smoothed_state_estimater)
#
#         return pred_state[:,0]
#
#     def filter_init(self, x):
#
#         filtered_state_estimater, nf_cov = self.ft_init.filter(x)
#         smoothed_state_estimater, ns_cov = self.ft_init.smooth(x)
#
#         pred_state = np.squeeze(smoothed_state_estimater)
#
#         return pred_state[:,0]
#
#     def kalman_1D(self,x):
#
#         x_Kal = []
#         print("Kalman1D  Filtering.....")
#
#         w_size = x.shape[1]
#         if w_size % 2 == 0:
#             w_size = w_size + 1
#
#         for i in range(x.shape[0]):
#             signal = np.array(x[i])
#             signal = np.squeeze(signal)
#
#             # WavePlot_Single(x[i],'signal')
#
#             # signal_sav = KalmanFilter(signal,len(signal))
#             signal_Kalman = self.filter(signal)
#             #signal_Kalman_init = self.filter_init(signal)
#
#             #WavePlot_Scatter(signal, signal_Kalman_init, signal_Kalman,'kalman')
#
#             x_Kal.append(signal_Kalman)
#
#         x_Kal = np.array(x_Kal)
#         x_Kal = np.expand_dims(x_Kal, 1)
#
#         return x_Kal, x_Kal.shape[-1]





# class EM_FK():
#
#     def __init__(self,initial_value_guess,observation_covariance, transition_covariance, transition_matrix):
#
#         self.initial_value_guess = initial_value_guess  #transition_matrix
#         self.observation_covariance = observation_covariance  #observation_matrix
#         self.transition_covariance = transition_covariance  #transition_covariance
#         self.transition_matrix = transition_matrix  #observation_covariance
#
#         self.ft = KalmanFilter(
#             initial_state_mean=self.initial_value_guess,
#             initial_state_covariance=self.observation_covariance,
#             observation_covariance=self.observation_covariance,
#             transition_covariance=self.transition_covariance,
#             transition_matrices=self.transition_matrix)
#
#
#         observation_covariance_init = 1
#         initial_value_guess_init = 0
#         transition_matrix_init = 1
#         transition_covariance_init = 0.01
#
#         self.ft_init = KalmanFilter(
#             initial_state_mean= initial_value_guess_init,
#             initial_state_covariance= observation_covariance_init,
#             observation_covariance= observation_covariance_init,
#             transition_covariance= transition_covariance_init,
#             transition_matrices=transition_matrix_init)
#
#
#     def filter(self, x):
#
#         filtered_state_estimater, nf_cov = self.ft.filter(x)
#         smoothed_state_estimater, ns_cov = self.ft.smooth(x)
#
#         pred_state = np.squeeze(smoothed_state_estimater)
#
#         return pred_state
#
#     def filter_init(self, x):
#
#
#         filtered_state_estimater, nf_cov = self.ft_init.filter(x)
#         smoothed_state_estimater, ns_cov = self.ft_init.smooth(x)
#
#         pred_state = np.squeeze(smoothed_state_estimater)
#
#         return pred_state
#
#
#     def kalman_1D(self,x):
#
#         x_Kal = []
#         print("Kalman1D  Filtering.....")
#
#         w_size = x.shape[1]
#         if w_size % 2 == 0:
#             w_size = w_size + 1
#
#         for i in range(x.shape[0]):
#             signal = np.array(x[i])
#             signal = np.squeeze(signal)
#
#             # WavePlot_Single(x[i],'signal')
#
#             # signal_sav = KalmanFilter(signal,len(signal))
#             signal_Kalman = self.filter(signal)
#             #signal_Kalman_init = self.filter_init(signal)
#
#             #WavePlot_Scatter(signal, signal_Kalman_init, signal_Kalman,'kalman')
#
#             x_Kal.append(signal_Kalman)
#
#         x_Kal = np.array(x_Kal)
#         x_Kal = np.expand_dims(x_Kal, 1)
#
#         return x_Kal, x_Kal.shape[-1]


class EM_FK():

    def __init__(self,initial_value_guess,initial_state_covariance,observation_covariance, transition_covariance, transition_matrix):

        self.initial_value_guess = initial_value_guess  #transition_matrix
        self.initial_state_covariance = initial_state_covariance  #initial_state_covariance
        self.observation_covariance = observation_covariance  #observation_matrix
        self.transition_covariance = transition_covariance  #transition_covariance
        self.transition_matrix = transition_matrix  #observation_covariance

        self.ft = KalmanFilter(
            initial_state_mean=self.initial_value_guess,
            initial_state_covariance=self.initial_state_covariance,
            observation_covariance=self.observation_covariance,
            transition_covariance=self.transition_covariance,
            transition_matrices=self.transition_matrix)

        initial_state_covariance_init = 1
        observation_covariance_init = 1
        initial_value_guess_init = 0
        transition_matrix_init = 1
        transition_covariance_init = 0.01

        self.ft_init = KalmanFilter(
            initial_state_mean= initial_value_guess_init,
            initial_state_covariance= initial_state_covariance_init,
            observation_covariance= observation_covariance_init,
            transition_covariance= transition_covariance_init,
            transition_matrices=transition_matrix_init)


    def filter(self, x):

        filtered_state_estimater, nf_cov = self.ft.filter(x)
        smoothed_state_estimater, ns_cov = self.ft.smooth(x)

        pred_state = np.squeeze(smoothed_state_estimater)

        return pred_state

    def filter_init(self, x):


        filtered_state_estimater, nf_cov = self.ft_init.filter(x)
        smoothed_state_estimater, ns_cov = self.ft_init.smooth(x)

        pred_state = np.squeeze(smoothed_state_estimater)

        return pred_state


    def kalman_1D_init(self,x):

        x_Kal = []
        print("Kalman1D  Filtering.....")

        w_size = x.shape[1]
        if w_size % 2 == 0:
            w_size = w_size + 1

        for i in range(x.shape[0]):
            signal = np.array(x[i])
            signal = np.squeeze(signal)

            #WavePlot_Single(x[i],'signal')

            # signal_sav = KalmanFilter(signal,len(signal))
            signal_Kalman = self.filter_init(signal)
            #signal_Kalman_init = self.filter_init(signal)
            #
            #WavePlot_Scatter(signal, signal_Kalman,'TwoLeadECG_label0_EM_5epoch{}'.format(i))
            #
            x_Kal.append(signal_Kalman)

        x_Kal = np.array(x_Kal)
        x_Kal = np.expand_dims(x_Kal, 1)

        return x_Kal, x_Kal.shape[-1]


    def kalman_1D(self,x):

        x_Kal = []
        print("Kalman1D  Filtering.....")

        w_size = x.shape[1]
        if w_size % 2 == 0:
            w_size = w_size + 1

        for i in range(x.shape[0]):
            signal = np.array(x[i])
            signal = np.squeeze(signal)

            #WavePlot_Single(x[i],'signal')

            # signal_sav = KalmanFilter(signal,len(signal))
            signal_Kalman = self.filter(signal)
            #signal_Kalman_init = self.filter_init(signal)
            #
            #WavePlot_Scatter(signal, signal_Kalman,'TwoLeadECG_label0_EM_5epoch{}'.format(i))
            #
            x_Kal.append(signal_Kalman)

        x_Kal = np.array(x_Kal)
        x_Kal = np.expand_dims(x_Kal, 1)

        return x_Kal, x_Kal.shape[-1]



def Kalman1D(observations,damping=1):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.01

    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state, state_cov = kf.smooth(observations)
    pred_state = np.squeeze(pred_state)
    return pred_state



def Kalman2D_Pykalman(measurements):
   # code by Huangxunhua
   # input shape:(1600,2)

   filter = []

   initial_state_mean = [measurements[0, 0],
                         0,
                         measurements[0, 1],
                         0]

   transition_matrix = [[1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 0, 1]]

   observation_matrix = [[1, 0, 0, 0],
                         [0, 0, 1, 0]]

   kf1 = KalmanFilter(transition_matrices=transition_matrix,
                      observation_matrices=observation_matrix,
                      initial_state_mean=initial_state_mean)

   kf1 = kf1.em(measurements, n_iter=5)
   (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)

   dim1 = smoothed_state_means[:, 0]
   dim2 = smoothed_state_means[:, 2]

   filter.append(dim1)
   filter.append(dim2)
   filter = np.array(filter)

   return filter




def kalman2D_Git(data):

    P = np.identity(2)
    X = np.array([[0], [0]])
    dt = 5
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[dt*dt/2], [dt]])
    Q = np.array([[.0001, .00002], [.00002, .0001]])
    R = np.array([[.01, .005], [.005, .02]])
    estimated = []
    H = np.identity(2)
    I = np.identity(2)
    # print("X")
    # print(X)
    for i in data:
        u1 = i[0]
        u2 = i[1]
        u_k = np.array([[u1], [u2]])
        u_k = np.squeeze(u_k)


        # z_k = np.squeeze(z_k)
        # prediction
        X = X + u_k
        P = A*P*A.T + Q
        # kalman gain/measurement
        K = P/(P + R)
        Y = np.dot(H, u_k).reshape(2, -1)

        # new X and P
        X = X + np.dot(K, Y - np.dot(H, X))
        P = (I - K*H)*P
        estimated.append(X)

    estimated = np.squeeze(np.array(estimated)[:,:,:1])

    return estimated



def Savitzky(x):

    x_sav = []
    print("SavitzkyFiltering.....")

    w_size = x.shape[1]
    if w_size % 2 == 0:
        w_size = w_size+1

    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)

        signal_sav = savgol_filter(signal,51,5)

        #WavePlot_Single(signal_sav,'kalman')

        
        x_sav.append(signal_sav)
        
    x_sav = np.array(x_sav)
    x_sav = np.expand_dims(x_sav, 1)
    
    return x_sav, x_sav.shape[-1]


def Wiener(x):
    x_Wie = []
    print("WienerFiltering.....")

    w_size = x.shape[1]
    if w_size % 2 == 0:
        w_size = w_size + 1

    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)

        # WavePlot_Single(x[i],'signal')

        signal_Wie = wiener(signal,81)

        # WavePlot_Single(signal_sav,'kalman')

        x_Wie.append(signal_Wie)

    x_Wie = np.array(x_Wie)
    x_Wie = np.expand_dims(x_Wie, 1)

    return x_Wie, x_Wie.shape[-1]


def Kalman_1D(x):
    x_Kal = []
    print("Kalman1D  Filtering.....")

    w_size = x.shape[1]
    if w_size % 2 == 0:
        w_size = w_size + 1

    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)

        # WavePlot_Single(x[i],'signal')

        # signal_sav = KalmanFilter(signal,len(signal))
        signal_Kalman = Kalman1D(signal)

        # WavePlot_Single(signal_sav,'kalman')

        x_Kal.append(signal_Kalman)

    x_Kal = np.array(x_Kal)
    x_Kal = np.expand_dims(x_Kal, 1)

    return x_Kal, x_Kal.shape[-1]

def Kalman_2D(x):
    x_Kal = []
    print("Kalman2D  Filtering.....")
    x = x.transpose(0,2,1)

    # w_size = x.shape[1]
    # if w_size % 2 == 0:
    #     w_size = w_size + 1

    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)
        print(i)

        # WavePlot_Single(x[i],'signal')

        # signal_sav = KalmanFilter(signal,len(signal))
        signal_Kalman = Kalman2D_Pykalman(signal)

        # WavePlot_Single(signal_sav,'kalman')

        x_Kal.append(signal_Kalman)

    x_Kal = np.array(x_Kal)
    #x_Kal = np.expand_dims(x_Kal, 1)

    return x_Kal, x_Kal.shape[-1]



def Kalman_2D_Git(x):
    x_Kal = []
    print("Kalman2D_Git  Filtering.....")
    x = x.transpose(0, 2, 1)

    # w_size = x.shape[1]
    # if w_size % 2 == 0:
    #     w_size = w_size + 1

    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)
        #print(i)

        # WavePlot_Single(x[i],'signal')

        # signal_sav = KalmanFilter(signal,len(signal))
        signal_Kalman = kalman2D_Git(signal)

        # WavePlot_Single(signal_sav,'kalman')

        x_Kal.append(signal_Kalman)

    x_Kal = np.array(x_Kal)
    x_Kal = x_Kal.transpose(0,2,1)
    # x_Kal = np.expand_dims(x_Kal, 1)

    return x_Kal, x_Kal.shape[-1]

# def KalmanFilter(z, n_iter=20):
#     # 这里是假设A=1，H=1的情况
#
#     # intial parameters
#
#     sz = (n_iter,)  # size of array
#
#     # Q = 1e-5 # process variance
#     Q = 1e-6  # process variance  变化量
#     # allocate space for arrays
#     xhat = np.zeros(sz)  # a posteri estimate of x    预测
#     P = np.zeros(sz)  # a posteri error estimate
#     xhatminus = np.zeros(sz)  # a priori estimate of x   观测
#     Pminus = np.zeros(sz)  # a priori error estimate
#     K = np.zeros(sz)  # gain or blending factor
#
#     R = 0.1 ** 2  # estimate of measurement variance, change to see effect
#
#     # intial guesses
#     xhat[0] = 0.0
#     P[0] = 1.0
#     A = 1
#     H = 1
#
#     for k in range(1, n_iter):
#         # time update
#         xhatminus[k] = A * xhat[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0   观测 = 上一个预测
#         Pminus[k] = A * P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1    观测误差 = 预测误差+变化量   观测准确度
#
#         # measurement update
#         K[k] = Pminus[k] / (Pminus[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1   均化系数 = 观测误差    卡尔曼增益
#         xhat[k] = xhatminus[k] + K[k] * (z[k] - H * xhatminus[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1  预测 = 观测 + 卡尔曼增益*（真实-观测(上一个预测)）
#         P[k] = (1 - K[k] * H) * Pminus[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1   预测误差=（1-均化系数）*观测误差   省去了预测？
#
#     return xhat



def Gussian_Noisy(x, snr):   # snr:信噪比


    x_gussian = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):

        signal = np.array(x[i])
        WavePlot_Single(signal,'signal')
        signal = np.squeeze(signal)
        #sum = np.sum(signal ** 2)

        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        #test = np.random.randn(len(signal))
        #WavePlot_Single(test,'random')

        gussian = np.random.randn(len(signal)) * np.sqrt(npower)
        WavePlot_Single(gussian,'gussian')

        x_gussian.append(x[i]+gussian)


    x_gussian = np.array(x_gussian)
    x_gussian = np.expand_dims(x_gussian,1)


    return x_gussian, x_gussian.shape[-1]





def Gamma_Noisy_return_N(x, snr):   # x:信号 snr:信噪比

    print('Gamma')
    x_gamma = []
    x_gamma_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):

        i = 3

        signal = np.array(x[i])   # signal
        #WavePlot_Single(signal, 'signal')
        signal = np.squeeze(signal)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        gamma = np.random.gamma(shape= 2, size = len(signal)) * np.sqrt(npower)  # noisy
        WavePlot_Single_2(gamma, 'gamma_5')

        x_gamma.append(x[i] + gamma)   # add
        WavePlot_Single(x[i] + gamma, 'gamma_5_add')
        x_gamma_only.append(gamma)

    x_gamma = np.array(x_gamma)
    x_gamma = np.expand_dims(x_gamma, 1)
    x_gamma_only = np.array(x_gamma_only)
    x_gamma_only = np.expand_dims(x_gamma_only ,1)

    return x_gamma_only, x_gamma, x_gamma.shape[-1]



def Rayleign_Noisy_return_N(x, snr):   # snr:信噪比

    print('Ralyeign')
    x_rayleign = []
    x_rayleign_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        i = 3

        signal = np.array(x[i])
        signal = np.squeeze(signal)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        rayleign = np.random.rayleigh(size = len(signal)) * np.sqrt(npower)
        WavePlot_Single_2(rayleign, 'rayleigh_5')


        x_rayleign.append(x[i] + rayleign)
        WavePlot_Single(x[i] + rayleign, 'rayleigh_5_add')
        x_rayleign_only.append(rayleign)

    x_rayleign = np.array(x_rayleign)
    x_rayleign = np.expand_dims(x_rayleign, 1)
    x_rayleign_only = np.array(x_rayleign_only)
    x_rayleign_only = np.expand_dims(x_rayleign_only, 1)

    return x_rayleign_only, x_rayleign, x_rayleign.shape[-1]

def Exponential_Noisy_return_N(x, snr):   # snr:信噪比

    print("Exponential")
    x_exponential = []
    x_exponential_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        i = 3

        signal = np.array(x[i])
        signal = np.squeeze(signal)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        exponential = np.random.exponential(size = len(signal)) * np.sqrt(npower)
        WavePlot_Single_2(exponential, 'exponential_5')


        x_exponential.append(x[i] + exponential)
        WavePlot_Single(x[i] + exponential, 'exponential_5_add')
        x_exponential_only.append(exponential)

    x_exponential = np.array(x_exponential)
    x_exponential = np.expand_dims(x_exponential, 1)
    x_exponential_only = np.array(x_exponential_only)
    x_exponential_only = np.expand_dims(x_exponential_only, 1)

    return x_exponential_only, x_exponential, x_exponential.shape[-1]

def Uniform_Noisy_return_N(x, snr):   # snr:信噪比

    print("Uniform")
    x_uniform = []
    x_uniform_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        i = 3

        signal = np.array(x[i])
        signal = np.squeeze(signal)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        uniform = np.random.uniform(size = len(signal)) * np.sqrt(npower)
        WavePlot_Single_2(uniform, 'uniform_5')


        x_uniform.append(x[i] + uniform)
        WavePlot_Single(x[i] + uniform, 'uniform_5_add')
        x_uniform_only.append(uniform)

    x_uniform = np.array(x_uniform)
    x_uniform = np.expand_dims(x_uniform, 1)
    x_uniform_only = np.array(x_uniform_only)
    x_uniform_only = np.expand_dims(x_uniform_only ,1)

    return x_uniform_only, x_uniform, x_uniform.shape[-1]

def Poisson_Noisy_return_N(x, snr):   # snr:信噪比

    print("possion")
    x_poisson = []
    x_poisson_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):

        i = 3

        signal = np.array(x[i])
        signal = np.squeeze(signal)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        poisson = np.random.poisson(1, len(signal)) * np.sqrt(npower)
        WavePlot_Single(poisson, 'poisson_5')


        x_poisson.append(x[i] + poisson)
        WavePlot_Single(x[i] + poisson, 'poisson_5_add')
        x_poisson_only.append(poisson)

    x_poisson = np.array(x_poisson)
    x_poisson = np.expand_dims(x_poisson, 1)
    x_poisson_only = np.array(x_poisson_only)
    x_poisson_only = np.expand_dims(x_poisson_only, 1)

    return x_poisson_only, x_poisson, x_poisson.shape[-1]

import time


def Gussian_Noisy_return_N(x, snr):   # snr:信噪比


    x_gussian = []
    x_gussian_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):

        #i = 3

        signal = np.array(x[i])
        #WavePlot_Single(signal, 'signal')
        signal = np.squeeze(signal)
        xpower = np.sum(signal ** 2) / len(signal)
        # print(np.sum(signal ** 2))
        # time.sleep(1)
        npower = xpower / snr
        gussian = np.random.randn(len(signal)) * np.sqrt(npower)
        # print(np.sum(signal ** 2))
        # time.sleep(1)
        WavePlot_Single_2(gussian, 'gussian_5')

        x_gussian.append(x[i] + gussian)
        #x_gussian_wie = wiener(x_gussian,5)
        #WavePlot_Single(x[i] + gussian, 'gussian_5_add')
        x_gussian_only.append(gussian)

    x_gussian = np.array(x_gussian)
    x_gussian = np.expand_dims(x_gussian, 1)
    x_gussian_only = np.array(x_gussian_only)
    x_gussian_only = np.expand_dims(x_gussian_only,1)

    return x_gussian_only, x_gussian, x_gussian.shape[-1]



def normalize(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''
    return 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1



global count
def normalized(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''

    if np.max(seq)-np.min(seq) != 0 :
        return (seq-np.min(seq))/(np.max(seq)-np.min(seq))
    else :


        return (seq-np.min(seq))/(np.max(seq)+1)


def getPercent(data_x, data_y, percent, seed):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,test_size=percent,random_state=seed)
    return train_x, test_x, train_y, test_y


def one_class_labeling(labels, normal_class:int,seed):
    normal_idx = np.where(labels == normal_class)[0]
    abnormal_idx = np.where(labels != normal_class)[0]

    labels[normal_idx] = 0
    labels[abnormal_idx] = 1
    np.random.seed(seed)


    np.random.shuffle(normal_idx)
    np.random.shuffle(abnormal_idx)

    return labels.astype("bool"), normal_idx, abnormal_idx

def one_class_labeling_sz(labels, abnormal_class:int, seed):
    normal_idx = np.where(labels != abnormal_class)[0]
    abnormal_idx = np.where(labels == abnormal_class)[0]

    labels[normal_idx] = 0
    labels[abnormal_idx] = 1
    np.random.seed(seed)
    np.random.shuffle(normal_idx)
    np.random.shuffle(abnormal_idx)

    return labels.astype("bool"), normal_idx, abnormal_idx


def one_class_labeling_multi(labels, normal_classes):

    all_idx = np.asarray(list(range(len(labels))))
    for normal_class in normal_classes:
        normal_idx = np.where(labels == normal_class)[0]

    abnormal_idx = np.delete(all_idx, normal_idx, axis=0)


    labels[normal_idx] = 0
    labels[abnormal_idx] = 1
    np.random.shuffle(normal_idx)
    np.random.shuffle(abnormal_idx)

    return labels.astype("bool"), normal_idx, abnormal_idx





class RawDataset(data.Dataset):
    def __init__(self, X, Y):
        """
        """

        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)



class SlidingWindowDataset(data.Dataset):
    def __init__(self, opt, data, label, window, target_dim=None, horizon=1):
        self.data = torch.Tensor(data)
        self.label = torch.Tensor(label)
        self.window = window # window = X_length
        self.target_dim = target_dim # target_dim = nc
        self.horizon = horizon
        self.opt = opt
        self.cut_len = int(self.data.shape[1]*0.9)
        if self.cut_len % 2 != 0: self.cut_len += 1

    def __getitem__(self, index):
        x_data = self.data[index]

        x = x_data[:self.cut_len,]
        # x_mask = zero_one(x)
        # x_mask = x * x_mask
        mask = noise_mask(x, masking_ratio = self.opt.mask_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None)
        mask = torch.from_numpy(mask) * x
        y = x_data[self.cut_len:,]
        return x, mask, y, self.label[index]

    def __len__(self):
        # return len(self.data) - self.window
        return len(self.data)



class RawDataset_EM_KF_init(data.Dataset):
    def __init__(self, X, Y):
        """
        """

        # specify parameters
        initial_state_covariance = 1
        observation_covariance = 1
        initial_value_guess = 0
        transition_matrix = 1
        transition_covariance = 0.001

        X_kal, _  = EM_FK(initial_value_guess, initial_state_covariance, observation_covariance, transition_covariance, transition_matrix).kalman_1D_init(X)
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.X_Kal = torch.Tensor(X_kal)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.X_Kal[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class WaveLetDataset(data.Dataset):
    def __init__(self, X, Y):
        """
        """

        X_wavelet, _ = wavelet_preprocessing_set(X)
        #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        self.X = torch.Tensor(X)
        self.X_wavelet = torch.Tensor(X_wavelet)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.X_wavelet[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class SavitzkyDataset(data.Dataset):
    def __init__(self, X, Y):
        """
        """

        X_Sav, _ = Savitzky(X)
        X = np.expand_dims(X, 1)
        #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        self.X = torch.Tensor(X)
        self.X_Sav = torch.Tensor(X_Sav)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index],self.X_Sav[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)



class WienerDataset(data.Dataset):
    def __init__(self, X, Y):
        """
        """

        X_Wie, _ = Wiener(X)
        X = np.expand_dims(X, 1)
        #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        self.X = torch.Tensor(X)
        self.X_Wie = torch.Tensor(X_Wie)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index],self.X_Wie[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)



# class KalmanDataset(data.Dataset):
#     def __init__(self, X, Y):
#         """
#         """
#         T = 0.1
#         A = np.array([[1, T, 0.5 * T * T], [0, 1, T], [0, 0, 1]])
#         B = [0, 0, 0]
#         C = [1, 0, 0]
#         D = [0]
#         Q = 0.02 * np.eye(3)
#         R = np.eye(1)
#         m0 = [0, 0, 1]
#         P0 = 5* np.eye(3)
#         random_state = np.random.RandomState(0)
#
#         #X_Kal, _ = EM_FK(A, C, Q, R, B, D, m0, P0, random_state=random_state).kalman_1D(X)
#         #X_Kal, _ = Kalman_1D(X)
#         X = np.expand_dims(X, 1)
#         #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
#         self.X = torch.Tensor(X)
#         self.X_Kal = torch.Tensor(X_Kal)
#         self.Y = torch.Tensor(Y)
#
#     def __getitem__(self, index):
#         # Get path of input image and ground truth
#
#         return self.X[index],self.X_Kal[index], self.Y[index]
#
#     def __len__(self):
#         return self.X.size(0)




class KalmanDataset(data.Dataset):
    def __init__(self, X, Y):
        """
        """
        observation_covariance = 1
        initial_value_guess = 0
        transition_matrix = 1
        transition_covariance = 0.01

        X_Kal, _ = EM_FK(initial_value_guess,observation_covariance, transition_covariance, transition_matrix).kalman_1D(X)
        #X_Kal, _ = Kalman_1D(X)
        X = np.expand_dims(X, 1)
        #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        self.X = torch.Tensor(X)
        self.X_Kal = torch.Tensor(X_Kal)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index],self.X_Kal[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)



class EMKFDataset(data.Dataset):
    def __init__(self, X, Y, EM):
        """
        """
        print('EM Kalman Start')

        X_Kal, _ = EM.kalman_1D(X)
        X = np.expand_dims(X, 1)
        #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        self.X = torch.Tensor(X)
        self.X_Kal = torch.Tensor(X_Kal)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index],self.X_Kal[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class GussianNoisyDataset(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        _, X_gussian, _= Gussian_Noisy_return_N(X, SNR)
        #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)

        self.X = torch.Tensor(X)
        self.X_Gussian = torch.Tensor(X_gussian)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.X_Gussian[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class GussianNoisyDataset_return_N(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        Noisy, Gussain, _= Gussian_Noisy_return_N(X, SNR)
        #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X,1)
        self.X = torch.Tensor(X)
        self.Nosiy_Only = torch.Tensor(Noisy)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Nosiy_Only[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)



class PossionNoisyDataset(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        _, Possion, _= Poisson_Noisy_return_N(X, SNR)
        #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Add_Noisy = torch.Tensor(Possion)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Add_Noisy[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)



class PossionNoisyDataset_return_N(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        Noisy, Possion, _= Poisson_Noisy_return_N(X, SNR)
        #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Noisy_Only = torch.Tensor(Noisy)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Noisy_Only[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class UniformNoisyDataset(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        _, Uniform, _= Uniform_Noisy_return_N(X, SNR)
        #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Add_Noisy = torch.Tensor(Uniform)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Add_Noisy[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)



class UniformNoisyDataset_return_N(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        Noisy, Uniform, _= Uniform_Noisy_return_N(X, SNR)
        #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Noisy_Only = torch.Tensor(Noisy)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Noisy_Only[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class ExponentialNoisyDataset(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        _, Exponential, _= Exponential_Noisy_return_N(X, SNR)
        #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Add_Nosiy = torch.Tensor(Exponential)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Add_Nosiy[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)



class ExponentialNoisyDataset_return_N(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        Noisy, Exponential, _= Exponential_Noisy_return_N(X, SNR)
        #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Noisy_Only = torch.Tensor(Noisy)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Noisy_Only[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class RayleignNoisyDataset(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        _, Rayleign, _= Rayleign_Noisy_return_N(X, SNR)
        #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Add_Noisy = torch.Tensor(Rayleign)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Add_Noisy[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)

class RayleignNoisyDataset_return_N(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        Noisy, Rayleign, _= Rayleign_Noisy_return_N(X, SNR)
        #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Noisy_Only = torch.Tensor(Noisy)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Noisy_Only[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)

class GammaNoisyDataset(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        _, Gamma, _= Gamma_Noisy_return_N(X, SNR)
        #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Add_Noisy = torch.Tensor(Gamma)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Add_Noisy[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class GammaNoisyDataset_return_N(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        Noisy, Gamma, _= Gamma_Noisy_return_N(X, SNR)
        #WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Noisy_Only = torch.Tensor(Noisy)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Noisy_Only[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class MELDataset(data.Dataset):
    def __init__(self, X, Y, Para):
        """
        """

        X_Mel, _ = mel_spectrogram_precessing_set(X,Para)
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.X_Mel = torch.Tensor(X_Mel)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.X_Mel[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)



class RPDataset(data.Dataset):
    def __init__(self, X, Y, transform):
        """
        """

        self.X_RP, _ = RP_preprocessing_set(X)

        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)
        self.transform = transform

    def __getitem__(self, index):
        # Get path of input image and ground truth

        X_raw = self.X[index]
        X_PR = self.X_RP[index]

        if self.transform is not None:
            X_PR = self.transform(X_PR)
        else:
            X_PR = torch.Tensor(X_PR)

        return X_raw, X_PR, self.Y[index]

    def __len__(self):
        return self.X.size(0)


class GTA_WindowDataset(data.Dataset):
    def __init__(self, opt, data, label, data_stamp):
        self.data = torch.Tensor(data)
        self.label = torch.Tensor(label)
        self.opt = opt
        # self.cut_len = int(self.data.shape[1]*0.9)
        self.data_stamp = data_stamp
        # if self.cut_len % 2 != 0: self.cut_len += 1

    def __getitem__(self, index):
        x_data = self.data[index]
        label = self.label[index]

        seq_x = x_data[:self.opt.seq_len,]
        seq_y = x_data[self.opt.seq_len - self.opt.label_len : self.opt.seq_len + self.opt.pred_len,]

        seq_x_mark = self.data_stamp[:self.opt.seq_len]
        seq_y_mark = self.data_stamp[self.opt.seq_len - self.opt.label_len : self.opt.seq_len + self.opt.pred_len]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, label

    def __len__(self):
        # return len(self.data) - self.window
        return len(self.data)

def WavePlot(x1,x2,x3,x4):
    x = np.linspace(0, len(x1)*4, len(x1)*4)
    a = list(x1)
    b = list(x2)
    d = list(x3)
    e = list(x4)
    a.extend(b)
    a.extend(d)
    a.extend(e)
    c= np.array(a)

    y = c



    plt.plot(x, y, ls="-", color="b", marker=",", lw=2)
    plt.axis('on')
    plt.legend()

    plt.show()
    plt.savefig("data_1.svg")


def WavePlot_Single(x1,name):
    x = np.linspace(0, len(x1), len(x1))
    a = list(x1)

    c= np.array(a)

    y = c



    plt.plot(x, y, ls="-", color="b", marker=",", lw=2)
    plt.axis('on')
    plt.tight_layout()
    plt.rcParams['figure.figsize'] = (5, 3)
    #plt.legend()

    #plt.show()
    plt.savefig('{}.svg'.format(name))
    plt.close()

# def WavePlot_Scatter(x1, x2, x3, name):
#     x = np.linspace(0, len(x1), len(x1))
#
#     a = list(x1)
#     b = list(x2)
#     c = list(x3)
#
#     d = np.array(a)
#     e = np.array(b)
#     f = np.array(c)
#
#     plt.figure()
#
#     plt.plot(d, color='r', linewidth=2.5,label='Signal')
#     #plt.scatter(x,d,color='y',label='True')
#     #plt.plot(e, 'b--', linewidth=2.5, label='KF')
#     plt.plot(f, 'c--', linewidth=2.5,label='EM_kF')
#
#     plt.axis('off')
#     plt.tight_layout()
#     plt.rcParams['figure.figsize'] = (5, 1)
#     plt.legend()
#
#     #plt.show()
#     plt.savefig('{}.eps'.format(name))
#     plt.close()


def WavePlot_Scatter(x1, x3, name):
    x = np.linspace(0, len(x1), len(x1))

    a = list(x1)

    c = list(x3)

    d = np.array(a)

    f = np.array(c)

    plt.figure(figsize=(8, 6), dpi=600)

    plt.plot(d, color='r', linewidth=2.5,label='Signal')
    #plt.scatter(x,d,color='y',label='True')
    #plt.plot(e, 'b--', linewidth=2.5, label='KF')
    plt.plot(f, 'c--', linewidth=2.5,label='EM_kF')

    plt.axis('off')
    plt.tight_layout()
    #plt.rcParams['figure.figsize'] = (48, 6)
    #plt.figure(figsize=(24, 4))
    #plt.figure(figsize=(24, 6), dpi=600)

    plt.legend()

    plt.show()
    plt.savefig('./Plot_Test/{}.svg'.format(name))
    plt.close()


def preprocess_signals(X_train, X_validation, X_test, outputfolder=None):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))

    # Save Standardizer data
    # with open(outputfolder + 'standard_scaler.pkl', 'wb') as ss_file:
    #   pickle.dump(ss, ss_file)

    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)


def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp


def get_EpilepticSeizure(dataset_path, dataset_name):
    data = []
    data_x = []
    data_y = []
    f = open('{}/{}/data.csv'.format(dataset_path, dataset_name), 'r')
    for line in range(0, 11501):
        if line == 0:
            f.readline()
            continue
        else:
            data.append(f.readline().strip())
    for i in range(0, 11500):
        tmp = data[i].split(",")
        del tmp[0]
        del tmp[178]
        data_x.append(tmp)
        data_y.append(data[i][-1])
    data_x = np.asfarray(data_x, dtype=np.float32)
    data_y = np.asarray([int(x) - 1 for x in data_y], dtype=np.int64)

    return data_x ,data_y

def load_data(opt, dataset_name, Filter, pre_train):



    if opt.data == 'UCR':



        if dataset_name in ['MFPT']:
            data_X = np.load("{}/{}/{}_data.npy".format(opt.data_UCR, dataset_name, dataset_name))
            # (2574,1024)
            data_Y = np.load("{}/{}/{}_label.npy".format(opt.data_UCR, dataset_name, dataset_name))


        elif dataset_name in ['CWRU']:
            with open('{}/{}/{}_data.pickle'.format(opt.data_UCR, dataset_name, dataset_name), 'rb') as handle1:
                data_X = pickle.load(handle1)
                # (8768,1024)

            with open('{}/{}/{}_label.pickle'.format(opt.data_UCR, dataset_name, dataset_name), 'rb') as handle2:
                data_Y = pickle.load(handle2)

        elif dataset_name in ['EpilepticSeizure']:

            data_X, data_Y = get_EpilepticSeizure(opt.data_UCR, dataset_name)

        elif dataset_name in ['cpsc']:

            data_X = np.load("{}/{}_data_pure_normal.npy".format(opt.data_CPSC, dataset_name))
            # (2574,1024)
            data_Y = np.load("{}/{}_label_pure_normal.npy".format(opt.data_CPSC, dataset_name))


            data_X = data_X[:,0:,:]

            data_X = np.squeeze(data_X)

            # data_X = data_X.reshape(-1,1,400)
            #
            # data_Y = np.repeat(data_Y, 4)

        elif dataset_name in ['DECG']:

            data_X = np.load("{}/{}_data.npy".format(opt.data_DECG, dataset_name))
            # (1881, 1000)
            data_Y = np.load("{}/{}_label.npy".format(opt.data_DECG, dataset_name))


            #data_X = data_X[:,0:1,:]

        elif dataset_name in ['CPSC']:

            data_X = np.load("{}/{}_Channel_0_Beat_Data.npy".format(opt.data_CPSC, dataset_name))
            # (13219, 120)
            data_Y = np.load("{}/{}_Channel_0_Beat_Label.npy".format(opt.data_CPSC, dataset_name))

            #WavePlot_Single(data_X[0], 'CPSC')



        elif dataset_name in ['zzu_MI']:
            with open('{}/{}_data.pickle'.format(opt.data_ZZU_MI, dataset_name), 'rb') as handle1:
                data_X = pickle.load(handle1)

                data_X = data_X.transpose(0,2,1)

                #data_X = data_X[:,:,:]
                # (64139,1000,12)

            with open('{}/{}_label.pickle'.format(opt.data_ZZU_MI, dataset_name), 'rb') as handle2:
                data_Y_5 = pickle.load(handle2)

                # (64139,5)
                data_Y=np.zeros(data_Y_5.shape[0])
                for i in range(data_Y_5.shape[0]):   # 二值化
                    if (data_Y_5[i] == [0,0,0,0,1]).all():
                        data_Y[i] = 0
                    else:
                        data_Y[i] = 1


        else:

            train_data = np.loadtxt(os.path.join(opt.data_UCR, dataset_name, (dataset_name + '_TRAIN.tsv')), delimiter='\t')  #
            test_data = np.loadtxt(os.path.join(opt.data_UCR, dataset_name, (dataset_name + '_TEST.tsv')), delimiter='\t')  #

            data_ALL = np.concatenate((train_data, test_data), axis=0)
            data_X = data_ALL[:, 1:]  #(16637,96)
            data_Y = data_ALL[:, 0]-min(data_ALL[:, 0])  #(16637,)

            if dataset_name == 'FordA':
                for i in range(len(data_Y)):
                    if data_Y[i] == 2:
                        data_Y[i] = 1


        #data_X = rescale(data_X)

        #data_X, data_Y = get_EpilepticSeizure(opt.data_UCR, dataset_name)

        label_idxs = np.unique(data_Y)



        class_stat={}
        for idx in label_idxs:
            class_stat[idx] = len(np.where(data_Y==idx)[0])

        if opt.normal_idx >= len(label_idxs):
            normal_idx = opt.normal_idx % len(label_idxs)
        else:
            normal_idx = opt.normal_idx



        if dataset_name == 'EpilepticSeizure':
            labels_binary, idx_normal, idx_abnormal = one_class_labeling_sz(data_Y, normal_idx, opt.seed)

        else:
            labels_binary, idx_normal, idx_abnormal = one_class_labeling(data_Y, normal_idx, opt.seed)


        #labels_binary, idx_normal, idx_abnormal = one_class_labeling_sz(data_Y, normal_idx, opt.seed)
        data_N_X = data_X[idx_normal]   #(4187,96)
        data_N_Y = labels_binary[idx_normal]  #(4187,)  1D
        data_A_X = data_X[idx_abnormal]   #(12450,96)
        data_A_Y = labels_binary[idx_abnormal]     # UCR end


        # Split normal samples
        n_normal = data_N_X.shape[0]
        train_X = data_N_X[:(int(n_normal * 0.6)), ]  # train 0.6
        train_Y = data_N_Y[:(int(n_normal * 0.6)), ]


        val_N_X = data_N_X[int(n_normal * 0.6):int(n_normal * 0.8)]  # train 0.2
        val_N_Y = data_N_Y[int(n_normal * 0.6):int(n_normal * 0.8)]


        test_N_X = data_N_X[int(n_normal * 0.8):]  # train 0.2
        test_N_Y = data_N_Y[int(n_normal * 0.8):]


        # val_N_X_len = val_N_X.shape[0]
        # test_N_X_len = test_N_X.shape[0]
        data_A_X_len = data_A_X.shape[0]


        # Split abnormal samples
        # data_A_X_idx = list(range(data_A_X_len))
        # # np.random.shuffle(data_A_X_idx)
        # np.random.shuffle(data_A_X)
        # val_A_X = data_A_X[data_A_X_idx[:val_N_X_len]]
        # val_A_Y = data_A_Y[data_A_X_idx[:val_N_X_len]]
        #
        # np.random.shuffle(data_A_X)
        # test_A_X = data_A_X[data_A_X_idx[:test_N_X_len]]
        # test_A_Y = data_A_Y[data_A_X_idx[:test_N_X_len]]
        ## val_A_X = data_A_X[data_A_X_idx[:val_N_X_len]]
        ## val_A_Y = data_A_Y[data_A_X_idx[:val_N_X_len]]
        ## test_A_X = data_A_X[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]
        ## test_A_Y = data_A_Y[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]

        ####### 正常与异常不平衡，采用异常全用的原则###########
        #
        val_N_X_len = data_A_X_len // 2
        test_N_X_len = data_A_X_len // 2


        data_A_X_idx = list(range(data_A_X_len))
        val_A_X = data_A_X[data_A_X_idx[:val_N_X_len]]
        val_A_Y = data_A_Y[data_A_X_idx[:val_N_X_len]]
        test_A_X = data_A_X[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]
        test_A_Y = data_A_Y[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]

        ####### 正常与异常平衡，采用异常不全用的原则###########


        # val_N_X_len = val_N_X.shape[0]
        # test_N_X_len = test_N_X.shape[0]
        # data_A_X_idx = list(range(data_A_X_len))
        # val_A_X = data_A_X[data_A_X_idx[:val_N_X_len]]
        # val_A_Y = data_A_Y[data_A_X_idx[:val_N_X_len]]
        # test_A_X = data_A_X[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]
        # test_A_Y = data_A_Y[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]





        val_X = np.concatenate((val_N_X, val_A_X))
        val_Y = np.concatenate((val_N_Y, val_A_Y))
        test_X = np.concatenate((test_N_X, test_A_X))
        test_Y = np.concatenate((test_N_Y, test_A_Y))


        # if opt.normalize:
        #     print("[INFO] Data Normalization!")
        #     # Normalize
        #     x_train_max = np.max(train_X)
        #     x_train_min = np.min(train_X)
        #     train_X = 2. * (train_X - x_train_min) / (x_train_max - x_train_min) - 1.
        #     # Test is secret
        #     val_X = 2. * (val_X - x_train_min) / (x_train_max - x_train_min) - 1.
        #     test_X = 2. * (test_X - x_train_min) / (x_train_max - x_train_min) - 1.



        if opt.normalize:
            print("[INFO] Data Normalization!")
            # Normalize

            #nb_dims = train_X.shape[1]

            #
            # for j in range(nb_dims):
            #     mean = np.mean(train_X[:, j])
            #     var = np.var(train_X[:, j],dtype=np.float64)
            #     train_X[:, j] = (train_X[:, j] - mean) / math.sqrt(var)
            #
            # for j in range(nb_dims):
            #     mean = np.mean(val_X[:, j])
            #     var = np.var(val_X[:, j],dtype=np.float64)
            #     val_X[:, j] = (val_X[:, j] - mean) / math.sqrt(var)
            #
            # for j in range(nb_dims):
            #     mean = np.mean(test_X[:, j])
            #     var = np.var(test_X[:, j],dtype=np.float64)
            #     test_X[:, j] = (test_X[:, j] - mean) / math.sqrt(var)


            #train_X, val_X, test_X = preprocess_signals(train_X, val_X, test_X)

                # 多通道归一化
            # global count
            # count = 0
            # # train_X, val_X, test_X = preprocess_signals(train_X, val_X, test_X)
            # for i in range(train_X.shape[0]):
            #     for j in range(12):
            #         train_X[i][j] = normalized(train_X[i][j][:])
            # train_X = train_X[:, :12, :]
            #
            # for i in range(val_X.shape[0]):
            #     for j in range(12):
            #         val_X[i][j] = normalized(val_X[i][j][:])
            # val_X = val_X[:, :12, :]
            #
            # for i in range(test_X.shape[0]):
            #     for j in range(12):
            #         test_X[i][j] = normalized(test_X[i][j][:])
            # test_X = test_X[:, :12, :]
            # print("count:", count)


            #train_X, val_X, test_X = preprocess_signals(train_X, val_X, test_X)


            # for i in range(train_X.shape[0]):
            #
            #     train_X[i] = normalize(train_X[i][:])
            #
            #
            # for i in range(val_X.shape[0]):
            #
            #     val_X[i] = normalize(val_X[i][:])
            #
            #
            # for i in range(test_X.shape[0]):
            #
            #     test_X[i] = normalize(test_X[i][:])



            # print("[INFO] Labels={}, normal label={}".format(label_idxs, opt.normal_idx))
            # print("[INFO] Train: normal={}".format(train_X.shape), )
            # print("[INFO] Val normal={}, abnormal={}".format(val_N_X.shape[0], val_A_X.shape[0]), )
            # print("[INFO] Test normal={}, abnormal={}".format(test_N_X.shape[0], test_A_X.shape[0]), )



        print("[INFO] Labels={}, normal label={}".format(label_idxs, opt.normal_idx))
        print("[INFO] Train: normal={}".format(train_X.shape), )
        print("[INFO] Val normal={}, abnormal={}".format(val_N_X.shape[0], val_A_X.shape[0]), )
        print("[INFO] Test normal={}, abnormal={}".format(test_N_X.shape[0], test_A_X.shape[0]), )

    if opt.data == 'ECG':

        N_samples = np.load(os.path.join(opt.data_ECG, "N_samples.npy")) #NxCxL  (86717,2,320)
        S_samples = np.load(os.path.join(opt.data_ECG, "S_samples.npy"))
        V_samples = np.load(os.path.join(opt.data_ECG, "V_samples.npy"))
        F_samples = np.load(os.path.join(opt.data_ECG, "F_samples.npy"))
        Q_samples = np.load(os.path.join(opt.data_ECG, "Q_samples.npy"))



        # normalize all
        for i in range(N_samples.shape[0]):
            for j in range(opt.nc):
                N_samples[i][j] = normalize(N_samples[i][j][:])
        N_samples = N_samples[:, :opt.nc, :]

        for i in range(S_samples.shape[0]):
            for j in range(opt.nc):
                S_samples[i][j] = normalize(S_samples[i][j][:])
        S_samples = S_samples[:, :opt.nc, :]

        for i in range(V_samples.shape[0]):
            for j in range(opt.nc):
                V_samples[i][j] = normalize(V_samples[i][j][:])
        V_samples = V_samples[:, :opt.nc, :]

        for i in range(F_samples.shape[0]):
            for j in range(opt.nc):
                F_samples[i][j] = normalize(F_samples[i][j][:])
        F_samples = F_samples[:, :opt.nc, :]

        for i in range(Q_samples.shape[0]):
            for j in range(opt.nc):
                Q_samples[i][j] = normalize(Q_samples[i][j][:])
        Q_samples = Q_samples[:, :opt.nc, :]

        test_N, test_N_y, train_N, train_N_y = getFloderK(N_samples, opt.folder, 0)
        # test_S,test_S_y, train_S,train_S_y = getFloderK(S_samples, opt.folder,1)
        # test_V,test_V_y, train_V,train_V_y = getFloderK(V_samples, opt.folder,1)
        # test_F,test_F_y, train_F,train_F_y = getFloderK(F_samples, opt.folder,1)
        # test_Q,test_Q_y, train_Q,train_Q_y = getFloderK(Q_samples, opt.folder,1)
        test_S, test_S_y = S_samples, np.ones((S_samples.shape[0], 1))
        test_V, test_V_y = V_samples, np.ones((V_samples.shape[0], 1))
        test_F, test_F_y = F_samples, np.ones((F_samples.shape[0], 1))
        test_Q, test_Q_y = Q_samples, np.ones((Q_samples.shape[0], 1))

        # train / val
        train_N, val_N, train_N_y, val_N_y = getPercent(train_N, train_N_y, 0.1, 0)

        test_S, val_S, test_S_y, val_S_y = getPercent(test_S, test_S_y, 0.1, 0)
        test_V, val_V, test_V_y, val_V_y = getPercent(test_V, test_V_y, 0.1, 0)
        test_F, val_F, test_F_y, val_F_y = getPercent(test_F, test_F_y, 0.1, 0)
        test_Q, val_Q, test_Q_y, val_Q_y = getPercent(test_Q, test_Q_y, 0.1, 0)

        val_data = np.concatenate([val_N, val_S, val_V, val_F, val_Q])
        val_y = np.concatenate([val_N_y, val_S_y, val_V_y, val_F_y, val_Q_y])

        test_data = np.concatenate([test_N,test_S,test_V,test_F,test_Q])
        test_y = np.concatenate([test_N_y,test_S_y,test_V_y,test_F_y,test_Q_y])

        train_N = train_N.reshape(train_N.shape[0],-1)    #(62436,320)
        val_data = val_data.reshape(val_data.shape[0],-1)
        test_data = test_data.reshape(test_data.shape[0],-1)   #(9674,320)


        train_N_y= train_N_y.flatten()
        val_Y = val_y.flatten()
        test_y = test_y.flatten()    # ECG   end


        # Split normal samples
        #n_normal = data_N_X.shape[0]
        train_X = train_N  # 62436
        train_Y = train_N_y
        val_X = val_data   # 8025
        val_Y = val_Y
        test_X = test_data  #27107
        test_Y = test_y


        #print("[INFO] Train: normal={}".format(train_X.shape), )
        # print("[INFO] Val normal={}, abnormal={}".format(val_N_X.shape[0], val_A_X.shape[0]), )
        # print("[INFO] Test normal={}, abnormal={}".format(test_N_X.shape[0], test_A_X.shape[0]), )


    # Wavelet transform
    X_length = train_X.shape[-1]
    #X_length = 400

    #transform = transforms.ToTensor()

    signal_length=[0]

    #a = MELDataset(train_X,train_Y,extract_params)

    if opt.model in ['AE_OSCNN_WL','AE_CNN_WL','AE_CNN_WL_ATT','AE_CNN_WL_SIN' ,"MM_GAN_OSCNN",'MM_GAN_CNN', "MM_GAN", 'MM_GAN_OSCNN_CAT']:
        _, signal_length = wavelet_preprocessing_set(train_X)

        train_X = np.expand_dims(train_X, 1)  # (292,1,140)
        test_X = np.expand_dims(test_X, 1)  # (4500,1,141)
        val_X = np.expand_dims(val_X, 1)

        train_dataset = WaveLetDataset(train_X, train_Y)
        val_dataset = WaveLetDataset(val_X, val_Y)
        test_dataset = WaveLetDataset(test_X, test_Y)
    elif opt.model in ['AE_OSCNN_RP','AE_CNN_RP','AE_CNN_RP_SIN','AE_CNN_EEG']:
        _, signal_length = RP_preprocessing_set(train_X)

        train_dataset = RPDataset(train_X, train_Y, None)
        val_dataset = RPDataset(val_X, val_Y, None)
        test_dataset = RPDataset(test_X, test_Y, None)

    elif opt.model in ['AE_CNN_MEL']:
        _,signal_length = mel_spectrogram_precessing_set(train_X,extract_params)

        train_dataset = MELDataset(train_X,train_Y,extract_params)
        val_dataset = MELDataset(val_X,val_Y,extract_params)
        test_dataset = MELDataset(test_X,test_Y,extract_params)


        print(opt.Snr)
        train_dataset = GussianNoisyDataset(train_X,train_Y,opt.Snr)
        val_dataset = GussianNoisyDataset(val_X,val_Y,opt.Snr)
        test_dataset = GussianNoisyDataset(test_X,test_Y,opt.Snr)

    elif opt.model in ['AE_CNN_Noisy_Only']:
        _, _, signal_length = Gussian_Noisy_return_N(train_X,snr)

        print(snr)
        train_dataset = GussianNoisyDataset_return_N(train_X,train_Y)
        val_dataset = GussianNoisyDataset_return_N(val_X,val_Y)
        test_dataset = GussianNoisyDataset_return_N(test_X,test_Y)

    elif opt.model in ['AE_CNN_noisy_multi']:
        print(opt.Snr)

        if opt.NoisyType == 'Gussian':

            _, _, signal_length = Gussian_Noisy_return_N(train_X, opt.Snr)
            train_dataset = GussianNoisyDataset_return_N(train_X, train_Y, opt.Snr)
            val_dataset = GussianNoisyDataset_return_N(val_X, val_Y, opt.Snr)
            test_dataset = GussianNoisyDataset_return_N(test_X, test_Y, opt.Snr)

        elif opt.NoisyType == 'Rayleign':
            _, _, signal_length = Rayleign_Noisy_return_N(train_X, opt.Snr)
            train_dataset = RayleignNoisyDataset_return_N(train_X, train_Y, opt.Snr)
            val_dataset = RayleignNoisyDataset_return_N(val_X, val_Y, opt.Snr)
            test_dataset = RayleignNoisyDataset_return_N(test_X, test_Y, opt.Snr)

        elif opt.NoisyType == 'Exponential':
            _, _, signal_length = Exponential_Noisy_return_N(train_X, opt.Snr)
            train_dataset = ExponentialNoisyDataset_return_N(train_X, train_Y, opt.Snr)
            val_dataset = ExponentialNoisyDataset_return_N(val_X, val_Y, opt.Snr)
            test_dataset = ExponentialNoisyDataset_return_N(test_X, test_Y, opt.Snr)

        elif opt.NoisyType == 'Uniform':
            _, _, signal_length = Uniform_Noisy_return_N(train_X, opt.Snr)
            train_dataset = UniformNoisyDataset_return_N(train_X, train_Y, opt.Snr)
            val_dataset = UniformNoisyDataset_return_N(val_X, val_Y, opt.Snr)
            test_dataset = UniformNoisyDataset_return_N(test_X, test_Y, opt.Snr)

        elif opt.NoisyType == 'Poisson':
            _, _, signal_length = Poisson_Noisy_return_N(train_X, opt.Snr)
            train_dataset = PossionNoisyDataset_return_N(train_X, train_Y, opt.Snr)
            val_dataset = PossionNoisyDataset_return_N(val_X, val_Y, opt.Snr)
            test_dataset = PossionNoisyDataset_return_N(test_X, test_Y, opt.Snr)

        elif opt.NoisyType =='Gamma':
            _, _, signal_length = Gamma_Noisy_return_N(train_X, opt.Snr)
            train_dataset = GammaNoisyDataset_return_N(train_X, train_Y, opt.Snr)
            val_dataset = GammaNoisyDataset_return_N(val_X, val_Y, opt.Snr)
            test_dataset = GammaNoisyDataset_return_N(test_X, test_Y, opt.Snr)
        else:

            print("illegal noisy type")



    elif opt.model in ['AE_CNN_Noisy']:
        print(opt.Snr)

        if opt.NoisyType == 'Gussian':

            _, _, signal_length = Gussian_Noisy_return_N(train_X, opt.Snr)
            train_dataset = GussianNoisyDataset(train_X, train_Y, opt.Snr)
            val_dataset = GussianNoisyDataset(val_X, val_Y, opt.Snr)
            test_dataset = GussianNoisyDataset(test_X, test_Y, opt.Snr)

        elif opt.NoisyType == 'Rayleign':
            _, _, signal_length = Rayleign_Noisy_return_N(train_X, opt.Snr)
            train_dataset = RayleignNoisyDataset(train_X, train_Y, opt.Snr)
            val_dataset = RayleignNoisyDataset(val_X, val_Y, opt.Snr)
            test_dataset = RayleignNoisyDataset(test_X, test_Y, opt.Snr)

        elif opt.NoisyType == 'Exponential':
            _, _, signal_length = Exponential_Noisy_return_N(train_X, opt.Snr)
            train_dataset = ExponentialNoisyDataset(train_X, train_Y, opt.Snr)
            val_dataset = ExponentialNoisyDataset(val_X, val_Y, opt.Snr)
            test_dataset = ExponentialNoisyDataset(test_X, test_Y, opt.Snr)

        elif opt.NoisyType == 'Uniform':
            _, _, signal_length = Uniform_Noisy_return_N(train_X, opt.Snr)
            train_dataset = UniformNoisyDataset(train_X, train_Y, opt.Snr)
            val_dataset = UniformNoisyDataset(val_X, val_Y, opt.Snr)
            test_dataset = UniformNoisyDataset(test_X, test_Y, opt.Snr)

        elif opt.NoisyType == 'Poisson':
            _, _, signal_length = Poisson_Noisy_return_N(train_X, opt.Snr)
            train_dataset = PossionNoisyDataset(train_X, train_Y, opt.Snr)
            val_dataset = PossionNoisyDataset(val_X, val_Y, opt.Snr)
            test_dataset = PossionNoisyDataset(test_X, test_Y, opt.Snr)

        elif opt.NoisyType =='Gamma':
            _, _, signal_length = Gamma_Noisy_return_N(train_X, opt.Snr)
            train_dataset = GammaNoisyDataset(train_X, train_Y, opt.Snr)
            val_dataset = GammaNoisyDataset(val_X, val_Y, opt.Snr)
            test_dataset = GammaNoisyDataset(test_X, test_Y, opt.Snr)
        else:

            print("illegal noisy type")


    elif opt.model in ['AE_CNN_Filter', 'AE_CNN_Filter_4indicator']:


        if opt.FilterType == 'Savitzky':

            train_dataset = SavitzkyDataset(train_X, train_Y)
            val_dataset = SavitzkyDataset(val_X, val_Y)
            test_dataset = SavitzkyDataset(test_X, test_Y)

        elif opt.FilterType == 'Wiener':
            train_dataset = WienerDataset(train_X, train_Y)
            val_dataset = WienerDataset(val_X, val_Y)
            test_dataset = WienerDataset(test_X, test_Y)

        elif opt.FilterType == 'Kalman':
            train_dataset = KalmanDataset(train_X, train_Y)
            val_dataset = KalmanDataset(val_X, val_Y)
            test_dataset = KalmanDataset(test_X, test_Y)

        elif opt.FilterType == 'EM-KF':

            if pre_train:
                # train_X = np.expand_dims(train_X, 1)  # (292,1,140)
                # test_X = np.expand_dims(test_X, 1)  # (4500,1,141)
                # val_X = np.expand_dims(val_X, 1)

                train_dataset = RawDataset_EM_KF_init(train_X, train_Y)
                val_dataset = RawDataset_EM_KF_init(val_X, val_Y)
                test_dataset = RawDataset_EM_KF_init(test_X, test_Y)
            else:

                train_dataset = EMKFDataset(train_X, train_Y, Filter)
                val_dataset = EMKFDataset(val_X, val_Y, Filter)
                test_dataset = EMKFDataset(test_X, test_Y, Filter)

                #train_dataset = EMKFDataset(data_X, data_Y, Filter)




        else:

            print("illegal Filter type")



    elif opt.model == "SLMR":
        target_dims = opt.nc
        window_size = X_length

        X_length = int(window_size * 0.9)
        if X_length % 2 != 0: X_length += 1
        opt.pre_len = window_size - X_length

        train_X = np.expand_dims(train_X, 2)  # (bs, X_length, 1)
        test_X = np.expand_dims(test_X, 2)  # (bs, X_length, 1)
        val_X = np.expand_dims(val_X, 2)

        train_dataset = SlidingWindowDataset(opt, train_X, train_Y, window_size, target_dims)
        test_dataset = SlidingWindowDataset(opt, test_X, test_Y, window_size, target_dims, val_Y)
        val_dataset = SlidingWindowDataset(opt, val_X, val_Y, window_size, target_dims, test_Y)

    elif opt.model == "ECOD":
        return train_X, test_X, train_Y.astype("float"), test_Y.astype("float")

    elif opt.model == "GTA":
        window_size = X_length

        X_length = int(window_size * 0.9)  # 序列长度

        if X_length % 2 != 0: X_length += 1

        opt.num_nodes = X_length
        opt.pred_len = window_size - X_length  # 需要预测长度
        opt.seq_len = X_length  # 序列长度
        opt.label_len = opt.seq_len // 2  # 用来预测的序列长度

        train_X = np.expand_dims(train_X, 2)  # (bs, X_length, 1)
        test_X = np.expand_dims(test_X, 2)  # (bs, X_length, 1)
        val_X = np.expand_dims(val_X, 2)

        df_stamp = pd.DataFrame(columns=['date'])
        date = pd.date_range(start='1/1/2015', periods=window_size, freq='4s')
        df_stamp['date'] = date
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        # df_stamp['minute'] = df_stamp.minute.map(lambda x:x//10)
        df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)
        # data_stamp = df_stamp.drop(['date'],1).values
        data_stamp = df_stamp.drop(['date'], axis=1).values

        train_dataset = GTA_WindowDataset(opt, train_X, train_Y, data_stamp)
        test_dataset = GTA_WindowDataset(opt, test_X, test_Y, data_stamp)
        val_dataset = GTA_WindowDataset(opt, val_X, val_Y, data_stamp)

    elif opt.model == "NSIBF":
        window_size = X_length

        X_length = int(window_size * 0.9)  # 序列长度

        if X_length % 2 != 0: X_length += 1

        opt.seqL = X_length
        # opt.pred_len = window_size-X_length #需要预测长度
        opt.input_range = window_size  # 序列长度
        # opt.label_len = opt.seq_len // 2 #用来预测的序列长度

        # train_X = train_X[:, :opt.seqL]
        # test_X = test_X[:, :opt.seqL]
        # val_X = val_X[:, :opt.seqL]

        return train_X, test_X, val_X, test_Y



    else:
        train_X = np.expand_dims(train_X, 1)  # (292,1,140)
        test_X = np.expand_dims(test_X, 1)  # (4500,1,141)
        val_X = np.expand_dims(val_X, 1)

        train_dataset = RawDataset(train_X, train_Y)
        val_dataset = RawDataset(val_X, val_Y)
        test_dataset = RawDataset(test_X, test_Y)

        # train_dataset = SavitzkyDataset(train_X, train_Y)
        # val_dataset = SavitzkyDataset(val_X, val_Y)
        # test_dataset = SavitzkyDataset(test_X, test_Y)

    # np.savetxt('log9/' + opt.dataloader + str(opt.normal_idx) + "_label.txt",
    #            val_Y.astype(int),
    #            delimiter=",",  fmt='%s')

    #WavePlot(train_X[0][0],train_X[1][0],train_X[2][0],train_X[3][0])



    dataloader = {"train": DataLoader(
                            dataset=train_dataset,  # torch TensorDataset format
                            batch_size=opt.batchsize,  # mini batch size
                            num_workers=int(opt.workers),
                            drop_last=True),

                    "val": DataLoader(
                            dataset=val_dataset,  # torch TensorDataset format
                            batch_size=opt.batchsize,  # mini batch size
                            num_workers=int(opt.workers),
                            drop_last=False),

                    "test":DataLoader(
                                dataset=test_dataset,  # torch TensorDataset format
                                batch_size=opt.batchsize,  # mini batch size
                                num_workers=int(opt.workers),
                                drop_last=False),


    }

    return dataloader, X_length, signal_length

# RP  raw (150,1,720)   RP (150,1,144,144)


def load_data_Multi(opt, dataset_name):

    if opt.data == 'UCR':

        train_data = np.loadtxt(os.path.join(opt.data_UCR, dataset_name, (dataset_name + '_TRAIN.tsv')), delimiter='\t')  #
        test_data = np.loadtxt(os.path.join(opt.data_UCR, dataset_name, (dataset_name + '_TEST.tsv')), delimiter='\t')  #

        data_ALL = np.concatenate((train_data, test_data), axis=0)
        data_X = data_ALL[:, 1:]
        data_Y = data_ALL[:, 0]-min(data_ALL[:, 0])

        data_X = rescale(data_X)

        label_idxs = np.unique(data_Y)
        class_stat={}
        for idx in label_idxs:
            class_stat[idx] = len(np.where(data_Y==idx)[0])

        # if opt.normal_idx >= len(label_idxs):
        #      normal_idx = opt.normal_idx % len(label_idxs)
        # else:
        #     normal_idx = opt.normal_idx

        labels_binary, idx_normal, idx_abnormal = one_class_labeling_multi(data_Y, opt.normal_idx)
        data_N_X = data_X[idx_normal]   #(4187,96)
        data_N_Y = labels_binary[idx_normal]  #(4187,)  1D
        data_A_X = data_X[idx_abnormal]   #(12450,96)
        data_A_Y = labels_binary[idx_abnormal]     # UCR end

        # Split normal samples
        n_normal = data_N_X.shape[0]
        train_X = data_N_X[:(int(n_normal * 0.6)), ]  # train 0.6
        train_Y = data_N_Y[:(int(n_normal * 0.6)), ]

        val_N_X = data_N_X[int(n_normal * 0.6):int(n_normal * 0.8)]  # train 0.2
        val_N_Y = data_N_Y[int(n_normal * 0.6):int(n_normal * 0.8)]
        test_N_X = data_N_X[int(n_normal * 0.8):]  # train 0.2
        test_N_Y = data_N_Y[int(n_normal * 0.8):]

        val_N_X_len = val_N_X.shape[0]
        test_N_X_len = test_N_X.shape[0]
        data_A_X_len = data_A_X.shape[0]

        # Split abnormal samples
        data_A_X_idx = list(range(data_A_X_len))
        # np.random.shuffle(data_A_X_idx)
        np.random.shuffle(data_A_X)
        val_A_X = data_A_X[data_A_X_idx[:val_N_X_len]]
        val_A_Y = data_A_Y[data_A_X_idx[:val_N_X_len]]

        np.random.shuffle(data_A_X)
        test_A_X = data_A_X[data_A_X_idx[:test_N_X_len]]
        test_A_Y = data_A_Y[data_A_X_idx[:test_N_X_len]]
        # val_A_X = data_A_X[data_A_X_idx[:val_N_X_len]]
        # val_A_Y = data_A_Y[data_A_X_idx[:val_N_X_len]]
        # test_A_X = data_A_X[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]
        # test_A_Y = data_A_Y[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]

        val_X = np.concatenate((val_N_X, val_A_X))
        val_Y = np.concatenate((val_N_Y, val_A_Y))
        test_X = np.concatenate((test_N_X, test_A_X))
        test_Y = np.concatenate((test_N_Y, test_A_Y))

        print("[INFO] Labels={}, normal label={}".format(label_idxs, opt.normal_idx))
        print("[INFO] Train: normal={}".format(train_X.shape), )
        print("[INFO] Val normal={}, abnormal={}".format(val_N_X.shape[0], val_A_X.shape[0]), )
        print("[INFO] Test normal={}, abnormal={}".format(test_N_X.shape[0], test_A_X.shape[0]), )

    if opt.data == 'ECG':

        N_samples = np.load(os.path.join(opt.data_ECG, "N_samples.npy")) #NxCxL  (86717,2,320)
        S_samples = np.load(os.path.join(opt.data_ECG, "S_samples.npy"))
        V_samples = np.load(os.path.join(opt.data_ECG, "V_samples.npy"))
        F_samples = np.load(os.path.join(opt.data_ECG, "F_samples.npy"))
        Q_samples = np.load(os.path.join(opt.data_ECG, "Q_samples.npy"))



        # normalize all
        for i in range(N_samples.shape[0]):
            for j in range(opt.nc):
                N_samples[i][j] = normalize(N_samples[i][j][:])
        N_samples = N_samples[:, :opt.nc, :]

        for i in range(S_samples.shape[0]):
            for j in range(opt.nc):
                S_samples[i][j] = normalize(S_samples[i][j][:])
        S_samples = S_samples[:, :opt.nc, :]

        for i in range(V_samples.shape[0]):
            for j in range(opt.nc):
                V_samples[i][j] = normalize(V_samples[i][j][:])
        V_samples = V_samples[:, :opt.nc, :]

        for i in range(F_samples.shape[0]):
            for j in range(opt.nc):
                F_samples[i][j] = normalize(F_samples[i][j][:])
        F_samples = F_samples[:, :opt.nc, :]

        for i in range(Q_samples.shape[0]):
            for j in range(opt.nc):
                Q_samples[i][j] = normalize(Q_samples[i][j][:])
        Q_samples = Q_samples[:, :opt.nc, :]

        test_N, test_N_y, train_N, train_N_y = getFloderK(N_samples, opt.folder, 0)
        # test_S,test_S_y, train_S,train_S_y = getFloderK(S_samples, opt.folder,1)
        # test_V,test_V_y, train_V,train_V_y = getFloderK(V_samples, opt.folder,1)
        # test_F,test_F_y, train_F,train_F_y = getFloderK(F_samples, opt.folder,1)
        # test_Q,test_Q_y, train_Q,train_Q_y = getFloderK(Q_samples, opt.folder,1)
        test_S, test_S_y = S_samples, np.ones((S_samples.shape[0], 1))
        test_V, test_V_y = V_samples, np.ones((V_samples.shape[0], 1))
        test_F, test_F_y = F_samples, np.ones((F_samples.shape[0], 1))
        test_Q, test_Q_y = Q_samples, np.ones((Q_samples.shape[0], 1))

        # train / val
        train_N, val_N, train_N_y, val_N_y = getPercent(train_N, train_N_y, 0.1, 0)

        test_S, val_S, test_S_y, val_S_y = getPercent(test_S, test_S_y, 0.1, 0)
        test_V, val_V, test_V_y, val_V_y = getPercent(test_V, test_V_y, 0.1, 0)
        test_F, val_F, test_F_y, val_F_y = getPercent(test_F, test_F_y, 0.1, 0)
        test_Q, val_Q, test_Q_y, val_Q_y = getPercent(test_Q, test_Q_y, 0.1, 0)

        val_data = np.concatenate([val_N, val_S, val_V, val_F, val_Q])
        val_y = np.concatenate([val_N_y, val_S_y, val_V_y, val_F_y, val_Q_y])

        test_data = np.concatenate([test_N,test_S,test_V,test_F,test_Q])
        test_y = np.concatenate([test_N_y,test_S_y,test_V_y,test_F_y,test_Q_y])

        train_N = train_N.reshape(train_N.shape[0],-1)    #(62436,320)
        val_data = val_data.reshape(val_data.shape[0],-1)
        test_data = test_data.reshape(test_data.shape[0],-1)   #(9674,320)


        train_N_y= train_N_y.flatten()
        val_Y = val_y.flatten()
        test_y = test_y.flatten()    # ECG   end


        # Split normal samples
        #n_normal = data_N_X.shape[0]
        train_X = train_N  # 62436
        train_Y = train_N_y
        val_X = val_data   # 8025
        val_Y = val_Y
        test_X = test_data  #27107
        test_Y = test_y


        #print("[INFO] Train: normal={}".format(train_X.shape), )
        # print("[INFO] Val normal={}, abnormal={}".format(val_N_X.shape[0], val_A_X.shape[0]), )
        # print("[INFO] Test normal={}, abnormal={}".format(test_N_X.shape[0], test_A_X.shape[0]), )


    # Wavelet transform
    X_length = train_X.shape[-1]

    #transform = transforms.ToTensor()

    signal_length=[0]

    #a = MELDataset(train_X,train_Y,extract_params)

    if opt.model in ['AE_OSCNN_WL','AE_CNN_WL', "MM_GAN_OSCNN",'MM_GAN_CNN', "MM_GAN_OSCNN_CAT"]:
        _, signal_length = wavelet_preprocessing_set(train_X)

        train_X = np.expand_dims(train_X, 1)  # (292,1,140)
        test_X = np.expand_dims(test_X, 1)  # (4500,1,141)
        val_X = np.expand_dims(val_X, 1)

        train_dataset = WaveLetDataset(train_X, train_Y)
        val_dataset = WaveLetDataset(val_X, val_Y)
        test_dataset = WaveLetDataset(test_X, test_Y)
    elif opt.model in ['AE_OSCNN_RP','AE_CNN_RP']:
        _, signal_length = RP_preprocessing_set(train_X)

        train_dataset = RPDataset(train_X, train_Y, None)
        val_dataset = RPDataset(val_X, val_Y, None)
        test_dataset = RPDataset(test_X, test_Y, None)

    elif opt.model in ['AE_CNN_MEL']:
        _,signal_length = mel_spectrogram_precessing_set(train_X,extract_params)

        train_dataset = MELDataset(train_X,train_Y,extract_params)
        val_dataset = MELDataset(val_X,val_Y,extract_params)
        test_dataset = MELDataset(test_X,test_Y,extract_params)

    else:
        train_X = np.expand_dims(train_X, 1)  # (292,1,140)
        test_X = np.expand_dims(test_X, 1)  # (4500,1,141)
        val_X = np.expand_dims(val_X, 1)

        train_dataset = RawDataset(train_X, train_Y)
        val_dataset = RawDataset(val_X, val_Y)
        test_dataset = RawDataset(test_X, test_Y)

    # np.savetxt('log9/' + opt.dataloader + str(opt.normal_idx) + "_label.txt",
    #            val_Y.astype(int),
    #            delimiter=",",  fmt='%s')

    #WavePlot(train_X[0][0],train_X[1][0],train_X[2][0])
    dataloader = {"train": DataLoader(
                            dataset=train_dataset,  # torch TensorDataset format
                            batch_size=opt.batchsize,  # mini batch size
                            shuffle=False,
                            num_workers=int(opt.workers),
                            drop_last=True),

                    "val": DataLoader(
                            dataset=val_dataset,  # torch TensorDataset format
                            batch_size=opt.batchsize,  # mini batch size
                            shuffle=False,
                            num_workers=int(opt.workers),
                            drop_last=False),

                    "test":DataLoader(
                                dataset=test_dataset,  # torch TensorDataset format
                                batch_size=opt.batchsize,  # mini batch size
                                shuffle=False,
                                num_workers=int(opt.workers),
                                drop_last=False)
    }

    return dataloader, X_length, signal_length


