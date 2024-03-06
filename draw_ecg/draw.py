import pickle as pkl
from get_AHLF import getAHLF
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import numpy as np
from draw_ecg import plot_sample, plot_sample_2

# data = pkl.load(open('/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL/data_0_1_normal.pickle', "rb"))
# data1 = pkl.load(open('/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL/data_10_10_fangchan.pickle', "rb"))
data = np.load('/home/chenpeng/workspace/dataset/11k/afib/00903_22142976.npy')
data1 = np.load('/home/chenpeng/workspace/dataset/11k/normal/02052_72040448.npy')
# data1 = data.flatten()
# data2 = data1[:1000]
# # # data1 = data[:18414000].reshape(-1, 1000)  # (18414, 1000)
# data2 = data[:400, :]
afi, heart_rate, lf, hf, sdnn, rmssd, nn20, nn50 = getAHLF(data)
afi1, heart_rate1, lf1, hf1, sdnn1, rmssd1, nn201, nn501 = getAHLF(data1)
# Total, LF, HF, LFnum, HFnum, LF_HF = getAHLF(data2)
# print("Total:		%f [ms^2]" % Total)
# print("LF:		    %f [ms^2]" % LF)
# print("HF :		    %f [ms^2]" % HF)
# print("LFnum:		%f [nU]" % LFnum)
# print("HFnum	:	%f [nU]" % HFnum)
# print("LF/HF ratio	: 	%f [-]" % LF_HF)
# # ecg = data2[1]

data4 = [afi, heart_rate, lf, hf, sdnn, rmssd, nn20, nn50]
df = pd.DataFrame(data4).T
df.columns = ['afi', 'hr', 'lf', 'hf', 'sdnn', 'rmssd', 'nn20', 'nn50']
filepath = '/home/gaoshaochen/data/DATA/data_normal.csv'
df.to_csv(filepath, index=False)

data5 = [afi1, heart_rate1, lf1, hf1, sdnn1, rmssd1, nn201, nn501]
df = pd.DataFrame(data5).T
df.columns = ['afi', 'hr', 'lf', 'hf', 'sdnn', 'rmssd', 'nn20', 'nn50']
filepath = '/home/gaoshaochen/data/DATA/data_AF.csv'
df.to_csv(filepath, index=False)

print('ok')
# data = '/home/gaoshaochen/data/DATA/data_AF.csv'
# df = pd.read_csv('/home/gaoshaochen/data/DATA/data_AF.csv')
# plot = sns.heatmap(df.corr(),  cmap="YlGnBu", annot=True, linewidths=.5, fmt=".3f")
# plt.savefig('relation2.svg')
# plot_sample_2(data, data1)
