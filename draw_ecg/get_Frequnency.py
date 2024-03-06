import pickle as pkl
from get_AHLF import getAHLF
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

#
# data = pkl.load(open('/home/gaoshaochen/data/Cinc2017_Accept.pickle', "rb"))
data = np.load('/data/ECG/data/ecg_img/235476.npy')
data1 = data.flatten()
print(data1.shape)
