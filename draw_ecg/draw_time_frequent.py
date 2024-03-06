import numpy as np
import pyhrv.tools as tools
import pyhrv.time_domain as td
import biosppy
import neurokit2 as nk
import matplotlib.pyplot as plt
from hrvanalysis.plot import plot_psd
from hrvanalysis.plot import plot_poincare
import pickle as pkl


def get_hr(signal_org, sp_rate):
    hr_mean = 0
    hr_std = 0
    heart_rates = [0]
    templates = [0]
    rpeaks = [0]

    try:
        if len(signal_org.shape) > 1:
            signal_org = signal_org.squeeze()

        signal, rpeaks, templates_ts, templates, heart_rate_ts, heart_rates \
            = biosppy.signals.ecg.ecg(signal_org, sampling_rate=sp_rate, show=False)[1:]

        templates = templates.tolist()
        rpeaks = rpeaks.tolist()

        # hr_mean = np.mean(heart_rates)
        # hr_std = np.std(heart_rates)
        # heart_rates = [int(hr) for hr in heart_rates]
    finally:
        return rpeaks


def get_nni(rpeaks, sp_rate=100):
    nni = [0]
    try:
        nni = tools.nn_intervals(rpeaks, sampling_rate=sp_rate)
    finally:
        return nni


# nn_intervals_list = [1000, 1050, 1020, 1080, ..., 1100, 1110, 1060]
if __name__ == "__main__":
    data = pkl.load(open("/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL_100HZ/data_0_1_normal.pickle", 'rb'))
    data1 = data[1][1]
    nn_intervals_list = get_hr(data1, 100)
    # plot_psd(nn_intervals_list, method="welch")
    plot_poincare(nn_intervals_list)
    # plot_psd(nn_intervals_list, method="lomb")
