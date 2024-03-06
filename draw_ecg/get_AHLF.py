import sys

sys.path.append('/home/gaoshaochen/Python/')
import numpy as np
import AF_Mul.experiments.MlultModal.tool.CPSC_dataloader.pyhrv.tools as tools
import pyhrv.time_domain as td
import biosppy
import neurokit2 as nk
import matplotlib.pyplot as plt
import pyhrv.time_domain as td
from AF_Mul.experiments.MlultModal.tool.CPSC_dataloader.pyhrv.frequency_domain import welch_psd


def get_hr(signal_org, sp_rate=100):
    # hr_mean=0
    # hr_std=0
    # heart_rates=[0]
    # templates=[0]
    # rpeaks = [0]

    if len(signal_org.shape) > 1:
        signal_org = signal_org.squeeze()

    signal, rpeaks, templates_ts, templates, heart_rate_ts, heart_rates \
        = biosppy.signals.ecg.ecg(signal_org, sampling_rate=sp_rate, show=False)[1:]

    templates = templates.tolist()
    rpeaks = rpeaks.tolist()

    hr_mean = np.mean(heart_rates)

    if np.isnan(hr_mean):
        rr_intervals = np.diff(rpeaks)
        heart_rates1 = 60 / (rr_intervals / 1000)
        hr_mean = np.mean(heart_rates1)

    # hr_std = np.std(heart_rates)
    # heart_rates = [int(hr) for hr in heart_rates]
    # return hr_mean, hr_std, heart_rates, templates, rpeaks
    return rpeaks, hr_mean


def get_nni(rpeaks):
    nni = tools.nn_intervals(rpeaks)

    return nni


def get_hrv(nni):
    sdnn, rmssd, nn20, pnn20, nn50, pnn50 = (0, 0, 0, 0, 0, 0)
    try:
        # Compute SDNN
        # results = td.time_domain(nni)
        sdnn_ = td.sdnn(nni)
        rmssd_ = td.rmssd(nni)
        nn20_ = td.nn20(nni)
        nn50_ = td.nn50(nni)

        sdnn = round(sdnn_['sdnn'], 1)
        rmssd = round(rmssd_['rmssd'], 1)
        nn20 = nn20_['nn20']
        pnn20 = round(nn20_['pnn20'], 1)
        nn50 = nn50_['nn50']
        pnn50 = round(nn50_['pnn50'], 1)
    finally:
        return sdnn, rmssd, nn20, pnn20, nn50, pnn50


def get_HLF(nni):
    lf, hf = (0, 0)

    results = welch_psd(nni, mode='normal')
    # Total = results['fft_total']
    lf = results['fft_abs'][1]
    hf = results['fft_abs'][2]
    # lfnum = (lf / (Total)) * 100
    # hfnum = (hf / (Total)) * 100
    # lf_hf = results['fft_ratio']

    return lf, hf


def getAHLF(signal):
    sdnn = []
    rmssd = []
    afi = []
    nn20 = []
    pnn20 = []
    nn50 = []
    pnn50 = []
    heart_rate = []
    lf = []
    hf = []
    err = 0

    for b, i in enumerate(signal):
        if b % 1000 == 0:
            print("已处理: " + str(b))
        if len(i.shape) > 1:
            i = i.squeeze()

            if len(i.shape) > 1 and i.shape[0] > 1:
                # 多道联
                '''total_heart_rate = []
                total_afi = []
                total_hf = []
                total_lf = []
                total_rounds = len(i)
                err = 0
                for round_data in i:

                    rpeaks, heart_mean = get_hr(round_data)
                    heart_rate0 = heart_mean

                    nni = get_nni(rpeaks)
                    sdnn0, rmssd0, _, _, _, _ = get_hrv(nni)
                    mean_nn = np.mean(nni)
                    afi0 = sdnn0 / mean_nn

                    lf0, hf0 = get_HLF(nni)

                    if sdnn0 == 0 and rmssd0 == 0:
                        err += 1

                    total_heart_rate.append(heart_rate0)
                    total_afi.append(afi0)
                    total_hf.append(hf0)
                    total_lf.append(lf0)
                afi1 = np.mean(total_afi)
                heart_rate1 = np.mean(total_heart_rate)
                lf1 = np.mean(total_lf)
                hf1 = np.mean(total_hf)
                afi.append(afi1)
                heart_rate.append(heart_rate1)
                lf.append(lf1)
                hf.append(hf1)'''
                rpeaks3, heart_mean3 = get_hr(i[0])  # 一导联
                heart_rate3 = heart_mean3
                nni3 = get_nni(rpeaks3)
                sdnn_3, rmssd_3, nn20_3, pnn20_3, nn50_3, pnn50_3 = get_hrv(nni3)
                lf_3, hf_3 = get_HLF(nni3)
                mean_nn3 = np.mean(nni3)
                afi3 = sdnn_3 / mean_nn3
                if sdnn_3 == 0 and rmssd_3 == 0:
                    err += 1
                # rpeaks0, heart_mean0 = get_hr(i[1]) # 二导联
                # heart_rate0 = heart_mean0
                # nni0 = get_nni(rpeaks0)
                # sdnn_1, rmssd_1, nn20_1, pnn20_1, nn50_1, pnn50_1 = get_hrv(nni0)
                # lf_1, hf_1 = get_HLF(nni0)
                # mean_nn1 = np.mean(nni0)
                # afi1 = sdnn_1 / mean_nn1
                # if sdnn_1 == 0 and rmssd_1 == 0:
                #     err += 1
                # rpeaks1, heart_mean1 = get_hr(i[7])
                # heart_rate1 = heart_mean1
                # nni1 = get_nni(rpeaks1)
                # sdnn_2, rmssd_2, nn20_2, pnn20_2, nn50_2, pnn50_2 = get_hrv(nni1)
                # lf_2, hf_2 = get_HLF(nni1)
                # if sdnn_1 == 0 and rmssd_1 == 0:
                #     err += 1
                #
                # mean_nn2 = np.mean(nni1)
                # afi2 = sdnn_2 / mean_nn2

                afi.append(afi3)
                heart_rate.append(heart_rate3)
                lf.append(lf_3)
                hf.append(hf_3)

            else:
                rpeaks, heart_mean = get_hr(i)
                # _, rpeaks = nk.ecg_peaks(i, sampling_rate=100)

                nni = get_nni(rpeaks)
                sdnn_1, rmssd_1, nn20_1, pnn20_1, nn50_1, pnn50_1 = get_hrv(nni)
                lf_3, hf_3 = get_HLF(nni)
                if sdnn_1 == 0 and rmssd_1 == 0:
                    err += 1
                mean_nn = np.mean(nni)
                afi.append(sdnn_1 / mean_nn)
                sdnn.append(sdnn_1)
                rmssd.append(rmssd_1)
                heart_rate.append(heart_mean)
                lf.append(lf_3)
                hf.append(hf_3)
        # except Exception as e:
        #     print(e)
        #     err += 1
        #     sdnn.append(0)
        #     rmssd.append(0)
        #     afi.append(0)
        #     heart_rate.append(0)
        #     lf.append(0)
        #     hf.append(0)
        #     # if len(i.shape)>1 and i.shape[0] > 1:
        #     #     sdnn.append([0,0])
        #     #     rmssd.append([0,0])
        #     #     afi.append(0)
        #     # else:
        #     #     sdnn.append(0)
        #     #     rmssd.append(0)
        #     #     afi.append(0)
    print("无法计算：" + str(err))
    return np.array(afi), np.array(heart_rate), np.array(lf), np.array(hf)
