from sklearn.manifold import TSNE
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE


# from ProcessingData import load_data


def do_hist(scores, true_labels, model=None, dataset=None, display=True, normal_idx=None, seed=None):
    plt.figure()
    plt.style.use('seaborn-darkgrid')  # 'seaborn-bright'

    idx_inliers = (true_labels == 0)
    idx_outliers = (true_labels == 1)
    # hrange1 = (0.0005, 0.02)
    # hrange2 = (0, 0.02)
    # hrange=(min(scores),max(scores))
    # hrange=(0.0015,0.0075)
    hrange = (min(scores), 0.3)
    # plt.hist(scores[idx_inliers], 50, facecolor='black',
    #          label="Normal samples", density=False, range=hrange)
    # plt.hist(scores[idx_outliers], 50, facecolor='silver',
    #          label="Anomalous samples", density=False, range=hrange)
    # plt.hist(scores[idx_inliers], 50, facecolor=(0, 0, 0, 1),
    #          label="Normal samples", density=True, range=hrange)
    # plt.hist(scores[idx_outliers], 50, facecolor=(0.5, 0.5, 0.5, 0.5),
    #          label="Anomalous samples", density=True, range=hrange)
    plt.ylim(0, 900)
    plt.hist(scores[idx_inliers], 50, facecolor=(0, 0.4, 1, 0.5),  # 浅绿色 0, 0.4, 1, 0.5
             label="Normal samples", density=False, range=hrange)
    plt.hist(scores[idx_outliers], 50, facecolor=(1, 0, 0, 0.5),  # 浅红色 1, 0, 0, 0.5
             label="Anomalous samples", density=False, range=hrange)

    plt.tick_params(labelsize=22)
    # ax = plt.gca()
    #
    # for i in ['top', 'right', 'bottom', 'left']:
    #     ax.spines[i].set_visible(False)
    # plt.title("Distribution of the anomaly score")
    # plt.grid()
    plt.rcParams.update({'font.size': 22})
    plt.xlabel('AnomalyScore', fontsize=22)
    plt.ylabel('Count', fontsize=22)
    plt.legend(loc="upper right")

    if display:
        plt.show()
    else:
        plt.savefig('/home/gaoshaochen/Python/AF_Mul/experiments/MlultModal/draw_ECG/plot/{}_{}_{}_{}.svg'.format(model, dataset, normal_idx, seed), transparent=False,
                    bbox_inches='tight')
        # plt.savefig(directory + 'histogram_{}_{}.pdf'.format(random_seed, dataloader),
        #             transparent=False, bbox_inches='tight')
        plt.close()
        print('Plot: ' + str(seed))

# do_hist(data, y_test, directory='./0/', dataloader='alad', random_seed='raw_1', display=False)
