import matplotlib

matplotlib.use('Agg')

import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from itertools import product
import numpy as np
import os


def create_logdir(model, dataset, classes):
    """ Directory to save training logs, weights, biases, etc."""
    return "../embeddings/{}/{}/{}".format(model, dataset, classes)


model = 'Dominant'  # 'OCGCN'#'deepwork'
# ['kdd','arrhythmia', 'satellite']
dataset = 'cora'
EPOCHS = [2000]

C = ['0', '1', '2', '3', '4', '5', '6']
# EPOCHS=np.arange(200,300,20)
PLOT_2D = True
for classes in C:
    save_dir = create_logdir(model, dataset, classes)

    vis_dir = os.path.sep.join(['../figure', 'embedding', '{}'.format(model), '{}'.format(dataset)])

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    for t in [10000]:

        for epoch in EPOCHS:
            fig_name = 'embeddings_class'
            if model in ['deepwork']:
                embeddingsA = np.load(os.path.sep.join([save_dir, 'embeddings.npy'.format(epoch)]))[:t]
                embeddingsS = np.load(os.path.sep.join([save_dir, 'embeddings.npy'.format(epoch)]))[:t]
                labels = np.load(os.path.sep.join([save_dir, 'label.npy'.format(epoch)]))[:t]
            elif model in ['OCGCN', 'ARGA', 'ARANE', 'mulGCN', 'GAE', 'Dominant']:
                embeddingsA = np.load(os.path.sep.join([save_dir, 'embeddings_max.npy'.format(epoch)]))[:t]
                embeddingsS = np.load(os.path.sep.join([save_dir, 'embeddings_max.npy'.format(epoch)]))[:t]
                labels = np.load(os.path.sep.join([save_dir, 'label_max.npy'.format(epoch)]))[:t]
            else:
                embeddingsA = np.load(os.path.sep.join([save_dir, 'embeddingsA_max.npy'.format(epoch)]))[:t]
                embeddingsS = np.load(os.path.sep.join([save_dir, 'embeddingsS_max.npy'.format(epoch)]))[:t]
                labels = np.load(os.path.sep.join([save_dir, 'label_max.npy'.format(epoch)]))[:t]
            labels = np.array(labels, np.int)
            for i in range(2):
                if i == 0:
                    embedding = embeddingsA
                else:
                    embedding = embeddingsS
                if len(embedding.shape) != 2:
                    continue

                pos = TSNE(n_components=2).fit_transform(embedding)
                df = pd.DataFrame()
                df['x'] = pos[:, 0]
                df['y'] = pos[:, 1]
                # df['z'] = pos[:, 2]
                legends = list(range(10000))
                df['class'] = [legends[l] for l in labels]
                if PLOT_2D:
                    sns.set_context("notebook", font_scale=1.5)
                    sns.set_style("ticks")

                    # Create scatterplot of dataframe
                    sns.lmplot('x',  # Horizontal axis
                               'y',  # Vertical axis
                               data=df,  # Data source
                               fit_reg=False,  # Don't fix a regression line
                               hue="class",  # Set color,
                               legend=False,
                               scatter_kws={"s": 25, 'alpha': 0.9})  # S marker size

                    # sns.despine(top=True, left=True, right=True, bottom=True)
                else:

                    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
                    ax.scatter(
                        xs=df["x"],
                        ys=df["y"],
                        # zs=df["z"],
                        c=df["class"],
                        cmap='tab10'
                    )
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                # ax.set_zlabel('pca-three')
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)

                plt.xlabel('')
                plt.ylabel('')
                plt.tight_layout()
                plt.savefig(vis_dir + '/' + fig_name + '{}_{}.png'.format(classes, i), bbox_inches='tight')
                plt.savefig(vis_dir + '/' + fig_name + '{}_{}.svg'.format(classes, i), bbox_inches='tight', dpi=600)
                print(vis_dir + '/' + fig_name + '{}_{}.png'.format(classes, i))
