from sklearn.manifold import TSNE
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import seaborn as sns

#rom ProcessingData import load_data
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

list_data=["Shuttle"]
# for dataloader in list_data:
#     _,_,y_test=load_data(dataloader)

#用TSNE进行数据降维并展示聚类结果
#plt.title('MOSGAN')
tsne = TSNE()

import matplotlib.pyplot as plt

def do_tsne(latent, true_labels, model=None, dataset=None, display=True, classes=None, seed=None):


    data=tsne.fit_transform(latent) #进行数据降维,并返回结果

    #不同类别用不同颜色和样式绘图
    colors1 = '#000080' #点的颜色  # 蓝色
    colors2 = '#FF0000'

    idx_normal = (true_labels == 0)
    idx_abnormal = (true_labels == 1)

    plt.scatter(data[idx_normal, 0], data[idx_normal, 1],s=8 ,c=colors1,alpha=0.6)
    plt.scatter(data[idx_abnormal, 0], data[idx_abnormal, 1],s=8 ,c=colors2,alpha=0.6)
    #plt.scatter(data[:, 0], data[:, 1], c=true_labels)


    #d = tsne[k.r[u'聚类类别'] == 2]
    #plt.plot(d[0], d[1], 'b*')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.axis('off')
    plt.tight_layout()
    plt.style.use('ggplot')
    if display:
       plt.show()
    else:
        plt.savefig( './Plot_Tsne/{}_{}_{}_{}.svg'.format(model,dataset, classes, seed),transparent=False, bbox_inches='tight',dpi=600)
        # plt.savefig(directory + 'histogram_{}_{}.pdf'.format(random_seed, dataloader),
        #             transparent=False, bbox_inches='tight')
        plt.close()
    plt.show()


def do_tsne_sns(latent, true_labels, model=None, dataset=None, display=True, classes=None, seed=None):


    pos = TSNE(n_components=2).fit_transform(latent)
    df = pd.DataFrame()
    df['x'] = pos[:, 0]
    df['y'] = pos[:, 1]
    #df['z'] = pos[:, 2]
    legends = list(range(10000))
    df['class'] = [legends[i] for i in true_labels]

    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")

    sns.lmplot(data=df,  # Data source
               x='x',  # Horizontal axis
               y='y',  # Vertical axis
               fit_reg=False,  # Don't fix a regression line
               hue="class",  # Set color,
               legend=False,
               scatter_kws={"s": 25, 'alpha': 0.8})  # S marker size

       # ax.set_zlabel('pca-three')
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)

    plt.xticks([])
    plt.yticks([])

    plt.xlabel('')
    plt.ylabel('')
    plt.axis('off')
    plt.tight_layout()
    if display:
        plt.show()
    else:
        plt.savefig('/home/gaoshaochen/Python/AF_Mul/experiments/MlultModal/draw_ECG/Plot_Tsne/{}_{}_{}_{}.svg'.format(model, dataset, classes, seed), transparent=False,bbox_inches='tight',dpi=600)
        plt.close()
def tsne_3D(latent,label):
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # 实例化 t-SNE 对象并将数据拟合到 3 维空间
    tsne = TSNE(n_components=3, random_state=0)
    X_3d = tsne.fit_transform(latent)

    # 将点按标签分组
    group1 = X_3d[label == 0]
    group2 = X_3d[label == 1]

    # 绘制 3D 散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(group1[:, 0], group1[:, 1], group1[:, 2], c='blue', label='Group 1')
    ax.scatter(group2[:, 0], group2[:, 1], group2[:, 2], c='red', label='Group 2')
    ax.legend()
    plt.show()

