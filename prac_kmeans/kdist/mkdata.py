import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from prac_kmeans.mypkg import PandasDisplay

PandasDisplay.custom()
sns.set(style='ticks', color_codes=True)


def read_data(show=False):
    file = '../../_data/sports_dataMidSc.txt'
    df = pd.read_csv(file, sep='\t')

    if show:
        show_dataframe(df)

    df.drop(columns='Student', inplace=True)
    print(df)
    return df


def show_dataframe(df):
    print()
    print(df.info())
    print()
    print(df.describe())

    sns.pairplot(df.loc[:, '50mRun':'stepping'])
    plt.show()


def standards(df):
    # 標準化
    sc = StandardScaler()
    df_std = sc.fit_transform(df)

    # 主成分分析を行う
    pca = PCA(svd_solver='full')
    pca.fit(df_std)
    df_pca = pd.DataFrame(pca.transform(df_std))

    print(df_pca)

    cumulative_contribution_ratio = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(10, 5))
    plt.plot(list(range(1, len(cumulative_contribution_ratio) + 1)),
             cumulative_contribution_ratio, "-o")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xlabel("components")
    plt.ylabel("cumulative contribution ratio")
    plt.show()


if __name__ == '__main__':
    def _main():
        df = read_data()
        standards(df)


    _main()