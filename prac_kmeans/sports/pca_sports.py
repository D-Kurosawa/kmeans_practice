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

    contribution_rate(pca)
    factor_loading(pca, df)
    show_biplot(pca, df_pca, df)


def contribution_rate(pca):
    """寄与率"""
    cumulative_contribution_ratio = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(10, 5))
    plt.plot(list(range(1, len(cumulative_contribution_ratio) + 1)),
             cumulative_contribution_ratio, "-o")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xlabel("components")
    plt.ylabel("cumulative contribution ratio")
    plt.show()


def factor_loading(pca, df):
    """因子負荷量（各主成分に対する相関係数）"""
    print('--- pc1 ---')
    print(pd.DataFrame(
        data={'factor_loading': pca.components_[0]},
        index=df.columns).sort_values(by='factor_loading', ascending=False))

    print('--- pc2 ---')
    print(pd.DataFrame(
        data={'factor_loading': pca.components_[1]},
        index=df.columns).sort_values(by='factor_loading', ascending=False))


def show_biplot(pca, df_pca, df):
    # グラフを表示する
    plt.figure(figsize=(10, 10))

    # 因子スコアのプロット
    plt.scatter(df_pca.iloc[:, 0], df_pca.iloc[:, 1])

    # 因子負荷量のプロット
    # 第一主成分と第2主成分の因子負荷量を取得
    pc0 = pca.components_[0]
    pc1 = pca.components_[1]
    arrow_mul = 4
    text_mul = 1.2
    for i in range(len(pc0)):
        plt.arrow(0, 0, pc0[i] * arrow_mul, pc1[i] * arrow_mul, color='r')
        plt.text(pc0[i] * arrow_mul * text_mul, pc1[i] * arrow_mul * text_mul,
                 df.columns[i], color='r', fontsize=14)
    plt.title('biplot')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    def _main():
        df = read_data()
        standards(df)


    _main()
