import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from prac_kmeans.mypkg import PandasDisplay

PandasDisplay.custom()
sns.set(style='ticks', color_codes=True)


class SportsData:
    def __init__(self, file=None):
        self._df = pd.DataFrame()
        self._df_std = pd.DataFrame()
        self._file = file
        if file is None:
            self._file = '../../_data/sports_dataMidSc.txt'

    @property
    def df(self):
        return self._df

    @property
    def df_std(self):
        return self._df_std

    def load(self):
        self._load_dataframe()
        self._to_standard()

    def _load_dataframe(self):
        self._df = pd.read_csv(self._file, sep='\t', index_col='Student')

    def _to_standard(self):
        sc = StandardScaler()
        self._df_std = pd.DataFrame(data=sc.fit_transform(self._df),
                                    index=self._df.index,
                                    columns=self._df.columns)

    def show(self, detail=False):
        print(f"\n>> {'-' * 30} DataFrame {'-' * 30}")
        self._dataframe()

        print(f"\n>> {'-' * 30} Standard DataFrame {'-' * 30}")
        self._std_dataframe()

        if not detail:
            return

        print(f"\n>> {'-' * 30} DataFrame Info {'-' * 30}")
        self._info()

        print(f"\n>> {'-' * 30} DataFrame Describe {'-' * 30}")
        self._describe()

        self._pair_plot()

    def _dataframe(self):
        print(self._df)

    def _std_dataframe(self):
        print(self._df_std)

    def _info(self):
        print(self._df.info())

    def _describe(self):
        print(self._df.describe())

    def _pair_plot(self):
        sns.pairplot(self._df)
        plt.show()


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
        spt = SportsData()
        spt.load()
        spt.show()


    _main()
