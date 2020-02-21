import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from prac_kmeans.mypkg import PandasDisplay

PandasDisplay.custom()
sns.set(style='ticks', color_codes=True)


class SportsData:
    """
    :type _df: pd.DataFrame
    :type _df_std: pd.DataFrame
    """

    def __init__(self, file=None):
        """
        :type file: str | None
        """
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
        self._df_std.to_csv('std1')

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


class PCAnalysis:

    def __init__(self, sport):
        """
        :type sport: SportsData
        """
        self._df = sport.df
        self._df_std = sport.df_std
        self._pca = PCA()
        self._df_pca = pd.DataFrame()

    @property
    def df(self):
        return self._df

    @property
    def df_std(self):
        return self._df_std

    @property
    def df_pca(self):
        return self._df_pca

    def run(self):
        self._pc_analysis()
        self._contribution_rate()
        self._factor_loading()
        self._factor_score()

    def _pc_analysis(self):
        self._pca = PCA(svd_solver='full')
        self._df_pca = pd.DataFrame(self._pca.fit_transform(self._df_std))
        self._df_pca.to_csv('pca')

    def _contribution_rate(self):
        """寄与率"""
        cumulative_contribution_ratio = np.cumsum(
            self._pca.explained_variance_ratio_)

        plt.figure(figsize=(10, 5))
        plt.plot(list(range(1, len(cumulative_contribution_ratio) + 1)),
                 cumulative_contribution_ratio, "-o")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.xlabel("components")
        plt.ylabel("cumulative contribution ratio")
        plt.show()

    def _factor_loading(self):
        """因子負荷量（各主成分に対する相関係数）"""
        print('--- pc1 ---')
        print(pd.DataFrame(
            data={'factor_loading': self._pca.components_[0]},
            index=self._df.columns
        ).sort_values(by='factor_loading', ascending=False))

        print('--- pc2 ---')
        print(pd.DataFrame(
            data={'factor_loading': self._pca.components_[1]},
            index=self._df.columns
        ).sort_values(by='factor_loading', ascending=False))

    def _factor_score(self):
        # グラフを表示する
        plt.figure(figsize=(10, 10))

        # 因子スコアのプロット
        plt.scatter(self._df_pca.iloc[:, 0], self._df_pca.iloc[:, 1])

        # 因子負荷量のプロット
        # 第一主成分と第2主成分の因子負荷量を取得
        pc0 = self._pca.components_[0]
        pc1 = self._pca.components_[1]

        arrow_mul = 4
        text_mul = 1.2
        for i in range(len(pc0)):
            plt.arrow(0, 0, pc0[i] * arrow_mul, pc1[i] * arrow_mul, color='r')
            plt.text(pc0[i] * arrow_mul * text_mul,
                     pc1[i] * arrow_mul * text_mul,
                     self._df.columns[i], color='r', fontsize=14)

        plt.title('biplot')
        plt.xlabel('pc1')
        plt.ylabel('pc2')
        plt.grid(True)
        plt.show()


class KmeansAnalysis:
    def __init__(self, pca):
        """
        :type pca: PCAnalysis
        """
        self._df = pca.df
        self._df_std = pca.df_std
        self._df_pca = pca.df_pca

    def elbow_method(self, df=None):
        """
        :type df: pd.DataFrame | None
        """
        if df is None:
            df = self._df_pca

        sse = self._get_sse(df)

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 20), sse, marker='o')
        plt.grid(True)
        plt.xlabel('Number of clusters')
        plt.ylabel('SSE')
        plt.show()

    @staticmethod
    def _get_sse(df):
        sse = []

        for n in range(1, 20):
            # KMeansクラスからkmインスタンスを作成
            km = KMeans(
                n_clusters=n,  # クラスターの個数
                init="k-means++",  # セントロイドの初期値
                n_init=10,  # 異なるセントロイドの初期値を用いたk-meansの実行回数
                max_iter=300,  # k-meansアルゴリズムを繰り返す最大回数
                random_state=0  # 乱数発生初期化
            )
            km.fit(df)
            sse.append(km.inertia_)

        return sse


if __name__ == '__main__':
    def _main():
        spt = SportsData()
        spt.load()
        spt.show()

        pca = PCAnalysis(spt)
        pca.run()

        km = KmeansAnalysis(pca)
        km.elbow_method()


    _main()
