"""
重み付きK-Meansによる教師なし学習

https://medium.com/@dey.mallika/unsupervised-learning-with-weighted-k-means
-3828b708d75d
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skewnorm
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from prac_kmeans.mypkg import PandasDisplay

sns.set()  # for plot styling
PandasDisplay.custom()

"""
STEP1
    すべてのライブラリをインポートし、演習用のランダムサンプルを生成します
    この場合、2つのカテゴリで1年間の支出シェアのBLOBを作成し、
    顧客の年間支出全体の歪んだ分布を作成しています
"""
X0, Y0 = make_blobs(n_samples=5000, centers=4, n_features=2, random_state=25)
df = pd.DataFrame(dict(Pct_Spend_in_Organic=(X0[:, 0]) + 10,
                       Pct_Spend_in_Local=(X0[:, 1]) + 10))
df['Total_Spend'] = (skewnorm.rvs(1000, size=5000) * 5000) + 100
ax = df.reset_index().plot(x='index', y="Total_Spend", kind="hist")

plt.show()

print()
print(df.head(10))
print(df.shape)
print()

"""
STEP2
    散布図で入力データを視覚化する
"""
plt.style.use('default')
x = np.array(df['Pct_Spend_in_Local'])
y = np.array(df['Pct_Spend_in_Organic'])

plt.figure(figsize=(15, 10))
plt.scatter(x, y, s=5, cmap='viridis', c='orange',
            label='Spend in Organic Products')

plt.title('Pct Spend in Local vs Organic Products', fontsize=18,
          fontweight='bold')
plt.xlabel('Pct Spend in Local', fontsize=15)
plt.ylabel('Pct Spend in Organic', fontsize=15)

plt.show()

"""
STEP3
    - 最大1000回の反復でKmeans定義する
    - 入力変数で配列「X」を定義します
    - 列 'Total_Spend'を観測重みとして配列 'Y'を定義します
"""
kmeans = KMeans(n_clusters=5, random_state=0, max_iter=1000)
X = np.array(df.drop(['Total_Spend'], axis=1).astype(float))
Y = np.array(df['Total_Spend'].astype(float))

print(X)
print(Y)

"""
STEP4
    加重k-meansクラスタリングを実行し、入力として「X」配列を、サンプルの重みとして
    「Y」配列を入力します
    すべてのデータポイントのクラスターレベルを生成する
"""
wt_kmeansclus = kmeans.fit(X, sample_weight=Y)
predicted_kmeans = kmeans.predict(X, sample_weight=Y)

"""
STEP5
    散布図でクラスターと重心を視覚化する
"""
plt.style.use('default')
plt.figure(figsize=(15, 10))
plt.scatter(X[:, 0], X[:, 1], c=wt_kmeansclus.labels_.astype(float), s=10,
            cmap='tab20b', marker='x')
plt.title('Customer Spend Local vs Organic - Weighted K-Means', fontsize=18,
          fontweight='bold')
plt.xlabel('Pct_Spend_in_Local', fontsize=15)
plt.ylabel('Pct_Spend_in_Organic', fontsize=15)

centers = wt_kmeansclus.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=500, alpha=0.5)

plt.show()

"""
STEP6
    データフレームでクラスターラベルと重心を結合する
"""
df['ClusterID_wt'] = predicted_kmeans
centersdf = pd.DataFrame(centers)
centersdf['ClusterID_wt'] = centersdf.index
centersdf = centersdf.rename(
    columns={0: 'Wt Centroid: Spend in Local',
             1: 'Wt Centroid: Spend in Organic'})
df = df.merge(centersdf, on='ClusterID_wt', how='left')

print()
print(df.head(10))
print(df.shape)
print()

"""
比較
    観測値の重み付けなしのK-Meansクラスタリング
"""
kmeans = KMeans(n_clusters=5, random_state=0, max_iter=1000)
kmeansclus_nw = kmeans.fit(X)
predicted_kmeans_nw = kmeans.predict(X)

centers_nw = kmeansclus_nw.cluster_centers_

df['ClusterID_unwt'] = predicted_kmeans_nw
centersdf_nw = pd.DataFrame(centers_nw)
centersdf_nw['ClusterID_unwt'] = centersdf_nw.index
centersdf_nw = centersdf_nw.rename(
    columns={0: 'Unwt Centroid: Spend in Local',
             1: 'Unwt Centroid: Spend in Organic'})
df_nw = df.merge(centersdf_nw, on='ClusterID_unwt', how='left')

print()
print(df_nw.head(10))
print(df_nw.shape)
print()

"""
比較
    重み付けされていないクラスターの散布図の生成
"""
plt.figure(figsize=(15, 10))
plt.scatter(X[:, 0], X[:, 1], c=predicted_kmeans_nw, s=10, cmap='tab20',
            marker='x')
plt.scatter(centers_nw[:, 0], centers_nw[:, 1], c='black', s=500, alpha=0.5)
plt.title('Customer Spend - Local vs Organic - Unweighted K-Means',
          fontsize=18, fontweight='bold')

plt.xlabel('Spend_in_Local', fontsize=15)
plt.ylabel('Spend_in_Organic', fontsize=15)

plt.show()
