"""
重み付きK-Meansによる教師なし学習

https://medium.com/@dey.mallika/unsupervised-learning-with-weighted-k-means
-3828b708d75d
"""

import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.stats import skewnorm
from sklearn.datasets import make_blobs

"""
STEP1
    すべてのライブラリをインポートし、演習用のランダムサンプルを生成します
    この場合、2つのカテゴリで1年間の支出シェアのBLOBを作成し、
    顧客の年間支出全体の歪んだ分布を作成しています
"""
X0, Y0 = make_blobs(n_samples=5000, centers=4, n_features=2, random_state=25)
df = DataFrame(dict(Pct_Spend_in_Organic=(X0[:, 0]) + 10,
                    Pct_Spend_in_Local=(X0[:, 1]) + 10))
df['Total_Spend'] = (skewnorm.rvs(1000, size=5000) * 5000) + 100
ax = df.reset_index().plot(x='index', y="Total_Spend", kind="hist")

plt.show()
print(df)
