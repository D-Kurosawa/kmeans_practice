"""
重み付きK-Meansによる教師なし学習

https://medium.com/@dey.mallika/unsupervised-learning-with-weighted-k-means
-3828b708d75d
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skewnorm
from sklearn.datasets import make_blobs

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
print(df)

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
