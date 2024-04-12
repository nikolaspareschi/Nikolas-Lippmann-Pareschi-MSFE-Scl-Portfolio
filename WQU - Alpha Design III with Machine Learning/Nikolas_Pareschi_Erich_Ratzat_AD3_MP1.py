# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 22:15:47 2018

@author: nikol
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:39:56 2018

@author: nikol
"""
#


import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
from pandas_datareader import data, wb
import numpy as np
import datetime

# Data

start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2018, 10, 10)
df = data.get_data_yahoo('SPY', start, end)

# ------------------------------



df['Vol'] = df['Adj Close'].rolling(20).std()
df['returns'] = df['Adj Close'].pct_change()
df['Futures_Ret'] = df.Open.shift(-2) - df.Open.shift(-1)
df['Target'] = 0
df.dropna(inplace=True)



df_train = df[df.index <= datetime.datetime(2015,1,1)]
df_test  = df[df.index > datetime.datetime(2015,1,1)]



# ----------------------------
#  K Means - Training
#
X = df_train[['Vol','returns']]
kmeans = KMeans(n_clusters=3).fit(X)
y_kmeans = kmeans.predict(X)
df_train['Target'] = y_kmeans


# --------------------------
#  Plot Training
#
centers = kmeans.cluster_centers_
plt.scatter(df_train['Vol'],df_train['returns'],c=y_kmeans)
plt.scatter(centers[:,0],centers[:,1],c='red',s=100,marker='x')
plt.show()



# -------------------------------
#  K Means - Testing
#
x = df_test[['Vol','returns']]
y_kmeans = kmeans.predict(x)
df_test['Target'] = y_kmeans

# --------------------------------
#  Plot Testing
#
plt.scatter(df_test['Vol'],df_test['returns'],c=y_kmeans)
plt.scatter(centers[:,0],centers[:,1],c='red',s=100,marker='x')
plt.show()



# ------------------------------------------
#  Compare Training and Testing
#
print("Total Points Earned by Cluster Prediction")

print("Cluster 1 Train: %.2f\tCluster 1 Test: %2.f" % (df_train['Futures_Ret'].loc[df_train['Target'] == 0].sum(),df_test['Futures_Ret'].loc[df_test['Target'] == 0].sum()))

print("Cluster 2 Train: %.2f\tCluster 2 Test: %.2f" % (df_train['Futures_Ret'].loc[df_train['Target'] == 1].sum(),df_test['Futures_Ret'].loc[df_test['Target'] == 1].sum()))

print("Cluster 3 Train: %.2f\tCluster 3 Test: %.2f" % (df_train['Futures_Ret'].loc[df_train['Target'] == 2].sum(),df_test['Futures_Ret'].loc[df_test['Target'] == 2].sum()))


# ------------------------------
#  Equity Curves
#
plt.plot(np.cumsum(df_test['Futures_Ret'].loc[df_test['Target'] == 0]),label='Low Volatility Environment')
plt.plot(np.cumsum(df_test['Futures_Ret'].loc[df_test['Target'] == 1]),label='Medium Volatility Environment')
plt.plot(np.cumsum(df_test['Futures_Ret'].loc[df_test['Target'] == 2]),label='High Volatility Environment')
plt.legend()
plt.show()
