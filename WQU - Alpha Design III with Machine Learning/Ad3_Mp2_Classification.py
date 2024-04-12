# -*- coding: utf-8 -*-
"""
Created on Sat Dec 08 15:02:44 2018

@author: nikol
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 05 21:28:44 2018

@author: nikol
"""



import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
from pandas_datareader import data, wb
import numpy as np
import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import talib as ta

# For this Project we will use 2 dataseries. SPY etf daily prices and TLT etf
# daily prices. We will use SVM and Decission Tree classifier to study the
# intermarket relationships between SP500 and treasuries notes with long duration.

start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2018, 10, 10)
df = data.get_data_yahoo('SPY', start, end)
df=df[['Open','High','Low','Close','Volume']]
df2 = data.get_data_yahoo('TLT', start, end)

df2[['Open_TLT','High_TLT','Low_TLT','Close_TLT','Volume_TLT']] = df2[['Open','High','Low','Close','Volume']]
df2=df2[['Open_TLT','High_TLT','Low_TLT','Close_TLT','Volume_TLT']]

# After Collecting data for both instruments we concatened them in just one dataframe

df3 = pd.concat([df, df2], axis=1, join_axes=[df.index])
Df=df3

# Future Returns and Target Variable

Df['Return'] = Df['Close'].pct_change(1).shift(-1)
Df['target'] = np.where(Df.Return > 0, 1, -1)
#Df['target'] = np.where(Df.Return < -0.01, -1, Df['target'])


# We will use the RSI on SPY and TLT as predictors

n=5
Df['RSI']=ta.RSI(np.array(Df['Close'].shift(1)), timeperiod=5)
Df['RSI_TLT']=ta.RSI(np.array(Df['Close_TLT'].shift(1)), timeperiod=5)

# We will use also the magnitude of absolute change in 5 days in these 2 instruments
# Legendary trader Victor Niederhoffer points out the aboluste points captures more
# wealth effects than percentual points.


Df['OC_SPY']= Df['Close'].shift(1)-Df['Open'].shift(6)
Df['OC_TLT']= Df['Close_TLT'].shift(1)-Df['Open_TLT'].shift(6)

Df.tail()
Df.head()

Df=Df.ffill()
Df=Df.bfill()
Df = Df.dropna()

predictors_list = Df[['RSI_TLT','RSI','OC_SPY','OC_TLT']]
X = predictors_list
X.tail()

y = Df.target
y.tail()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_test=X_test.ffill()
X_test=X_test.bfill()


print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=5, class_weight = 'balanced') 
clf

X_train = X_train.ffill()
X_train = X_train.bfill()
clf = clf.fit(X_train, y_train)


from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None,filled=True,feature_names=predictors_list.columns)   
graphviz.Source(dot_data) 


y_pred = clf.predict(X_test)


# Run the code to view the classification report metrics

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)

