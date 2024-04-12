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


# Let´s split the data to train and then test the SVM algorithm

t=.8
split = int(t*len(Df))

# We will use the RSI on SPY and TLT as predictors

n=5
Df['RSI']=ta.RSI(np.array(Df['Close'].shift(1)), timeperiod=5)
Df['RSI_TLT']=ta.RSI(np.array(Df['Close_TLT'].shift(1)), timeperiod=5)

# We will use also the magnitude of absolute change in 5 days in these 2 instruments
# Legendary trader Victor Niederhoffer points out the aboluste points captures more
# wealth effects than percentual points.


Df['OC_SPY']= Df['Close'].shift(1)-Df['Open'].shift(6)
Df['OC_TLT']= Df['Close_TLT'].shift(1)-Df['Open_TLT'].shift(6)

# We need to compute the future returns, which are what we want to predict


Df['Ret']=np.log(Df['Open'].shift(-1)/Df['Open'])

# Dealing with the Nan values

Df= Df.fillna(method = 'backfill')
Df= Df.fillna(method = 'ffill')

# The objective here is to be long only in the 50% better days. We train on that
# and the plan is to be invested in the test data on the better days, cutting
# at least the drawdowns. 


Df['Signal']=0
Df.loc[Df['Ret']>Df['Ret'][:split].quantile(q=0.66),'Signal']=1
Df.loc[Df['Ret']<Df['Ret'][:split].quantile(q=0.34),'Signal']=-1


X=Df.drop(['Close','Signal','High','Low','Volume','Ret'],axis=1)
X=Df[['RSI','RSI_TLT','OC_SPY','OC_TLT']]
#X=Df[['RSI','RSI_TLT','OO_SPY','OO_TLT']]
#X=Df[['OC_SPY','OC_TLT']]
X = X.fillna(method = 'backfill')
y=Df['Signal']

# We define the steps that will be followed by the pipeline function. We need
# first to standardize our data so outliers and asymmetrical points will affect
# less the algorith

steps = [
         		('scaler',StandardScaler()),
        		('svc',SVC())
              ]
pipeline =Pipeline(steps)

# To improve the result we will perform also a Randomized Grid Search in the
# hyperparamet space

c =[10,100,1000,10000]
g= [1e-2,1e-1,1e0]


parameters = {
              		'svc__C':c,
              		'svc__gamma':g,
              		'svc__kernel': ['rbf']
             	           }


rcv = RandomizedSearchCV(pipeline, parameters,cv=7)

# After founding the optimal hyperparameter we fit the train data in the SVC
# algorithm

rcv.fit(X.iloc[:split],y.iloc[:split])
best_C = rcv.best_params_['svc__C']
best_kernel =rcv.best_params_['svc__kernel']
best_gamma=rcv.best_params_['svc__gamma']
cls = SVC(C =best_C,kernel=best_kernel, gamma=best_gamma)
ss1= StandardScaler()
cls.fit(ss1.fit_transform(X.iloc[:split]),y.iloc[:split])

# After the fit we make our predictions

y_predict =cls.predict(ss1.transform(X.iloc[split:]))

Df['Pred_Signal']=0


Df.iloc[:split,Df.columns.get_loc('Pred_Signal')]\
       =pd.Series(cls.predict(ss1.transform(X.iloc[:split])).tolist())
Df.iloc[split:,Df.columns.get_loc('Pred_Signal')]=y_predict

# We determine the returns under the predicted signals

Df['Ret1']=Df['Ret']*Df['Pred_Signal'] 
Df['Cu_Ret1']=0
Df['Cu_Ret1']=np.cumsum(Df['Ret1'].iloc[split:])
Df['Cu_Ret']=0
Df['Cu_Ret']=np.cumsum(Df['Ret'].iloc[split:])


Std =np.std(Df['Cu_Ret1'])
Sharpe = (Df['Cu_Ret1'].iloc[-1]-Df['Cu_Ret'].iloc[-1])/Std
print'Sharpe Ratio:',Sharpe

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
print "Current size:", fig_size
 
# Set figure width to 12 and height to 9
fig_size[0] = 9
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size

plt.plot(Df['Cu_Ret1'] ,color='r',label='Strategy Returns')
plt.plot(Df['Cu_Ret'],color='g',label='Market Returns')
#plt.figure(figsize=(4,3))
plt.figtext(0.14,0.7,s='Sharpe ratio: %.2f'%Sharpe)
plt.legend(loc='best')
plt.show()

#Df['Cu_Ret1'].plot()
