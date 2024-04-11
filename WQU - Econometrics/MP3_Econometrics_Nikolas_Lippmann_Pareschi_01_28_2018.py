# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 16:41:54 2018

@author: Nikolas
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


# First we need to read the data. I saved the xls file in: C:\Users\Nikolas
# The user need to have the file in his/her Python working directory

houses = pd.read_csv('CSUSHPINSA.csv', parse_dates=['DATE'], index_col='DATE')

#convert to time series:

time_series = houses['CSUSHPINSA']

# LEt's plot the data 

plt.plot(time_series)
plt.show

# Checking stationarity with Adfuller

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    
    
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

test_stationarity(time_series)

# Making TS stationary - Trend elimination using log

time_series_log = np.log(time_series)
plt.plot(time_series_log)
plt.show
test_stationarity(time_series_log)
#Eliminating Seasonality by differenciation


time_series_log_diff = time_series_log - time_series_log.shift()
time_series_log_diff.dropna(inplace=True)


test_stationarity(time_series_log_diff)

print test_stationarity(time_series_log_diff)

print "\n This is not good yet. The mean is clearly not stationary and we don't have a p-value lower than 0.05 \n"

# Lets take the difference one more time


time_series_log_diff_diff = time_series_log_diff - time_series_log_diff.shift()
time_series_log_diff_diff.dropna(inplace=True)

print test_stationarity(time_series_log_diff_diff)
print "\n We have a p-value lower than 0.01. Our mean seems stationary. The standard deviation is not greatly stationary, but considering subprime it is ok \n"


'''
Implementation of ARIMA(p,d,q) model using Box-Jenkins methodology. 
 
'''



pyplot.figure()
pyplot.subplot(211)
plot_acf(time_series_log_diff_diff, ax=pyplot.gca(), lags=50)
pyplot.subplot(212)
plot_pacf(time_series_log_diff_diff, ax=pyplot.gca(), lags=50)
pyplot.show()

# Accordingly to the picures we believe an ARIMA of (1,2,0) is a good choice


model = ARIMA(time_series_log_diff_diff, order=(1, 2, 0))  
results_AR = model.fit(disp=0)  
plt.plot(time_series_log_diff_diff)
plt.plot(results_AR.fittedvalues, color='red')


print results_AR.summary()


# 3 months predictions

X3 = time_series.values
size = int(len(X3) * 0.995)
train, test = X3[0:size], X3[size:len(X3)]
history = [x for x in train]
predictions = list()


print "\n Let`s predict 3 months of house prices: \n"

for t in range(len(test)):
	model = ARIMA(history, order=(1,2,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)


# 2 months predictions



X2 = time_series.values
size = int(len(X2) * 0.997)
train, test = X2[0:size], X2[size:len(X2)]
history = [x for x in train]
predictions = list()

print "\n Let`s predict 2 months of house prices: \n"

for t in range(len(test)):
	model = ARIMA(history, order=(1,2,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)


# 1 month prediction

X1 = time_series.values
size = int(len(X1) * 0.999)
train, test = X1[0:size], X1[size:len(X1)]
history = [x for x in train]
predictions = list()

print "\n Let`s predict 1 month of house prices: \n"

for t in range(len(test)):
	model = ARIMA(history, order=(1,2,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)




