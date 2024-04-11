# -*- coding: utf-8 -*-
"""
Created on Sat Feb 03 20:44:24 2018

@author: Nikolas
"""

import pandas_datareader.data as web
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
from datetime import datetime as dt
from scipy.stats import norm
import numpy as np
import pandas as pd
from scipy import optimize
from pylab import *
import statsmodels.api as sm
from scipy.stats import norm

'''
Value-at-Risk
An investment fund has 100,000 Apple shares.

1.       Quantify the Maximum Expected Loss for the next day using a Value-at-Risk (VaR) model.

Confidence level: 95%
Volatility: 2.5%
Current stock price: $126
Implement VaR in Python and Excel. 

'''
aapl_price = 126
shares = 100000
ci = 0.95
vol = 0.025


# Computing the Z-score to calculate the value atr risk with a 5% drawdown chance

z = norm.ppf(ci)


value_at_risk = shares*z*vol*aapl_price


print ("\n The Maximum expected loss for the next day is", round(value_at_risk,2))


'''
2.       Quantify the Maximum Expected Loss for the next day using a Value-at-Risk (VaR) model.

Confidence level: 95%
Volatility: Forecasted
Forecast the volatility using a GARCH(1,1) programmed in Python.
Stock price: closing price from Google Finance, Yahoo Finance, Quandl, CityFALCON, or another similar source

'''

# Defining the period to download the data

start_date = dt(2015,1,1)
end_date = dt(2018,02,01)

# Downloading the daya for SP500

data = web.DataReader('spy', data_source="yahoo", start=start_date, end=end_date)['Adj Close']

# Computing the returns

returns = data.pct_change().dropna()#(fill_method='pad')
plot(data)
data2 =  returns

def GARCH11_logL(param, data2):
    omega, alpha, beta = param
    n = len(data2)
    s = np.ones(n)
    for i in range(3, n):
        s[i] = omega + alpha*data2[i-1]**2 +beta*(s[i-1])
    logL = -((-np.log(s) - data2**2/s).sum())
    return logL

R = optimize.fmin(GARCH11_logL, np.array([.1,.1,.1]), args = (data2,), full_output = 1)

print("omega = %.6f\nbeta = %.6f\nalpha = %.6f\n") % (R[0][0], R[0][2], R[0][1])


volatility2 = np.zeros(len(data2))

for i in range(2,len(data2)):

    volatility2[i] = R[0][0]+R[0][1]*data2[i-1]**2+R[0][2]*(volatility2[i-1])
    print volatility2[i]
    print ("volatility forecast for day %i is %.6f") % (i, R[0][0]+R[0][1]*data2[i-1]**2+R[0][2]*(data2[i-1]))
    print R[0][0]+R[0][1]*data2[i-1]**2+R[0][2]*(data2[i-1])

   

Var = data[-1]*volatility2[-1]*z
print('The Maximum Expected Loss for the next day using a Value-at-Risk (VaR) model will be its return*stockprice*z*forecasted volatility which is %.6f') %Var


plt.title('Volatility FORESCAST USING GARCH (1,1)') 
plt.plot(volatility2)
plt.subplot(1,2,1) # This is crazy, the graph only works with this
plt.plot(volatility2)

plt.show()