# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 02:01:45 2018

@author: Nikolas
"""

import pandas as pd
from pandas_datareader import data as web
import numpy as np
import datetime
import matplotlib.pyplot as plt



# Global variables declaration

######## Question 1 ##########


index = "^GSPC"

start = datetime.datetime(2007, 2, 01)
end = datetime.datetime(2018, 2, 28)


def download_data(index):

    sp500 = web.DataReader(index, 'yahoo', start, end)
    
    return sp500

####### Question 2 #################
    
sp500_data = download_data(index)
plt.title("SP500 price plot", fontsize=20)
sp500_data['Adj Close'].plot(grid = True)
plt.show()


'''Daily Returns '''

sp500_daily_return = sp500_data['Adj Close'].pct_change().dropna()

plt.title("Daily SP500 returns", fontsize=20)
sp500_daily_return.plot(legend=True, subplots=False,figsize=(12,6))
plt.show()


'''Computing Monthly Returns'''
def data_treatment():

    sp500_monthly = sp500_data['Adj Close'].groupby(pd.Grouper(freq='MS')).last()
#    sp500_monthly = sp500.groupby(pd.Grouper(freq='MS')).last()

    sp500_mon_returns = sp500_monthly.pct_change().dropna()
#    sp500_mon_returns = sp500_monthly.pct_change()

    return sp500_monthly, sp500_mon_returns

sp500_monthly, sp500_mon_returns = data_treatment()


print "Biggest 5 monthly drawdowns \n ", sp500_mon_returns.nsmallest(5)


'''Drawdowns'''

# Define a trailing 252 trading day window
window = 252

# Calculate the max drawdown in the past window days for each day 
rolling_max = sp500_data['Adj Close'].rolling(window, min_periods=1).max()
daily_drawdown = sp500_data['Adj Close']/rolling_max - 1.0

# Calculate the minimum (negative) daily drawdown
max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()

# Plot the results
daily_drawdown.plot()
max_daily_drawdown.plot()
plt.title("Rolling Maximum Drawdown in a year", fontsize=20)
daily_drawdown.min()
# Show the plot
plt.show()

############# Question 3 ###############

sp500_cum_daily_return = (1 + sp500_daily_return).cumprod()
print(sp500_cum_daily_return)
sp500_cum_daily_return.plot(figsize=(12,8))
plt.title("Cumulative SP500 returns", fontsize=20)
plt.show()



days = (sp500_data.index[-1] - sp500_data.index[0]).days

CAGR = (((sp500_data['Adj Close'][-1]/sp500_data['Adj Close'][0]))**(365.0/days))-1

Calmar_Ratio = CAGR/abs(daily_drawdown.min())

print "\n The Calmar Ratio is: \n ", Calmar_Ratio


################## Question 4 and Question 5 #####################

signals = pd.DataFrame(index=sp500_data.index)
signals['signal'] = 0.0
signals['close_above'] = sp500_data['Adj Close']
signals['mavg'] = sp500_data['Adj Close'].rolling(30, min_periods=1, center=False).mean()
signals['close_above'].plot(figsize=(12,6))
signals['mavg'].plot(figsize=(12,6))
plt.title("30 days Simple Moving Average Crossover")
plt.show()



signals['signal'] = np.where(signals['close_above'] > signals['mavg'], 1.0, 0)
signals.dropna(inplace=True) 
signals['signal'].plot(ylim=[-1.1, 1.1], title='Market Positioning')

signals['returns'] = np.log(signals['close_above'] / signals['close_above'].shift(1))


signals['returns'].hist(bins=35)

signals['strategy'] = signals['signal'].shift(1) * signals['returns'] 
signals[['returns', 'strategy']].sum() 

signals[['returns', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))


fig = plt.figure(figsize=(12,6))

ax1 = fig.add_subplot(111,  ylabel='Price in $')
signals['close_above'].plot(figsize=(12,6))
signals['mavg'].plot(figsize=(12,6))
plt.title("Gree = Bought / Red = Flat")
# Buy signals
ax1.plot(signals.loc[signals.signal == 1.0].index, 
         signals.mavg[signals.signal == 1.0],
         '^', markersize=5, color='g')
# Sell signals
ax1.plot(signals.loc[signals.signal == 0].index, 
         signals.mavg[signals.signal == 0],
         'v', markersize=5, color='r')


plt.show()

signals['positions'] = signals['signal'].diff()
signals['strategy2'] = signals['strategy']+1
signals['10000port'] = signals['strategy2'].dropna().cumprod()*10000

print signals
plt.title("Cummulative Returns - 10.000 dollars portfolio", fontsize=20)
signals['10000port'].plot(grid = True)
plt.show()
####### Question 6 ##########

########## Lake Ratio ####################


st = signals['strategy'].cumsum().apply(np.exp)
st.head()
lake_ratio = pd.DataFrame(st)
lake_ratio = lake_ratio.dropna()
lake_ratio.head()
lake_ratio.plot(figsize=(10, 6))
window = 2520
rolling_max = lake_ratio.rolling(window, min_periods=1).max()
lake_ratio["Rolling_Maximum"] = rolling_max
lake_ratio["Water"] = lake_ratio["Rolling_Maximum"] - lake_ratio["strategy"]
lake_ratio.head()
rolling_max = lake_ratio.rolling(window, min_periods=1).max()
lake_ratio.plot(figsize=(10, 6))

plt.show()


######## Gain to pain  ####


rets = signals['strategy'] 

total_loses = np.sum(rets[rets < 0])
#print(total_loses)

total_return = np.sum(rets)

gain_to_pain = total_return/abs(total_loses)
print "\n The gain to pain ratio is: \n", gain_to_pain

