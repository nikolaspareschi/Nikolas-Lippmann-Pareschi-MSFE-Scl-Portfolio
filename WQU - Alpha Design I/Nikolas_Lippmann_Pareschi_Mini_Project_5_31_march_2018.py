# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:30:44 2018

@author: Nikolas
"""


import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import datetime


'''1.	Write a Python program to download the historical data of Dow Jones
 Industrial Average (DJIA) over the last 15 years.'''

start = datetime.datetime(2003, 3, 1)
end = datetime.datetime(2018, 3, 1)
dow = data.DataReader('^DJI',  'yahoo', start, end)
dow.head()


'''2.	Construct a simple trading system that goes long when DJIA closes above
 its 20 Day Exponential Moving Average of Close Prices (20 DEMA) and closes in
 position and goes short when prices close below the 20 DEMA.'''

dow['DEMA20'] = dow['Close'].ewm(span=20, adjust=False).mean()
dow['10k'] = 10000


# Ploting to see if the EMA and the DJ were corretly calculated and updated

dow.plot(y=['Close', 'DEMA20'],figsize=(15,8))
plt.title('DIA Close Prices & Simple Moving Average DEMA(20)')
plt.legend(loc='upper left')
plt.show()


# Signal construction CASE 1 ############################


dow['Close(-1)'] = dow['Close'].shift(1)
dow['DEMA20(-1)'] = dow['DEMA20'].shift(1)

dow['close_above_now'] = dow['Close'] - dow['DEMA20']
dow['close_above_before'] = dow['Close(-1)'] - dow['DEMA20(-1)']


dow['Stance2'] = np.where((dow['close_above_now'] > 0), 1, -1)
dow['Stance2'].value_counts()

# Checking the signals thourgh plot (-1 we will be selling, + 1 we will be buying)

dow['Stance2'].plot(lw=1.5,ylim=[-1.1,1.1], title= 'Checking the signals through plot (-1 we will be selling, + 1 we will be buying', figsize=(15,8))
plt.show()

# Returns without considering money

dow['Market Returns'] = np.log(dow['Close'] / dow['Close'].shift(1))
dow['Strategy'] = dow['Market Returns'] * dow['Stance2'].shift(1)
dow[['Market Returns','Strategy']].cumsum().plot(grid=True,figsize=(8,5))


''' 3.	Exit from Trades are made by stop Market orders at the price of the
 20 DEMA. Start with 10,000$ and always invest 1,000$ in every single trade'''
 


'''Case 1: Consider frictionless trading'''
 
# Returns considered that we will always be buying selling 1000 dollars with
# a 10.000 dollars of initial capital

dow['Market Returns1000'] = dow['Market Returns']*1000 
dow['Strategy1000'] = dow['Strategy']*1000
dow['Market Returns1000_10k'] = dow['Market Returns1000'].fillna(10000)
dow['Strategy1000_10k'] = dow['Strategy1000'].fillna(10000)

# Plot of capital evolution


dow[['Market Returns1000_10k','Strategy1000_10k']].cumsum().plot(grid=True, title = 'Frictionless Trading', figsize=(8,5))

'''5.	Case 2: Consider Real Trading'''

# Computing comissions and slippage. The slippage -2 dollars for every 100 units of the ETF traded accordingly to our model in A5

dow['Stance3'] = dow['Stance2'] - dow['Stance2'].shift(1) 
dow['trade_day_slippage'] = np.where(dow['Stance3'] != 0, -2, 0)

# Broker Comissions are stimated in 1 dollar per trade

dow['trade_day_comissions'] = np.where(dow['Stance3'] != 0, -1, 0)



# We will just buy sell in the next day, to simulate real trade conditions

dow['Market Returns2'] = np.log(dow['Open'].shift(-1) / dow['Open'])
dow['Market Returns2'] = dow['Market Returns2'].fillna(0)
dow['Market Returns1000_2'] = dow['Market Returns2']*1000 
dow['Market Returns1000_10k_2'] = dow['Market Returns1000_2']
dow['Market Returns1000_10k_2'][0] =dow['Market Returns1000_10k_2'][0] + 10000
dow['Strategy2'] = dow['Market Returns2'] * dow['Stance2'].shift(1)
dow['Strategy1000_2'] = dow['Strategy2']*1000
dow['Strategy1000_10k_2'] = dow['Strategy1000_2'].fillna(10000)
dow[['Market Returns1000','Strategy1000']].cumsum().plot(grid=True,figsize=(8,5))



dow[['Market Returns1000_10k_2','Strategy1000_10k_2']].cumsum().plot(grid=True, title = 'Next day price on open but without Slippage and Comissions', figsize=(8,5))
dow['Strategy1000_10k_2_slippage+comission'] = dow['Strategy1000_10k_2'] + dow['trade_day_slippage']* (1000/((dow['Close'][0]+dow['Close'][-1])/2)) + dow['trade_day_comissions']
dow[['Market Returns1000_10k_2','Strategy1000_10k_2_slippage+comission']].cumsum().plot(grid=True, title = 'With Slippage and Comissions', figsize=(8,5))



