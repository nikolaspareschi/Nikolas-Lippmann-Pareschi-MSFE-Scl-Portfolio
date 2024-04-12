# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 17:37:17 2018

@author: Nikolas
"""

import pandas_datareader.data as web
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


'''
1 - Gather the daily high, low, and closing prices for McDonald's stock
 (ticker symbol MCD) for January 2004 through July 2005 from an appropriate
 financial website such as Google Finance, Yahoo Finance, Quandl, CityFALCON,
 or another similar source.
'''


# Global Variables Declaration

symbol = "MCD"
start = dt.datetime(2004, 1, 1)
end = dt.datetime(2005, 7, 31)

def download_data():

    df = web.DataReader(symbol, 'yahoo', start, end)
    returns_mac = df['Adj Close'].pct_change()

    return df, returns_mac


bigmac, bigmac_ret = download_data()


'''

2 - Calculate 10-day and 60-day SMAs. Plot these two curves with a bar chart of
 the stock prices. 
 
 '''


bigmac['10sma'] = bigmac['Adj Close'].rolling(10).mean()
bigmac['60sma'] = bigmac['Adj Close'].rolling(60).mean()



fig = plt.figure(figsize=(12, 4))
ax = bigmac[["Open", "High", "Low", "Close"]].transpose().plot.box(figsize=(12, 4))  
ax2 = plt.axes()

ax2.yaxis.set_major_locator(plt.NullLocator())
ax2.xaxis.set_major_formatter(plt.NullFormatter())
ax2 = ax.twinx()
ax2.plot(ax.get_xticks(), bigmac[['10sma', '60sma']]) 
plt.title('Plot of SMA 10, SMA60 and Bars of Mac Donalds')
plt.show()


'''

3. Compare and contrast the 10-day and the 60-day SMA.

'''


bigmac['sma_diff'] = (bigmac['60sma'] - bigmac['10sma'])
bigmac['Adj Close'].plot(figsize=(12, 4))
bigmac['10sma'].plot(figsize=(12, 4))
bigmac['60sma'].plot(figsize=(12, 4))

plt.legend(['Adj Close', '10sma', '60sma'])
plt.title('Plot of SMA 10, SMA60 and Adjusted Closes of Mac Donalds')
plt.show()

bigmac['sma_diff'].plot(figsize=(12, 4))
plt.legend(['sma_diff'])
plt.title('Simple Moving Averages Difference')
plt.show()

'''

4. Explain the relationship between the market trend and the 60-day SMA during
the following periods:
May 2004-October 2004
October 2004-May 2005
May 2005-July 2005

'''

# May 2004-October 2004

bigmac['2004-05-01':'2004-10-31']['Adj Close'].plot(figsize=(12, 4))
bigmac['2004-05-01':'2004-10-31']['10sma'].plot(figsize=(12, 4))
bigmac['2004-05-01':'2004-10-31']['60sma'].plot(figsize=(12, 4))
plt.legend(['Adj Close', 'SMA 10', 'SMA 60'])
plt.title('May 2004-October 2004')
plt.show()

# October 2004-May 2005

bigmac['2004-10-01':'2005-05-31']['Adj Close'].plot(figsize=(12, 4))
bigmac['2004-10-01':'2005-05-31']['10sma'].plot(figsize=(12, 4))
bigmac['2004-10-01':'2005-05-31']['60sma'].plot(figsize=(12, 4))
plt.legend(['Adj Close', 'SMA 10', 'SMA 60'])
plt.title('October 2004-May 2005')
plt.show()

# May 2005-July 2005

bigmac['2005-05-31':'2005-07-31']['Adj Close'].plot(figsize=(12, 4))
bigmac['2005-05-31':'2005-07-31']['10sma'].plot(figsize=(12, 4))
bigmac['2005-05-31':'2005-07-31']['60sma'].plot(figsize=(12, 4))
plt.legend(['Adj Close', 'SMA 10', 'SMA 60'])
plt.title('May 2005-July 2005')
plt.show()

# The analysis was done in the PDF file

'''
5 - Draw the moving average oscillator of the price chart.


'''


bigmac['sma_diff'].plot(figsize=(12, 4))
plt.legend(['SMA_oscilator'])
plt.title('Simple Moving Averages Difference - SMA Oscilator')
plt.show()

'''

6 - Bollinger Band is a band plotted 1.5 standard deviations away from a simple
moving average. Calculate the Bollinger bands of 10-day simple moving average
for Mac Donald share. 

'''

'''

7 - Develop a trading strategy based on the relation between price and 
Bollinger Bands. Graphically represent the risk-return profile of such a trading strategy

'''
# This part of the code was based in the following tutorial:
# http://www.pythonforfinance.net/2017/07/31/bollinger-band-trading-strategy-backtest-in-python/

def bollinger_strat(df,window,std):
    rolling_mean = df['Adj Close'].rolling(window).mean()
    rolling_std = df['Adj Close'].rolling(window).std()
    
    df['Bollinger High'] = rolling_mean + (rolling_std * std)
    df['Bollinger Low'] = rolling_mean - (rolling_std * std)
    
    df['Short'] = None
    df['Long'] = None
    df['Position'] = None
    
    for row in range(len(df)):
    
        if (df['Adj Close'].iloc[row] > df['Bollinger High'].iloc[row]) and (df['Close'].iloc[row-1] < df['Bollinger High'].iloc[row-1]):
            df['Position'].iloc[row] = -1
        
        if (df['Adj Close'].iloc[row] < df['Bollinger Low'].iloc[row]) and (df['Adj Close'].iloc[row-1] > df['Bollinger Low'].iloc[row-1]):
            df['Position'].iloc[row] = 1
            
    df['Position'].fillna(method='ffill',inplace=True)
    
    df['Market Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['Strategy Return'] = df['Market Return'] * df['Position']
    
    df['Strategy Return'].cumsum().plot()
    rolling_std.plot()
    plt.title('Cumulative Returns x Rolling Standard Deviation')
    plt.show()
    print ' \n The sharp ratio of this stragy is', (df['Strategy Return'].mean()/df['Strategy Return'].std())
    
	
bollinger_strat(bigmac,10,1.5)
