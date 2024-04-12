# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 12:08:15 2018

@author: Nikolas
"""

import pandas as pd
import pandas_datareader.data as web
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns  


# 1.	Download data for Dow Jones Index (DJIA) for the last 15 years.

# Global variables declation

dj = "^DJI"

start = datetime.datetime(2003, 3, 1)
end = datetime.datetime(2018, 3, 1)



def download_data():

    dj_data = web.DataReader(dj, 'yahoo', start, end)
    returns_dj = dj_data['Adj Close'].pct_change()

    return dj_data, returns_dj


dj_data, returns_dj = download_data()




# 2.	Compute daily percentage deviations of mid- price of DJIA from the 200 DAY Exponential Moving average (200DEMA) of close prices.


dj_data = dj_data.iloc[:, :-2]
dj_data['midsum'] = dj_data.sum(axis=1)
dj_data['midp'] = dj_data['midsum'] / 4
dj_data['ema'] = dj_data['Close'].ewm(200).mean()
dj_data['diff_pc'] = (dj_data['midp'] / dj_data['ema']) - 1

# 3.	Prices above the 200-DEMA would get a positive value while those below would get a negative value

treeshold = 0
dj_data['Signal'] = np.where(dj_data['diff_pc'] > treeshold, 1, 0)

# We need this line of code because preices can be exactly ath the 200-DEMA

dj_data['Signal'] = np.where(dj_data['diff_pc'] < treeshold, -1, dj_data['Signal'])



# 4.	Whenever price is above 200-DEMA the market is generally considered to be in an up-move (and vice versa). Graphically represent the historical deviations and mark out clear periods of overall bullish and bearish regimes in DJIA.
# 5.	For each of these regimes, clearly fit separate linear trend-lines and plot it on the same graph.
# The logic of this code was based in: https://stackoverflow.com/questions/41906679/how-to-calculate-and-plot-multiple-linear-trends-for-a-time-series

signal = dj_data['Signal']
plt.plot(signal)
plt.title("Buy Above 200 EMA Sell Below")
plt.show()

# We are defining a minimum number of signals in one direction to cut the shipsaw moves

min_signal = 5


limits = (np.diff(signal) != 0) & (signal[1:] != 0)
limits = np.concatenate(([signal[0] != 0], limits))
limits_index = np.where(limits)[0]
limits2_index = np.array([index for index in limits_index if np.all(signal[index] == signal[index:index + min_signal])])

# Including first day and last day

if limits2_index[0] != 0:
    limits2_index = np.concatenate(([0], limits2_index))
if limits2_index[-1] != len(signal) - 1:
    limits2_index = np.concatenate((limits2_index, [len(signal) - 1]))


for start_index, end_index in zip(limits2_index[:-1], limits2_index[1:]):

    segment = dj_data.iloc[start_index:end_index + 1, :]
    x = np.array(mdates.date2num(segment.index.to_pydatetime()))
    data_color = 'green' if signal[start_index] > 0 else 'red'
    plt.plot(segment.index, segment['Close'], color=data_color)
    coef, intercept = np.polyfit(x, segment['Close'], 1)
    fit_val = coef * x + intercept

    if coef > 2:
        fit_color = 'brown'
    elif coef > 0:
        fit_color = 'yellow'
    elif coef > -2:
        fit_color = 'blue'
    else:
        fit_color = 'red'
        
    plt.plot(segment.index, fit_val, color=fit_color)
    
plt.show()
    