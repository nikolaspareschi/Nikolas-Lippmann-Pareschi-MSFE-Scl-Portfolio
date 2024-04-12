# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 21:19:52 2018

@author: Nikolas
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from datetime import datetime as dt
import matplotlib.pyplot as plt



'''
Write a Python program to download data for Dow Jones Transportation Average
 and Dow Jones Industrial Average for the last 5 Years.
'''

start = dt(2013, 1, 1)
end = dt(2018, 3, 20)
df = web.DataReader(['DIA','IYT'], "yahoo", start, end)
df = df['Close']

'''Create and calculate any one indicator that would allow you to decide
 on making a pairs trade between these 2 indices.'''


'''
Based on the historical values of that indicator, calculate and graphically
 represent the return profile of a pairs trading strategy

'''
# This function is based on the following tutorial:
# https://medium.com/auquan/pairs-trading-data-science-7dbedafcfe5a

def trade(S1, S2, window1, window2):
    
# The indicator will be the ratio between DJ Industrial and DJ Transports normalized. First 
# let's plot and check if the data series are right
    
    plt.figure(figsize=(18,9))
    S1.plot(color='b')
    S2.plot(color='c')
    plt.title('DJ Industrial in Dark Blue, DJ Transports in light blue')
    plt.show()
    
    if (window1 == 0) or (window2 == 0):
        return 0
    
    ratios = S1/S2
    ma1 = ratios.rolling(window=window1,
                               center=False).mean()
    ma2 = ratios.rolling(window=window2,
                               center=False).mean()
    std = ratios.rolling(window=window2,
                        center=False).std()
    zscore = (ma1 - ma2)/std
    zscore.plot()
    plt.title('Z-score indicator of the Ratio between DJ Industials x DJ Transportation')
    plt.show()
 
    # The Strategy buys DJ Industrails and sells DJ Transports when the z-score is below -2
    # The Strategy sells DJ Industrails and buy DJ Transports when the z-score is above 2
    # We close the opened positions with the Z-Score between -1 and 1
    
    money = 0
    money2 = []
    countS1 = 0
    countS2 = 0
    for i in range(len(ratios)):

        if zscore[i] > 2:
            money += S1[i] - S2[i] * ratios[i]
            money2.append(S1[i] - S2[i] * ratios[i])
            countS1 -= 1
            countS2 += ratios[i]

        elif zscore[i] < -2:
            money -= S1[i] - S2[i] * ratios[i]
            money2.append(-S1[i] + S2[i] * ratios[i])
            countS1 += 1
            countS2 -= ratios[i]

        elif abs(zscore[i]) < 1:
            money += countS1*S1[i] - S2[i] * countS2
            money2.append(countS1*S1[i] - S2[i] * countS2)
            count = 0

    plt.plot(money2)
    plt.title('Cumulative Return Graph')
    plt.ylabel('Cumulative Returns with scale 10 powered to 7')
    plt.xlabel('Number of days')
    plt.show()
    print 'Not the best strategy, our profit is:', money
    return money

pairs_trading = trade(df['DIA'], df['IYT'], 60, 5)
