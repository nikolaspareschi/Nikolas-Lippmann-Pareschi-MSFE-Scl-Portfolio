# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 17:40:27 2017

@author: Nikolas
"""
#######################################################
# PLEASE CHECK THE WORD FILE FOR THE COMPLETE ANALYSIS#
#######################################################

# If the user is using Anaconda Distribution, he may need to instal the 
# pandas_datareader. To do so, go into Anaconda prompt and insert:
# 'conda instal -c https://conda.anaconda.org/anaconda pandas-datareader'
# You may need to install other packages if you are not using anaconda.

# This project is running in Python 3.6 / Spyder Version 4.20

# Part of the code for this project was provided by Douglas Kelly on Piazza
# Discussion Group and by WQU in his lab file for the Mini Project II

import pandas_datareader.data as web
import numpy as np
import scipy.signal as sc
from scipy.fftpack import rfft
import matplotlib.pyplot as plt

# Commodity chosen: NATURAL GAS
# The reasons for the choice are explained in the word file

# We have used the code provided by Quandl to download the Natural Gas Prices

symbol = 'CHRIS/CME_QG2'
df = web.DataReader(symbol, 'quandl')

# We printed the imported data and we have discovered that the data is from:
# 2014-02-18 to 2017-08-18

print(df)

# First we make a plot of natural gas close (last) prices:

plt.figure(1)
plt.plot(df['Last'])
plt.title('Natural gas price movement')


# We then apply the Python code provided to SMOOTH DETREND and then plot 
#  graph:

detrend = sc.detrend(df['Last'])
print(sc.detrend(df['Last']))
plt.figure(2)
plt.plot(detrend)
plt.title('Natural gas prices detrended')

#We then apply the Python code for the Blackman window function 
# from numpymodule and then plot graph:

w = np.blackman(20) # We select 20 the parameter of the blackman window function
y = np.convolve(w/w.sum(), detrend, mode = 'same')
plt.figure(3)
plt.plot(y)
plt.title('Blackman window function for detrended Natural Gas price')


# We then Apply the FFT Algorithm and plot the graph 

fft = abs(rfft(y))
plt.figure(4)
plt.plot(fft)
plt.title('FFT algorithm applied to Natural Gas Prices')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')

# Now we diminish the scope of the frequency so we can analyse lower frequencies

fft = abs(rfft(y))
plt.figure(5)
plt.plot(fft)
plt.xlim(0,20)
plt.title('FFT algorithm applied to Natural Gas Prices')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')

# PLEASE CHECK THE WORD FILE FOR THE COMPLETE ANALYSIS

