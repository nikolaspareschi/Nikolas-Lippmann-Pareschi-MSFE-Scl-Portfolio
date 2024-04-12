# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 20:40:10 2018

@author: Nikolas
"""

import pandas_datareader.data as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns


# Global variables declaration

symbol1 = "AAPL"
symbol2 = "MSFT"
index = "^GSPC"
fiveyear = "^FVS"

start = datetime.datetime(2015, 2, 28)
end = datetime.datetime(2018, 2, 28)


def download_data(symbol1, symbol2, index, fiveyear):

    aapl = web.DataReader(symbol1, 'yahoo', start, end)
    msft = web.DataReader(symbol2, 'yahoo', start, end)
    sp500 = web.DataReader(index, 'yahoo', start, end)
    fiveyear = web.DataReader(fiveyear, 'yahoo', start, end)
#    returns_jpm = df['Adj Close'].pct_change()
#    returns_sp500 = df2['Adj Close'].pct_change()

    return aapl, msft, sp500, fiveyear


(aapl_data, msft_data, sp500_data, fiveyear) = download_data(symbol1, symbol2, index, fiveyear)


aapl_data_close = aapl_data['Adj Close']
msft_data_close = msft_data['Adj Close']
sp500_data_close = sp500_data['Adj Close']
fiveyear_data_close = fiveyear['Adj Close']

aapl_data_close_returns = aapl_data_close.pct_change()
msft_data_close_returns = msft_data_close.pct_change()

aapl_data_close_returns2 = aapl_data_close_returns.dropna()
msft_data_close_returns2 = msft_data_close_returns.dropna()

aapl_data_close_returns2.plot()
plt.title("Apple Daily Returns", fontsize=20)
plt.show()
msft_data_close_returns2.plot()
plt.title("Microsoft daily returns", fontsize=20)
plt.show()

area = np.pi*20.0
sns.set(style='darkgrid')
plt.figure(figsize=(9,9))
plt.scatter(aapl_data_close_returns2, msft_data_close_returns2, s=area)
plt.xlabel("Apple Daily Returns", fontsize=15)
plt.ylabel("Microsoft Daily Returns", fontsize=15)
plt.title("Scatter Plot of MSFT x AAPL daily returns", fontsize=20)

########## Question 3 ###############

annualized_appl = aapl_data_close_returns2.mean()*252
annualized_msft = msft_data_close_returns2.mean()*252

print "\n Apple annualized return: \n ", annualized_appl
print "\n Microsoft annualized return: \n ", annualized_msft

area = np.pi*20.0
sns.set(style='darkgrid')
plt.figure(figsize=(9,9))
plt.scatter(annualized_appl, annualized_msft, s=area)
plt.xlabel("Apple annualized return:", fontsize=15)
plt.ylabel("Microsoft annualized return", fontsize=15)
plt.title("MSFT x AAPL Annualized return", fontsize=20)
plt.show()

print " \n Microsoft statistical measures:  \n ", msft_data_close_returns2.describe()
print " \n Apple statistical measures:  \n",aapl_data_close_returns2.describe()

portfolio = pd.concat([aapl_data_close_returns2, msft_data_close_returns2], axis = 1)
portfolio.describe()

########### Question 4 #############

portfolio2 = (aapl_data_close + msft_data_close)/2
print "\n Portfolio stats:\n ", portfolio2.describe()
port_std = portfolio2.std() 
port_var = portfolio2.var()

result_var = (0.5**2)*(aapl_data_close.std()**2)+(0.5**2)*(msft_data_close.std()**2)+2*0.5*0.5*aapl_data_close.cov(msft_data_close)

print "\n Our portfolio variance is: \n ", result_var

########## Question 5 ##############


fiveyear_returns = fiveyear_data_close.pct_change().dropna()
fiveyear_returns.mean()*252

fiveyear_data_close.mean()


portfolio_returns = portfolio2.pct_change().dropna()
sharp = (portfolio_returns.mean()-fiveyear_data_close.mean())/portfolio_returns.std()
annualized_sharp = sharp*np.sqrt(252)



################# Question 5 ##########################

rets = portfolio2.pct_change().dropna().mean()
area = np.pi*20.0


plt.figure(figsize=(9,9))
plt.scatter(rets*252, portfolio2.pct_change().dropna().std()*np.sqrt(252), s=area)
plt.scatter(fiveyear_returns.mean()*252, fiveyear_returns.std()*np.sqrt(252), s=area)
plt.xlabel("Expected Return", fontsize=15)
plt.ylabel("Risk", fontsize=15)
plt.title("Return/Risk for Port vs. Real return of 5 year bunds ", fontsize=20)
plt.show()

sns.set(style='darkgrid')
plt.figure(figsize=(9,9))
plt.scatter(rets*252, portfolio2.pct_change().dropna().std()*np.sqrt(252), s=area)
plt.scatter(0.0172, fiveyear_returns.std()*np.sqrt(252), s=area)
plt.xlabel("Expected Return", fontsize=15)
plt.ylabel("Risk", fontsize=15)
plt.title("Return/Risk for Port vs. 5 year returns given ", fontsize=20)
plt.show()

########### Question 6 ##################

print "KPI METRICS"
print "" 
print "The annualized sharp ratio of the portfolio is: " , annualized_sharp



count = 0
count2 =0

for i in portfolio_returns:
#    print i
    if i >= 0:
        count = count + 1
    else:
        count2 = count2 + 1
        
#print count    
    
winrate = float(count)/float(len(portfolio_returns))
lossrate = float(count2)/float(len(portfolio_returns))

print "The winrate of our portfolio is: ", winrate
print "The lossrate of our portfolio is: ", lossrate


sns.set(style='darkgrid')
plt.title("Violin plot for portfolio returns ", fontsize=20)
sns.violinplot(portfolio_returns)
plt.show()
plt.title("Violin plot for 5 year notes returns ", fontsize=20)
sns.violinplot(fiveyear_returns)
plt.show()


