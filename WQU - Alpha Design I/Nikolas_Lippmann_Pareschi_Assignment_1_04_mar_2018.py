# -*- coding: utf-8 -*-
"""
Created on Sun Mar 04 14:33:28 2018

@author: Nikolas
"""

# Global variables declaration

import pandas_datareader.data as web
import pandas as pd
import datetime as dt
import statsmodels.api as sm
import matplotlib.pyplot as plt

symbol = "DIS"
index = "^GSPC"
start = dt.datetime(2008, 10, 1)
end = dt.datetime(2013, 9, 30)

def download_data():

    df = web.DataReader(symbol, 'yahoo', start, end)
    df2 = web.DataReader(index, 'yahoo', start, end)
    returns_dis = df['Adj Close'].pct_change()
    returns_sp500 = df2['Adj Close'].pct_change()

    return df, df2, returns_dis, returns_sp500


disney, sp500, returns_dis, returns_sp500 = download_data()

def data_treatment():

    disney_monthly = disney.groupby(pd.Grouper(freq='MS')).last()
    sp500_monthly = sp500.groupby(pd.Grouper(freq='MS')).last()

    disney_mon_returns = disney_monthly.pct_change()
    sp500_mon_returns = sp500_monthly.pct_change()

    return disney_monthly, sp500_monthly, disney_mon_returns, sp500_mon_returns

disney_monthly, sp500_monthly, disney_mon_returns, sp500_mon_returns = data_treatment()

sp500_mon_returns = sp500_mon_returns['Adj Close']

sp500_mon_returns = sp500_mon_returns.dropna()

def plot():

    plt.subplot(2,1,1)
    plt.plot(disney_mon_returns, color='red', label='Disney')
    plt.legend(loc='upper right')
    plt.subplot(2,1,2)
    plt.plot(sp500_mon_returns, color='green', label='SP500')
    plt.legend(loc='upper right')
    plt.show()


def data_manipulation(a, b):

    returns_dis_treated = a.dropna()
    returns_sp500_treated = b.dropna()
    returns_sp500_treated = sm.add_constant(b)

    return returns_dis_treated, returns_sp500_treated

returns_dis_treated, returns_sp500_treated = data_manipulation(disney_mon_returns, sp500_mon_returns)

print 'Disney Returns: ', returns_dis_treated
print 'SP500 Returns: ',  returns_sp500_treated
disney_annual_return = ((1+returns_dis_treated.mean())**12)-1
print "\n Disney Annual Return", disney_annual_return

def main():

    plot()
    y, x = data_manipulation(returns_dis_treated, returns_sp500_treated)
    est2 = sm.OLS(y['Adj Close'], x).fit()
    print est2.summary()
    print 'Parameters: ', est2.params
    print 'R2: ', est2.rsquared
    print 'Standard errors: ', est2.bse
    print 'Significance of standard erros: ', est2.tvalues

if __name__ == '__main__':
    main()

