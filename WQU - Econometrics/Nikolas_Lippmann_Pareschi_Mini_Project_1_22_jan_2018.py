# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 19:31:24 2018

@author: nikol
"""


import pandas_datareader.data as web
import datetime
import statsmodels.api as sm

# Global variables declaration

symbol = "JPM"
index = "^GSPC"

start = datetime.datetime(2015, 4, 1)
end = datetime.datetime(2015, 6, 25)
risk_free = 0.001


def download_data():

    df = web.DataReader(symbol, 'yahoo', start, end)
    df2 = web.DataReader(index, 'yahoo', start, end)
    returns_jpm = df['Adj Close'].pct_change()
    returns_sp500 = df2['Adj Close'].pct_change()

    return returns_jpm, returns_sp500


returns_jpm, returns_sp500 = download_data()


def data_manipulation(a, b):

    returns_jpm_treated = returns_jpm.dropna()
    returns_sp500_treated = returns_sp500.dropna()
    returns_sp500_treated = sm.add_constant(returns_sp500_treated)

    return returns_jpm_treated, returns_sp500_treated


def jp_statistics():

    jpm_returns = returns_jpm.dropna()
    daily_mean = jpm_returns.mean()
    std = jpm_returns.std()

    return jpm_returns, daily_mean, std


jpm_returns, daily_mean, std = jp_statistics()


def main():

    y, x = data_manipulation(returns_jpm, returns_sp500)
    est2 = sm.OLS(y, x).fit()
    print est2.summary()
    print ""
    print 'Daily Returns of JPM: \n'
    print jpm_returns
    print ""
    print 'Mean of daily returns of JPM: \n'
    print daily_mean
    print ""
    print 'Daily Volatility of JPM: \n'
    print std


if __name__ == '__main__':
    main()
