# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 21:21:07 2018

@author: nikol
"""


import pandas_datareader.data as web
import datetime
import statsmodels.api as sm

# Global variables declaration

symbol = "ORCL"
index = "^GSPC"

start = datetime.datetime(2015, 3, 25)
end = datetime.datetime(2015, 5, 22)

""" Accordingly to the Fama-French website the riske free rate in the period
was zero """

risk_free = 0


def download_data():

    df = web.DataReader(symbol, 'yahoo', start, end)
    df2 = web.DataReader(index, 'yahoo', start, end)
    returns_orcl = df['Adj Close'].pct_change()
    returns_sp500 = df2['Adj Close'].pct_change()

    return returns_orcl, returns_sp500


returns_orcl, returns_sp500 = download_data()


def data_manipulation(a, b):

    returns_orcl_treated = returns_orcl.dropna()
    returns_sp500_treated = returns_sp500.dropna()
    returns_sp500_treated = sm.add_constant(returns_sp500_treated)

    return returns_orcl_treated, returns_sp500_treated


def main():

    y, x = data_manipulation(returns_orcl, returns_sp500)
    est2 = sm.OLS(y, x).fit()
    print est2.summary()


if __name__ == '__main__':
    main()
