# -*- coding: utf-8 -*-
"""
Created on Sun Aug 05 12:48:07 2018

@author: Nikolas
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr
sns.set_style('whitegrid')


'''1.	Write a program that prompts the user to enter any valid stock symbol
 available in an appropriate financial website such as Google Finance, Yahoo
 Finance, Quandl, CityFALCON, or another similar source for NYSE & NASDAQ.
 Ensure proper error handling for wrong user inputs.'''


def download_data():

    # Creating dataframes with the symbol for posterior check

    # NYSE

    yf.pdr_override()
    url_nyse = 'http://www.nasdaq.com/screening/companies-by-name.aspx?'\
               'letter=0&exchange=nyse&render=download'

    # Nasdaq

    url_nasdaq = "http://www.nasdaq.com/screening/companies-by-name.aspx?"\
                 "letter=0&exchange=nasdaq&render=download"

    nyse = pd.read_csv(url_nyse)
    nasdaq = pd.read_csv(url_nasdaq)
    symbol = raw_input("Please enter symbol for the stock that"
                       "you want to download \n")

    while (nyse['Symbol'] == symbol).any() != True and ((nasdaq['Symbol']
                                                         == symbol).any() !=
                                                        True):
        symbol = raw_input("Please enter a VALID symbol for the stock"
                           "that you want to download \n")

# Error handling

    while (nyse['Symbol'] == symbol).any() != True and (
            (nasdaq['Symbol'] == symbol).any() != True):
        symbol = raw_input("Please enter a VALID"
                           "symbol for the stock that you want to download \n")

    try:

        data = pdr.get_data_yahoo(symbol, start="2007-07-31", end="2018-07-31")
    except:
        print("Got an Error : ")
        exit()

    return data


data_ = download_data()


def RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean(u[:period])
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean(d[:period])
    d = d.drop(d.index[:(period-1)])
    rs = pd.stats.moments.ewma(u, com=period-1, adjust=False) / \
        pd.stats.moments.ewma(d, com=period-1, adjust=False)
    return 100 - 100 / (1 + rs)


class Output(object):
    def __init__(self, returns_df, date_freq='D'):
        self.returns_df = returns_df if isinstance(
                returns_df, pd.DataFrame) else pd.DataFrame(returns_df)
        self.wealthpaths = self.returns_df.apply(self._calc_wealthpath)
        self._date_freq = str(date_freq).upper()
        if self._date_freq == 'D':
            self._freq = 252
        elif self._date_freq == 'M':
            self._freq = 12

    def _calc_annualized_return(self, series):
        avg_daily_return = series.mean()
        ann_return = avg_daily_return * self._freq
        return ann_return

    def _calc_annualized_std_dev(self, series):
        series_std = series.std()
        ann_std = series_std * (np.sqrt(self._freq))
        return ann_std

    def _calc_sharpe(self, ann_returns, ann_stds):
        sharpe = ann_returns.divide(ann_stds)
        return sharpe

    def _calc_hwm(self, wealthpath):
        hwm = wealthpath.expanding().max()
        return hwm

    def _calc_wealthpath(self, series):
        if series.iloc[0] != 0:
            first_dt = series.index[0]
            set_dt = first_dt - dt.timedelta(days=1)
            series.ix[set_dt] = 0.0
            series = series.sort_index()

        cum_prod = (1.0 + series).cumprod()
        return cum_prod

    def _calc_drawdowns(self, wealthpath):
        hwm = self._calc_hwm(wealthpath)
        drawdowns = wealthpath.divide(hwm).subtract(1.0)
        return drawdowns

    def _calc_lake_ratios(self, hwm, wps):
        lakes = hwm.subtract(wps)
        mountains = hwm.subtract(lakes)
        lake_ratios = lakes.sum() / mountains.sum()
        return lake_ratios

    def _calc_gain_to_pain_ratio(self, series):
        total_return_series = (1.0 + series).cumprod().subtract(1.0)
        total_return = total_return_series.iloc[-1]

        loss_returns_series = self.__get_loss_returns(series).abs()
        if not loss_returns_series.empty:
            total_loss_return_series = (
                    (1.0 + loss_returns_series).cumprod().subtract(1.0))
            total_loss_return = total_loss_return_series.iloc[-1]

            gpr = total_return / total_loss_return
        else:
            gpr = np.nan
        return gpr

    def __get_win_returns(self, series):
        win_returns = series[series >= 0.0]
        return win_returns

    def __get_loss_returns(self, series):
        loss_returns = series[series < 0.0]
        return loss_returns

    def _calc_win_rate(self, series):
        win_returns = self.__get_win_returns(series)
        rate = float(len(win_returns)) / float(len(series))
        return rate

    def _calc_loss_rate(self, series):
        loss_returns = self.__get_loss_returns(series)
        rate = float(len(loss_returns)) / float(len(series))
        return rate

    def _calc_avg_win_return(self, series):
        win_returns = self.__get_win_returns(series)
        avg = win_returns.mean()
        return avg

    def _calc_avg_loss_return(self, series):
        loss_returns = self.__get_loss_returns(series)
        avg = loss_returns.mean()
        return avg

    def _calc_winloss_ratio(self, series):
        wins = self.__get_win_returns(series)
        losses = self.__get_loss_returns(series)
        if len(losses) == 0.0:
            wl_ratio = np.nan
        else:
            wl_ratio = len(wins) / len(losses)
        return wl_ratio

    def _calc_expectancy(self, win_rates, avg_win, loss_rates, avg_loss):
        w_win = win_rates.multiply(avg_win)
        w_loss = loss_rates.multiply(avg_loss)
        exp = w_win.subtract(w_loss)
        return exp

    def generate_output(self):
        hwms = self.wealthpaths.apply(self._calc_hwm)
        lake_ratios = self._calc_lake_ratios(hwms, self.wealthpaths)
        lake_ratios.name = "Lake Ratio"

        drawdowns = self.wealthpaths.apply(self._calc_drawdowns)
        max_dds = drawdowns.min()
        max_dds.name = "Max Drawdown"

        ann_returns = self.returns_df.apply(self._calc_annualized_return)
        ann_returns.name = "Annualized Return"

        ann_stds = self.returns_df.apply(self._calc_annualized_std_dev)
        ann_stds.name = "Annualized Std Dev"

        sharpes = self._calc_sharpe(ann_returns, ann_stds)
        sharpes.name = "Sharpe Ratio"

        win_rates = self.returns_df.apply(self._calc_win_rate)
        win_rates.name = "Win Rate"

        loss_rates = self.returns_df.apply(self._calc_loss_rate)
        loss_rates.name = "Loss Rate"

        avg_win_returns = self.returns_df.apply(self._calc_avg_win_return)
        avg_win_returns.name = "Avg Win Return"

        avg_loss_returns = self.returns_df.apply(self._calc_avg_loss_return)
        avg_loss_returns.name = "Avg Loss Return"

        win_loss_ratio = self.returns_df.apply(self._calc_winloss_ratio)
        win_loss_ratio.name = "Win Loss Ratio"

        expectancy = self._calc_expectancy(win_rates, avg_win_returns,
                                           loss_rates, avg_loss_returns)
        expectancy.name = "Trade Expectancy"

        gpr = self.returns_df.apply(self._calc_gain_to_pain_ratio)
        gpr.name = 'Gain to Pain Ratio'

        output_df = pd.concat([lake_ratios, max_dds, ann_returns,
                               ann_stds, sharpes, win_rates,
                               loss_rates, avg_win_returns,
                               avg_loss_returns, expectancy,
                               gpr, ], axis=1).round(4)

        return output_df.T.sort_index()


def main():

    # Ploting the data

    data_['Close'].plot(legend=True, figsize=(15, 8))
    plt.show()

    data_['SMA1'] = data_['Close'].rolling(50).mean()
    data_.tail()
    data_['SMA2'] = data_['Close'].rolling(200).mean()
    data_.tail()

    columns = ['Close', 'SMA1', 'SMA2']
    df_ = pd.DataFrame(index=data_.index, columns=columns)
    df_['SMA1'] = data_['SMA1']
    df_['SMA2'] = data_['SMA2']
    df_['Close'] = data_['Close']

    df_.plot(title='Stock price | 50 & 200 days SMAs', figsize=(10, 6))
    plt.show()

    # Moving Averages Trading System

    df_['position'] = np.where(df_['SMA1'] > df_['SMA2'], 1, -1)
    df_.dropna(inplace=True)
    df_['position'].plot(ylim=[-1.1, 1.1], title='SMA System Positioning')
    plt.show()

    df_['returns'] = (df_['Close'] / df_['Close'].shift(1) - 1)
    df_['returns2'] = (df_['Close'] / df_['Close'].shift(1))
    df_['returns'].hist(bins=35)

    df_['strategy'] = df_['position'].shift(1) * df_['returns']
    df_['strategy2'] = df_['strategy'] + 1

    df_[['returns2', 'strategy2']].prod()

    df_[['returns2', 'strategy2']].cumprod().plot(figsize=(10, 6))
    plt.show()

    # Annual returns

    df_[['returns', 'strategy']].mean() * 252

    # Annual Standard Deviations

    df_[['returns', 'strategy']].std() * 252 ** 0.5
    df_.head()

    # RSI Calculation

    df_['RSI'] = RSI(df_['Close'], 14)
    df_.tail()

    df_['RSI'].plot(legend=True, figsize=(15, 8))
    plt.show()

    # RSI TRADING SYSTEM

    df_['position_2'] = np.where((df_['SMA1'] > df_['SMA2']) &
                                 (df_['RSI'] < 40), 1, np.nan)
    df_['position_2'] = np.where((df_['SMA1'] < df_['SMA2']) &
                                 (df_['RSI'] > 50), -1, df_['position_2'])
    df_['position_2'].fillna(method='ffill', inplace=True)
    df_['strategy_777'] = df_['position_2'].shift(1) * df_['returns']

    df_['strategy_777_2'] = df_['strategy_777'] + 1
    df_[['returns2', 'strategy_777_2']].prod()
    df_[['returns2', 'strategy_777_2']].cumprod().plot(figsize=(10, 6))
    plt.show()

    # Annual returns

    df_[['returns', 'strategy_777']].mean() * 252

    # Annual Standard Deviations

    df_[['returns', 'strategy_777']].std() * 252 ** 0.5
    df_.head()
    df_['position_2'].plot(ylim=[-1.1, 1.1], title='RSI + SMA - Positions')

    # KPIs for the moving average trading system:

    xxx = Output(df_['returns'])
    print "\n The KPIs for buy and hold strategy are: ",  xxx.generate_output()

    yyy = Output(df_['strategy'])
    print "\n KPIs - moving average trading system: ",  yyy.generate_output()

    zzz = Output(df_['strategy_777'])
    print "\n The KPIs for the RSI + SMA system are: ",  zzz.generate_output()


if __name__ == '__main__':
    main()
