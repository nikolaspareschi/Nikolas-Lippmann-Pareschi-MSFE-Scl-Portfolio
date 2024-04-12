# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:07:11 2018

@author: Nikolas
"""

# For data manipulation

import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
import numpy as np
import datetime as dt
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr



# These functions will be used in the main function. They are used to get
# the fundamental data from finviz.


def get_fundamental_data(df):
    for symbol in df.index:
        try:
            url = ("http://finviz.com/quote.ashx?t=" + symbol.lower())
            soup = bs(requests.get(url).content, features='html5lib')
            for m in df.columns:
                df.loc[symbol, m] = fundamental_metric(soup, m)
        except Exception as e:
            print (symbol, 'not found')
    return df


def fundamental_metric(soup, metric):
    return soup.find(text=metric).find_next(class_='snapshot-td2').text


# Global Variables Declaration

stock_list = ['ABT', 'ACN', 'ADBE', 'AES', 'AET', 'AFL', 'A', 'APD', 'AKAM',
              'ALXN', 'ALLE', 'AGN', 'ADS', 'ALL', 'AEE', 'AXP', 'AIG', 'AMT',
              'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI', 'ANDV', 'ANTM',
              'AON', 'AIV', 'AMAT', 'BAK', 'BK', 'T', 'ADSK', 'BDX', 'BLK',
              'GS', 'BSX', 'AMZN', 'GOOG', 'PG', 'KO', 'IBM', 'DG', 'COG',
              'CBRE', 'CBS', 'CELG', 'CF', 'CI', 'CSCO', 'C', 'KO', 'CTSH',
              'CMA', 'CAG', 'COST', 'CCI', 'STZ', 'DVA', 'DAL', 'DOV', 'DWDP',
              'DUK', 'ETFC', 'EMN', 'ECL', 'EW', 'EA', 'EMR', 'EOG', 'EQT',
              'EXPE', 'EXPD', 'ESRX', 'EXR', 'FB', 'FIS', 'FISV', 'FLIR',
              'FLS', 'FLR', 'GRMN', 'GGP', 'GIS', 'GM', 'GPC', 'GILD', 'GPN',
              'GWW', 'HBI', 'HRS', 'HAS', 'HSIC', 'XOM', 'KO', 'PEP', 'MT',
              'NL', 'ALDW', 'DCM', 'GSB', 'LPL']

metric = ['P/B', 'P/E', 'EPS (ttm)', 'ROI', 'P/FCF']

# We will use this class to compute our KPIs. This class was provided in
# Piazza by Professor Steven Stelk in Risk Management classes


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
            total_loss_return_series = (1.0 + loss_returns_series).cumprod().subtract(1.0)
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

    yf.pdr_override()
    df = pd.DataFrame(index=stock_list, columns=metric)
    df = get_fundamental_data(df)
    print df.head()
    df['ROI'].replace('-', regex=True, inplace=True)
    df['ROI'].replace('%', '', regex=True, inplace=True)
    df['P/B'].replace('-', regex=False, inplace=True)
    df['P/E'].replace('-', regex=False, inplace=True)
    df['P/FCF'].replace('-', regex=True, inplace=True)
    df['EPS (ttm)'].replace('-', regex=True, inplace=True)
    df_top_decile = df[(df['P/E'].astype(float) < 35) &
                       (df['P/B'].astype(float) < 5) &
                       (df['ROI'].astype(float) > 8) &
                       (df['P/FCF'].astype(float) < 35) &
                       (df['EPS (ttm)'].astype(float) > 1)]

    df_bottom_decile = df[(df['P/E'].astype(float) > 15) &
                          (df['P/B'].astype(float) > 3) &
                          (df['ROI'].astype(float) < 8) &
                          (df['P/FCF'].astype(float) > 15) &
                          (df['EPS (ttm)'].astype(float) < 5)]
    top_stocks = pdr.get_data_yahoo(list(df_top_decile.index),
                                    start="2017-07-31", end="2018-07-31")
    bottom_stocks = pdr.get_data_yahoo(list(df_bottom_decile.index),
                                       start="2017-07-31", end="2018-07-31")
    top = top_stocks['Close']
    bottom = bottom_stocks['Close']
    top_daily_returns = top.resample('D').last().pct_change().dropna()
    top_daily_returns['Portfolio'] = top_daily_returns.sum(axis=1)/19
    xxx = Output(top_daily_returns)
    print "\n The KPIs for our top stocks and the portfolio with 19 stocks"
    print "are: \n\n", xxx.generate_output()
    bottom_daily_returns = bottom.resample('D').last().pct_change().dropna()
    bottom_daily_returns['Portfolio'] = bottom_daily_returns.sum(axis=1)/14
    yyy = Output(bottom_daily_returns)
    print "\n The KPIs for our bottom stocks and the portfolio with 14"
    print "stocks are: \n\n", yyy.generate_output()


if __name__ == '__main__':
    main()
