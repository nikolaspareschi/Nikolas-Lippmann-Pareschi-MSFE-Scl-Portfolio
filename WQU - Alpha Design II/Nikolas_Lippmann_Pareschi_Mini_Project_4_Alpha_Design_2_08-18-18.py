# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:57:14 2018

@author: nikolasp
"""

import datetime
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


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

    # Reading the data

    df = pd.read_csv('sp500intra.csv', index_col='Date', parse_dates=True)
    df = df.set_index('Timestamp')

# Eliminating unecessary columns

    df.drop(['OpenPrice', 'HighPrice', 'LowPrice', 'TotalVolume',
             'TotalQuantity', 'TotalTradeCount'], axis=1, inplace=True)

# Making the tickers columns

    df3 = df.pivot(columns='Ticker', values='ClosePrice')

# Eliminating the timestamps without operations anc creatin in and out samples

    df5 = df3[223:615]
    df5_in_sample = df3[223:323]
    df5_out_sample = df3[324:615]
    df5 = df5.ffill()
    df5 = df5.bfill()
    df5_in_sample = df5_in_sample.ffill()
    df5_in_sample = df5_in_sample.bfill()
    df5_out_sample = df5_out_sample.ffill()
    df5_out_sample = df5_out_sample.bfill()

# Computing returns

    df7_returns = df5.pct_change()
    df7_returns_in_sample = df5_in_sample.pct_change()
    df7_returns_out_sample = df5_out_sample.pct_change()

# Creating the SP500 returns as we do not have in the file

    df7_returns['sp500'] = df7_returns.sum(axis=1)/len(df7_returns.columns)
    df7_returns_in_sample['sp500'] = (
            df7_returns_in_sample.sum(
                    axis=1)/len(df7_returns_in_sample.columns))
    df7_returns_out_sample['sp500'] = (
            df7_returns_out_sample.sum(
                    axis=1)/len(df7_returns_out_sample.columns))

    df8_in_sample = df7_returns_in_sample.corr()
    df8_in_sample['sp500'].sort_values(ascending=True)
    print df8_in_sample['sp500'].sort_values(ascending=True)

# The Stock with highest intraday correlation in the in sample history is FLR

    df7_returns_out_sample['pair'] = (df7_returns_out_sample['sp500']
                                      - df7_returns_out_sample['FLR'])
    df7_returns_out_sample['pair'].tail()
    df7_returns_out_sample['pair'].mean()
    df7_returns_out_sample['pair'].std()

# Trade system logic. Buy above -X std below mean, sell X std above mean. Flat
# Other wise

    df7_returns_out_sample['position'] = (np.where(
            df7_returns_out_sample['pair'] >
            (df7_returns_out_sample['pair'].mean()
             + 1*df7_returns_out_sample['pair'].std()), -1, 0))
    df7_returns_out_sample['position'] = (np.where(
            df7_returns_out_sample['pair'] <
            (df7_returns_out_sample['pair'].mean()
             - 1*df7_returns_out_sample['pair'].std()), 1,
            df7_returns_out_sample['position']))

# Plot of returns and positions

    df7_returns_out_sample['position'].plot(
            title="Strategy positioning - Out Sample", figsize=(10, 6))
    plt.show()

    df7_returns_out_sample['strategy'] = (
            df7_returns_out_sample['position'].shift(1) *
            df7_returns_out_sample['pair'])
    df7_returns_out_sample['strategy2'] = (
            df7_returns_out_sample['strategy'] + 1)
    df7_returns_out_sample['strategy2'].cumprod().plot(
            title="Cummulative Out Sample ROI for our Index Arbitrage System")
    plt.show()

    df7_returns_out_sample['strategy'].index = (
            pd.Timestamp(datetime.date.today()) +
            pd.TimedeltaIndex(
                    df7_returns_out_sample['strategy'].index, unit='s'))

# KPIs calculation

    df7_returns_out_sample['strategy'].dropna()
    xx = Output(df7_returns_out_sample['strategy'])
    print "\nThe KPIs for our index arbitrage system:\n", xx.generate_output()


if __name__ == '__main__':
    main()
