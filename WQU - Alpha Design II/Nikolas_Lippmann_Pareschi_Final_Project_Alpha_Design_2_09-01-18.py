# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 11:43:04 2018

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

    # 25 years Of stocks
    yf.pdr_override()

    tickers = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS',
               'DWDP', 'XOM', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ',
               'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UTX',
               'UNH', 'VZ', 'V', 'WMT']

    data = pdr.get_data_yahoo(tickers, start="1993-07-31", end="2018-07-31")
    df = data
    n = 5

    # ATR calculation

    df['n_day_high'] = df['Close'].shift(1).rolling(window=n).max()
    df['n_day_low'] = df['Close'].shift(1).rolling(window=n).min()
    df['avg'] = df['Close'].shift(1).rolling(window=n).mean()

    df['H-L'] = abs(df['High']-df['Low'])
    df['H-PC'] = abs(df['High']-df['Close'].shift(1))
    df['L-PC'] = abs(df['Low']-df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=0)
    df['ATR'] = df['TR'].rolling(window=n).mean()

    df['21ma'] = df['Close'].rolling(window=21).mean()
    df['45ma'] = df['Close'].rolling(window=45).mean()

    # Trade System 1

    df['position1'] = np.nan
    df['position1'] = (np.where(df['Close'] > (df['21ma'] + 0.5*df['ATR']),
                       1, df['position1']))
    df['position1'] = (np.where((df['Close'] < df['21ma']) & (
                        df['position1'].shift(1) == 1), 0, df['position1']))
    df['position1'] = (np.where(df['Close'] < (
                       df['21ma'] - 0.5*df['ATR']), -1, df['position1']))
    df['position1'] = (np.where((df['Close'] > df['21ma']) & (
                       df['position1'].shift(1) == -1), 0, df['position1']))
    df['position1'] = df['position1'].ffill()
    df['position1'].plot(title="Trade System 1 positioning", figsize=(20, 12))
    plt.show()

    # Trade System 2

    df['position2'] = np.nan
    df['position2'] = (np.where(df['Close'] > (
            df['45ma'] + 0.5*df['ATR']), 1, df['position2']))
    df['position2'] = (np.where((df['Close'] < df['45ma']) & (
            df['position2'].shift(1) == 1), 0, df['position2']))
    df['position2'] = (np.where(df['Close'] < (
            df['45ma'] - 0.5*df['ATR']), -1, df['position2']))
    df['position2'] = (np.where((df['Close'] > df['45ma']) & (
            df['position2'].shift(1) == -1), 0, df['position2']))
    df['position2'] = df['position2'].ffill()
    df['position2'].plot(title="Trade System 2 positioning", figsize=(20, 12))
    plt.show()

    # Computing returns - Money Market - Trade System 1

    df['returns'] = df['Close'].pct_change()
    df['returns2'] = df['Close'].pct_change()
    df['strategy1'] = (df['position1'].shift(1) * df['returns'])
    df['strategy_capital1'] = df['strategy1'] + 1
    df['Money_Market_1'] = df['strategy_capital1'].cumprod()*10000
    df['Money_Market_1'].plot(title="Cummulative ROI $ 10.0000 capital\
  for our Trading System 1 - Money Market- All data", figsize=(20, 12))
    plt.show()

    # Computing returns - Money Market - Trade System 2

    df['returns'] = df['Close'].pct_change()
    df['returns2'] = df['Close'].pct_change()
    df['strategy2'] = (df['position2'].shift(1) * df['returns'])
    df['strategy_capital2'] = df['strategy2'] + 1
    df['Money_Market_2'] = df['strategy_capital2'].cumprod()*10000
    df['Money_Market_2'].plot(title="Cummulative ROI $ 10.0000 capital\
  for our Trading System 2 - Money Market- All data", figsize=(20, 12))
    plt.show()

    # Multiple Tiers System 1

    df['position1_mul_tier'] = (
                np.where((df['position1'].shift(4) == 1) & (
                        df['position1'].shift(3) == 1) & (
                                df['position1'].shift(2) == 1) & (
                                        df['position1'].shift(1) == 1) & (
                                                df['position1'] == 1), 2,
                                                              df['position1']))
    df['position1_mul_tier'] = (
                np.where(
                        (df['position1'].shift(4) == -1) & (
                                df['position1'].shift(3) == -1) & (
                                        df['position1'].shift(2) == -1) & (
                                            df['position1'].shift(1) == -1) & (
                                                        df['position1'] == -1),
                            -2, df['position1_mul_tier']))

    df['position1_mul_tier'].plot(title="Trade System 1\
 Multiple Tiers Positioning", figsize=(20, 12))
    plt.show()

    df['s1_multiple_tier'] = (
                             df['position1_mul_tier'].shift(1) * df['returns'])
    df['s1_multiple_tier_capital'] = df['s1_multiple_tier'] + 1
    df['s1_multiple_tier_capital'] .cumprod(
                ).plot(
                    title="Cummulative\
  ROI for Trade System 1 with Multiple Tiers", figsize=(20, 12))
    plt.show()

    # Multiple Tiers System 2

    df['position2_mul_tier'] = (
                np.where((df['position2'].shift(4) == 1) & (
                        df['position2'].shift(3) == 1) & (
                                df['position2'].shift(2) == 1) & (
                                        df['position2'].shift(1) == 1) & (
                                                df['position2'] == 1), 2,
                                                           df['position2']))
    df['position2_mul_tier'] = (
                np.where((df['position2'].shift(4) == -1) & (
                        df['position2'].shift(3) == -1) & (
                                df['position2'].shift(2) == -1) & (
                                        df['position2'].shift(1) == -1) & (
                                                df['position2'] == -1), -2,
                                                     df['position2_mul_tier']))

    df['position2_mul_tier'].plot(title="Cummulative\
 ROI for Trade System 2 with Multiple Tiers", figsize=(20, 12))
    plt.show()

    # Percent Volatility Model - System 1

    df['returns'] = df['Close'].pct_change()
    df['vol_21'] = df['Close'].rolling(21).std()
    df['returns2'] = df['Close'].pct_change()
    df['strategy1_vol'] = (
            df['position1'].shift(1) * df['returns']/df['vol_21'])
    df['strategy_capital1_vol'] = df['strategy1_vol'] + 1
    df['Money_Market_1_vol'] = df['strategy_capital1_vol'].cumprod()*10000
    df['Money_Market_1_vol'].plot(title="Cummulative\
 ROI for Trading System 1 with Volatility Model", figsize=(20, 12))
    plt.show()

    # Percent Volatility Model - System 2

    df['returns'] = df['Close'].pct_change()
    df['vol_45'] = df['Close'].rolling(45).std()
    df['returns2'] = df['Close'].pct_change()
    df['strategy2_vol'] = (
            df['position2'].shift(1) * df['returns']/df['vol_45'])
    df['strategy_capital2_vol'] = df['strategy2_vol'] + 1
    df['Money_Market_2_vol'] = df['strategy_capital2_vol'].cumprod()*10000
    df['Money_Market_2_vol'].plot(title="Cummulative\
 ROI for Trading System 2 with Volatility Model", figsize=(20, 12))
    plt.show()

    # Upright Pyramid - System 1

    df['position1_pyramid'] = (np.where((df['position1'].shift(1) == 1) & (
                                                df['position1'] == 1), 1.5,
                               df['position1']))

    df['position1_pyramid'] = (
            np.where((df['position1_pyramid'].shift(2) == 1) & (
                    df['position1_pyramid'].shift(1) == 1.5) & (
                            df['position1_pyramid'] == 1.5), 1.8,
                               df['position1_pyramid']))

    df['position1_pyramid'] = (np.where((df['position1'].shift(1) == -1) & (
                                                df['position1'] == -1), -1.5,
                               df['position1_pyramid']))

    df['position1_pyramid'] = (
            np.where((df['position1_pyramid'].shift(2) == -1) & (
                    df['position1_pyramid'].shift(1) == -1.5) & (
                            df['position1_pyramid'] == -1.5), -1.8,
                               df['position1_pyramid']))

    df['position1_pyramid'].plot(title="Trade\
 System 1 upright pyramid positioning", figsize=(20, 12))
    plt.show()
    # Inverted Pyramid - System 1

    df['position1_pyramid_inv'] = (np.where((df['position1'].shift(1) == 1) & (
                                                df['position1'] == 1), 2,
                                   df['position1']))

    df['position1_pyramid_inv'] = (
            np.where((df['position1_pyramid_inv'].shift(2) == 1) & (
                    df['position1_pyramid_inv'].shift(1) == 2) & (
                            df['position1_pyramid_inv'] == 2), 3,
                                   df['position1_pyramid_inv']))

    df['position1_pyramid_inv'] = (
            np.where((df['position1'].shift(1) == -1) & (
                                                df['position1'] == -1), -2,
                     df['position1_pyramid_inv']))

    df['position1_pyramid_inv'] = (
            np.where((df['position1_pyramid_inv'].shift(2) == -1) & (
                    df['position1_pyramid_inv'].shift(1) == -2) & (
                            df['position1_pyramid_inv'] == -2), -3,
                                  df['position1_pyramid_inv']))

    df['position1_pyramid_inv'].plot(title="Trade\
 System 1 inverted pyramid positioning", figsize=(20, 12))
    plt.show()

    # Reflected Pyramid - System 1

    df['position1_pyramid_ref'] = (np.where((df['position1'].shift(1) == 1) & (
                                                df['position1'] == 1), 2,
                                   df['position1']))

    df['position1_pyramid_ref'] = (
            np.where((df['position1_pyramid_ref'].shift(2) == 1) & (
                    df['position1_pyramid_ref'].shift(1) == 2) & (
                            df['position1_pyramid_ref'] == 2), 1.5,
                                   df['position1_pyramid_ref']))

    df['position1_pyramid_ref'] = (
            np.where((df['position1'].shift(1) == -1) & (
                                                df['position1'] == -1), -2,
                     df['position1_pyramid_ref']))

    df['position1_pyramid_ref'] = (
            np.where((df['position1_pyramid_ref'].shift(2) == -1) & (
                    df['position1_pyramid_ref'].shift(1) == -2) & (
                            df['position1_pyramid_ref'] == -2), -1.5,
                                  df['position1_pyramid_ref']))

    df['position1_pyramid_ref'].plot(title="Trade\
 System 1 reflected pyramid positioning", figsize=(20, 12))
    plt.show()

    # Now let's apply the upright pyramid in each
    # sub-system 1 (money market, tiers and volatility)

    # Percent Volatility Model with upright pyramid - System 1

    df['returns'] = df['Close'].pct_change()
    df['vol_21'] = df['Close'].rolling(21).std()
    df['returns2'] = df['Close'].pct_change()
    df['strategy1_vol_pyr_up'] = (
            df['position1_pyramid'].shift(1) * df['returns']/df['vol_21'])
    df['strategy_capital1_vol_pyr_up'] = df['strategy1_vol_pyr_up'] + 1
    df['Money_Market_1_vol_pyr_up'] = (
            df['strategy_capital1_vol_pyr_up'].cumprod()*10000)
    df['Money_Market_1_vol_pyr_up'].plot(
            title="Cummulative\
 ROI Trading System 1 - Volatility Model - Upright Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy1_vol_pyr_up']['portfolio'] = df['strategy1_vol_pyr_up'].sum(
            axis=1)/len(df['strategy1_vol_pyr_up'].columns)
    zz_insample_strategy1_vol_pyr_up = Output(
            df['strategy1_vol_pyr_up']['portfolio'][0:len(df)*3/5])
    zz_insample_strategy1_vol_pyr_up.generate_output()
    print "\nThe KPIs\
 for Trading System 1 - Volatility Model - Upright Pyramid in the in sample \
    data are:\n", zz_insample_strategy1_vol_pyr_up.generate_output()

    sns.heatmap(zz_insample_strategy1_vol_pyr_up.generate_output())
    plt.show()

    # Money Market Model with upright pyramid - System 1

    df['returns'] = df['Close'].pct_change()
    df['returns2'] = df['Close'].pct_change()
    df['strategy1_money_market_up_pyr'] = (
            df['position1_pyramid'].shift(1) * df['returns'])
    df['strategy_capital1_money_market_up_pyr'] = (
            df['strategy1_money_market_up_pyr'] + 1)
    df['Money_Market_1_up_pyr'] = (
            df['strategy_capital1_money_market_up_pyr'].cumprod()*10000)
    df['Money_Market_1_up_pyr'].plot(
            title="Cummulative ROI Trading System 1 - Market Model - Upright\
 Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy1_money_market_up_pyr']['portfolio'] = (
            df['strategy1_money_market_up_pyr'].sum(
                    axis=1)/len(df['strategy1_money_market_up_pyr'].columns))
    zz_insample_strategy1_money_market_up_pyr = Output(
            df['strategy1_money_market_up_pyr']['portfolio'][0:len(df)*3/5])
    zz_insample_strategy1_money_market_up_pyr.generate_output()
    print "\nThe KPIs for Trading System 1 - Market Model - Upright Pyramid in \
 the insample data \
 are:\n", zz_insample_strategy1_money_market_up_pyr.generate_output()

    sns.heatmap(zz_insample_strategy1_money_market_up_pyr.generate_output())

    # Multiple Tier with upright pyramid - System 1

    df['returns'] = df['Close'].pct_change()
    df['returns2'] = df['Close'].pct_change()
    df['strategy1_mul_tier_up_pyr'] = (
            (
                    df['position1_pyramid']+df['position1_mul_tier']).shift(
                            1) * df['returns'])
    df['strategy_capital1_money_market_up_pyr'] = (
            df['strategy1_mul_tier_up_pyr'] + 1)
    df['Money_Market_1_up_pyr'] = (
            df['strategy_capital1_money_market_up_pyr'].cumprod()*10000)
    df['Money_Market_1_up_pyr'].plot(
            title="Cummulative ROI Trading System 1 - Multiple Tier - Upright\
 Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy1_mul_tier_up_pyr']['portfolio'] = (
            df['strategy1_mul_tier_up_pyr'].sum(
                    axis=1)/len(df['strategy1_mul_tier_up_pyr'].columns))
    zz_insample_strategy1_mul_tier_up_pyr = Output(
            df['strategy1_mul_tier_up_pyr']['portfolio'][0:len(df)*3/5])
    zz_insample_strategy1_mul_tier_up_pyr.generate_output()
    print "\nThe KPIs for Trading System 1 - Multiple Tier - Upright Pyramid \
 in the insample data are:\n",
    print zz_insample_strategy1_mul_tier_up_pyr.generate_output()
    sns.heatmap(zz_insample_strategy1_mul_tier_up_pyr.generate_output())

    # Now let's apply the inverted pyramid in each sub-system 1 (money market,
    # tiers and volatility)

    # Percent Volatility Model with inverted pyramid - System 1

    df['returns'] = df['Close'].pct_change()
    df['vol_21'] = df['Close'].rolling(21).std()
    df['returns2'] = df['Close'].pct_change()
    df['strategy1_vol_pyr_inv'] = (
            df['position1_pyramid_inv'].shift(1) * df['returns']/df['vol_21'])
    df['strategy_capital1_vol_pyr_inv'] = df['strategy1_vol_pyr_inv'] + 1
    df['Money_Market_1_vol_pyr_inv'] = (
            df['strategy_capital1_vol_pyr_inv'].cumprod()*10000)
    df['Money_Market_1_vol_pyr_inv'].plot(
            title="Cummulative ROI Trading System 1 - Volatility Model -\
 Inverted Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy1_vol_pyr_inv']['portfolio'] = (
            df['strategy1_vol_pyr_inv'].sum(
                    axis=1)/len(df['strategy1_vol_pyr_inv'].columns))
    zz_insample_strategy1_vol_pyr_inv = Output(
            df['strategy1_vol_pyr_inv']['portfolio'][0:len(df)*3/5])
    zz_insample_strategy1_vol_pyr_inv.generate_output()
    print "\nThe KPIs for Trading System 1 - Volatility Model - Inverted\
 Pyramid in the insample data \
 are:\n", zz_insample_strategy1_vol_pyr_inv.generate_output()
    sns.heatmap(zz_insample_strategy1_vol_pyr_inv.generate_output())

    # Money Market Model with inverted pyramid - System 1

    df['returns'] = df['Close'].pct_change()
    df['returns2'] = df['Close'].pct_change()
    df['strategy1_money_market_inv_pyr'] = (
            df['position1_pyramid_inv'].shift(1) * df['returns'])
    df['strategy_capital1_money_market_inv_pyr'] = (
            df['strategy1_money_market_inv_pyr'] + 1)
    df['Money_Market_1_inv_pyr'] = (
            df['strategy_capital1_money_market_inv_pyr'].cumprod()*10000)
    df['Money_Market_1_inv_pyr'].plot(
            title="Cummulative ROI Trading System 1 - Money Market Model -\
 Inverted Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy1_money_market_inv_pyr']['portfolio'] = (
            df['strategy1_money_market_inv_pyr'].sum(axis=1)/len(
                    df['strategy1_money_market_inv_pyr'].columns))
    zz_insample_strategy1_money_market_inv_pyr = Output(
            df['strategy1_money_market_inv_pyr']['portfolio'][0:len(df)*3/5])
    zz_insample_strategy1_money_market_inv_pyr.generate_output()
    print "\nThe KPIs for Trading System 1 - Money Market Model - Inverted\
 Pyramid in the insample data\
 are:\n", zz_insample_strategy1_money_market_inv_pyr.generate_output()

    sns.heatmap(zz_insample_strategy1_money_market_inv_pyr.generate_output())

    # Multiple Tier with inverted pyramid - System 1

    df['returns'] = df['Close'].pct_change()
    df['returns2'] = df['Close'].pct_change()
    df['strategy1_mul_tier_inv_pyr'] = (
            (df['position1_pyramid_inv']+df['position1_mul_tier']).shift(
                    1) * df['returns'])
    df['strategy_capital1_money_market_inv_pyr'] = (
            df['strategy1_mul_tier_inv_pyr'] + 1)
    df['Money_Market_1_inv_pyr'] = (
            df['strategy_capital1_money_market_inv_pyr'].cumprod()*10000)
    df['Money_Market_1_inv_pyr'].plot(
            title="Cummulative ROI Trading System 1 - Multiple Tier Model -\
 Inverted Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy1_mul_tier_inv_pyr']['portfolio'] = (
            df['strategy1_mul_tier_inv_pyr'].sum(axis=1)/len(
                    df['strategy1_mul_tier_inv_pyr'].columns))
    zz_insample_strategy1_mul_tier_inv_pyr = Output(
            df['strategy1_mul_tier_inv_pyr']['portfolio'][0:len(df)*3/5])
    zz_insample_strategy1_mul_tier_inv_pyr.generate_output()
    print "\nThe KPIs for Trading System 1 - Multiple Tier Model - Inverted\
 Pyramid in the insample data \
 are:\n", zz_insample_strategy1_mul_tier_inv_pyr.generate_output()
    sns.heatmap(zz_insample_strategy1_mul_tier_inv_pyr.generate_output())

    # Now let's apply the reflected pyramid in each sub-system 1
    # (money market, tiers and volatility)

    # Percent Volatility Model with reflected pyramid - System 1

    df['returns'] = df['Close'].pct_change()
    df['vol_21'] = df['Close'].rolling(21).std()
    df['returns2'] = df['Close'].pct_change()
    df['strategy1_vol_pyr_ref'] = (
            df['position1_pyramid_ref'].shift(1) * df['returns']/df['vol_21'])
    df['strategy_capital1_vol_pyr_ref'] = df['strategy1_vol_pyr_ref'] + 1
    df['Money_Market_1_vol_pyr_ref'] = (
            df['strategy_capital1_vol_pyr_ref'].cumprod()*10000)
    df['Money_Market_1_vol_pyr_ref'].plot(
            title="Cummulative ROI Trading System 1 - Volatility Model\
 - Reflected Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy1_vol_pyr_ref']['portfolio'] = df['strategy1_vol_pyr_ref'].sum(
            axis=1)/len(df['strategy1_vol_pyr_ref'].columns)
    zz_insample_strategy1_vol_pyr_ref = Output(
            df['strategy1_vol_pyr_ref']['portfolio'][0:len(df)*3/5])
    zz_insample_strategy1_vol_pyr_ref.generate_output()
    print "\nThe KPIs for Trading System 1 - Volatility Model - Reflected\
 Pyramid in the insample data \
 are:\n", zz_insample_strategy1_vol_pyr_ref.generate_output()
    sns.heatmap(zz_insample_strategy1_vol_pyr_ref.generate_output())

    # Money Market Model with reflected pyramid - System 1

    df['returns'] = df['Close'].pct_change()
    df['returns2'] = df['Close'].pct_change()
    df['strategy1_money_market_ref_pyr'] = (
            df['position1_pyramid_ref'].shift(1) * df['returns'])
    df['strategy_capital1_money_market_ref_pyr'] = (
            df['strategy1_money_market_ref_pyr'] + 1)
    df['Money_Market_1_ref_pyr'] = (
            df['strategy_capital1_money_market_ref_pyr'].cumprod()*10000)
    df['Money_Market_1_ref_pyr'].plot(
            title="Cummulative ROI Trading System 1 - Money Market Model -\
 Reflected Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy1_money_market_ref_pyr']['portfolio'] = (
            df['strategy1_money_market_ref_pyr'].sum(
                    axis=1)/len(df['strategy1_money_market_ref_pyr'].columns))
    zz_insample_strategy1_money_market_ref_pyr = Output(
            df['strategy1_money_market_ref_pyr']['portfolio'][0:len(df)*3/5])
    zz_insample_strategy1_money_market_ref_pyr.generate_output()
    print "\nThe KPIs for Trading System 1 - Money Market Model - Reflected\
 Pyramid in the insample data \
 are:\n", zz_insample_strategy1_money_market_ref_pyr.generate_output()
    sns.heatmap(zz_insample_strategy1_money_market_ref_pyr.generate_output())

    # Multiple Tier with reflected pyramid - System 1

    df['returns'] = df['Close'].pct_change()
    df['returns2'] = df['Close'].pct_change()
    df['strategy1_mul_tier_ref_pyr'] = (
            (df['position1_pyramid_ref']+df['position1_mul_tier']).shift(
                    1) * df['returns'])
    df['strategy_capital1_money_market_ref_pyr'] = (
            df['strategy1_mul_tier_ref_pyr'] + 1)
    df['Money_Market_1_ref_pyr'] = (
            df['strategy_capital1_money_market_ref_pyr'].cumprod()*10000)
    df['Money_Market_1_ref_pyr'].plot(
            title="Cummulative ROI Trading System 1 - Multiple Tier Model\
 - Reflected Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy1_mul_tier_ref_pyr']['portfolio'] = (
            df['strategy1_mul_tier_ref_pyr'].sum(
                    axis=1)/len(df['strategy1_mul_tier_ref_pyr'].columns))
    zz_insample_strategy1_mul_tier_ref_pyr = Output(
            df['strategy1_mul_tier_ref_pyr']['portfolio'][0:len(df)*3/5])
    zz_insample_strategy1_mul_tier_ref_pyr.generate_output()
    print "\nThe KPIs for Trading System 1 - Multiple Tier Model - Reflected \
 Pyramid in the insample data \
 are:\n", zz_insample_strategy1_mul_tier_ref_pyr.generate_output()
    sns.heatmap(zz_insample_strategy1_mul_tier_ref_pyr.generate_output())

    # Upright Pyramid - System 2

    df['position2_pyramid'] = (np.where((df['position2'].shift(1) == 1) & (
                                                df['position2'] == 1), 1.5,
                               df['position2']))

    df['position2_pyramid'] = (
            np.where((df['position2_pyramid'].shift(2) == 1) & (
                    df['position2_pyramid'].shift(1) == 1.5) & (
                            df['position2_pyramid'] == 1.5), 1.8,
                    df['position2_pyramid']))

    df['position2_pyramid'] = (np.where((df['position2'].shift(1) == -1) & (
                                                df['position2'] == -1), -1.5,
                               df['position2_pyramid']))

    df['position2_pyramid'] = (
            np.where((df['position2_pyramid'].shift(2) == -1) & (
                    df['position2_pyramid'].shift(1) == -1.5) & (
                            df['position2_pyramid'] == -1.5), -1.8,
                      df['position2_pyramid']))

    df['position2_pyramid'].plot(
            title="Trade System 2 upright pyramid\
 positioning", figsize=(20, 12))
    plt.show()

    # Inverted Pyramid - System 2

    df['position2_pyramid_inv2'] = (
            np.where((df['position2'].shift(1) == 1) & (
                                                df['position2'] == 1), 2,
                     df['position2']))

    df['position2_pyramid_inv2'] = (
            np.where((df['position2_pyramid_inv2'].shift(2) == 1) & (
                    df['position2_pyramid_inv2'].shift(1) == 2) & (
                            df['position2_pyramid_inv2'] == 2), 3,
                       df['position2_pyramid_inv2']))

    df['position2_pyramid_inv2'] = (
            np.where((df['position2'].shift(1) == -1) & (
                                                df['position2'] == -1), -2,
                     df['position2_pyramid_inv2']))

    df['position2_pyramid_inv2'] = (
            np.where((df['position2_pyramid_inv2'].shift(2) == -1) & (
                    df['position2_pyramid_inv2'].shift(1) == -2) & (
                            df['position2_pyramid_inv2'] == -2), -3,
                     df['position2_pyramid_inv2']))

    df['position2_pyramid_inv2'].plot(
            title="Trade System 2 inverted pyramid positioning", figsize=(
                    20, 12))
    plt.show()

    # Reflected Pyramid - System 2

    df['position2_pyramid_ref'] = (np.where((df['position2'].shift(1) == 1) & (
                                                df['position2'] == 1), 2,
                                   df['position2']))

    df['position2_pyramid_ref'] = (
            np.where((df['position2_pyramid_ref'].shift(2) == 1) & (
                    df['position2_pyramid_ref'].shift(1) == 2) & (
                            df['position2_pyramid_ref'] == 2), 1.5,
                                   df['position2_pyramid_ref']))

    df['position2_pyramid_ref'] = (np.where((df['position2'].shift(
            1) == -1) & (
                                                df['position2'] == -1), -2,
                                   df['position2_pyramid_ref']))

    df['position2_pyramid_ref'] = (
            np.where((df['position2_pyramid_ref'].shift(2) == -1) & (
                    df['position2_pyramid_ref'].shift(1) == -2) & (
                            df['position2_pyramid_ref'] == -2), -1.5,
                     df['position2_pyramid_ref']))

    df['position2_pyramid_ref'].plot(
            title="Trade System 2 reflected pyramid positioning", figsize=(
                    20, 12))
    plt.show()

    # Now let's apply the upright pyramid in each sub-system 2
    # (money market, tiers and volatility)

    # Percent Volatility Model with upright pyramid - System 2

    df['returns'] = df['Close'].pct_change()
    df['vol_45'] = df['Close'].rolling(45).std()
    df['returns2'] = df['Close'].pct_change()
    df['strategy2_vol_pyr_up'] = (
            df['position2_pyramid'].shift(1) * df['returns']/df['vol_45'])
    df['strategy_capital2_vol_pyr_up'] = df['strategy2_vol_pyr_up'] + 1
    df['Money_Market_2_vol_pyr_up'] = (
            df['strategy_capital2_vol_pyr_up'].cumprod()*10000)
    df['Money_Market_2_vol_pyr_up'].plot(
            title="Cummulative ROI - Trading System 2 - Percent Volatility \
 - Upright Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy2_vol_pyr_up']['portfolio'] = df['strategy2_vol_pyr_up'].sum(
            axis=1)/len(df['strategy2_vol_pyr_up'].columns)
    zz_insample_strategy2_vol_pyr_up = Output(
            df['strategy2_vol_pyr_up']['portfolio'][0:len(df)*5/5])
    zz_insample_strategy2_vol_pyr_up.generate_output()
    print "\nThe KPIs for Trading System 2 - Percent Volatility - Upright \
 Pyramid in the insample data \
 are:\n", zz_insample_strategy2_vol_pyr_up.generate_output()
    sns.heatmap(zz_insample_strategy2_vol_pyr_up.generate_output())

    # Money Market Model with upright pyramid - System 2

    df['returns'] = df['Close'].pct_change()
    df['returns2'] = df['Close'].pct_change()
    df['strategy2_money_market_up_pyr'] = (
            df['position2_pyramid'].shift(1) * df['returns'])
    df['strategy_capital2_money_market_up_pyr'] = (
            df['strategy2_money_market_up_pyr'] + 1)
    df['Money_Market_2_up_pyr'] = (
            df['strategy_capital2_money_market_up_pyr'].cumprod()*10000)
    df['Money_Market_2_up_pyr'].plot(title="Cummulative ROI - Trading System 2\
 - Money Market Model - Upright Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy2_money_market_up_pyr']['portfolio'] = (
            df['strategy2_money_market_up_pyr'].sum(
                    axis=1)/len(df['strategy2_money_market_up_pyr'].columns))
    zz_insample_strategy2_money_market_up_pyr = Output(
            df['strategy2_money_market_up_pyr']['portfolio'][0:len(df)*4/5])
    zz_insample_strategy2_money_market_up_pyr.generate_output()
    print "\nThe KPIs for Trading System 2 - Money Market Model - Upright\
 Pyramid in the insample data\
 are:\n", zz_insample_strategy2_money_market_up_pyr.generate_output()
    sns.heatmap(zz_insample_strategy2_money_market_up_pyr.generate_output())

    # Multiple Tier with upright pyramid - System 2

    df['position2_mul_tier']
    df['returns'] = df['Close'].pct_change()
    df['returns2'] = df['Close'].pct_change()
    df['strategy2_mul_tier_up_pyr'] = (
            (df['position2_pyramid']+df['position2_mul_tier']).shift(
                    1) * df['returns'])
    df['strategy_capital2_money_market_up_pyr'] = (
            df['strategy2_mul_tier_up_pyr'] + 1)
    df['Money_Market_2_up_pyr'] = (
            df['strategy_capital2_money_market_up_pyr'].cumprod()*10000)
    df['Money_Market_2_up_pyr'].plot(
            title="Cummulative ROI - Trading System 2 - Multiple Tier -\
 Upright Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy2_mul_tier_up_pyr']['portfolio'] = (
            df['strategy2_mul_tier_up_pyr'].sum(
                    axis=1)/len(df['strategy2_mul_tier_up_pyr'].columns))
    zz_insample_strategy2_mul_tier_up_pyr = Output(
            df['strategy2_mul_tier_up_pyr']['portfolio'][0:len(df)*4/5])
    zz_insample_strategy2_mul_tier_up_pyr.generate_output()
    print "\nThe KPIs for Trading System 2 - Multiple Tier - Upright Pyramid \
 in the insample data \
 are:\n", zz_insample_strategy2_mul_tier_up_pyr.generate_output()
    sns.heatmap(zz_insample_strategy2_mul_tier_up_pyr.generate_output())

    # Now let's apply the inverted pyramid in each sub-system 2
    # (money market, tiers and volatility)

    # Percent Volatility Model with inverted pyramid - System 2

    df['returns'] = df['Close'].pct_change()
    df['vol_45'] = df['Close'].rolling(45).std()
    df['returns2'] = df['Close'].pct_change()
    df['strategy2_vol_pyr_inv'] = (
            df['position2_pyramid_inv2'].shift(1) * df['returns']/df['vol_45'])
    df['strategy_capital2_vol_pyr_inv'] = df['strategy2_vol_pyr_inv'] + 1
    df['Money_Market_2_vol_pyr_inv'] = (
            df['strategy_capital2_vol_pyr_inv'].cumprod()*10000)
    df['Money_Market_2_vol_pyr_inv'].plot(
            title="Cummulative ROI - Trading System 2 - Percent Volatility\
 - Inverted Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy2_vol_pyr_inv']['portfolio'] = (
            df['strategy2_vol_pyr_inv'].sum(axis=1)/len(
                    df['strategy2_vol_pyr_inv'].columns))
    zz_insample_strategy2_vol_pyr_inv = Output(
            df['strategy2_vol_pyr_inv']['portfolio'][0:len(df)*3/5])
    zz_insample_strategy2_vol_pyr_inv.generate_output()
    print "\nThe KPIs for Trading System 2 - Percent Volatility - Inverted \
 Pyramid in the insample data \
 are:\n", zz_insample_strategy2_vol_pyr_inv.generate_output()
    sns.heatmap(zz_insample_strategy2_vol_pyr_inv.generate_output())

    # Money Market Model with inverted pyramid - System 2

    df['returns'] = df['Close'].pct_change()
    df['returns2'] = df['Close'].pct_change()
    df['strategy2_money_market_inv_pyr'] = (
            df['position2_pyramid_inv2'].shift(1) * df['returns'])
    df['strategy_capital2_money_market_inv_pyr'] = (
            df['strategy2_money_market_inv_pyr'] + 1)
    df['Money_Market_2_inv_pyr'] = (
            df['strategy_capital2_money_market_inv_pyr'].cumprod()*10000)
    df['Money_Market_2_inv_pyr'].plot(
            title="Cummulative ROI - Trading System 2 - Money Market Model\
 - Inverted Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy2_money_market_inv_pyr']['portfolio'] = (
            df['strategy2_money_market_inv_pyr'].sum(
                    axis=1)/len(df['strategy2_money_market_inv_pyr'].columns))
    zz_insample_strategy2_money_market_inv_pyr = Output(
            df['strategy2_money_market_inv_pyr']['portfolio'][0:len(df)*3/5])
    zz_insample_strategy2_money_market_inv_pyr.generate_output()
    print "\nThe KPIs for Trading System 2 - Money Market Model - Inverted \
 Pyramid in the insample data \
 are:\n", zz_insample_strategy2_money_market_inv_pyr.generate_output()
    sns.heatmap(zz_insample_strategy2_money_market_inv_pyr.generate_output())

    # Multiple Tier with inverted pyramid - System 2

    df['returns'] = df['Close'].pct_change()
    df['returns2'] = df['Close'].pct_change()
    df['strategy2_mul_tier_inv_pyr'] = (
            (df['position2_pyramid_inv2']+df['position2_mul_tier']).shift(
                    1) * df['returns'])
    df['strategy_capital2_money_market_inv_pyr'] = (
            df['strategy2_mul_tier_inv_pyr'] + 1)
    df['Money_Market_2_inv_pyr'] = (
            df['strategy_capital2_money_market_inv_pyr'].cumprod()*10000)
    df['Money_Market_2_inv_pyr'].plot(
            title="Cummulative ROI - Trading System 2 - Multiple Tier\
 - Inverted Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy2_mul_tier_inv_pyr']['portfolio'] = (
            df['strategy2_mul_tier_inv_pyr'].sum(
                    axis=1)/len(df['strategy2_mul_tier_inv_pyr'].columns))
    zz_insample_strategy2_mul_tier_inv_pyr = Output(
            df['strategy2_mul_tier_inv_pyr']['portfolio'][0:len(df)*3/5])
    zz_insample_strategy2_mul_tier_inv_pyr.generate_output()
    print "\nThe KPIs for Trading System 2 - Multiple Tier - Inverted Pyramid \
 in the insample data\
 are:\n", zz_insample_strategy2_mul_tier_inv_pyr.generate_output()
    sns.heatmap(zz_insample_strategy2_mul_tier_inv_pyr.generate_output())

    # Now let's apply the reflected pyramid in each
    # sub-system 2 (money market, tiers and volatility)

    # Percent Volatility Model with reflected pyramid - System 2

    df['returns'] = df['Close'].pct_change()
    df['vol_45'] = df['Close'].rolling(45).std()
    df['returns2'] = df['Close'].pct_change()
    df['strategy2_vol_pyr_ref'] = (
            df['position2_pyramid_ref'].shift(1) * df['returns']/df['vol_45'])
    df['strategy_capital2_vol_pyr_ref'] = df['strategy2_vol_pyr_ref'] + 1
    df['Money_Market_2_vol_pyr_ref'] = (
            df['strategy_capital2_vol_pyr_ref'].cumprod()*10000)
    df['Money_Market_2_vol_pyr_ref'].plot(
            title="Cummulative ROI - Trading System 2 - Percent Volatility \
 - Reflected Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy2_vol_pyr_ref']['portfolio'] = (
            df['strategy2_vol_pyr_ref'].sum(
                    axis=1)/len(df['strategy2_vol_pyr_ref'].columns))
    zz_insample_strategy2_vol_pyr_ref = Output(
            df['strategy2_vol_pyr_ref']['portfolio'][0:len(df)*3/5])
    zz_insample_strategy2_vol_pyr_ref.generate_output()
    print "\nThe KPIs for Trading System 2 - Percent Volatility - Reflected \
 Pyramid in the insample data \
 are:\n", zz_insample_strategy2_vol_pyr_ref.generate_output()
    sns.heatmap(zz_insample_strategy2_vol_pyr_ref.generate_output())

    # Money Market Model with reflected pyramid - System 2

    df['returns'] = df['Close'].pct_change()
    df['returns2'] = df['Close'].pct_change()
    df['strategy2_money_market_ref_pyr'] = (
            df['position2_pyramid_ref'].shift(1) * df['returns'])
    df['strategy_capital2_money_market_ref_pyr'] = (
            df['strategy2_money_market_ref_pyr'] + 1)
    df['Money_Market_2_ref_pyr'] = (
            df['strategy_capital2_money_market_ref_pyr'].cumprod()*10000)
    df['Money_Market_2_ref_pyr'].plot(
            title="Cummulative ROI - Trading System 2 - Money Market Model \
 - Reflected Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy2_money_market_ref_pyr']['portfolio'] = (
            df['strategy2_money_market_ref_pyr'].sum(
                    axis=1)/len(df['strategy2_money_market_ref_pyr'].columns))
    zz_insample_strategy2_money_market_ref_pyr = Output(
            df['strategy2_money_market_ref_pyr']['portfolio'][0:len(df)*3/5])
    zz_insample_strategy2_money_market_ref_pyr.generate_output()
    print "\nThe KPIs for Trading System 2 - Money Market Model - Reflected \
 Pyramid in the insample data\
 are:\n", zz_insample_strategy2_money_market_ref_pyr.generate_output()
    sns.heatmap(zz_insample_strategy2_money_market_ref_pyr.generate_output())

    # Multiple Tier with reflected pyramid - System 2

    df['returns'] = df['Close'].pct_change()
    df['returns2'] = df['Close'].pct_change()
    df['strategy2_mul_tier_ref_pyr'] = (
            (df['position2_pyramid_ref']+df['position2_mul_tier']).shift(
                    1) * df['returns'])
    df['strategy_capital2_money_market_ref_pyr'] = (
            df['strategy2_mul_tier_ref_pyr'] + 1)
    df['Money_Market_2_ref_pyr'] = (
            df['strategy_capital2_money_market_ref_pyr'].cumprod()*10000)
    df['Money_Market_2_ref_pyr'].plot(
            title="Cummulative ROI - Trading System 2 - Multiple\
 Tier - Reflected Pyramid", figsize=(20, 12))
    plt.show()

    # KPIs and heat map

    df['strategy2_mul_tier_ref_pyr']['portfolio'] = (
            df['strategy2_mul_tier_ref_pyr'].sum(
                    axis=1)/len(df['strategy2_mul_tier_ref_pyr'].columns))
    zz_insample_strategy2_mul_tier_ref_pyr = Output(
            df['strategy2_mul_tier_ref_pyr']['portfolio'][0:len(df)*3/5])
    zz_insample_strategy2_mul_tier_ref_pyr.generate_output()
    print "\nThe KPIs for Trading System 2 - Multiple Tier - Reflected Pyramid\
 in the insample data\
 are:\n", zz_insample_strategy2_mul_tier_ref_pyr.generate_output()
    sns.heatmap(zz_insample_strategy2_mul_tier_ref_pyr.generate_output())

    # KPIs and heat map for the Best System Out Sample

    df['strategy1_vol_pyr_inv']['portfolio'] = (
            df['strategy1_vol_pyr_inv'].sum(
                    axis=1)/len(df['strategy1_vol_pyr_inv'].columns))
    zz_outsample_strategy1_vol_pyr_inv = Output(
            df['strategy1_vol_pyr_inv']['portfolio'][len(df)*3/5:])
    zz_outsample_strategy1_vol_pyr_inv.generate_output()
    print "\nThe KPIs for Trading System 1 - Volatility Model - Inverted \
 Pyramid in the out sample data \
 are:\n", zz_outsample_strategy1_vol_pyr_inv.generate_output()
    sns.heatmap(zz_outsample_strategy1_vol_pyr_inv.generate_output())

    # Computing Correlations

    df['returns2']['portfolio'] = df['returns'].sum(
            axis=1)/len(df['returns'].columns)

    print df['returns2'].corr()['portfolio'].sort_values(ascending=True)

    tickers_non_correlated = ['AAPL', 'KO', 'JNJ', 'MCD', 'MRK', 'NKE',
                              'PG', 'UNH', 'VZ', 'WMT']

    data_2 = pdr.get_data_yahoo(
            tickers_non_correlated, start="1993-07-31", end="2018-07-31")

    df2 = data_2['Close']
    data_2['returns'] = df2.pct_change()
    data_2['returns2'] = df2.pct_change()

    # KPIs and heat map of the 10 least correlated stocks

    data_2['returns2']['portfolio'] = data_2['returns2'].sum(
            axis=1)/len(data_2['returns2'].columns)
    zz_10_least_correlated = Output(data_2['returns2']['portfolio'])
    zz_10_least_correlated.generate_output()
    print "\nThe KPIs for our portfolio of the 10 least correlated stocks\
 are:\n", zz_10_least_correlated.generate_output()
    sns.heatmap(zz_10_least_correlated.generate_output())
    plt.show()

    # KPIs and heat map of the Dow Jones

    df['returns2']['portfolio'] = df['returns2'].sum(
            axis=1)/len(df['returns2'].columns)
    zz_dow_jones = Output(df['returns2']['portfolio'])
    zz_dow_jones.generate_output()
    print "\nThe KPIs for Dow Jones are:\n", zz_dow_jones.generate_output()
    sns.heatmap(zz_dow_jones.generate_output())
    plt.show()


if __name__ == '__main__':
    main()
