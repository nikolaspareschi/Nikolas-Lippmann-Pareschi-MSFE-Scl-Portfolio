# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 14:37:01 2018

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

        data = pdr.get_data_yahoo(symbol, start="1993-07-31", end="2018-07-31")
    except:
        print("Got an Error : ")
        exit()

    return data


df = download_data()


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

    # Trade Systen 1
    n = 5

    # ATR calculation

    df['n_day_high'] = df['Close'].shift(1).rolling(window=n).max()
    df['n_day_low'] = df['Close'].shift(1).rolling(window=n).min()
    df['avg'] = df['Close'].shift(1).rolling(window=n).mean()

    df['H-L'] = abs(df['High']-df['Low'])
    df['H-PC'] = abs(df['High']-df['Close'].shift(1))
    df['L-PC'] = abs(df['Low']-df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=n).mean()

    # Volatility calculation

    df['volatility'] = df['Close'].rolling(window=n).std()

    df['vol_stop_long'] = df['n_day_high'] - df['volatility']
    df['vol_stop_short'] = df['n_day_low'] + df['volatility']

    # Trade System Logic

    df['position'] = np.nan
    df['position'] = (np.where(df['Close'] >
                      df['n_day_high'], 1, df['position']))
    df['position'] = (np.where((df['Close'] <
                      df['vol_stop_long']) & (
                              df['position'].shift(1) == 1),
                             0, df['position']))

    df['position'] = (np.where(df['Close'] <
                      df['n_day_low'], -1, df['position']))
    df['position'] = (np.where((df['Close'] >
                      df['vol_stop_short']) &
                      df['position'].shift(1) == -1, 0, df['position']))

    df['position'] = df['position'].ffill()

    df['position'].plot(title="System positioning", figsize=(10, 6))
    plt.show()

    # Trade Systen 2

    df['N_ema'] = pd.Series.ewm(df['Close'], span=n).mean()
    df['position2'] = np.nan
    df['position2'] = (np.where(df['Close'] >
                       (df['N_ema'] + df['ATR']), 1, df['position2']))

    df['position2'] = (np.where((df['Close'] <
                       df['N_ema']) & (
                               df['position2'].shift(1) == 1),
                                0, df['position2']))

    df['position2'] = (np.where(df['Close'] <
                       (df['N_ema'] - df['ATR']), -1, df['position2']))

    df['position2'] = (np.where((df['Close'] >
                       df['N_ema']) & (
                               df['position2'].shift(1) == -1),
                               0, df['position2']))

    df['position2'] = df['position2'].ffill()
    df['position2'].plot(title="System 2 positioning", figsize=(10, 6))
    plt.show()

    # Computing returns - Trade System 1

    df['returns'] = df['Close'].pct_change()
    df['strategy'] = (df['position'].shift(1) * df['returns'])
    df['strategy_capital'] = df['strategy'] + 1
    df['strategy_capital'].cumprod().plot(
        title="Cummulative Out Sample ROI for our Trading System 1 - All data")
    plt.show()

    # Computing returns - Trade System 2

    df['returns2'] = df['Close'].pct_change()
    df['strategy2'] = (df['position2'].shift(1) * df['returns'])
    df['strategy_capital2'] = df['strategy2'] + 1
    df['strategy_capital2'].cumprod(
                                    ).plot(
        title="Cummulative Out Sample ROI for our Trading System 2 - All data")
    plt.show()

    # KPI Buy and Hold

    df['returns'].dropna()
    xx = Output(df['returns'])
    print "\nThe KPIs for buying and holding:\n", xx.generate_output()

    # KPI System 1

    yy = Output(df['strategy'])
    print "\nThe KPIs for our trade system 1 are:\n", yy.generate_output()

    # KPI System 1 In Sample

    yy_insample = Output(df['strategy'][0:len(df)*4/5])
    print "\nThe KPIs for our trade system 1 in sample are:\n",
    yy_insample.generate_output()

    # Optimal f for TS 1

    y_winrate_in_sample = yy_insample._calc_win_rate(
            df['strategy'][0:len(df)*4/5])
    y_avg_win_in_sample = yy_insample._calc_avg_win_return(
            df['strategy'][0:len(df)*4/5])
    y_avg_loss_in_sample = yy_insample._calc_avg_loss_return(
            df['strategy'][0:len(df)*4/5])

    f_y = (y_winrate_in_sample*(
            y_avg_win_in_sample/y_avg_loss_in_sample) - 1)/(
           y_avg_win_in_sample/y_avg_loss_in_sample)

    print "\nOptimal f for the in sample data is:\n", f_y

    # KPI System 2

    zz = Output(df['strategy2'])
    print "\nThe KPIs for our trading system 2 are:\n", zz.generate_output()

    # KPI System 2 In Sample

    zz_insample = Output(df['strategy2'][0:len(df)*4/5])
    print "\nThe KPIs for our trading system 2 in the insample data are:\n",
    zz_insample.generate_output()

    # Optimal f for TS 2

    z_winrate_in_sample = zz_insample._calc_win_rate(
            df['strategy2'][0:len(df)*4/5])
    z_avg_win_in_sample = zz_insample._calc_avg_win_return(
            df['strategy2'][0:len(df)*4/5])
    z_avg_loss_in_sample = zz_insample._calc_avg_loss_return(
            df['strategy2'][0:len(df)*4/5])

    f_z = (z_winrate_in_sample*(
            z_avg_win_in_sample/z_avg_loss_in_sample) - 1)/(
             z_avg_win_in_sample/z_avg_loss_in_sample)

    print "\nOptimal f for the in sample data is:\n", f_z

    # Pyramiding System 1 without optimal f

    df['position_pyramid'] = (
            np.where((df['position'].shift(4) == 1) & (
                    df['position'].shift(3) == 1) & (
                            df['position'].shift(2) == 1) & (
                                    df['position'].shift(1) == 1) & (
                                            df['position'] == 1), 2,
                                                               df['position']))
    df['position_pyramid'] = (
            np.where(
                    (df['position'].shift(4) == -1) & (
                            df['position'].shift(3) == -1) & (
                                    df['position'].shift(2) == -1) & (
                                            df['position'].shift(1) == -1) & (
                                                    df['position'] == -1),
                                                   -2, df['position_pyramid']))

    df['position_pyramid'].plot(title="Trade System 1 pyramid positioning")
    plt.show()

    # Pyramiding System 2 without optimal f

    df['position2_pyramid'] = (
            np.where((df['position2'].shift(4) == 1) & (
                    df['position2'].shift(3) == 1) & (
                            df['position2'].shift(2) == 1) & (
                                    df['position2'].shift(1) == 1) & (
                                            df['position2'] == 1), 2,
                                                       df['position2']))
    df['position2_pyramid'] = (
            np.where((df['position2'].shift(4) == -1) & (
                    df['position2'].shift(3) == -1) & (
                            df['position2'].shift(2) == -1) & (
                                    df['position2'].shift(1) == -1) & (
                                            df['position2'] == -1), -2,
                                                     df['position2_pyramid']))

    df['position2_pyramid'].plot(title="Trade System 2 pyramid positioning")
    plt.show()

    # Pyramiding System 1 with optimal f

    df['position_pyramid_of'] = (
            np.where(
                    (df['position'].shift(4) == 1) & (
                            df['position'].shift(3) == 1) & (
                                    df['position'].shift(2) == 1) & (
                                            df['position'].shift(1) == 1) & (
                                                    df['position'] == 1),
                                                          f_y, df['position']))
    df['position_pyramid_of'] = (
            np.where((df['position'].shift(4) == -1) & (
                    df['position'].shift(3) == -1) & (
                            df['position'].shift(2) == -1) & (
                                    df['position'].shift(1) == -1) & (
                                            df['position'] == -1),
                    -f_y, df['position_pyramid_of']))

    df['position_pyramid_of'].plot()
    plt.show()

    # Pyramiding System 2 with optimal f

    df['position2_pyramid_of'] = (
            np.where((df['position2'].shift(4) == 1) & (
                    df['position2'].shift(3) == 1) & (
                            df['position2'].shift(2) == 1) & (
                                    df['position2'].shift(1) == 1) & (
                                            df['position2'] == 1),
                     f_z, df['position2']))
    df['position2_pyramid_of'] = (
            np.where(
                    (df['position2'].shift(4) == -1) & (
                     df['position2'].shift(3) == -1) & (
                     df['position2'].shift(2) == -1) & (
                     df['position2'].shift(1) == -1) & (
                        df['position2'] == -1),
                            -f_z, df['position2_pyramid_of']))

    df['position2_pyramid_of'].plot()
    plt.show()

    # Computing returns - Trade System 1 - Normal Pyramid

    df['returns'] = df['Close'].pct_change()
    df['strategy_pyramid'] = (df['position_pyramid'].shift(1) * df['returns'])
    df['strategy_capital_pyramid'] = df['strategy_pyramid'] + 1
    df['strategy_capital_pyramid'].cumprod(
            ).plot(
                title="Cummulative ROI for Trade System 1 with normal Pyramid")
    plt.show()

    # Computing returns - Trade System 2 - Normal Pyramid

    df['strategy2_pyramid'] = (df['position2_pyramid'].shift(1)*df['returns'])
    df['strategy_capital2_pyramid'] = df['strategy2_pyramid'] + 1
    df['strategy_capital2_pyramid'].cumprod(
            ).plot(
                title="Cummulative ROI for Trade System 2 with normal Pyramid")
    plt.show()

    # Computing returns - Trade System 1 - Optimal f Pyramid

    df['returns'] = df['Close'].pct_change()
    df['strategy_pyramid_of'] = (
            df['position_pyramid_of'].shift(1) * df['returns'][len(df)*4/5:])
    df['strategy_capital_pyramid_of'] = df['strategy_pyramid_of'] + 1
    df['strategy_capital_pyramid_of'].cumprod(
            ).plot(
                   title="Cummulative Out Sample ROI for Trade \
 System 1 Pyramided with Optimal f")
    plt.show()

    # Computing returns - Trade System 2 - Optimal f Pyramid

    df['strategy2_pyramid_of'] = (
            df['position2_pyramid_of'].shift(1) * df['returns'][len(df)*4/5:])
    df['strategy_capital2_pyramid_of'] = df['strategy2_pyramid_of'] + 1
    df['strategy_capital2_pyramid_of'].cumprod(
            ).plot(
                   title="Cummulative Out Sample ROI for Trade\
 System 2 Pyramided with Optimal f")
    plt.show()

    yy_of_outsample = Output(df['strategy_pyramid_of'][len(df)*4/5:])
    print "\nThe KPIs for our trading system 1 using optimal f in out sample\
 data are:\n", yy_of_outsample.generate_output()

    zz_of_outsample = Output(df['strategy2_pyramid_of'][len(df)*4/5:])
    print "\nThe KPIs for our trading system 2 using optimal f in out sample\
 data are:\n", zz_of_outsample.generate_output()

    yy_outsample_pyramid = Output(df['strategy_pyramid'][len(df)*4/5:])
    print "\nThe KPIs for our trading system 1 in out sample data using\
 a 2 pyramid are:\n", yy_outsample_pyramid.generate_output()

    zz_outsample_pyramid = Output(df['strategy2_pyramid'][len(df)*4/5:])
    print "\nThe KPIs for our trading system 2 in out sample data using\
 a 2 pyramid are:\n", zz_outsample_pyramid.generate_output()

    yy_outsample = Output(df['strategy'][len(df)*4/5:])
    print "\nThe KPIs for our trading system 1 in out sample data\
 are:\n", yy_outsample.generate_output()

    zz_outsample = Output(df['strategy2'][len(df)*4/5:])
    print "\nThe KPIs for our trading system 2 in out sample data\
 are:\n", zz_outsample.generate_output()


if __name__ == '__main__':
    main()
