# -*- coding: utf-8 -*-
"""
Created on Sat Mar 02 11:16:29 2019

@author: nikol
"""


import matplotlib.pyplot as plt
from pandas_datareader import data
import numpy as np
import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import talib as ta
import datetime as dt

# Before the main program we define a class to compute the strategies KPIs


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

    # For this Project we will use 2 dataseries. SPY etf daily prices and
    # TLT etf daily prices. We will use SVM and Decission Tree classifier to
    # study the intermarket relationships between SP500 and treasuries notes
    # with long duration.

    start = datetime.datetime(2006, 1, 1)
    end = datetime.datetime(2019, 2, 23)
    df = data.get_data_yahoo('SPY', start, end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df2 = data.get_data_yahoo('TLT', start, end)

    df2[['Open_TLT', 'High_TLT', 'Low_TLT', 'Close_TLT',
         'Volume_TLT']] = df2[['Open', 'High', 'Low', 'Close', 'Volume']]
    df2 = df2[['Open_TLT', 'High_TLT', 'Low_TLT', 'Close_TLT', 'Volume_TLT']]

    # After Collecting data for both instruments we concatened them in just
    # one dataframe

    df3 = pd.concat([df, df2], axis=1, join_axes=[df.index])
    Df = df3

    # Let´s split the data to train and then test the SVM algorithm

    t = .7
    split = int(t*len(Df))

    # We will use the RSI on SPY and TLT as predictors

#    n = 5
    Df['RSI'] = ta.RSI(np.array(Df['Close'].shift(1)), timeperiod=5)
    Df['RSI_TLT'] = ta.RSI(np.array(Df['Close_TLT'].shift(1)), timeperiod=5)

    # We will use also the magnitude of absolute change in 5 days in these 2
    # instruments Legendary trader Victor Niederhoffer points out the aboluste
    # points captures more wealth effects than percentual points.

    Df['OC_SPY'] = Df['Close'].shift(1)-Df['Open'].shift(6)
    Df['OC_TLT'] = Df['Close_TLT'].shift(1)-Df['Open_TLT'].shift(6)

    # We need to compute the future returns, which are what we want to predict

    Df['Ret'] = np.log(Df['Open_TLT'].shift(-1)/Df['Open_TLT'])

    # Dealing with the Nan values

    Df = Df.fillna(method='backfill')
    Df = Df.fillna(method='ffill')

    # The objective here is to be long only in the 50% better days. We train on
    # that and the plan is to be invested in the test data on the better days,
    # cutting at least the drawdowns.

    Df['Signal'] = 0
    Df.loc[Df['Ret'] > Df['Ret'][:split].quantile(q=0.50), 'Signal'] = 1
    Df.loc[Df['Ret'] < Df['Ret'][:split].quantile(q=0.50), 'Signal'] = 0

    X = Df.drop(['Close', 'Signal', 'High', 'Low', 'Volume', 'Ret'], axis=1)
    # X=Df[['RSI','RSI_TLT','OC_SPY','OC_TLT']]
    # X=Df[['RSI','RSI_TLT','OO_SPY','OO_TLT']] , 'OC_TLT'
    X = Df[['OC_SPY']]
    X = X.fillna(method='backfill')
    y = Df['Signal']

    # We define the steps that will be followed by the pipeline function.
    # We need first to standardize our data so outliers and asymmetrical
    # points will affect less the algorithm

    steps = [('scaler', StandardScaler()), ('svc', SVC())]
    pipeline = Pipeline(steps)

    # To improve the result we will perform also a Randomized Grid Search in
    # the hyperparamet space

    c = [10, 100, 1000, 10000]
    g = [1e-2, 1e-1, 1e0]

    parameters = {'svc__C': c, 'svc__gamma': g, 'svc__kernel': ['rbf']}

    rcv = RandomizedSearchCV(pipeline, parameters, cv=7)

    # After founding the optimal hyperparameter we fit the train data in the
    # SVC algorithm

    rcv.fit(X.iloc[:split], y.iloc[:split])
    best_C = rcv.best_params_['svc__C']
    best_kernel = rcv.best_params_['svc__kernel']
    best_gamma = rcv.best_params_['svc__gamma']
    cls = SVC(C=best_C, kernel=best_kernel, gamma=best_gamma)
    ss1 = StandardScaler()
    cls.fit(ss1.fit_transform(X.iloc[:split]), y.iloc[:split])

    # After the fit we make our predictions

    y_predict = cls.predict(ss1.transform(X.iloc[split:]))

    Df['Pred_Signal'] = 0

    Df.iloc[:split, Df.columns.get_loc('Pred_Signal')]\
        = pd.Series(cls.predict(ss1.transform(X.iloc[:split])).tolist())
    Df.iloc[split:, Df.columns.get_loc('Pred_Signal')] = y_predict

    # We determine the returns under the predicted signals

    Df['Ret1'] = Df['Ret'] * Df['Pred_Signal']
    Df['Cu_Ret1'] = 0
    Df['Cu_Ret1'] = np.cumsum(Df['Ret1'].iloc[split:])
    Df['Cu_Ret'] = 0
    Df['Cu_Ret'] = np.cumsum(Df['Ret'].iloc[split:])
    Df['Ret1_out_sample'] = Df['Ret1'].iloc[split:]

    split_percentage = 0.7
    split = int(split_percentage*len(Df))
    Df['Strategy_Return2'] = Df.Ret[split:] * Df['Pred_Signal'][split:]

    Df['Strategy_Return2'].cumsum().plot(
            color='b', label='TLT Strategy returns')

    Df['Close_TLT'].pct_change(1)[split:].cumsum().plot(
            color='g', label='TLT returns')
    plt.legend(loc='best')
    plt.title('Out Sample Cummulative Returns')
    plt.show()

    yyy_tlt = Output(Df['Ret1'].iloc[split:])
    print "\n The KPIs for our TLT strategy are:"
    print "are: \n\n", yyy_tlt.generate_output()


if __name__ == '__main__':
    main()
