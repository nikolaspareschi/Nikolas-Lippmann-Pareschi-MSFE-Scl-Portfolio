# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:57:13 2018

@author: nikolasp
"""

import pandas as pd
from IPython import get_ipython
import numpy as np
from pandas_datareader import data
import datetime as dt
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import scipy



class Output(object):
    def __init__(self, returns_df, date_freq = 'D'):
        self.returns_df = returns_df if isinstance(returns_df, pd.DataFrame) else pd.DataFrame(returns_df)
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
            set_dt = first_dt - dt.timedelta(days = 1)
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

        expectancy = self._calc_expectancy(win_rates, avg_win_returns, loss_rates, avg_loss_returns)
        expectancy.name = "Trade Expectancy"

        gpr = self.returns_df.apply(self._calc_gain_to_pain_ratio)
        gpr.name = 'Gain to Pain Ratio'

        output_df = pd.concat([lake_ratios, max_dds, ann_returns,
                               ann_stds, sharpes, win_rates,
                               loss_rates, avg_win_returns,
                               avg_loss_returns, expectancy,
                               gpr, ], axis = 1).round(4)

        return output_df.T.sort_index()

# Saving the stocks from the XLS file to our dataframe

df = pd.read_csv('DJIA_Stocks_Close.csv', index_col = 'Date', parse_dates = True)

# Checking if everything is ok:

print df.head()
print df.tail()

# Computing the daily returns to use in the KPI class

daily_returns = df.resample('D').last().pct_change().dropna()

# Creating a column to see the KPIs for the portfolio considering the 30 stocks:

daily_returns['Portfolio'] = daily_returns.sum(axis=1)/30



std_rolling = daily_returns.rolling(30).std().dropna()
std_rolling.plot(figsize = (15, 12));
plt.show()


monthly_returns = df.resample('M').last().pct_change().dropna()
print "\n The monthly returns are: \n",  monthly_returns
positive_return_months = monthly_returns.apply(lambda x: x[x > 0].mean())
print "\n The mean of the positive monthly returns is: \n\n", positive_return_months
negative_return_months = monthly_returns.apply(lambda x: x[x < 0].mean())
print"\n The mean of the negative monthly returns is: \n\n", negative_return_months
freq = monthly_returns.apply(lambda x: float(len(x[x > 0])) / float(len(monthly_returns)))
print "\n The frequency of positive months returns is: \n\n", freq

xxx = Output(daily_returns)
print "\n The KPIs for our stocks and the portfolio are: \n\n", xxx.generate_output()


#
#
#get_ipython().run_line_magic('matplotlib', 'inline')    


#set up empty list to hold our ending values for each simulated price series
    
result = []
    
    
mean_monthly = monthly_returns.mean()
mu_single_monthly = mean_monthly.mean()
#mu_single_monthly = companies2.mean().mean()
sigma_monthly = monthly_returns.std()
sigma_single_montlhy = sigma_monthly.mean()




monthly_returns.hist(bins=40, normed=True, histtype='stepfilled', alpha=0.5);

    
months = 60   # time horizon
runs = 100
dt2 = 1/float(runs)



def random_walk(startprice, mu, sigma):
    price = np.zeros(runs)
    shock = np.zeros(runs)
    price[0] = startprice
    for i in range(1, runs):
        shock[i] = np.random.normal(loc=mu * dt2, scale=sigma * np.sqrt(dt2))
        price[i] = max(0, price[i-1] + shock[i] * price[i-1])
    return price

##### Plot of price considering DF #####

(df / df.iloc[0] * 100).plot(figsize = (15, 12));
plt.show()

##### Here we are ploting the price evolution of the 6 portfolios considering 100 invested #######
    
for i in range(15, 21, 1):
    
    companies = list(monthly_returns.sample(i,axis=1).columns)
    companies2 = monthly_returns.sample(i,axis=1)
    companies2_day = daily_returns.sample(i,axis=1)
    companies2_day['Portfolio'] = companies2_day.sum(axis=1)/i
    zzz = Output(companies2_day)
    print zzz.generate_output().sample(i,axis=1)

    companies3 = df.sample(i, axis =1)

    (companies3/ companies3.iloc[0] * 1000000).plot(figsize = (15, 12));
    plt.title(u"Portfolio evolution ".format(days), weight='bold');
    plt.show()
     
    
    dr = companies3.resample('D').last().pct_change().dropna()

    std_rolling = dr.rolling(30).std().dropna()
    std_rolling.plot(figsize = (15, 12));
    plt.title(u"Rolling 30 days risk / deviation graph ".format(days), weight='bold');
    plt.show()

    monthly_returns_i = companies3.resample('M').last().pct_change().dropna()
    positive_return_months_i = monthly_returns.apply(lambda x: x[x > 0].mean())
    negative_return_months_i = monthly_returns.apply(lambda x: x[x < 0].mean())
    freq_i = monthly_returns.apply(lambda x: float(len(x[x > 0])) / float(len(monthly_returns)))

    
    
    mean_monthly = monthly_returns.mean()
    mu_single_monthly = companies2.mean().mean()
    mu_total = mu_single_monthly*60
    sigma_monthly = companies2.std().mean()
    sigma_total = sigma_monthly*np.sqrt(60)
    
    for run in range(100):
        plt.plot(random_walk(1000000.0, mu_total, sigma_total))
    plt.xlabel("Portfolios simulated for 60 months")
    plt.ylabel("Portfolio Value in dollars");
    plt.show()
    
    print "Random Portfolio chosen", companies
    
    
    
    
    for run in range(runs):
        simulations[run] = random_walk(1000000.0, mu_total, sigma_total)[runs-1]
#    print np.percentile(simulations,5)
        q = np.percentile(simulations, 1)
    plt.hist(simulations, normed=True, bins=30, histtype='stepfilled', alpha=0.2)
    plt.figtext(0.5, 0.8, u"Start price: $ 1M")
    plt.figtext(0.5, 0.7, u"Mean final price: $ {:.8}".format(simulations.mean()))
    plt.figtext(0.5, 0.6, u"VaR(0.99): $ {:.7}".format(10 - q))
    plt.figtext(0.15, 0.6, u"q(0.99): $ {:.7}".format(q))
    plt.axvline(x=q, linewidth=4, color='r')
    plt.title(u"Portfolio value after 60 months".format(days), weight='bold');
    plt.show()


    for run in range(runs):
        if run == 15:
            simulations[run] = random_walk(900000.0, mu_total, sigma_total)[runs-1]
    #    print np.percentile(simulations,5)
            q = np.percentile(simulations, 1)
        elif run == 25:
            simulations[run] = random_walk(900000.0, mu_total, sigma_total)[runs-1]
    #    print np.percentile(simulations,5)
            q = np.percentile(simulations, 1)
        else:
            simulations[run] = random_walk(1000000.0, mu_total, sigma_total)[runs-1]
    #    print np.percentile(simulations,5)
            q = np.percentile(simulations, 1)
    plt.hist(simulations, normed=True, bins=30, histtype='stepfilled', alpha=0.2)
    plt.figtext(0.5, 0.8, u"Start price: $ 1M")
    plt.figtext(0.5, 0.7, u"Mean with shocks: $ {:.8}".format(simulations.mean()))
    plt.figtext(0.5, 0.6, u"VaR(0.99): $ {:.7}".format(10 - q))
    plt.figtext(0.15, 0.6, u"q(0.99): $ {:.7}".format(q))
    plt.axvline(x=q, linewidth=4, color='r')
    plt.title(u"Portfolio value after 60 months considering shocks".format(days), weight='bold');
    plt.show()




    ####  Here we are showing the returns ####
    
(df / df.iloc[0] * 100).plot(figsize = (15, 12));
plt.show()

