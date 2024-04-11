# -*- coding: utf-8 -*-
"""
Created on Fri Dec 08 20:26:27 2017

@author: Nikolas
"""

import numpy as np
import pandas as pd
import pandas_datareader as data
from datetime import datetime as dt
from scipy import stats
import matplotlib.pyplot as plot
import matplotlib.mlab as mlab
import numpy.random as npr
import pylab

from numpy import log, polyfit, std, subtract


""" 1.	Write a python program(s) to download end-of-day data last 25 years the
major global stock market indices from Google Finance, Yahoo Finance, Quandl,
CityFALCON, or another similar source."""


def download():

    start = dt(1992, 10, 30)
    end = dt(2017, 10, 30)

    start2 = dt(1980, 5, 31)
    end2 = dt(2017, 10, 30)

    indexes = ['^SP500TR', '^IXIC', '^DJI', '^GDAXI', '^FTSE',
               '^HSI', '^KS11', '^N225', '^NSEI']
    indexes_data = data.DataReader(indexes, 'yahoo', start, end)['Adj Close']

    black_monday = data.DataReader('^DJI', 'yahoo', start2, end2)['Adj Close']
    black_monday_ret = black_monday.pct_change()
    black_monday_ret = min(black_monday_ret.dropna())

    indexes_ret = indexes_data.pct_change()

    indexes_data_ln = pd.DataFrame()
    # indexes_data_ln = np.log(indexes_data).diff()
    indexes_data_ln = np.log(indexes_data)
    return indexes_data, black_monday_ret, indexes_ret, indexes_data_ln


""" 2.	It is a common assumption in quantitative finance that stock returns
follow a normal distribution whereas prices follow a lognormal distribution
For all these indices check how closely price movements followed a log-normal
 distribution.

3.	Verify whether returns from these broad market indices followed a normal
 distribution?"""

indexes_data, black_monday_ret, indexes_ret, indexes_data_ln = download()


def normal_log_test():

    print '\n\n'
    print "Normality test applied to prices returns \n"
    print '\n'
    print indexes_ret.apply(stats.normaltest, axis=0, nan_policy='omit')
    print '\n'
    print "The normality of price returns was rejected"
    print '\n'
    print ("Normality test applied to the difference of log of prices \n")
    print '\n'
    print indexes_data_ln.apply(stats.normaltest, axis=0, nan_policy='omit')
    print '\n'
    print "The premise of prices following a log normal distribution was",
    print "rejected"
    print '\n'


""" 4. For each of the above two parameters (price movements and stock returns)
 come up with specific statistical measures that clearly identify the degree
 of deviation from the ideal distributions. Graphically represent the degree
 of correspondence."""


def statistics(a1, a2):

    sta1 = a1.describe()
    sta2 = a2.describe()
    sta1skew = a1.skew()
    sta2skew = a2.skew()
    sta1kurtosis = a1.kurtosis()
    sta2kurtosis = a2.kurtosis()

    print sta1
    print '\n'
    print sta2
    print '\n'
    print ("Skewness from log of prices")
    print '\n'
    print sta1skew
    print '\n'
    print ("Skewness from indexes returns")
    print '\n'
    print sta2skew
    print '\n'
    print ("Kurtosis from log of prices")
    print '\n'
    print sta1kurtosis
    print '\n'
    print ("Kurtosis from indexes returns")
    print '\n'
    print sta2kurtosis
    print '\n'


def error_from_normal(a1):

    expected_skew_normal = 0
    expected_kurtosis_normal = 3
    skew_error = abs(a1.skew() - expected_skew_normal)
    kurtosis_error = abs(a1.kurtosis() - expected_kurtosis_normal)
    print ("The skewness error from the expected normal distribution is")
    print ("\n \n")
    print skew_error
    print ("\n \n")
    print("The kurtosis error from the expected normal distribution is")
    print ("\n \n")
    print kurtosis_error


def error_from_lognormal(a2):

    expected_skew_normal = (np.exp((a2.std())**2)
                            + 2)*np.sqrt(np.exp((a2.std())**2)-1)
    expected_kurt_lognormal = (np.exp(4*a2.std()) +
                               2*np.exp(3*a2.std()) + 3*np.exp(2*a2.std())-6)
    skew_error = abs(a2.skew() - expected_skew_normal)
    kurtosis_error = abs(a2.kurtosis() - expected_kurt_lognormal)
    print '\n'
    print("The skewness error from the expected lognormal distribution is")
    print ("\n \n")
    print skew_error
    print '\n'
    print("The kurtosis error from the expected lognormal distribution is")
    print ("\n \n")
    print kurtosis_error


def price_returns_graph(a1):

    mu, sigma = indexes_ret['^DJI'].mean(), indexes_ret['^DJI'].std()

    n, bins, patches = pylab.hist(indexes_ret['^DJI'].dropna(),
                                  500, normed=1, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)

    plot.plot(bins, y, 'r--', linewidth=1)

    plot.xlabel('Returns')
    plot.ylabel('Frequency')
    plot.title(r'Frequency of Returns and Normal distribution')
    plot.grid(True)
    plot.show()

    print 'The Kurtosis and the appearence of outliers makes the distribution',
    print 'substantially different from the normal'

# Now we will do the same but for the lognormal distribution


def log_of_price_graph(a2):

    mu, sigma = indexes_data_ln['^DJI'].mean(), indexes_data_ln['^DJI'].std()
    n, bins, patches = pylab.hist(indexes_data_ln['^DJI'].dropna(), 500,
                                  normed=1, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)

    plot.plot(bins, y, 'r--', linewidth=1)
    plot.xlabel('Log of Prices')
    plot.ylabel('Frequency')
    plot.title(r'Log of Prices and Normal distribution')
    plot.grid(True)
    plot.show()

    print 'The log of prices clearly do not follow a normal distribution',
    print 'as they should if the prices were lognoramlly distributed'


""" 5. One of the most notable hypothesis about stock market behavior is the
“Efficient market hypothesis” which also internally assume that market price
 follows a random-walk process. Assuming that Stock Index prices follow a
 geometric Brownian motion and hence index returns were normally distributed
 with about 20% historical volatility, write a program sub-module to calculate
 the probability of an event like the 1987 stock market crash happening ?
 Explain in simple terms what the results imply."""


def black_monday():

    mu = indexes_ret['^DJI'].mean()
    sd_yearly = 0.2
    sd_daily = sd_yearly/np.sqrt(252)

    prob_1987 = stats.norm.cdf(black_monday_ret, mu, sd_daily)
    print 'Probability of a day like tha black monday'

    print prob_1987
    prob_not_1987_in_100_years_of_dow = (1 - prob_1987)**(252*100)
    print '\n'
    print 'Probability of not having in 100 years a black monday'
    print '\n'
    print prob_not_1987_in_100_years_of_dow
    print '\n'
    print 'In the Python documentation we have on a typical machine running',
    print 'Python, there are 53 bits of precision available for a float.',
    print 'So our result is 1. We power the probability of a 1987 to 60 to',
    print 'avoid the round'

    prob_not_1987_fake = (1 - prob_1987*(10)**60)**(252*100)

    print 'This value ', prob_not_1987_fake, 'is actually much lower than the',
    print 'real probability, which in practice says that day was impossible',
    print 'considering the normal distribution. The results imply that if we',
    print 'want real precision we cannot use the normal distribution on price',
    print 'returns.'


"""Now we will simulate daily geometric brownian movements of price considering
the lognormal premise to visualize several simulations and how close we get
from the 1987 black monday"""


def geom_move_prices():

    S0 = 100  # initial value
    r = 0.0001  # Daily rate
    sd_yearly = 0.2
    sd_daily = sd_yearly/np.sqrt(252)
    sigma = sd_daily  # constant volatility
    T = 1.0  # in years
    It = 252  # Number of all simulations
    M = 420  # Minutes per trading day
    dt = T / M
    S = np.zeros((M + 1, It))
    S[0] = S0

    for t in range(1, M + 1):
        S[t] = (S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt
                + sigma * np.sqrt(dt) * npr.standard_normal(It)))

    plot.plot(S[:, :252], lw=1.5)
    plot.title('252 days of index movements simulated')
    plot.xlabel('One Trading day (420 minutes) of stock movements')
    plot.ylabel('Index level with open at 100')
    plot.grid(True)
    plot.show()


""" I simulated below 25200 days of trading with geometric brownian movement
to see if we had a single day in which we had a 20,5% drop in prices. I placed
the code under comment because it took 5 minutes to generate the graph,
nevertheless I have printed the picture in the word file. The results were
very far from the black monday event. The worst day was near a 4% movement.
This is a strong evidence agains the log normality of prices """

#
# def geom_move_prices2():
#
#     S0 = 100  # initial value
#     r = 0.0001  # Daily rate
#     sigma = sd_daily  # constant volatility
#     T = 1.0  # in years
#     It = 25200  # Number of all simulations
#     M = 420  # Minutes per trading day
#     dt = T / M
#     S = np.zeros((M + 1, It))
#     S[0] = S0
#
#     for t in range(1, M + 1):
#         S[t] = (S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt
#                 + sigma * np.sqrt(dt) * npr.standard_normal(It)))
#
#     plot.plot(S[:, :25200], lw=1.5)
#     plot.title('25200 days of index movements simulated')
#     plot.xlabel('One Trading day (420 minutes) of stock movements')
#     plot.ylabel('Index level with open at 100')
#     plot.grid(True)
#     plot.show()
#
#
# geom_move_prices2()

""" 6. What does "fat tail" mean? Plot the distribution of price movements for
the downloaded indices (in separate subplot panes of a graph) and identify
fat tail locations if any."""


def z_scores(ys):

    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return z_scores


def outliers_numbers(ind):

    a = []
    for i in ind:
        if i > 4 or i < -4:
            print i
            a.append(i)
    print len(a)
    print 'We had', len(a), '4sigma event if we consider the normal premise',
    print 'is true in'


def sigma_event_prob(a):

    mu = a.mean()
    sd_yearly = 0.2
    sd_daily = sd_yearly/np.sqrt(252)

    prob_4sigma = stats.norm.cdf(4*sd_daily, mu, sd_daily)

    print '\n'
    print 'The probability of a daily 4 sigma event not',
    print 'happening in a single year is', prob_4sigma**252
    print '\n'
    print 'The probability of a daily 4 sigma event not',
    print 'occuring in 25 years is', prob_4sigma**(252*25)


def normal_hist_kurtosis(index):

    stats.probplot(index, dist="norm", plot=pylab)
    print 'The Theorical Quantiles are adapted from the distribution. If,'
    print 'the distribution is normal, the graph should resemble a straight',
    print 'and 45 degree line'
    pylab.show()

    mu, sigma = index.mean(), index.std()
    n, bins, patches = pylab.hist(index.dropna(),
                                  500, normed=1, facecolor='blue', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    plot.plot(bins, y, 'r--', linewidth=1)
    plot.xlabel('Returns')
    plot.ylabel('Frequency')
    plot.title(r'Frequency of Returns and Normal distribution')
    plot.grid(True)
    plot.show()

    print 'The Kurtosis is:', index.kurtosis()
    print 'The Kurtosis > 3 and the exitence of 4 sigma outliers makes the',
    print ' distribution substantially different from the normal'


"""7.  It is often claimed that fractals and multi-fractals generate a more
realistic picture of market risks than log-normal distribution. Considering
last 10 year daily price movements of NASDAQ, write a program to check
whether fractal geometrics could have better predicted stock market movements
than log-normal distribution assumption. Explain your findings with suitable
graphs."""


def hurst_higher_timeframe(ts):

    """Returns the Hurst Exponent of the time series vector ts"""
# Create the range of lag values
    lags = range(180, 200)
    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)
    hurst = poly[0]*2.0
    fractal_dimension = 2 - hurst
    # Return the Hurst exponent from the polyfit output
    return hurst, fractal_dimension


def hurst_lower_timeframe(ts):

    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 10)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)
    hurst = poly[0]*2.0
    fractal_dimension = 2 - hurst
    # Return the Hurst exponent from the polyfit output
    return hurst, fractal_dimension


def buy_sell_3sigma():

    mudji = indexes_ret['^DJI'].mean()
    sddji = indexes_ret['^DJI'].std()
    selldji = mudji + sddji*3
    buydji = mudji - sddji*3

    plot.plot(indexes_ret['^DJI'])
    plot.axhline(y=buydji, color='r')
    plot.axhline(y=selldji, color='g')
    plot.title('Buy in red, sell in green - 3 sigmas Dow Jones')
    plot.show()

    musp500 = indexes_ret['^SP500TR'].mean()
    sdsp500 = indexes_ret['^SP500TR'].std()
    sellsp500 = musp500 + sdsp500*3
    buysp500 = musp500 - sdsp500*3

    plot.plot(indexes_ret['^SP500TR'])
    plot.axhline(y=buysp500, color='r')
    plot.axhline(y=sellsp500, color='g')
    plot.title('Buy in red, sell in green - 3 sigmas SP500')
    plot.show()

    munasdaq = indexes_ret['^IXIC'].mean()
    sdnasdaq = indexes_ret['^IXIC'].std()
    sellnasdaq = munasdaq + sdnasdaq*3
    buynasdaq = munasdaq - sdnasdaq*3

    plot.plot(indexes_ret['^IXIC'])
    plot.axhline(y=buynasdaq, color='r')
    plot.axhline(y=sellnasdaq, color='g')
    plot.title('Buy in red, sell in green - 3 sigmas Nasdaq')
    plot.show()

    mudax = indexes_ret['^GDAXI'].mean()
    sddax = indexes_ret['^GDAXI'].std()
    selldax = mudax + sddax*3
    buydax = mudax - sddax*3

    plot.plot(indexes_ret['^GDAXI'])
    plot.axhline(y=buydax, color='r')
    plot.axhline(y=selldax, color='g')
    plot.title('Buy in red, sell in green - 3 sigmas DAX')
    plot.show()

    muftse = indexes_ret['^FTSE'].mean()
    sdftse = indexes_ret['^FTSE'].std()
    sellftse = muftse + sdftse*3
    buyftse = muftse - sdftse*3

    plot.plot(indexes_ret['^FTSE'])
    plot.axhline(y=buyftse, color='r')
    plot.axhline(y=sellftse, color='g')
    plot.title('Buy in red, sell in green - 3 sigmas FTSE')
    plot.show()

    muhsi = indexes_ret['^HSI'].mean()
    sdhsi = indexes_ret['^HSI'].std()
    sellhsi = muhsi + sdhsi*3
    buyhsi = muhsi - sdhsi*3

    plot.plot(indexes_ret['^HSI'])
    plot.axhline(y=buyhsi, color='r')
    plot.axhline(y=sellhsi, color='g')
    plot.title('Buy in red, sell in green - 3 sigmas Hong Kong')
    plot.show()

    muks11 = indexes_ret['^KS11'].mean()
    sdks11 = indexes_ret['^KS11'].std()
    sellks11 = muks11 + sdks11*3
    buyks11 = muks11 - sdks11*3

    plot.plot(indexes_ret['^KS11'])
    plot.axhline(y=buyks11, color='r')
    plot.axhline(y=sellks11, color='g')
    plot.title('Buy in red, sell in green - 3 sigmas South Korea')
    plot.show()

    munikkei = indexes_ret['^N225'].mean()
    sdnikkei = indexes_ret['^N225'].std()
    sellnikkei = munikkei + sdnikkei*3
    buynikkei = munikkei - sdnikkei*3

    plot.plot(indexes_ret['^N225'])
    plot.axhline(y=buynikkei, color='r')
    plot.axhline(y=sellnikkei, color='g')
    plot.title('Buy in red, sell in green - 3 sigmas Nikkei')
    plot.show()

    munifty = indexes_ret['^NSEI'].mean()
    sdnifty = indexes_ret['^NSEI'].std()
    sellnifty = munifty + sdnifty*3
    buynifty = munifty - sdnifty*3

    plot.plot(indexes_ret['^NSEI'])
    plot.axhline(y=buynifty, color='r')
    plot.axhline(y=sellnifty, color='g')
    plot.title('Buy in red, sell in green - 3 sigmas Nifty')
    plot.show()


def main():

    indexes_data, black_monday_ret, indexes_ret, indexes_data_ln = download()
    normal_log_test()
    statistics(indexes_data_ln, indexes_ret)

    print '\n'
    print '-- Dow Jones Analysis -- '
    print '\n'
    price_returns_graph(indexes_ret['^DJI'])
    log_of_price_graph(indexes_data_ln['^DJI'])
    print '\n'
    print '-- SP500 Analysis -- '
    print '\n'
    price_returns_graph(indexes_ret['^SP500TR'])
    log_of_price_graph(indexes_data_ln['^SP500TR'])
    print '\n'
    print '-- Nasdaq Analysis -- '
    print '\n'
    price_returns_graph(indexes_ret['^IXIC'])
    log_of_price_graph(indexes_data_ln['^IXIC'])
    print '\n'
    print '-- DAX Analysis -- '
    print '\n'
    price_returns_graph(indexes_ret['^GDAXI'])
    log_of_price_graph(indexes_data_ln['^GDAXI'])
    print '\n'
    print '-- FTSE Analysis -- '
    print '\n'
    price_returns_graph(indexes_ret['^FTSE'])
    log_of_price_graph(indexes_data_ln['^FTSE'])
    print '\n'
    print '-- Hong Kong Analysis -- '
    print '\n'
    price_returns_graph(indexes_ret['^HSI'])
    log_of_price_graph(indexes_data_ln['^HSI'])
    print '\n'
    print '-- South Korea Analysis -- '
    print '\n'
    price_returns_graph(indexes_ret['^KS11'])
    log_of_price_graph(indexes_data_ln['^KS11'])
    print '\n'
    print '-- Nifty Analysis -- '
    print '\n'
    price_returns_graph(indexes_ret['^NSEI'])
    log_of_price_graph(indexes_data_ln['^NSEI'])

    error_from_normal(indexes_ret)
    error_from_lognormal(indexes_data_ln)

    black_monday()
    geom_move_prices()

    outliers_dji = z_scores(indexes_ret['^DJI'])
    outliers_nasdaq = z_scores(indexes_ret['^IXIC'])
    outliers_sp500 = z_scores(indexes_ret['^SP500TR'])
    outliers_gdaxi = z_scores(indexes_ret['^GDAXI'])
    outliers_hsi = z_scores(indexes_ret['^HSI'])
    outliers_ks11 = z_scores(indexes_ret['^KS11'])
    outliers_n225 = z_scores(indexes_ret['^N225'])
    outliers_nsei = z_scores(indexes_ret['^NSEI'])

    outliers_numbers(outliers_dji)
    print 'Dow Jones'
    outliers_numbers(outliers_nasdaq)
    print 'Nasdaq'
    outliers_numbers(outliers_sp500)
    print 'SP500'
    outliers_numbers(outliers_gdaxi)
    print 'DAX'
    outliers_numbers(outliers_hsi)
    print 'Hong Kong'
    outliers_numbers(outliers_ks11)
    print 'South Korea'
    outliers_numbers(outliers_n225)
    print 'Nikkei'
    outliers_numbers(outliers_nsei)
    print 'Nifty'

    print '\n'
    print '---DOW JONES---'
    sigma_event_prob(indexes_ret['^DJI'])
    print '\n'
    print '---NASDAQ---'
    sigma_event_prob(indexes_ret['^IXIC'])
    print '\n'
    print '---SP500---'
    sigma_event_prob(indexes_ret['^SP500TR'])
    print '\n'
    print '---DAX---'
    sigma_event_prob(indexes_ret['^GDAXI'])
    print '\n'
    print '---HONG KONG---'
    sigma_event_prob(indexes_ret['^HSI'])
    print '\n'
    print '---SOUTH KOREA---'
    sigma_event_prob(indexes_ret['^KS11'])
    print '\n'
    print '---NIKKEI---'
    sigma_event_prob(indexes_ret['^N225'])
    print '\n'
    print '---NIFTY---'
    sigma_event_prob(indexes_ret['^NSEI'])

    print '\n'
    print '---- Dow Jones analysis ----'
    print '\n'
    normal_hist_kurtosis(indexes_ret['^DJI'])
    print '\n'
    print '---- Nasdaq analysis ----'
    print '\n'
    normal_hist_kurtosis(indexes_ret['^IXIC'])
    print '\n'
    print '---- SP500 analysis ----'
    print '\n'
    normal_hist_kurtosis(indexes_ret['^SP500TR'])
    print '\n'
    print '---- Dax analysis ----'
    print '\n'
    normal_hist_kurtosis(indexes_ret['^GDAXI'])
    print '\n'
    print '---- Hong Kong analysis ----'
    print '\n'
    normal_hist_kurtosis(indexes_ret['^HSI'])
    print '\n'
    print '---- South Korea analysis ----'
    print '\n'
    normal_hist_kurtosis(indexes_ret['^KS11'])
    print '\n'
    print '---- Nikkei analysis ----'
    print '\n'
    normal_hist_kurtosis(indexes_ret['^N225'])
    print '\n'
    print '---- Nifty analysis ----'
    print '\n'
    normal_hist_kurtosis(indexes_ret['^NSEI'])

    print '\n'
    print '-- Hurst and Fractal Analysis -- Dow Jones -- '
    print '\n'
    print '            Higher Timeframe'
    print '  ---- Hurst ----      Fractal Dimension'
    print hurst_higher_timeframe(indexes_data['^DJI'].dropna())
    print '            Lower Timeframe'
    print '  ---- Hurst ----      Fractal Dimension'
    print hurst_lower_timeframe(indexes_data['^DJI'].dropna())

    print '\n'
    print '-- Hurst and Fractal Analysis -- Nasdaq -- '
    print '\n'
    print '            Higher Timeframe'
    print '  ---- Hurst ----      Fractal Dimension'
    print hurst_higher_timeframe(indexes_data['^IXIC'].dropna())
    print '            Lower Timeframe'
    print '  ---- Hurst ----      Fractal Dimension'
    print hurst_lower_timeframe(indexes_data['^IXIC'].dropna())

    print '\n'
    print '-- Hurst and Fractal Analysis -- SP500 -- '
    print '\n'
    print '            Higher Timeframe'
    print '  ---- Hurst ----      Fractal Dimension'
    print hurst_higher_timeframe(indexes_data['^SP500TR'].dropna())
    print '            Lower Timeframe'
    print '  ---- Hurst ----      Fractal Dimension'
    print hurst_lower_timeframe(indexes_data['^SP500TR'].dropna())

    print '\n'
    print '-- Hurst and Fractal Analysis ---  DAX  --- '
    print '\n'
    print '            Higher Timeframe'
    print '  ---- Hurst ----      Fractal Dimension'
    print hurst_higher_timeframe(indexes_data['^GDAXI'].dropna())
    print '            Lower Timeframe'
    print '  ---- Hurst ----      Fractal Dimension'
    print hurst_lower_timeframe(indexes_data['^GDAXI'].dropna())

    print '\n'
    print '-- Hurst and Fractal Analysis -- Hong Kong -- '
    print '\n'
    print '            Higher Timeframe'
    print '  ---- Hurst ----      Fractal Dimension'
    print hurst_higher_timeframe(indexes_data['^HSI'].dropna())
    print '            Lower Timeframe'
    print '  ---- Hurst ----      Fractal Dimension'
    print hurst_lower_timeframe(indexes_data['^HSI'].dropna())

    print '\n'
    print '-- Hurst and Fractal Analysis -- South Korea -- '
    print '\n'
    print '            Higher Timeframe'
    print '  ---- Hurst ----      Fractal Dimension'
    print hurst_higher_timeframe(indexes_data['^KS11'].dropna())
    print '            Lower Timeframe'
    print '  ---- Hurst ----      Fractal Dimension'
    print hurst_lower_timeframe(indexes_data['^KS11'].dropna())

    print '\n'
    print '-- Hurst and Fractal Analysis -- Nikkei -- '
    print '\n'
    print '            Higher Timeframe'
    print '  ---- Hurst ----      Fractal Dimension'
    print hurst_higher_timeframe(indexes_data['^N225'].dropna())
    print '            Lower Timeframe'
    print '  ---- Hurst ----      Fractal Dimension'
    print hurst_lower_timeframe(indexes_data['^N225'].dropna())

    print '\n'
    print '-- Hurst and Fractal Analysis -- Nifty -- '
    print '\n'
    print '            Higher Timeframe'
    print '  ---- Hurst ----      Fractal Dimension'
    print hurst_higher_timeframe(indexes_data['^NSEI'].dropna())
    print '            Lower Timeframe'
    print '  ---- Hurst ----      Fractal Dimension'
    print hurst_lower_timeframe(indexes_data['^NSEI'].dropna())

    print '\n'
    print 'As one can see most indexes mean revert in small timeframes and ',
    print 'exhibit trending behavior in the long term. One strategy is to',
    print ' uy sudden dips in a long term uptrend and quit the markets in',
    print 'sudden up move.This makes sense considering the theory of fractals',
    print '(patterns that repeat itself in low and higher frequencies) and',
    print 'the paper posted in Piazza about long term momentum and short',
    print 'term momentum. These graphs consider a longterm uptrend and point ',
    print 'out entry and sell points considering 3 sigma events, which should',
    print ' not happen in the observed frequency under the normal premise.',
    print 'Hurst and Fractals say that when these sudden movements happen we',
    print 'have a return to the mean phenomena.'

    buy_sell_3sigma()


if __name__ == '__main__':
    main()
