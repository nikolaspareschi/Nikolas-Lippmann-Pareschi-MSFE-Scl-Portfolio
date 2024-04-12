
import datetime
from datetime import timedelta
import calendar
import numpy as np
import pandas as pd
from math import exp, log
from scipy.stats import norm
import mibian  
import pandas_datareader.data as web
import math
from math import exp, log
from IPython import get_ipython
from pandas_datareader import data
import datetime as dt
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import scipy



df = pd.read_csv('DJI.csv', index_col = 'Date', parse_dates = True)

# Take Out-of-Money strick price (X) based on stepsize of N points

N_step = 250               # Assume stepsize of N points
N_trading_per_year = 252   # Number of trading days in one year excluding holidays and weekends
N_per_year = 365           # Number of days in a calendar year
N_per_month = 30           # Nubmer of days in a calendar month
N_per_week = 7             # Number of days in a calendar week
N_per_two_weeks = 14       # Number of days in calendar two weeks
N_Friday = 4               # Number for Friday

Strike  = "Strike"         # Add new column for strike price
Strike1 = "Strike1"        # Add new column for another strike price
Sigma   = "Volatility"     # Add new column for 30-day volatility
Expire  = "Expires"        # Add new column for option expiration date
Close   = "Adj Close"      # Adjusted closing price used to compute strick price and volatility
Open = "Open"

# Compute and store stike price and volatility and save as new column

df[Strike] = ((df[Close]-N_step/2)/N_step).round()*N_step
df[Sigma] = df[Close].pct_change().rolling(N_per_month).std()*np.sqrt(N_trading_per_year)

# Function: get_expire_dates - Compute days to expiration
#
# Input:  dates - list of dates
# Output: expire_date_list - list of expiration dates
#

def get_expire_days(dates):
    expire_date_list = [] # Start with empty list
    
    # For each row (day), determine option expiration date  (do two weeks from 1st to 3rd friday of the month -- of course, this can be adjusted)  
    for name in dates:
        now = name                                                      # Current date
        first_day_of_month = datetime.datetime(now.year, now.month, 1)  # Save 1st day of month
        first_friday = first_day_of_month + timedelta(days=((N_Friday-calendar.monthrange(now.year,now.month)[0]) + N_per_week) % N_per_week) # 1st Friday
        third_friday = first_friday + timedelta(days=N_per_two_weeks)   # 3rd Friday
        expire_date_list.append(third_friday)                           # Add 2-week expiration date to table
    return expire_date_list

# Add expiration date of option (default is 2 weeks)
df[Expire] = get_expire_days(df.index)

# Not all days will have expiration dates so populate with NAN value
df.loc[:,Expire][df.loc[:,Expire]<=df.index] = np.nan

# Fill expiration day backwards.  Since last row may be NaT, use ffill instead of bfill
df[Expire].bfill(inplace=True)
df[Expire].ffill(inplace=True)

# Set strick price (Strick 1) equal to 1st day of changed expiration date
df[Strike1] = df.groupby(Expire)[Strike].first()

# Organize by expiration date per strike price
df.groupby(Expire)[Strike].first()

# Source: https://www.quantstart.com/articles/European-Vanilla-Call-Put-Option-Pricing-with-Python

# Define function to compute d1 and d2 values
def d_j(j, S, K, r, v, T):
    """d_j = \frac{log(\frac{S}{K})+(r+(-1)^{j-1} \frac{1}{2}v^2)T}{v sqrt(T)}"""
    return (log(S/K) + (r + ((-1)**(j-1))*0.5*v*v)*T)/(v*(T**0.5))
    
# Use these functions to help compute put option values

# Define function to compute vanilla (European) Put Price using Black-Scholas formula (again can do same for call options)
def vanilla_put_price(S, K, r, v, T):
    """Price of a European put option struck at K, with spot S, constant rate r, constant vol v (over the life of the option) and time to maturity T"""
    return -S*norm.cdf(-d_j(1, S, K, r, v, T))+K*exp(-r*T) * norm.cdf(-d_j(2, S, K, r, v, T))

# Pick the month to compute put option prices - in this case, June 2016 (recommend changing)
My_Month_Choice = "June 2016"  # Pick a month to examine - June 2016

# Get ready to add put options columns to table (price, mibian price, delta, and hedge)
Put   = "put_price"             # Put price field name
Put_M = "put_price_mibian"      # Mibian put price field name
Put_D = "put_delta"             # Put delta field name
Put_H = "put_hedge"             # Put hedge field name
Put_M_Open = "put_price_mibian_open"
Put_D_Open = "put_delta_open"
# Calculate put option price (can do the same for call option price)
for row in df.index:
    S = df.loc[row, Close]    # Close Price
    K = df.loc[row, Strike]   # Strike Price
    r = 0.01                  # Interest Free Rate
    v = df.loc[row, Sigma]    # Computed Volatility (30 days)
    T = (df.loc[row, Expire] - row).days # Time to expiration in days
    
    # If non-zero volatility, compute the put option prices and delta
    if v > 0:
        df.loc[row, Put] = vanilla_put_price(S, K, r, v, T/N_per_year)
        p = mibian.BS([S, K, r*100, T], volatility=v*100)
        df.loc[row, Put_M] = p.putPrice
        df.loc[row, Put_D] = p.putDelta
        

     
# Calculate put option price (can do the same for call option price)
for row in df.index:
    S2 = df.loc[row, Open]    # Open Price
    K = df.loc[row, Strike]   # Strike Price
    r = 0.01                  # Interest Free Rate
    v = df.loc[row, Sigma]    # Computed Volatility (30 days)
    T = (df.loc[row, Expire] - row).days # Time to expiration in days
    
    # If non-zero volatility, compute the put option prices and delta
    if v > 0:
        df.loc[row, Put] = vanilla_put_price(S2, K, r, v, T/N_per_year)
        p2 = mibian.BS([S2, K, r*100, T], volatility=v*100)
        df.loc[row, Put_M_Open] = p2.putPrice
        df.loc[row, Put_D_Open] = p2.putDelta


    
# III. Consider any particular trading month during the past 2 years (choose a month with few holidays).  Start trading on 1st day of the trading month.
    
df[Put_H] = np.round(df[Put_D]*100,1) 
df.ix[My_Month_Choice].head()          # View first rows of hedging

# Create new dataframe with only roll dates (on previous expiration date) and skip first date (since no historical volatility available)

df.groupby(Expire).first()



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
  

df['Daily_Delta_DJ_unhedged']  = -df['Close']*df['put_delta']
df['Number_of_futures_contracts_to_hedge'] =  df['Daily_Delta_DJ_unhedged']/df['Close']
print "\n The number of contracts that we need to buy each day to hedge our puts are: \n\n"
print df['Number_of_futures_contracts_to_hedge']

df['Daily_Delta_DJ_close']  = -df['Close']*df['put_delta']
df['Daily_Delta_DJ_open']  = -df['Open']*df['put_delta_open']

# As one will buy the ammount of DJs to hedge the delta in the options, in the open
# one will have zero delta. In a perfect world with constant delta heding the delta will be zero. 
# As we are doing only 2 times we will have the delta that will be the difference betweeen the end of the day
# delta and the beginning of the day delta

df['Number_of_futures_contracts_to_hedge_open'] =  df['Daily_Delta_DJ_unhedged']/df['Open']
df['Number_of_futures_contracts_to_hedge_Close'] =  df['Daily_Delta_DJ_unhedged']/df['Close']

df['Daily_hedged_net_delta'] = df['put_delta'] - df['put_delta_open']
df['Hedged_twice'] = df['Daily_hedged_net_delta']*(-df['Close'])
df['Hedged_twice_daily_returns'] = df['Hedged_twice'].resample('D').last().pct_change().dropna()


df['daily_returns_unhedged'] = df['Daily_Delta_DJ_unhedged'].resample('D').last().pct_change().dropna()

#std_rolling = df['daily_returns_unhedged'].rolling(365).std().dropna()
#std_rolling.plot(figsize = (15, 12));
#plt.title(u"Rolling 30 days risk / deviation graph ", weight='bold');
#plt.show()

df['Unhedged'] = -df['put_delta']/df['Close']

df['Unhedged'].plot(figsize = (12, 9));
plt.title(u"Returns of the un-hedged short puts", weight='bold');
plt.show()

std_rolling = df['Unhedged'].rolling(30).std().dropna()
std_rolling.plot(figsize = (12, 9));
plt.title(u"Rolling 30 days risk un-hedged/ deviation graph ", weight='bold');
plt.show()



df['Daily_Returns_hedged_once'] = -df['Close']*df['put_delta']/df['Close'].shift(1)
std_rolling = df['Daily_Returns_hedged_once'].rolling(30).std().dropna()
std_rolling.plot(figsize = (12, 9));
plt.title(u"Rolling 30 days risk hedged once / deviation graph ", weight='bold');
plt.show()


df['Daily_Returns_hedged_once'].plot(figsize = (12, 9));
plt.title(u"Returns of Hedged Once Portfolio ", weight='bold');
plt.show()

df['Daily_Returns'] = df['Hedged_twice']/df['Close'].shift(1)
std_rolling = df['Daily_Returns'].rolling(30).std().dropna()
std_rolling.plot(figsize = (12, 9));
plt.title(u"Rolling 30 days risk hedged twice: open and close / deviation graph ", weight='bold');
plt.show()


df['Daily_Returns'].plot(figsize = (12, 9));
plt.title(u"Returns of Hedged Twice Times: Open and Close ", weight='bold');
plt.show()

df['Daily_hedged_net_delta'].plot(figsize = (12, 9));
plt.title(u"Portfolio Delta", weight='bold');
plt.show()


df['Number_of_futures_contracts_cumulative'] = df['Number_of_futures_contracts_to_hedge_open'].cumsum() + df['Number_of_futures_contracts_to_hedge_Close'].cumsum() 
df['Number_of_futures_contracts_cumulative'].plot(figsize = (12, 9));
plt.title(u"Cumulative number of contracts used to hedge", weight='bold');
plt.show()

#www = Output(df['daily_returns_unhedged'])
#print "\n The KPIs for our unhedged Portfolio are: \n\n", www.generate_output()
#
#zzz = Output(df['daily_returns_unhedged'])
#print "\n The KPIs for our unhedged Portfolio are: \n\n", zzz.generate_output()

df['daily_net_delta'] = df['Daily_Delta_DJ_close'] - df['Daily_Delta_DJ_open']
df['daily_returns_unhedged'] = df['Daily_Delta_DJ_unhedged'].resample('D').last().pct_change().dropna()
df['daily_returns_2'] = df['daily_net_delta'].resample('D').last().pct_change().dropna()


df['daily_returns_Close'] = df['Close'].resample('D').last().pct_change().dropna()
df['daily_returns_Open'] = df['Open'].resample('D').last().pct_change().dropna()

#xxx = Output(df['daily_returns_2'])
#print "\n The KPIs for our unhedged Portfolio are: \n\n", xxx.generate_output()




df['return_unhedged']= -df['put_delta']*df['daily_returns_Close']
xxx = Output(df['return_unhedged'])
print "\n The KPIs for our unhedged Portfolio using close prices are: \n\n", xxx.generate_output()


df['return_full_hedge']=df['Daily_hedged_net_delta']*df['daily_returns_Close']
yyy = Output(df['return_full_hedge'])
print "\n The KPIs for our hedged Portfolio are: \n\n", yyy.generate_output()
