# -*- coding: utf-8 -*-
"""
Created on Sun Dec 09 13:11:15 2018

@author: nikol
"""


import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import datetime as dt
import numpy as np
import datetime
import talib as ta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# For this Project we will use 2 dataseries. GLD etf daily prices and DOL etf
# daily prices. We will use a regression linear algorithm in a return to the mean
# trading system

start = datetime.datetime(2007, 3, 1)
end = datetime.datetime(2018, 10, 10)
df = data.get_data_yahoo('GLD', start, end)
df=df[['Open','High','Low','Close','Volume']]
df2 = data.get_data_yahoo('UUP', start, end)

df2[['Open_DOL','High_DOL','Low_DOL','Close_DOL','Volume_DOL']] = df2[['Open','High','Low','Close','Volume']]
df2=df2[['Open_DOL','High_DOL','Low_DOL','Close_DOL','Volume_DOL']]

# After Collecting data for both instruments we concatened them in just one dataframe

df3 = pd.concat([df, df2], axis=1, join_axes=[df.index])
Df=df3

n=5
Df['RSI']=ta.RSI(np.array(Df['Close'].shift(1)), timeperiod=5)
Df['RSI_DOL']=ta.RSI(np.array(Df['Close_DOL'].shift(1)), timeperiod=5)

# We will use also the magnitude of absolute change in 5 days in these 2 instruments
# Legendary trader Victor Niederhoffer points out the aboluste points captures more
# wealth effects than percentual points.


Df['OC_GLD']= Df['Close'].shift(1)-Df['Open'].shift(22)
Df['OC_DOL']= Df['Close_DOL'].shift(1)-Df['Open_DOL'].shift(22)
Df['Std_U']=Df['High']-Df['Open']
Df['Std_D']=Df['Open']-Df['Low']

# Drop rows with missing values
Df= Df.dropna()

# Plot the closing price of GLD
Df.Close.plot(figsize=(10,5))
plt.show()

# We now define the predictor variables

X = Df[['OC_DOL','OC_GLD','RSI','RSI_DOL']]
X.tail()

# This is one targer variable, the predicted maximum

yU = Df['Std_U']
yU.tail()

# This is one targer variable, the predicted minimum

yD = Df['Std_D']
yD.tail()

# In pipeline we will compute the following sequence: 1) The input function which
# deals with the NaN values. 2) The scaler that deal with asymmetries and outliers.
# 3) The Linear Regression algorith that will be used to predict yU and yD

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
steps = [('imputation', imp),
         ('scaler',StandardScaler()),
         ('linear',LinearRegression())]     
pipeline = Pipeline(steps)


# We define also the search space for the hyperparameter Intercept. The cross
# validation method will be used in the training data with a number o 5.

parameters = {'linear__fit_intercept':[0,1]}
reg = GridSearchCV(pipeline, parameters,cv=5)

# We will use 80% of our data as the train datase

t=.80
split = int(t*len(Df))
reg.fit(X[:split],yU[:split])

# Training the algorith with the best hyperparameter value

best_fit = reg.best_params_['linear__fit_intercept']
reg = LinearRegression(fit_intercept = best_fit)
X=imp.fit_transform(X,yU)

# We use the best parameter discovered to predict our yUs

reg.fit(X[:split],yU[:split])
yU_predict = reg.predict(X[split:])


# Assigning the predicted values to a new column in the data frame
Df = Df.assign(Max_U =pd.Series(np.zeros(len(X))).values)
Df['Max_U'][split:]=yU_predict
Df['Max_U'][Df['Max_U']<0]=0

# we do now the Steps above but for the yD

reg = GridSearchCV(pipeline, parameters,cv=5)
reg.fit(X[:split],yD[:split])
best_fit = reg.best_params_['linear__fit_intercept']
reg = LinearRegression(fit_intercept =best_fit)
X = imp.fit_transform(X,yD)
reg.fit(X[:split],yD[:split])
yD_predict = reg.predict(X[split:])

# Assign the predicted values to a new column in the data frame

Df = Df.assign(Max_D = pd.Series(np.zeros(len(X))).values)
Df['Max_D'][split:]=yD_predict
Df['Max_D'][Df['Max_D']<0]=0



# We will use the predicted upside deviation values to calculate the High price

Df['Predicted_High'] = Df['Open']+Df['Max_U']
Df['Predicted_Low'] = Df['Open']-Df['Max_D']
Df[['High','Predicted_High', 'Low','Predicted_Low']].tail()


# Trading Signal

Df['Signal'] = 0
Df['Signal'][(Df['High']>Df['Predicted_High']) &(Df['Low']>Df['Predicted_Low'])]=-1
Df['Signal'][(Df['High']<Df['Predicted_High']) &(Df['Low']<Df['Predicted_Low'])]=1


# Compute GLD returns & cumulative GLD returns 
Df['GLD_Returns']=np.log(Df['Close']/Df['Close'].shift(1))
Df['Cum_GLD_Returns']=np.cumsum(Df['GLD_Returns'][split:])

# Compute strategy returns & cumulative strategy returns

Df['Strategy_Returns'] = Df['GLD_Returns'] * Df['Signal'].shift(1)
Df['Cum_Strategy_Returns']=np.cumsum(Df['Strategy_Returns'][split:])

# Ploting the gold returns and the Strategy Returns

plt.figure(figsize=(10,5))
plt.plot(Df['Cum_GLD_Returns'],color='r',label='GLD Returns')
plt.plot(Df['Cum_Strategy_Returns'],color='g',label='Strategy Returns')
plt.legend()
plt.show()

# Now we will create a class to analyze the performance of the Machine Learning algorithm

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

# These are the stats for the Strategy returns in our out sample

xxx = Output(Df['Strategy_Returns'][split:])
print "\n The KPIs for our strategy are:"
print "are: \n\n", xxx.generate_output()

# These are the stats for the Gold returns in our out sample

yyy = Output(Df['GLD_Returns'][split:])
print "\n The KPIs for buying gold are:"
print "are: \n\n", yyy.generate_output()


