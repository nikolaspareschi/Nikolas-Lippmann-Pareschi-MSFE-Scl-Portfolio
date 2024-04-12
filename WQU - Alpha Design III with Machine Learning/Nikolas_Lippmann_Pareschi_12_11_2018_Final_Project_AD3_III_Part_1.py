# -*- coding: utf-8 -*-
"""
Created on Sat Dec 08 15:02:44 2018

@author: nikol
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 05 21:28:44 2018

@author: nikol
"""



import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
from pandas_datareader import data, wb
import numpy as np
import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import talib as ta
import datetime as dt
import numpy as np
import datetime

# For the part 1 of the Final Project we will use 2 dataseries. SPY etf daily prices and TLT etf
# daily prices. We will use Decission Tree classifier to study the
# intermarket relationships between SP500 and treasuries notes with long duration.
# We combine a SPY strategy and a TLT strategy in a portfolio. The idea is to use
# the treasuries to hedge the SP500 due to the fact that in sell offs SP500 and Treasuries have
# negative correlation

# Data Handling

start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2018, 10, 10)
df = data.get_data_yahoo('SPY', start, end)
df=df[['Open','High','Low','Close','Volume']]
df2 = data.get_data_yahoo('TLT', start, end)

df2[['Open_TLT','High_TLT','Low_TLT','Close_TLT','Volume_TLT']] = df2[['Open','High','Low','Close','Volume']]
df2=df2[['Open_TLT','High_TLT','Low_TLT','Close_TLT','Volume_TLT']]

# After Collecting data for both instruments we concatened them in just one dataframe

df3 = pd.concat([df, df2], axis=1, join_axes=[df.index])
Df=df3

# Future Returns and Target Variable

Df['Return'] = Df['Close'].pct_change(1).shift(-1)
Df['target'] = np.where(Df.Return > -0.01, 1, 0)
#Df['target'] = np.where(Df.Return < -0.01, -1, Df['target'])


# We will use the RSI on SPY and TLT as predictors

n=5
Df['RSI']=ta.RSI(np.array(Df['Close'].shift(1)), timeperiod=5)
Df['RSI_TLT']=ta.RSI(np.array(Df['Close_TLT'].shift(1)), timeperiod=5)

# We will use also the magnitude of absolute change in 5 days in these 2 instruments
# Legendary trader Victor Niederhoffer points out the aboluste points captures more
# wealth effects than percentual points.


Df['OC_SPY']= Df['Close'].shift(1)-Df['Open'].shift(6)
Df['OC_TLT']= Df['Close_TLT'].shift(1)-Df['Open_TLT'].shift(6)

Df.tail()
Df.head()

Df=Df.ffill()
Df=Df.bfill()
Df = Df.dropna()

# These are the predictors that will be used in the tree classifier algorithm.

predictors_list = Df[['RSI_TLT','RSI','OC_SPY','OC_TLT']]
X = predictors_list
X.tail()

y = Df.target
y.tail()

# Splitting the data


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_test=X_test.ffill()
X_test=X_test.bfill()


# Testing if shapes are ok:

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

# Importing the Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=5, class_weight = 'balanced') 
clf

# Fitting the train data

X_train = X_train.ffill()
X_train = X_train.bfill()
clf = clf.fit(X_train, y_train)

# This is how our tree looks like:

from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None,filled=True,feature_names=predictors_list.columns)   
graphviz.Source(dot_data) 

# Making the predictions:

y_pred = clf.predict(X_test)


# Run the code to view the classification report metrics

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)

# Strategy returns on SPY without Hedging

split_percentage = 0.7
split = int(split_percentage*len(Df))
Df['Strategy_Return'] = Df.Return[split:] * y_pred


# Now we will use the same algorithm to predict TLT. Treasuries bunds have negative correlation
# to equities in crisis and many funds use it as a form of hedging and as a way to improve the
# sharp ratio. We will buy TLT accordingly to the algoritm.

# Future Returns and Target Variable

Df['Return2'] = Df['Close_TLT'].pct_change(1).shift(-1)
Df['target2'] = np.where(Df.Return2 > -0.002, 1, 0)



Df=Df.ffill()
Df=Df.bfill()
Df = Df.dropna()

predictors_list2 = Df[['RSI_TLT','RSI','OC_SPY','OC_TLT']]
X2 = predictors_list2
X2.tail()

y2 = Df.target2
y2.tail()


from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42, stratify=y2)
X2_test=X2_test.ffill()
X2_test=X2_test.bfill()


print (X2_train.shape, y2_train.shape)
print (X2_test.shape, y2_test.shape)


from sklearn.tree import DecisionTreeClassifier
clf2 = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=5, class_weight = 'balanced') 
clf2

X2_train = X2_train.ffill()
X2_train = X2_train.bfill()
clf2 = clf2.fit(X2_train, y2_train)


from sklearn import tree
import graphviz
dot_data2 = tree.export_graphviz(clf2, out_file=None,filled=True,feature_names=predictors_list2.columns)   
graphviz.Source(dot_data2) 


y2_pred = clf2.predict(X2_test)


# Run the code to view the classification report metrics

from sklearn.metrics import classification_report
report2 = classification_report(y2_test, y2_pred)
print(report2)

# Without Hedging

split_percentage = 0.7
split = int(split_percentage*len(Df))
Df['Strategy_Return2'] = Df.Return2[split:] * y2_pred
Df['SPY_Hedged_Return'] = (Df['Strategy_Return2']+Df['Strategy_Return'])/2
Df['Strategy_Return'][split:].cumsum().plot()
Df['Close'].pct_change(1)[split:].cumsum().plot(label='SPY Buy and Hold returns')
Df['Strategy_Return2'].cumsum().plot(color='b',label='TLT Strategy returns')
Df['SPY_Hedged_Return'].cumsum().plot()
Df['Close_TLT'].pct_change(1)[split:].cumsum().plot(color='g',label='TLT returns')
plt.legend(loc='best')
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

xxx_spy = Output(Df['Close'].pct_change(1)[split:])
print "\n The KPIs for SPY buy and hold are:"
print "are: \n\n", xxx_spy.generate_output()

xxx_tlt = Output(Df['Close_TLT'].pct_change(1)[split:])
print "\n The KPIs for TLT buy and hold are:"
print "are: \n\n", xxx_tlt.generate_output()

yyy_spy_strategy = Output(Df['Strategy_Return'][split:])
print "\n The KPIs for our SPY strategy are:"
print "are: \n\n", yyy_spy_strategy.generate_output()

yyy_tlt = Output(Df['Strategy_Return2'][split:])
print "\n The KPIs for our TLT strategy are:"
print "are: \n\n", yyy_tlt.generate_output()

xxx_strategy_hedged = Output(Df['SPY_Hedged_Return'][split:])
print "\n The KPIs for our SPY strategy with hedge:"
print "are: \n\n", xxx_strategy_hedged.generate_output()

