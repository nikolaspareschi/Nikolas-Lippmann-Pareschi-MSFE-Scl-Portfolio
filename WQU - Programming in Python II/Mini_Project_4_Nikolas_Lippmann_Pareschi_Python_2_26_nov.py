# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 16:10:17 2017

@author: Nikolas

1) Create a csv file with a list of all presidents, their parties
from 1920 onwards
2) Using Pandas load the .csv file into a Pandas dataframe.
3) Download data from an appropriate financial website such as Google Finance,
Yahoo Finance, Quandl, CityFALCON, or another similar source.
4) Calculate yearly returns for both the downloaded indices from 1920 onwards
5) Segregate returns in terms of Presidency – i.e. stock market returns
during Democratic and Republican years
6) Calculate measures of central tendency (mean return, median return,
variance of returns) for each of the two groups.
7) Represent the findings through suitable comparative graphical studies

"""

import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as data

'''First we set up the data that will be used in our functions'''

# Equities Data

sp500 = data.DataReader('YALE/SPCOMP', 'quandl', "1920-01-01", "2017-01-21")
dj = data.DataReader('BCB/UDJIAD1', 'quandl', "1920-01-01", "2017-01-20")


# Presidential data

presidents = pd.read_csv("presidents2.csv").dropna()
presidents['Took office '] = pd.to_datetime(presidents['Took office '])
presidents['Left office '] = pd.to_datetime(presidents['Left office '])
presidents2 = presidents.loc[26:, :]
presidents3 = presidents2.groupby(presidents2['Took office '],
                                  as_index=True, sort=False,
                                  group_keys=False).size()


def djrets(dj):

    djanual = dj.resample('A').bfill()
    djanual['Anualreturns'] = djanual['Value'].pct_change()
    print djanual['Anualreturns']
    djanual['Party'] = 'Democrats'

    djanual.set_value(djanual.index[1:13], 'Party', 'Republican')
    djanual.set_value(djanual.index[33:41], 'Party', 'Republican')
    djanual.set_value(djanual.index[49:57], 'Party', 'Republican')
    djanual.set_value(djanual.index[61:73], 'Party', 'Republican')
    djanual.set_value(djanual.index[81:89], 'Party', 'Republican')

    djanualrepublicans = djanual[djanual['Party'] == 'Republican']
    djanualdemocratic = djanual[djanual['Party'] == 'Democrats']

    djanualrepmean = djanualrepublicans["Anualreturns"].mean()
    djanualdemmean = djanualdemocratic["Anualreturns"].mean()

    djanualrepmedian = djanualrepublicans["Anualreturns"].median()
    djanualdemmedian = djanualdemocratic["Anualreturns"].median()

    djanualrepkurt = djanualrepublicans["Anualreturns"].kurtosis()
    djanualdemkurt = djanualdemocratic["Anualreturns"].kurtosis()

    djanualrepskew = djanualrepublicans["Anualreturns"].skew()
    djanualdemskew = djanualdemocratic["Anualreturns"].skew()

    djanualrepskew = djanualrepublicans["Anualreturns"].skew()
    djanualdemskew = djanualdemocratic["Anualreturns"].skew()

    djanualrepstd = djanualrepublicans["Anualreturns"].std()
    djanualdemstd = djanualdemocratic["Anualreturns"].std()

    plt.scatter(djanualrepublicans["Anualreturns"].index,
                djanualrepublicans["Anualreturns"], alpha=0.3, color='red')
    plt.scatter(djanualdemocratic["Anualreturns"].index,
                djanualdemocratic["Anualreturns"], alpha=0.3)

    plt.title("Annual DJ Mean Returns: Republicans x Democrats")
    plt.legend('Model length', 'upper center')

    plt.text('1910', 0.95,
             "Republicans DJ annual mean and median return (%f , %f)"
             % (djanualrepmean, djanualrepmedian))
    plt.text('1910', 0.87,
             "Democrats DJ annual mean and median return (%f , %f) "
             % (djanualdemmean, djanualdemmedian))

    plt.text('1910', -0.82,
             "Republicans DJ annual skew, kurtosis and SD (%f, %f, %f) "
             % (djanualrepskew, djanualrepkurt, djanualrepstd))
    plt.text('1910', -0.9,
             "Democrats DJ annual skew, kurtosis and SD (%f , %f, %f)"
             % (djanualdemskew, djanualdemkurt, djanualdemstd))

    plt.show()


''' We now compute the returns and we create a column with the correct
parties in our pandas dataframe'''


def sp500rets(sp500):

    sp500anual = sp500.resample('A').bfill()
    sp500anual['Anualreturns'] = sp500anual['S&PComposite'].pct_change()
    print sp500anual['Anualreturns']

    sp500['monthreturns'] = sp500['S&PComposite'].pct_change(-1)

    sp5002 = sp500[['S&PComposite', 'monthreturns']]
    sp5002['Party'] = 'Democrats'

# Every party is Democrat, we change that hard coding,
# considering the data from the csv. I tried to make change the index from a
# dataframe created from the csv so I woul merge then for several hours
# without sucess. So I went for the non elegant way. I would love to
# see the program from a veteran or a coleague that manged to do that

    sp5002.set_value(sp5002.index[96:180], 'Party', 'Republican')
    sp5002.set_value(sp5002.index[288:432], 'Party', 'Republican')
    sp5002.set_value(sp5002.index[480:576], 'Party', 'Republican')
    sp5002.set_value(sp5002.index[672:768], 'Party', 'Republican')
    sp5002.set_value(sp5002.index[1006:1150], 'Party', 'Republican')

    sp500republicans = sp5002[sp5002['Party'] == 'Republican']
    sp500democratic = sp5002[sp5002['Party'] == 'Democrats']

    sp500repmean = sp500republicans["monthreturns"].mean()
    sp500demmean = sp500democratic["monthreturns"].mean()

    sp500repmedian = sp500republicans["monthreturns"].median()
    sp500demmedian = sp500democratic["monthreturns"].median()

    sp500repkurt = sp500republicans["monthreturns"].kurtosis()
    sp500demkurt = sp500democratic["monthreturns"].kurtosis()

    sp500repskew = sp500republicans["monthreturns"].skew()
    sp500demskew = sp500democratic["monthreturns"].skew()

    sp500repskew = sp500republicans["monthreturns"].skew()
    sp500demskew = sp500democratic["monthreturns"].skew()

    sp500repstd = sp500republicans["monthreturns"].std()
    sp500demstd = sp500democratic["monthreturns"].std()


#######################################################################
#
#                     Computing the returns
#
#######################################################################

    plt.scatter(sp500republicans["monthreturns"].index,
                sp500republicans["monthreturns"], alpha=0.3, color='red')
    plt.scatter(sp500democratic["monthreturns"].index,
                sp500democratic["monthreturns"], alpha=0.3)

    plt.title("Monthly SP500 Mean Returns: Republicans x Democrats")
    plt.legend('Model length', 'upper center')

    plt.text('1910', 0.72, "Republicans sp500 mean and median return (%f , %f)"
             % (sp500repmean, sp500repmedian))
    plt.text('1910', 0.67, "Democrats sp500 mean and median return (%f , %f) "
             % (sp500demmean, sp500demmedian))

    plt.text('1910', -0.45,
             "Republicans sp500 skew, kurtosis and SD (%f, %f, %f) "
             % (sp500repskew, sp500repkurt, sp500repstd))
    plt.text('1910', -0.5,
             "Democrats sp500 skew, kurtosis and SD (%f , %f, %f)"
             % (sp500demskew, sp500demkurt, sp500demstd))

    plt.show()


def main():

    djrets(dj)
    sp500rets(sp500)


if __name__ == '__main__':
    main()
