# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 15:00:46 2017

@author: Nikolas
"""




from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime
import scipy as sp
import pandas as pd


#################################################################
## Creating dataframes with the symbol for posterior check ######

#NYSE

url_nyse = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nyse&render=download"

# Nasdaq

url_nasdaq = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"

nyse = pd.read_csv(url_nyse)
nasdaq = pd.read_csv(url_nasdaq)

########################################


symbol = raw_input("Please enter symbol for the stock that you want to download \n")


start = datetime.datetime(2017, 10, 1)
end = datetime.datetime(2017, 10, 31)

inicio = pd.Timestamp('2017-10-01')
fim = pd.Timestamp('2017-10-31')


########### Error handling ###############



while (nyse['Symbol'] == symbol).any() != True and (nasdaq['Symbol'] == symbol).any() != True:
    symbol = raw_input("Please enter a VALID symbol for the stock that you want to download \n")


try:
    df = web.DataReader(symbol, 'yahoo', start, end)
    
except:
    print("Got an Error : ")
    exit()

   
###############Basic Linear Interpolation #######################
    
x = np.linspace(inicio.value, fim.value, df['Close'].count())
signal = df['Close']


interpreted2 = sp.interpolate.interp1d(x, signal, 'quadratic')
x2 = np.linspace(inicio.value, fim.value, df['Close'].count())
yy = interpreted2(x2)


plt.plot(pd.to_datetime(x), signal, 'o', label= symbol)
plt.plot(pd.to_datetime(x2), yy, '-', label= "Basic Linear Interpolation")
plt.legend()
plt.show()


############################ Least_Squares #############################


#Lambda functions for the Quadratic fit

funcQuad=lambda tpl,x : tpl[0]*x**2+tpl[1]*x+tpl[2]

# ErrorFunc is the diference between the modeled function and the real data


ErrorFunc=lambda tpl,x,signal: funcQuad(tpl,x)-signal
  
#tplInitial contains the "first guess" of the parameters 


tplInitial=(1.0,2.0,3.0)

tplFinal,success=leastsq(ErrorFunc,tplInitial[:],args=(x,signal))
print "quadratic fit" ,tplFinal

yy2=funcQuad(tplFinal,x2)


plt.plot(pd.to_datetime(x),signal,'bo',label = symbol)
plt.plot(pd.to_datetime(x),yy2,'g-', label = 'Quadratic fit')
plt.legend()
plt.show()

################### Absolute Errors ###################################

plt.plot(pd.to_datetime(x), abs(signal-yy2), 'ro', label = 'Absolute Errors')
plt.legend()
plt.show()
