# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:31:10 2017

@author: Nikolas
"""

import numpy as np, pandas as pd, pandas_datareader as data
from datetime import datetime as dt
import seaborn as sns

start = dt(2007, 5, 31)
start
end = dt(2017, 10, 30)

####### Create a Dictionary with the Countries and Symbols ######

dictionary = {'brazil':['^BVSP'], 'united states':['^GSPC'], 'germany':['^GDAXI'],
              'france':['^FCHI'],'russia':['MICEXINDEXCF.ME'],
              'belgium':['^BFX'],
              'japan':['^N225'], 'china':['^SSEC'], 
              'singapore':['^STI'],
              'australia':['^AXJO'], 'india':['^BSESN'],
              'indonesia':['^JKSE'], 'malaysia':['^KLSE'], 'new zealand':['^NZ50'],
              'south korea':['^KS11'], 'taiwan':['^TWII'], 'canada':['^GSPTSE'],
              'mexico':['^MXX'], 'chile':['^IPSA'], 
              'hong kong':['^HSI'],
              'argentina':['^MERV'], 'united kingdom':['^FTSE']}

############################## USER INPUT ############################

print "Please enter 5 countries from this list to analyse its exchange correlation \n"
print "Select your 1st country"
print dictionary.keys()
country1 = raw_input("\n")

    
while country1 not in dictionary:
    country1 = raw_input("Please Select your 1st again exactly as it is in the list")
    
print "1st loaded sucessfuly"
    
 
    
print "Please enter a 2nd country from this list to analyse its exchange correlation \n"
print dictionary.keys()
country2 = raw_input("\n")

    
while country2 not in dictionary:
    country2 = raw_input("Please Select your 2nd again exactly as it is in the list")
    
print "2nd country loaded sucessfuly"



print "Please enter a 3rd country from this list to analyse its exchange correlation \n"
print dictionary.keys()
country3 = raw_input("\n")

    
while country3 not in dictionary:
    country3 = raw_input("Please Select your 3rd again exactly as it is in the list")
    
print "3rd country loaded sucessfuly"



print "Please enter a 4th country from this list to analyse its exchange correlation \n"
print dictionary.keys()
country4 = raw_input("\n")

    
while country4 not in dictionary:
    country4 = raw_input("Please Select your 4th again exactly as it is in the list")
    
print "4th country loaded sucessfuly"
    

print "Please enter a 5th country from this list to analyse its exchange correlation \n"
print dictionary.keys()
country5 = raw_input("\n")

    
while country5 not in dictionary:
    country5 = raw_input("Please Select your 5th again exactly as it is in the list")
    
print "5th country loaded sucessfuly"

    
#### Let's keep the exchange symbols in these var symbols #####

symbol1 = dictionary[country1]
symbol2 = dictionary[country2]
symbol3 = dictionary[country3]
symbol4 = dictionary[country4]
symbol5 = dictionary[country5]



countries = [country1, country2, country3, country4, country5]
stocks = [symbol1[0], symbol2[0],  symbol3[0], symbol4[0], symbol5[0]]
stocks_data = data.DataReader(stocks, 'yahoo', start, end)['Close']

stocks_data.describe()

stocks_data.head()

stocks_ln = pd.DataFrame()
for col in stocks_data:
    if col not in  stocks_ln:
        stocks_ln[col] = np.log(stocks_data[col]).diff()

stocks_ln.head()

corr_stocks = stocks_ln.corr()
corr_stocks
print corr_stocks

###########################################################################

stocks_data2 = stocks_data.asfreq('M').ffill()
stocks_ln2 = pd.DataFrame()
for col in stocks_data2:
    if col not in  stocks_ln2:
        stocks_ln2[col] = np.log(stocks_data2[col]).diff()

stocks_ln2.head()

corr_stocks2 = stocks_ln2.corr()
corr_stocks2
print corr_stocks2



sns.heatmap(corr_stocks2, 
            xticklabels=countries,
            yticklabels=countries, cmap="YlGnBu")




from pandas.tools.plotting import scatter_matrix
scatter_matrix(stocks_ln, figsize=(16,12), alpha = 0.3)
#plt.show()



print "Now we will plot a heat map for the matrix of correlations"
print "\n"


######################CORRELATION GRAPHS##############################

import matplotlib.pyplot as plt

# Plot the rolling correlation
#return_data['MSFT'].rolling(window=252).corr(return_data['AAPL']).plot()

#xxx = stocks_ln2.rolling(3).corr()
plt.show()

correlation12 = stocks_ln2[symbol1[0]].rolling(12).corr(stocks_ln2[symbol2[0]])
correlation12.plot(title = "12 month rolling correlation between %s and %s" %(country1, country2))

plt.show()

correlation13 = stocks_ln2[symbol1[0]].rolling(12).corr(stocks_ln2[symbol3[0]])
correlation13.plot(title = "12 month rolling correlation between %s and %s" %(country1, country3))

plt.show()

correlation14 = stocks_ln2[symbol1[0]].rolling(12).corr(stocks_ln2[symbol4[0]])
correlation14.plot(title = "12 month rolling correlation between %s and %s" %(country1, country4))

plt.show()

correlation15 = stocks_ln2[symbol1[0]].rolling(12).corr(stocks_ln2[symbol5[0]])
correlation15.plot(title = "12 month rolling correlation between %s and %s" %(country1, country5))

plt.show()

correlation23 = stocks_ln2[symbol2[0]].rolling(12).corr(stocks_ln2[symbol3[0]])
correlation23.plot(title = "12 month rolling correlation between %s and %s" %(country2, country3))

plt.show()

correlation24 = stocks_ln2[symbol2[0]].rolling(12).corr(stocks_ln2[symbol4[0]])
correlation24.plot(title = "12 month rolling correlation between %s and %s" %(country2, country4))

plt.show()

correlation25 = stocks_ln2[symbol2[0]].rolling(12).corr(stocks_ln2[symbol5[0]])
correlation25.plot(title = "12 month rolling correlation between %s and %s" %(country2, country5))

plt.show()

correlation34 = stocks_ln2[symbol3[0]].rolling(12).corr(stocks_ln2[symbol4[0]])
correlation34.plot(title = "12 month rolling correlation between %s and %s" %(country3, country4))


plt.show()

correlation35 = stocks_ln2[symbol3[0]].rolling(12).corr(stocks_ln2[symbol5[0]])
correlation35.plot(title = "12 month rolling correlation between %s and %s" %(country3, country5))

plt.show()

correlation45 = stocks_ln2[symbol4[0]].rolling(12).corr(stocks_ln2[symbol5[0]])
correlation45.plot(title = "12 month rolling correlation between %s and %s" %(country4, country5))

# Show the plot
plt.show()

######################################################################################################
################################ EVERYTHING IN THE SAME GRAPH ########################################
######################################################################################################

correlation12 = stocks_ln2[symbol1[0]].rolling(12).corr(stocks_ln2[symbol2[0]])
graph1 = correlation12.plot(label = "%s and %s" %(country1, country2))
graph1


correlation13 = stocks_ln2[symbol1[0]].rolling(12).corr(stocks_ln2[symbol3[0]])
graph2 = correlation13.plot(label = "%s and %s" %(country1, country3))
graph2


correlation14 = stocks_ln2[symbol1[0]].rolling(12).corr(stocks_ln2[symbol4[0]])
graph3 = correlation14.plot(label = "%s and %s" %(country1, country4))
graph3


correlation15 = stocks_ln2[symbol1[0]].rolling(12).corr(stocks_ln2[symbol5[0]])
graph4 = correlation15.plot(label = "%s and %s" %(country1, country5))
graph4


correlation23 = stocks_ln2[symbol2[0]].rolling(12).corr(stocks_ln2[symbol3[0]])
graph5 = correlation23.plot(label = "%s and %s" %(country2, country3))
graph5


correlation24 = stocks_ln2[symbol2[0]].rolling(12).corr(stocks_ln2[symbol4[0]])
graph6 = correlation24.plot(label = "%s and %s" %(country2, country4))
graph6


correlation25 = stocks_ln2[symbol2[0]].rolling(12).corr(stocks_ln2[symbol5[0]])
graph7 = correlation25.plot(label = "%s and %s" %(country2, country5))
graph7


correlation34 = stocks_ln2[symbol3[0]].rolling(12).corr(stocks_ln2[symbol4[0]])
graph8 = correlation34.plot(label = "%s and %s" %(country3, country4))
graph8



correlation35 = stocks_ln2[symbol3[0]].rolling(12).corr(stocks_ln2[symbol5[0]])
graph9 = correlation35.plot(label = "%s and %s" %(country3, country5))
graph9

correlation45 = stocks_ln2[symbol4[0]].rolling(12).corr(stocks_ln2[symbol5[0]])
graph10 = correlation45.plot(label = "%s and %s" %(country4, country5))
graph10

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)


plt.show()
