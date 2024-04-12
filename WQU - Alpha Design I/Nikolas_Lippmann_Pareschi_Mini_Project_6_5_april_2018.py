# -*- coding: utf-8 -*-
"""
Created on Thu Apr 05 13:30:46 2018

@author: Nikolas
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


'''

Part 1

Consider a mutual fund with beta of 0.8 which has an expected rate of return
 of 14%. If risk-free rate of return is rf = 5%, and you expect the rate of 
 return on market portfolio to be 15%.
Use Python to address the following questions:
    
1.	Would you be interested in investing in the fund? If so, what is the Alpha of the fund.
2.	What passive portfolio comprised of a market-index portfolio and a
 money-market account would have the same beta as the fund? 
 
Note: show that the difference between the expected rate of return on this
 passive portfolio and that of the fund equals the alpha from question 1.

'''

beta = 0.8
expected_return = 0.14
risk_free = 0.05
market_return = 0.15


# Accordingly to CAPM

return_asset_capm = risk_free + beta*(market_return - risk_free)
print "CAPM predicts a return of", return_asset_capm
print "The real return of the asset was", expected_return

alpha = expected_return - return_asset_capm

print "We have an alpha of", alpha
print "It is a good investment that has a beta lower than 1 (low variance / risk) and returns above CAPM. I would invest in it."

#
# 0.14 = 0.15*x + (1-x)*0.05 
# 0.14 = 0.15*x + 0.05 - 0.05*x
# 0.09 = 0.1*x
# x = 0.9
 
'''

Part 2

Consider the following data for a one-factor economy. All portfolios are assumed to be well diversified.

Portfolio A has an ER of 12%-20% and a Beta of 1.2 - 1.7
Portfolio B has an ER of 6%-9% and a Beta of 0.0 - 0.9


Note: Consider that Expected Return changes in step sizes of 1% (e.g. Portfolio
 A can have expected returns of 12%,13%,…19%,20%) and Beta change in step sizes
 of 0.1 (e.g Portfolio F can have Beta values of 0.0,0.1,0.2,….0.8,0.9)
 
Suppose that another portfolio, portfolio E, is well diversified with a beta of
 0.6 and expected return of 8%.
For which range of values of Expected Return and Beta would an arbitrage
opportunity exist? 

Develop a simple strategy in Python to exploit the most of the juice out of the
arbitrage opportunity strategy for each of the cases 
 
Plot the risk-reward profiles of these strategies (for each set of combination
 of Expected Return and Beta for Portfolio A & F) and discuss.


'''


def CAPM(rf,market,beta):
    return rf+(beta*(market-rf))



betas = []
returns = []
beta = 1.2
rf = -7.72
market = 8.8
for x in range(6):
    betas+=[beta]
    r = CAPM(rf,market,beta)
    returns +=[r]
    beta+=.1
plt.title("Portfolio A in blue Portfolio B in Red")
plt.legend(loc='upper left')
plt.plot(betas,returns,"blue")

betas = []
returns = []
beta = 0
rf = 6.0
market = 9.3333
for x in range(10):
    betas+=[beta]
    r = CAPM(rf,market,beta)
    returns +=[r]
    beta+=.1
plt.plot(betas,returns,"red")
plt.show()

'''
Part 3

Suppose the economy can be in one of the following two states:
1.	Boom or “good” state and
2.	Recession or “bad” state.
Each state can occur with an equal opportunity. The annual return on the market
 and a certain security X in the two states of the economy are as follows:

•	Market: at the end of the year, the market is expected to yield a return of 
30% in the good state and a return of (-10%) in the bad state;
•	Security X: at the end of the year, the security is expected to yield a
 return of 40% in the good state and a return of (-35%) in the bad state;
Furthermore, assume that annual risk-free rate of return is 5%.

1.	Write a Python Program to calculate the beta of security X relative to the
market.
2.	Furthermore, calculate the alpha of security X using CAPM.
3.	Draw the security market line (SML). Label the axes and all points
(including the market portfolio, the risk-free security, and security X) in
 the graph clearly. Identify alpha in the graph.

'''

good_market = .30
bad_market = -.10

good_security = .40
bad_security= -.35

risk_free = .05




data = {'Security': [0.40, -0.35], 'Market': [0.30, -0.15]}
                   
df = pd.DataFrame(data, index = ['Good', 'Bad'])
df

cov = df.cov() * 250
cov

cov_with_market = cov.iloc[0,1]
cov_with_market

market_var = df['Market'].var() * 250
market_var

beta = cov_with_market / market_var
beta

df['Market']
df['Security']


# CAPM EQUATION: df['Security'] = alpha2 + risk_free + beta*(df['Market'] - risk_free)

alpha = df['Security'] - risk_free - beta*(df['Market'] - risk_free)
print "The alpha of our security is: \n", alpha

plt.scatter(df['Market'],df['Security'])
plt.ylabel('Market Returns')
plt.xlabel('Security Returns')
plt.show()
