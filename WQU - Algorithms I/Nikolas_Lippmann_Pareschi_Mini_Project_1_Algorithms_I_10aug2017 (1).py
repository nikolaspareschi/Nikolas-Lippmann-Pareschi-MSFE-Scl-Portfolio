# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:35:45 2017

@author: nikolasp
"""


import math
from scipy import stats

#Our first step is to define the vars as given in the assignment:

s0 = 34
E = 34
r = 0.001
t = 1
c = 2.7240

# Where S0 is the Stock Price,  E is the exercise price of the call option,
# r is the risk free rate , t is the time to expiry and c is the call option
# price according to Black-Scholes Model.


# Now let´s define an initial value of volatility to be used
# in the Newton Raphson algorithm:

sigma = 0.10

#Let`s define a list. 

sig = list()
sig.append(sigma)

# We defined the list in this way so we can append values
# to it continuously and doing so we avoid the error:
# list assignment index out of range if there is no quick convergence when
# we apply the Newton-Raphson algorithm.





#Newton-Raphson algorithm

for i in range (2, 100):
    d1 = (math.log(s0/E) + (r + (sigma**2)/2)*t)/(sigma*math.sqrt(t))
    d2 = d1 - sigma*math.sqrt(t)
    
    # We need to be carefull here, we use the normal CDF to compute the f(sigma)
    
    f = s0*stats.norm.cdf(d1, 0.0, 1.0) - E * math.exp(-r * t) * stats.norm.cdf(d2, 0.0, 1.0) - c

    # Derivatives of d1 and d2:
        
    d11 = ((sigma**2)*t*math.sqrt(t)-(math.log(s0/E)+(r + (sigma**2)/2)*t)*math.sqrt(t))/(sigma**2*t)
    d22 = d11 - math.sqrt(t)
    
    # Derivative of f(sigma)
    # We need to be carefull here, it uses the normal pdf
    
    f1 = s0*stats.norm.pdf(d1, 0.0, 1.0)*d11-E*math.exp(-r*t)*stats.norm.pdf(d2, 0.0, 1.0)*d22
                     
    #Update sigma:
        
    sigma = sigma - f/f1
    sig.append(sigma)
    if (abs(sig[i-1]-sig[i-2]) <  10**-8):
        break

# Here we print the list of the updated volatilities
   
print('The list of our calulated implied volatilities is ', sig)

# And now only the last volatility wthin the desired margin of error

print('Implied vol: %.13f%%' % (sig[i-1] * 100))

