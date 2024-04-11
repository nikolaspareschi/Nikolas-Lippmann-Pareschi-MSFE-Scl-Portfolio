# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 12:30:54 2018

@author: Nikolas
"""

# I took the CPI and IP data from www.tradingeconomics. Forex data from reuters

BRLUSD = [0.3174, 0.3216, 0.3202, 0.3148, 0.3099, 0.3024, 0.3199, 0.3177, 0.3163, 0.3045, 0.3056, 0.3019, 0.3139]
BRL_CPI = [4793.85, 4809.67, 4821.69, 4828.44, 4843.41, 4832.27, 4843.87, 4853.07, 4860.83, 4881.25, 4894.92, 4916.46, 4930.72  ]
BRL_IP = [0, 2, -0.2, 2, -4.4, 4.6, 0.8, 2.9, 4, 2.6, 5.5, 4.7, 4.3]



# Linear Regression meethod is used for prediction

# Model 1 for calculating the equilibrium in BRLUSD
# BRLUSD movement is determined by Brazil industrial production

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


PD = [BRL_CPI, BRL_IP]

linear_regression1=sm.OLS(BRLUSD,BRL_IP)
estimation1=linear_regression1.fit()
estimation1.summary()
equilibrium_fx1=estimation1.fittedvalues
equilibrium_fx1
print estimation1.summary()
plt.plot(equilibrium_fx1)
plt.plot(BRLUSD)
plt.title('BRLUSD and Equilibrium predicted by Industrial Production')
plt.show()

#Model 2 for calculating the equilibrium in BRLUSD
# BRLUSD movement is determined by Brazil CPI evolution

linear_regression2=sm.OLS(BRLUSD,BRL_CPI)
estimation2=linear_regression2.fit()
print estimation2.summary()
equilibrium_fx2=estimation2.fittedvalues
equilibrium_fx2
plt.plot(equilibrium_fx2)
plt.plot(BRLUSD)
plt.title('BRLUSD and Equilibrium predicted by CPI') 
plt.show()



def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results

zz = reg_m(BRLUSD, PD)
equilibrium_fx3=zz.fittedvalues
print zz.summary()

plt.plot(equilibrium_fx3)
plt.plot(BRLUSD)
plt.title('BRLUSD and Equilibrium predicted by Industrial Production and CPI')
plt.show()