# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:39:25 2018

@author: Nikolas
"""

from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import statsmodels.tsa.stattools as sttool
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Part 1 - Discuss the results provided by the regression (alpha, beta, R-squared). Would you use the current model in predicting gold prices?

# Code given:


gold=[5.264967387, 5.719262046, 6.420808929, 6.129616498, 5.927725706, 
      6.048931247, 5.888268354, 5.759847699, 5.907675246, 6.100812104,
      6.079612778, 5.942326823, 5.949496062, 5.892362186, 5.840496298,
      5.885603906,5.951033101, 5.950772752, 5.960670232, 5.802994125,
      5.683885843, 5.629669374, 5.631570141, 5.602266411, 5.735539506,
      5.895283989, 6.014130718, 6.096837563, 6.403193331, 6.544472839,
      6.770743551, 6.879715822, 7.110304209, 7.359798583, 7.41996794,
      7.252216944, 7.143933509] #logarithm of gold price

gold2=[193, 305, 615, 459, 375, 424, 361, 317, 368, 446, 437, 381, 384, 362,
       344, 360, 384, 384, 388, 331, 294, 279, 279, 271, 310, 363, 409, 444,
       604, 695, 872, 972, 1225, 1572, 1669, 1411, 1266] #gold price



plt.plot(gold2)
plt.grid()
plt.title("Gold price evolution from 1978 until 2014")
plt.xlabel("Period")
plt.ylabel("USD per troy ounce")
plt.show()

lag=gold[:-1] # gold price evolution in the previous period
lag
gold = gold[1:]

beta, alpha, r_value, p_value, std_err = stats.linregress(gold, lag)
print("Beta =",beta, "Alpha = ", alpha) # the estimated parameters

print("R-squared =", r_value**2) 

print("p-value =", p_value)

forecast_gold=np.exp(0.922226946193*7.143933509+0.428123903567) # (BETA, last month price and Alpha)
print("Forecast gold price =",forecast_gold)


# Part 2 - Multiple Regression

# In the trading community we can see that nowdays the dollar trade is a proxy for several days, commodities and equities.
# When dollar values equities and gold falls in price. When dollar devalues equities and silver rise in price.
# So we will try to improve our R2 with a multiple regression using lagged gold values and EUR/USD (not lagged)




# New gold data 10 years - monthly from Index Mundi

new_gold = [922.3,
968.43,
909.71,
888.66,
889.49,
939.77,
839.03,
829.93,
806.62,
760.86,
816.09,
858.69,
943,
924.27,
890.2,
928.65,
945.67,
934.23,
949.38,
996.59,
1043.16,
1127.04,
1134.72,
1117.96,
1095.41,
1113.34,
1148.69,
1205.43,
1232.92,
1192.97,
1215.81,
1270.98,
1342.02,
1369.89,
1390.55,
1360.46,
1374.68,
1423.26,
1480.89,
1512.58,
1529.36,
1572.75,
1759.01,
1772.14,
1666.43,
1739.00,
1639.97,
1654.05,
1744.82,
1675.95,
1649.20,
1589.04,
1598.76,
1594.29,
1630.31,
1744.81,
1746.58,
1721.64,
1684.76,
1671.85,
1627.57,
1593.09,
1487.86,
1414.03,
1343.35,
1285.52,
1351.74,
1348.60,
1316.58,
1275.86,
1221.51,
1244.27,
1299.58,
1336.08,
1298.45,
1288.74,
1277.38,
1310.59,
1295.13,
1236.55,
1222.49,
1175.33,
1200.62,
1250.75,
1227.08,
1178.63,
1198.93,
1198.63,
1181.50,
1128.31,
1117.93,
1124.77,
1159.25,
1086.44,
1075.74,
1097.91,
1199.50,
1245.14,
1242.26,
1260.95,
1276.40,
1336.65,
1340.17,
1326.61,
1266.55,
1238.35,
1157.36,
1192.10,
1234.20,
1231.42,
1266.88,
1246.04,
1260.26,
1236.85,
1283.04,
1314.07,
1279.51,
1281.90,
1264.45,
1331.30]


# EURUSD data from investing.com 10 years monthly



#EURUSD = [1.5892,
EURUSD2 = [
1.5774,
1.5617,
1.5554,
1.5756,
1.5601,
1.4674,
1.4104,
1.2733,
1.2698,
1.398,
1.2782,
1.2669,
1.3251,
1.3226,
1.4154,
1.4036,
1.425,
1.4331,
1.4636,
1.4718,
1.5009,
1.4318,
1.3864,
1.3626,
1.3512,
1.3298,
1.2306,
1.2236,
1.3048,
1.2687,
1.3633,
1.395,
1.298,
1.3379,
1.3686,
1.3802,
1.4167,
1.4802,
1.4396,
1.4506,
1.4396,
1.4379,
1.3386,
1.3857,
1.3443,
1.2948,
1.3079,
1.3326,
1.3344,
1.3241,
1.2359,
1.266,
1.2304,
1.2577,
1.2858,
1.296,
1.2986,
1.3196,
1.3579,
1.3056,
1.282,
1.3167,
1.2999,
1.301,
1.3302,
1.3222,
1.3526,
1.3584,
1.359,
1.3746,
1.3487,
1.3803,
1.3771,
1.3868,
1.3631,
1.3692,
1.3389,
1.3133,
1.2632,
1.2525,
1.2452,
1.2098,
1.1288,
1.1195,
1.0731,
1.1225,
1.0988,
1.1138,
1.0988,
1.1214,
1.1177,
1.1006,
1.0564,
1.0861,
1.0837,
1.0873,
1.138,
1.1456,
1.1132,
1.1105,
1.1174,
1.1158,
1.1241,
1.0981,
1.0588,
1.0516,
1.0798,
1.0577,
1.0652,
1.0897,
1.1244,
1.1426,
1.1842,
1.191,
1.1814,
1.1646,
1.1904,
1.1998,
1.2421]

new_gold2 = np.log(new_gold)
new_gold3 = new_gold2[:-1]
new_gold4 = new_gold2[1:120]

EURUSD2 = np.log(EURUSD2)


PD = [new_gold3, EURUSD2]

def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results

zz = reg_m(new_gold4, PD)
equilibrium_gl=zz.fittedvalues
print zz.summary()

plt.plot(equilibrium_gl)
plt.plot(new_gold2)
plt.title('Gold x Predicted gold - Log prices')
plt.show()



forecast_gold2=np.exp(0.0314*0.2168035+0.9732*7.19391119+0.1595) # this is beta*gold price from the  current period + alpha
print 'Forecasted gold price for next month using - Multiple Regression =',forecast_gold2 
