# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 16:56:09 2017

@author: Nikolas
"""

#Library Functions:
    
import math

L1 = [1, 2, 3, 4 , 5, 6, 7, 8]
L2 = [3, 7, 6, 4, 6, 65, 44, 13] 

def Mean(L1):
    
 """Calculates mean, L1 is a list of numbers"""
 
 tsum = 0
 for x in range(0,len(L1)):
     tsum = tsum + L1[x]
 return tsum / len(L1)


def Sample_Standard_Deviation(L1):
    
 """Calculates sample standard deviation, L1 is a list of numbers"""
 
 Mean_of_L1 = Mean(L1)
 tsum = 0
 for x in range(0, len(L1)):
     tsum = tsum + (L1[x] - Mean_of_L1) ** 2
 return math.sqrt(tsum / (len(L1) - 1))


def Correlation_Coefficient(L1,L2): 
 meanx = Mean(L1) 
 meany = Mean(L2)
 stdx = Sample_Standard_Deviation(L1)
 stdy = Sample_Standard_Deviation(L2)
 tsum = 0
 for idx in range(0,len(L1)):
     tsum = tsum + ((L1[idx] - meanx)/stdx) * ((L2[idx] - meany) / stdy)
 return tsum / (len(L1) - 1)



def Least_Squares_Regression_Line(L1,L2):
 
 b1 = Correlation_Coefficient(L1,L2) * (Sample_Standard_Deviation(L2) / Sample_Standard_Deviation(L1))
 b0 = Mean(L2) - b1 * Mean(L1)
 return b1, b0



def Sum_of_the_Squared_Residuals(L1,L2):
    
 a, b = Least_Squares_Regression_Line(L1,L2)

 tsum = 0
 for idx in range(0,len(L2)):
     tsum = tsum + (L2[idx] - (a * L1[idx] + b)) ** 2 
 return tsum
