# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 13:13:51 2018

@author: Nikolas
"""


from pandas_datareader import data
import datetime

'''
Write a simple Python program that implements your model. The program should
be able print out the expected slippages in the different order types based on
values of 9 data points (Do not ask the user to input these values manually,
rather define them as variables in your python file with proper commenting).

'''

start = datetime.datetime(2002, 3, 1)
end = datetime.datetime(2018, 3, 1)

dow = data.DataReader('^DJI',  'yahoo', start, end)
dow.head()


dow['Best_bid'] = dow['Close'] - 0.01
dow['Best_ask'] = dow['Close'] + 0.02 
dow['Slippage'] = dow['Best_bid'] - dow['Best_ask'] 
dow['Highest_Bid_Quantity'] = 3000
dow['Highest_Ask_Quantity'] = 3000
dow['Volume_that_will_be_traded'] = 9000
dow['Mean_bid_ask_volume'] = (dow['Highest_Bid_Quantity']+dow['Highest_Ask_Quantity'])/2
dow['Final_Slippage'] = dow['Slippage']*(dow['Volume_that_will_be_traded']/(dow['Mean_bid_ask_volume']))

print dow.head()

print 'The slippage that will be paid considering the volume that will be negotiated is \n', dow['Final_Slippage'].tail()
