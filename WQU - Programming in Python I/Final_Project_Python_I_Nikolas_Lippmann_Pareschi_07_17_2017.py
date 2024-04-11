# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 00:29:01 2017

@author: Nikolas
"""


# Part 1 - Data Collection

# I have chosen TSLA. I write into the csv file using Python to explain the why,
# as asked in the assignment.
#
# I have used yahoo finance to download the monthly prices
# First I entered in yahoo: https://finance.yahoo.com/quote/TSLA/history?p=TSLA
# Then I changed the frequency to monthly. Then pressed apply. Then downloaded data.
# After that I have moved the file to the working directory in Python. 

# The yahoo data comes with the close prices of the first day of each month,
# except for the last day of the data, which is not used in our analysis.
# The data is of sequential months.

# I have saved the data as an CSV / Excel file and attached.   
    
# Part 2 - The program performs the exponential smooth analysis asking the 
# user input for the alpha. A value is predicted for the next month and the 
# graph is ploted. If the user does not like the forecasted value
# he can change the alpha. 

# I have coded in a way that the graph shoud be displayed BEFORE the user
# answers if it is a good forecast. But the graph only really appears
# after all the program is executed. Maybe it is a Spyder bug or maybe I did
# something wrong. Until the program is terminated all the graph windows
# appear blank with the text not responding. Python 3.6 / Spyder Anaconda
# distribution. UPDATE (07/17/2017): I tested the py files in another computer
# and the error didn´t happen.
#
# We have received an e-mail from Professor Daniel Yoo on 12 july in which he asked us
# to use lists, dictionaries, for and while loops, exceptions, classes and
# object oriented programming.
#
# I have programmed the linear regression equation using the statslibrary.py file
# The statslibrary.py file contains the functions used in the calculus (standard
# deviation, mean, coefficient of correlation) of the linear regression and
# was programmed by Professor David Hays in one of our Python I / WQU lectures.

# Checklist from the e-mail:
    
# USE OF LISTS = DONE
# USE OF DICTIONARIES = DONE
# EXCEPTION HANDLING = DONE
# WHILE LOOP = DONE
# FOR LOOP = DONE
# USE OF CLASSES = DONE
# IMPORT = DONE
# LOOP THROUGH THE PROGRAM ASKING TO REPEAT = DONE

# Questions from the file:
    
# What do you will believe it will happen when alpha = 1?


#Ans: When alpha = 1 the predicted F(t+1) will be the value from the
# actual series in time (t).


import csv
import statslib as s
from matplotlib import pyplot as plt
from initalpha import initalpha as initalpha


# First we create a function to calculate the exponential_smoothing

def exponential_smoothing(alpha, s):
    
    s2 = s[:]   
    for i in range(1, len(s)):
        s2[i] = alpha*s[i-1]+(1-alpha)*s2[i-1]
    return s2



# We will create this list like in the excel file with the Final Project 
# instructions to calculate the time series of the exponential smooth
# and to plot all the graphs.

TS = [1, 2 , 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# This second list will be used to calculate the linear regression equation to predict
# the future. If we use the 12th month we will be using the future, so we must
# stop at the 11th month. This problem does not occur in the exponential
# smoothing function.

TS2 = [1, 2 , 3, 4, 5, 6, 7, 8, 9, 10, 11]

# We will now open out TSLA and save to a list called your_list

with open('TSLA.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)
    f.close()
  

# We will now open the file again and write into it the reasons for picking TSLA.      

with open('TSLA.csv','a',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(["# I have chosen Tesla Motors, its nasdaq code is TSLA. I have chosen TSLA because it was one of the companies that have risen more last years and several analysts are saying that its value is unreal. It will be interesting to see what will happen in next months or even years."])
    f.close()
  
  
# Let´s check if the dates provided from yahoo are from the first day of the month

Date = list()
for i in range (1, 13):
    Date.append(your_list[i][0])

print("The dates for the close values of TSLA are", Date)
print('\n')


# We want the close values, the CSV file has other values that we are not
# interested.

Closes_tsla = list()

for i in range (1, 13):
    Closes_tsla.append(float(your_list[i][4]))

print("The values of the close prices of TESLA are:", Closes_tsla)
print('\n')

# Let´s put those dates and close values into a dictionary!


Tesla_Motors = {Date[0]:Closes_tsla[0], Date[1]:Closes_tsla[1], Date[2]:Closes_tsla[2], Date[3]:Closes_tsla[3], Date[4]:Closes_tsla[4], Date[5]:Closes_tsla[5], Date[6]:Closes_tsla[6], Date[7]:Closes_tsla[7], Date[8]:Closes_tsla[8], Date[9]:Closes_tsla[9], Date[10]:Closes_tsla[10], Date[11]:Closes_tsla[11]}
print("Dictionary with Dates and Close prices from TESLA", Tesla_Motors)
print('\n')

# Let's put the close values of TSLA into a list without the last month so we
# can use these value to calculate the linear regression equation and then
# use it to predict the last month.

Closestsla_Linear_Regression = list()

for i in range (1, 12):
    Closestsla_Linear_Regression.append(float(your_list[i][4]))


print("The values from the close prices of TESLA that we will use to calculate the Linear Regression equation are:", Closestsla_Linear_Regression)
print('\n')

# We have created the control_var so we will quit the while loop when the user
# press any key different from 0.

Control_var = str(0)
alphauser = initalpha()

while Control_var == str(0):

    
    
    try:
    
# We will initialize the alpha using the initalpha class from initalpha.py file.
# Alpha is initialized with a value of 2
        alphauser = initalpha()

        while alphauser.getalpha() > 1.0 or alphauser.getalpha() < 0.0:
            alphauser.setalpha((input("Please enter a value for alpha between 0 included and 1 included \n")))
            print("We will make our forecast with the alpha choosen:", alphauser.getalpha())
            print('\n')
            if alphauser.getalpha() > 1.0 or alphauser.getalpha() < 0.0:
                print("You have not entered a value for alpha between 0 included and 1 included")
                print('\n')

# If the user presses a letter the ValueError occurs.
    
    except ValueError:
        print("You have not inserted a number as requested. The program will assume alpha to be 1. Please run the program again if you would like to see the analysis with a different alpha.")
        print('\n')
        alphauser.setalpha(1)

# For other errors we have this except:
   
    except:
        print("You have not inserted a number as requested. The program will assume alpha to be 1. Please run the program again if you would like to see the analysis with a different alpha.")
        print('\n')
        alphauser.setalpha(1)
        
    
    Closes_tsla2 = Closes_tsla[:]
    Exp_smoo = list()
    Exp_smoo = exponential_smoothing(alphauser.getalpha(), Closes_tsla2)
    print("Exponential Smoothing forecast:", Exp_smoo)
    print('\n')
    
    print(s.Least_Squares_Regression_Line(TS2, Closestsla_Linear_Regression))
    print('\n')
    
    b, a = s.Least_Squares_Regression_Line(TS2, Closestsla_Linear_Regression)
    
    LS = list()
    for i in range(0,12):
        LS.append(float(a + b*TS[i]))
    
    print("The linear regression coefficients are:",a, b)
    print('\n')
    print("Linear Regression forecast:",LS)
    print('\n')


    
    plt.figure(figsize=(14, 6), dpi=80)
    plt.plot(TS, Closes_tsla, color='blue', label="Actual Value")
    plt.plot(TS, LS, color='red', label="Linear Regression Predicted Value")
    plt.plot(TS, Exp_smoo, color='green', label="Exponential Smooth Predicted Value")
    plt.legend(loc='lower right')
    plt.title('Projects')
    plt.xlabel('Time Series')
    plt.ylabel('TSLA values')
    plt.show()
    
    print("The predicted price for the next month using exponential smoothing is:", Exp_smoo[11])
    print('\n')
    print("The predicted price for the next month using linear regression is:", LS[11])
    print('\n')
    print("The real value at the month in which we are predicting the price is: ", Closes_tsla[11])
    print('\n')
    print("The correlation coefficient between the Linear Regression Equation Prices and the Real Prices are", s.Correlation_Coefficient(LS,Closes_tsla))
    print('\n')
 
# We ask the user to input 0 if they want to do another simulation.    
    
    Control_var = input("Do you think the alpha chosen is correcting predicting the future? If not, please enter 0 to choose another alpha. Please enter anything different from zero if you want to quit. \n")

