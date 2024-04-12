# -*- coding: utf-8 -*-
"""
Created on Fri May 04 19:47:50 2018

@author: Nikolas

A company is investing in two securities, x1 and x2.  The risk management division of the company indicated the following constraints to the investment strategy:

Short selling is not allowed

The company must not buy more than 400 units of x1

The total volume must not exceed 800 for every unit of x1 and x2 invested

The total volume must not exceed 1,000 for every 2 units of x1 invested and 1 unit of x2 invested

The total number of units is maximized considering that, for each 3 units of x1 security, 2 units of x2 security must be bought

The company requests the following from you:

Indicate the objective function.
Write the optimization problem.
Find x1 and x2 values that maximize the objective function and explain the algorithm.

"""

from pulp import *


problem_1 = LpProblem('Simplex Algorithm - P1', LpMaximize)

'''2 constraints were placed in our definition of X1 and X2, not short selling
allowed, so the second argument is 0, meaning that this is the lower bound and
the company must not buy more than 400 units of x1, so the third argument of X1 is
400'''

x1 = LpVariable('X1', 0, 400, LpInteger)
x2 = LpVariable('X2', 0, None, LpInteger)

'''Let's define our objective funcion'''

problem_1  += x1 + x2


'''
Constraints
	
'''
# The total volume must not exceed 800 for every unit of x1 and x2 invested

problem_1 += x1 + x2 <= 800

# The total volume must not exceed 1,000 for every 2 units of x1 invested and 1 unit of x2 invested

problem_1 += 2*x1 + 1*x2 <= 1000

# The total number of units is maximized considering that, for each 3 units of x1 security, 2 units of x2 security must be bought

problem_1 += x1*0.333333333 - x2*0.5 == 0

# Solving the problem

problem_1.writeLP('problem_1.lp')
problem_1.solve()

print "\n","The numbers shares of X1 and X2 that maximize the objective function are:", "\n"

for i in problem_1.variables():
    print "\t", i.name, "-", i.varValue, "Shares", "\n"