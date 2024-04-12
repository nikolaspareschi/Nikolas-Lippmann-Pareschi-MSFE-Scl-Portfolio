# -*- coding: utf-8 -*-
"""
Created on Sat May 05 21:17:37 2018

@author: Nikolas
"""


from pulp import *

'''First let's solve the problem with the integer constraint (Without Relaxation)'''

prob = LpProblem('Problem1', LpMaximize)


'''
When we define the variable we also create the constraints: x1, x2, x3 > 0
and x1 <= 3
'''

x1 = LpVariable('X1', 0, 3, LpInteger)
x2 = LpVariable('X2', 0, None, LpInteger)
x3 = LpVariable('X3', 0, None, LpInteger)

'''
The objective function

'''
prob += 3*x1 - x2 + 2*x3

'''
The other constraints

'''


prob += 1*x1 - x2 + x3 <= 5
prob += 2*x2 + x3 <= 4

prob.writeLP('problem1.lp')

prob.solve()

print "\n","Status:", LpStatus[prob.status], "\n"


for i in prob.variables():
    print "\t", i.name, ":", i.varValue, "\n"
    
print "Optimal Xi values for integer programming: ", value(prob.objective)

'''

Now let's do the relaxation

'''
'''First let's solve the problem with the integer constraint (Without Relaxation)'''

prob2 = LpProblem('Problem1', LpMaximize)


'''
When we define the variable we also create the constraints: x1, x2, x3 > 0
and x1 <= 3
'''

x11 = LpVariable('X11', 0, 3, LpContinuous)
x22 = LpVariable('X22', 0, None, LpContinuous)
x33 = LpVariable('X33', 0, None, LpContinuous)

'''
The objective function

'''
prob2 += 3*x11 - x22 + 2*x33

'''
The other constraints

'''


prob2 += 1*x11 - x22 + x33 <= 5
prob2 += 2*x22 + x33 <= 4

prob2.writeLP('problem1.lp')

prob2.solve()

print "\n","Status:", LpStatus[prob.status], "\n"


for i in prob2.variables():
    print "\t", i.name, ":", i.varValue, "\n"
    
print "Optimal Xi values: ", value(prob.objective)

