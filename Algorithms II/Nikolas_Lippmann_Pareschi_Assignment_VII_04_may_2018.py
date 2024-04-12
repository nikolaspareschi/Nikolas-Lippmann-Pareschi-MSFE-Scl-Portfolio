# -*- coding: utf-8 -*-
"""
Created on Sat May 05 20:54:07 2018

@author: Nikolas
"""


from pulp import *


problem_3 = LpProblem('Assignment 7: Branch and Bound', LpMinimize)

'''In the definition of our variables we define that the constraints should be
 at least equal to zero. Cutting plane methods work by solving a non-integer
 linear program, the linear relaxation of the given integer program. We so solve the
 continuous problem'''

x1 = LpVariable('X1', 0, None, LpContinuous)
x2 = LpVariable('X2', 0, None, LpContinuous)



'''Let's define our objective funcion'''


problem_3  += 4*x1 +5*x2


'''
Constraints
	
'''

# Constraint 1

problem_3 += x1 + 4*x2 >= 5

# Constraint 2

problem_3 += 3*x1 + 2*x2 >= 7



# Solving the problem

problem_3.writeLP('problem_3.lp')
problem_3.solve()

print "\n","The numbers for X1 and X2 that minimize the relaxed objective function are:", "\n"

for i in problem_3.variables():
    print "\t", i.name, "-", i.varValue, "\n"
    
print "\n","We have from our material that if the objective function coefficients are integer, then for minimization, the optimal objective for (IP) is greater than or equal to the “round up” of the optimal objective for (LR).", "\n"

print "\n","As this is our case the optimal objective may be be X1 = 2 and X2 = 1 . These numbers obey the constraints and as the objective function is a sum, any other number for X1 and X2 will imply a higher f(x) by induction. It follows that X1 and X2 minimizes the objective.", "\n"




problem_4 = LpProblem('Assignment 7: Branch and Bound', LpMinimize)

'''In the definition of our variables we define that the constraints should be
 at least equal to zero. Cutting plane methods work by solving a non-integer
 linear program, the linear relaxation of the given integer program. We so solve the
 continuous problem'''

x11 = LpVariable('X1', 0, None, LpInteger)
x22 = LpVariable('X2', 0, None, LpInteger)



'''Let's define our objective funcion'''


problem_4  += 4*x11 +5*x22


'''
Constraints
	
'''

# Constraint 1

problem_4 += x11 + 4*x22 >= 5

# Constraint 2

problem_4 += 3*x11 + 2*x22 >= 7



# Solving the problem

problem_4.writeLP('problem_3.lp')
problem_4.solve()

print "\n","The numbers for X1 and X2 that minimize the objective function are:", "\n"

for i in problem_4.variables():
    print "\t", i.name, "-", i.varValue, "\n"