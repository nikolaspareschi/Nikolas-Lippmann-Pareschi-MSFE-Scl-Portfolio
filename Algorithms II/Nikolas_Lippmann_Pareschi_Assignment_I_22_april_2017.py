# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 01:21:48 2018

@author: nikol
"""

def longest_palindrome(str):
    n = len(str)

    memory = [[0 for column in range(n)] for row in range(n)]

    for i in range(n):
        memory[i][i] = str[i]

    for sub_problem in range(2, n+1):
        for row in range(n-sub_problem+1):
            column = row+sub_problem-1
            if str[row] == str[column] and sub_problem==2 :
                memory[row][column] = str[row]+str[column]
            elif str[row] == str[column]:
                memory[row][column] = str[row] + memory[row+1][column-1] + str[column];
            else:
                if(len(memory[row][column-1]) > len(memory[row+1][column]) ):
                    memory[row][column] = memory[row][column-1]
                else:
                    memory[row][column] = memory[row+1][column]
    return memory[0][n-1]

print(longest_palindrome('character'))