# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 18:23:07 2018

@author: Nikolas
"""


import numpy as np



A = np.array([[25, 33], [30, 36]])
B = np.array([[9, 10], [13, 12]])
sigma_r = np.array([0, 1])
sigma_c = np.array([.5, .5])

print 'Case 1 utilities: Firm 2 randomizing 50%/50% and Firm 1 staying passive'
print np.dot(sigma_r, np.dot(A, sigma_c)), np.dot(sigma_r, np.dot(B, sigma_c))

# Case 2: Firm 2 randomizing 50%/50% and Firm 1 staying agressive

A = np.array([[25, 33], [30, 36]])
B = np.array([[9, 10], [13, 12]])
sigma_r = np.array([1, 0])
sigma_c = np.array([.5, .5])
print 'Case 2 utilities: Firm 2 randomizing 50%/50% and Firm 1 staying agressive'
print np.dot(sigma_r, np.dot(A, sigma_c)), np.dot(sigma_r, np.dot(B, sigma_c))


# Case 3: Firm 2 staying passive and Firm 1 staying agressive

A = np.array([[25, 33], [30, 36]])
B = np.array([[9, 10], [13, 12]])
sigma_r = np.array([1, 0])
sigma_c = np.array([0, 1])
print 'Case 3 utilities: Firm 2 staying passive and Firm 1 staying agressive'
print np.dot(sigma_r, np.dot(A, sigma_c)), np.dot(sigma_r, np.dot(B, sigma_c))

# Case 4: Firm 2 staying agressive and Firm 1 staying agressive

A = np.array([[25, 33], [30, 36]])
B = np.array([[9, 10], [13, 12]])
sigma_r = np.array([1, 0])
sigma_c = np.array([1, 0])
print 'Case 4 utilities: Firm 2 staying agressive and Firm 1 staying agressive'
print np.dot(sigma_r, np.dot(A, sigma_c)), np.dot(sigma_r, np.dot(B, sigma_c))


# Case 5: Firm 2 staying passive and Firm 1 staying passive

A = np.array([[25, 33], [30, 36]])
B = np.array([[9, 10], [13, 12]])
sigma_r = np.array([0, 1])
sigma_c = np.array([0, 1])
print 'Case 5 utilities: Firm 2 staying passive and Firm 1 staying passive'
print np.dot(sigma_r, np.dot(A, sigma_c)), np.dot(sigma_r, np.dot(B, sigma_c))


# Case 6: Firm 2 staying agressive and Firm 1 staying passive

A = np.array([[25, 33], [30, 36]])
B = np.array([[9, 10], [13, 12]])
sigma_r = np.array([0, 1])
sigma_c = np.array([1, 0])
print 'Case 6 utilities: Firm 2 staying agressive and Firm 1 staying passive'
print np.dot(sigma_r, np.dot(A, sigma_c)), np.dot(sigma_r, np.dot(B, sigma_c))


print ''
print 'Per inspection we see that the equlibrium is achived in case 6. If Firm 1 goes agressive it has a lower utility. If Firm 2 goes passive it has a lower utility also'
