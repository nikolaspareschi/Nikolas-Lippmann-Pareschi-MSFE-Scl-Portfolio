# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 01:22:41 2018

@author: nikol
"""

def change(n, coins_to_trade, coins_summed):
    if sum(coins_summed) == n:
        yield coins_summed
    elif sum(coins_summed) > n:
        pass
    elif coins_to_trade == []:
        pass
    else:
        for c in change (n, coins_to_trade[:], coins_summed + [coins_to_trade[0]]):
            yield c
        for c in change (n, coins_to_trade[1:], coins_summed):
            yield c
            
n = 100
coins = [1, 5, 10, 25]
    
solutions = [s for s in change(n, coins, [])]
for s in solutions:
    print s

print 'The solution that uses less coins is:', min(solutions, key = len)