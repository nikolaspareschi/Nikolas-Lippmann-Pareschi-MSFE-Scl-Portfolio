

# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 16:56:09 2017

"""

class initalpha:
    
    def __init__(self):
        self.alpha = float(2)
    
    def getalpha(self):
        return self.alpha
    
    def setalpha(self, alpha):
        self.alpha = float(alpha)
    

joao = initalpha()
#print(joao.alpha)
#print(joao.getalpha())
#print(joao.setalpha(5))
#print(joao.getalpha())