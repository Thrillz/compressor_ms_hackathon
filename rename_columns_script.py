# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:51:08 2021

@author: U378246
"""

class renamer():
    def __init__(self):
        self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        
        else:
            self.d[x] += 1
            return "%s %d" % (x, self.d[x])