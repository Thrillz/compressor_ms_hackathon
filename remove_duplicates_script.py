# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 13:04:44 2021

@author: U378246
"""
import pandas as pd

class remove_duplicates():
    def __init__(self, df, site_id):
        self.df = df
        self.site_id = site_id
        
    def merge_columns(self, a, b):
        merged = self.df[a].fillna(self.df[b])
        return merged
    
    def find_duplicates(self):
        for i in self.df.columns:
            name = i + ' 1'
            if name in self.df.columns:
                self.df[i] = self.merge_columns(i, name)
                self.df = self.df.drop(name, axis=1)
                
            if self.site_id != 52253:
                if ' Outside Temp' in i:
                    self.df['Outside Temp'] = self.merge_columns('Outside Temp', i)
                    self.df = self.df.drop(i, axis=1)
                
        return self.df