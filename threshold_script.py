# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:08:27 2021

@author: U378246
"""

import pandas as pd
import os
    
def threshold(df, filename, alarm=False):
    threshold_list = [0]*14
    counter = 0
    
    for i,j in enumerate(df):
        if j == -1:
            if counter == len(threshold_list):
                counter = 0
                list_df = pd.DataFrame(threshold_list, columns=['Timetag'])
                alarm = True
                list_df['alarm'] = alarm
                appendDFToCSV_void(list_df, filename)
                return(threshold(df.iloc[i:], filename, alarm))

            threshold_list[counter] = df.index[i]
            if counter >= 1:
                diff = threshold_list[counter] - threshold_list[counter-1]
                if diff.days > 1:
                    list_df = pd.DataFrame(threshold_list, columns=['Timetag'])
                    alarm = False
                    list_df['alarm'] = alarm
                    appendDFToCSV_void(list_df, filename)
                    threshold_list = [0]*14
                    counter = 0

                else:
                    counter += 1
            else:
                counter += 1
                
    return (threshold_list,alarm)

def appendDFToCSV_void(df, csvFilePath, sep=","):
    
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)