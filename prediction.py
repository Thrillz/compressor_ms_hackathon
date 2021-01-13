# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 08:36:35 2021

@author: U378246
"""

from joblib import load
import pandas as pd
import numpy as np
import os

def scaling(df, filename):
    scaler = load(filename)
    scaler_data = pd.DataFrame(scaler.transform(df), 
                              columns=df.columns, 
                              index=df.index)
    
    return scaler_data

def predict(df, filename1, filename2):
    
    model = load(filename1)
    model_pred = model.predict(np.array(df)).tolist()
    df = df.assign(isolation_forest_pred=model_pred)
    
    appendDFToCSV_void(df, filename2)
    return df

def run_predict(df, filename, filename1, filename2):
    df = df.dropna()
    data = scaling(df, filename)
    result = predict(data, filename1, filename2)
    return result

def appendDFToCSV_void(df, csvFilePath, sep=","):
    
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=True, sep=sep)
    else:
        df.to_csv(csvFilePath, mode='a', index=True, sep=sep, header=False)