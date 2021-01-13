# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 16:26:36 2021

@author: U378246
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn import model_selection
from scipy.cluster.hierarchy import linkage, dendrogram
from joblib import dump

class model_train():
    def __init__(self, df, rack_name, site_id):
        self.df = df.dropna()
        self.rack_name = rack_name.replace(" ", "_")
        self.site_id = str(site_id)
        
    def scaling(self):
        scaler = StandardScaler()
        scaler_data = pd.DataFrame(scaler.fit_transform(self.df), 
                              columns=self.df.columns, 
                              index=self.df.index)
        
        filename = self.site_id + '_'+ self.rack_name + '_std_scaler.bin'
        dump(scaler, filename, compress=True)
        return scaler_data
    
    def dendrogram_plot(self, data):
        den = dendrogram(linkage(data, method='ward'), labels = data.index, no_plot=True)
        
        temp = pd.DataFrame(den['ivl'], columns=['Dates'])
        temp['Dates'] = pd.to_datetime(temp['Dates'])
        temp['cluster_list'] = pd.Series(den['color_list'])
        temp = temp.fillna(method='ffill')
        temp.set_index('Dates', inplace=True)
        temp = temp.sort_index()
        self.df = self.df.assign(cluster_output=temp['cluster_list'])
        
        return self.df
    
    def scorer_f(self, estimator, X):
        return np.mean(estimator.score_samples(X))
    
    def isoForest(self, train_data, test_data):
    
        rng = np.random.RandomState(42)
        param_grid = {'contamination': ['auto', 0.1, 0.2]}
        
        clf = IsolationForest(random_state=rng)
        iso_clf = model_selection.GridSearchCV(clf, param_grid,  scoring=self.scorer_f, refit=True, cv=10, return_train_score=True)
        model = iso_clf.fit(train_data)
        model_pred = model.predict(test_data).tolist()
        
        filename = self.site_id + '_'+ self.rack_name + '_model.bin'
        dump(model, filename, compress=True)
        
        self.df = self.df.assign(isolation_forest_pred=model_pred)
        return self.df
    
    def train_model(self):
        scaled_data = self.scaling()
        scaled_data = scaled_data[scaled_data.columns.drop(list(scaled_data.filter(regex='Target')))]
        clustered_data = self.dendrogram_plot(scaled_data)
        
        
        biggest_cluster = clustered_data.cluster_output.mode()[0]
        train_data = clustered_data[clustered_data['cluster_output'] == biggest_cluster]
        
        train_data = np.array(train_data.drop('cluster_output', axis=1))
        test_data = np.array(clustered_data.drop('cluster_output', axis=1))
        
        answer = self.isoForest(train_data, test_data)
        
        return answer
    
    

