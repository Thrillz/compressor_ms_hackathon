# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# This Jupyter file runs the entire anomaly detection code for the compressor insights use case.
# 
# from the import statements below, the main_util_script, does all the data cleaning and transformation and provides a dataframe and a list as output. To run the util script, the pressure ranges of the racks need to be supplied as input. The dataframe provided as output here can be split for different uses, those needed for compressor quadrants are selected and written to file, while ml_df, refers to data used as input for anomaly detection.
# 
# Training_script trains the model, saves the scaling parameters used, model trained and results of the prediction to a file unique to the rack that calls the script. Prediction script is run if a result.csv file exists for the rack.
# 
# The threshold script contains the dates that were flagged as anomalies as well as the alarm parameter that is either set to true or false. Alarm parameter is set to true when we have 14 consecutive days flagged as anomalies.

# %%
import os.path
import pandas as pd
import matplotlib.pyplot as plt
from pressure_range_script import pressure_range_dict
from main_util_script import aggregate_dataframe
# from training_script import model_train
from threshold_script import threshold
from prediction import run_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn import model_selection
from joblib import dump
import argparse
import numpy as np
from azureml.core import  Run

# 0.0 Parse input arguments


def init():
    global current_run
    current_run = Run.get_context()
    parser = argparse.ArgumentParser("split")
    parser.add_argument("--output_path", type=str, required=True, help="input target column")
    global args
    args, _ = parser.parse_known_args()
    

def appendDFToCSV_void(df, csvFilePath, sep=","):
    if not os.path.isfile(csvFilePath):
        df.to_csv(os.path.join(args.output_path, csvFilePath), mode='a', index=True, sep=sep)
    else:
        df.to_csv(os.path.join(args.output_path, csvFilePath), mode='a', index=True, sep=sep, header=False)

def get_racks(df, rack_var):
    var_name = rack_var[-1:].upper()
    rack_name = df.loc[:, df.columns.str.contains('{}$|{}\s[0-9]*$|{}\s'.format(var_name, var_name, var_name))]
    return rack_name


def run(input_data):
    seperator = '_'
    result_list = []
    # 1.0 Set up output directory and the results list
    for idx, csv_file_path in enumerate(input_data):
        overall_result = {}
        colnames = ['SiteID', 'SiteName', 'AssetID', 'AssetName', 'PointName', 'DataValue', 'Timetag', 'Units', 'PropertyName']
        df = pd.read_csv(csv_file_path, names=colnames, header=None)
        site_id = df.SiteID.unique()[0]
        pressure_range = pressure_range_dict.get(site_id)
        

        data_agg = aggregate_dataframe(df,site_id, suc_pres_range=pressure_range)
        data,rack_names = data_agg.aggregate_data()

        quadrants_df = data.filter(regex='Runtime|Switch|% Capacity|Neutral|Quadrant')
        quadrants_df['SiteID'] = site_id
        quadrant_filename = 'quadrants_'+str(site_id)+'.csv'
        appendDFToCSV_void(quadrants_df, quadrant_filename)

        ml_df = data[data.columns.drop(list(data.filter(regex='% Capacity|Neutral|Quadrant')))]

        racks = [0]*len(rack_names)
        result = [0]*len(rack_names)

        for i,j in enumerate(rack_names):
            rack_names[i] = rack_names[i].replace(" ", "_")
            alarm_file = str(site_id)+ seperator+ rack_names[i]+ '_flags.csv'
            result_file= str(site_id)+ seperator+ rack_names[i]+ '_results.csv'
            racks[i] = get_racks(ml_df, j)

            train_instance = model_train(racks[i], rack_names[i], site_id)
            result[i]= train_instance.train_model()
            appendDFToCSV_void(result[i], result_file)
            threshold(result[i]['isolation_forest_pred'], os.path.join(args.output_path, alarm_file))

        overall_result = {'file_processed': result_file,
                        'rack_names': rack_names,
                        'site_id': site_id}

        result_list.append(overall_result)
            
                

    return pd.DataFrame(result_list)

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

    def scorer_f(self, estimator, X):
        return np.mean(estimator.score_samples(X))

    def isoForest(self, train_data):

        rng = np.random.RandomState(42)
        param_grid = {'contamination': ['auto', 0.1, 0.2]}
        
        clf = IsolationForest(random_state=rng)
        iso_clf = model_selection.GridSearchCV(clf, param_grid,  scoring=self.scorer_f, refit=True, cv=5, return_train_score=True)
        model = iso_clf.fit(train_data)
        model_pred = model.predict(train_data).tolist()
        
        filename = self.site_id + '_'+ self.rack_name + '_model.bin'
        dump(model, filename, compress=True)
        
        self.df = self.df.assign(isolation_forest_pred=model_pred)
        return self.df

    def train_model(self):
        scaled_data = self.scaling()
        scaled_data = scaled_data[scaled_data.columns.drop(list(scaled_data.filter(regex='Target')))]
        
        train_data = np.array(scaled_data)
        
        answer = self.isoForest(train_data)
        
        return answer

