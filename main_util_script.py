# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 14:54:30 2021

@author: U378246
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, entropy
import logging
from remove_duplicates_script import remove_duplicates
from rename_columns_script import renamer
from collections import Counter

class aggregate_dataframe():

    def __init__(self, df, site_id, win='24h', suc_pres_range = [(11,19), (11,19)]):

        self.df = df
        self.site_id = site_id
        self.win = str(win)
        self.sub = 'compressor'
        self.suc_pres_range=suc_pres_range

    def check_df_type(self):
        if isinstance(self.df, pd.DataFrame):
            return(True)
        else:
            return(False)


    def pivot_data(self):
        self.df['Timetag'] = pd.to_datetime(self.df['Timetag'])
        self.df = self.df.assign(new_id = self.df['PointName'])
        self.df = self.df.drop_duplicates(subset = ['new_id','Timetag'])
        self.df = self.df.pivot(index = 'Timetag',values = 'DataValue',columns = 'new_id')
        self.df.columns = self.df.columns.str.rstrip()
        self.df = self.df.rename(columns=renamer())
        self.df = self.df.replace(',','.', regex=True)
        self.df = self.df.apply(pd.to_numeric)
        self.df = self.df[self.df.columns.drop(list(self.df.filter(regex='Misc|Suction Temp Target| AG')))]
        if self.site_id == 52253:
            self.df = self.df.rename(columns={'Compressor B4': "Compressor B 4 1"})
        
        if self.site_id == 18641:
            self.df = self.df.rename(columns={'Disch Pres': "Disch Pres D 1"})
            
        rd = remove_duplicates(self.df, self.site_id)
        self.df = rd.find_duplicates()

        return(self)

    def capacity(self, df, variable):
        a = df.groupby(pd.Grouper(freq = self.win)).count()
        b = df.groupby(pd.Grouper(freq = self.win)).apply(lambda x: (x == variable).sum())
        c = (b/a)*100
        return c

    def quad(self, df, variable, variable2, variable3, great=True):
        if great == True:
            answer = df[(df == variable) & (variable2 >= variable3)]
            answer = self.quadrant_percent(df, answer)

        if great != True:
            answer = df[(df == variable) & (variable2 <= variable3)]
            answer = self.quadrant_percent(df, answer)

        return(answer)

    def quadrant_percent(self, df, variable):
        days = df.groupby(pd.Grouper(freq = self.win)).count()
        quad_percent = (variable.groupby(pd.Grouper(freq = self.win)).count()/days)*100
        return quad_percent

    def get_switches(self, series):
        new_series = series.dropna()
        new_series = new_series.diff()
        new_series = new_series[new_series>-0.1]
        return(new_series)

    def cycle_count_runtime(self):
        df_cycle_runtime = pd.DataFrame()

        for i in self.df.columns:
            if self.sub in i.lower():
                temp = self.df[i]
                name = str(i + ' ' + 'Switch')
                run_name = str(i + ' ' + 'Runtime (%)')

                switches = self.get_switches(temp).groupby(pd.Grouper(freq = self.win)).sum()
                c = temp.groupby(pd.Grouper(freq = self.win)).agg(np.nanmean)*100

                df_cycle_runtime[run_name] = c
                df_cycle_runtime[name] = switches

        return df_cycle_runtime

    def calculate_entropy(self, list_values):
        counter_values = Counter(list_values).most_common()
        probabilities = [elem[1]/len(list_values) for elem in counter_values]
        entropy_res=entropy(probabilities)
        return entropy_res

    def calculate_statistics(self, list_values):
        n5 = np.nanpercentile(list_values, 5)
        n25 = np.nanpercentile(list_values, 25)
        n75 = np.nanpercentile(list_values, 75)
        n95 = np.nanpercentile(list_values, 95)
        median = np.nanpercentile(list_values, 50)
        mean = np.nanmean(list_values)
        std = np.nanstd(list_values)
        var = np.nanvar(list_values)
        rms = np.nanmean(np.sqrt(list_values**2))
        kurt = kurtosis(list_values.dropna())
        sk   = skew(list_values.dropna())
        ent = self.calculate_entropy(list_values)
        return [n5, n25, n75, n95, median, mean, std, var, rms, kurt, sk, ent]

    def diff(self, df, rack_id, variable = 'Suction Pres'):
        ans_diff = pd.DataFrame(df.filter(like='Suction Adj').groupby(pd.Grouper(freq = '24h')).agg(np.nanmean).to_numpy()\
                    - df.filter(like=variable).groupby(pd.Grouper(freq = '24h')).agg(np.nanmean).to_numpy()\
                    , columns=[variable + ' ' + rack_id +' Diff'], index=df.groupby(pd.Grouper(freq = '24h')).agg(np.nanmean).index)
        return ans_diff

    def get_df_values(self, df, rack_ident):
        temp_df = df
        combined = pd.DataFrame()
        values = temp_df.filter(regex='Suction Temp\s[A-Z]$|Disch|Capacity|Suction Pres').columns.unique()

       


        for i in values:
            for j in temp_df.columns:
                if i.lower() in j.lower():
                    answer = temp_df[j].groupby(pd.Grouper(freq = '24h')).agg(self.calculate_statistics)
                    answer = pd.DataFrame.from_dict(dict(zip(answer.index, answer.values))).T
                    answer.columns = [(j+' n5'), (j+' n25'), (j+' n75'), (j+' n95'), (j+' median'), (j+' mean'), (j+' std'),\
                                      (j+' var'), (j+' rms'), (j+' kurtosis'), (j+' skew'), (j+' entropy')]
                    combined = pd.concat([combined, answer], axis=1)

        mean_targets = temp_df.filter(regex='Compressor|Target|Outside Temp').groupby(pd.Grouper(freq = '24h')).agg(np.nanmean)
        
        if self.site_id != 52253:
            press_diff = self.diff(temp_df, rack_ident, variable='Suction Pres')
            float_diff = self.diff(temp_df, rack_ident, variable='Suction PSI')
            combined = pd.concat([combined, mean_targets], axis=1)
            combined = pd.concat([combined, press_diff], axis=1)
            combined = pd.concat([combined, float_diff], axis=1)
        else:
            mean_targets = mean_targets[mean_targets.columns.drop(list(mean_targets.filter(regex='Suction Adj.')))]
            combined = pd.concat([combined, mean_targets], axis=1)

        return combined

    def get_racks(self, rack_var):
        var_name = rack_var[-1:].upper()
        rack_name = self.df.loc[:, self.df.columns.str.contains('{}$|{}\s[0-9]*$|{}[0-9]$'.format(var_name, var_name, var_name))]
        return(rack_name)

    def quads_and_cap(self):
        rack_names = []
        rack_list  = []
        test_frame = pd.DataFrame()
        combined_frame = pd.DataFrame()
        suc_cap = [0, 100, 1, 99]

        for x in self.df.AssetName.unique():
            if 'Rack' in x:
                if x == 'Rack X':
                    continue
                rack_names.append(x)
        rack_names.sort()
        assert(len(rack_names) == len(self.suc_pres_range))
        self.pivot_data()
        ##########################################################################################################


        for i,j in enumerate(rack_names):
            rack_list.append(self.get_racks(j))
            if rack_list[i].filter(regex='Outside Temp').empty:
                rack_list[i]['Outside Temp'] = self.df.filter(regex='(Outside Temp)$')
            temp_name = j[-1:]
            parameter = self.get_df_values(rack_list[i], temp_name)

            for k in rack_list[i].columns:
                if 'Suction Capacity {}'.format(temp_name) in k:
                    name_pres = 'Suction Pres {}'.format(temp_name)
                    name_cap = 'Suction Capacity {}'.format(temp_name)
                    temp_i = rack_list[i][name_cap]
                    temp_j = rack_list[i][name_pres]

                    nz = temp_i[(temp_i.between(suc_cap[2], suc_cap[3])) & (temp_j.between(self.suc_pres_range[i][0], self.suc_pres_range[i][1]))]
                    q1_percent = self.quad(temp_i, suc_cap[0], temp_j, self.suc_pres_range[i][1], great=True)
                    q2_percent = self.quad(temp_i, suc_cap[0], temp_j, self.suc_pres_range[i][0], great=False)
                    q3_percent = self.quad(temp_i, suc_cap[1], temp_j, self.suc_pres_range[i][1], great=True)
                    q4_percent = self.quad(temp_i, suc_cap[1], temp_j, self.suc_pres_range[i][0], great=False)
                    nz_percent = self.quadrant_percent(temp_i, nz)

                    zero_cap = self.capacity(temp_i, suc_cap[0])
                    hund_cap = self.capacity(temp_i, suc_cap[1])

                    test_frame[temp_name + ' Neutral Zone (%)'] = nz_percent
                    test_frame[temp_name + ' Quadrant 1 (%)'] = q1_percent
                    test_frame[temp_name + ' Quadrant 2 (%)'] = q2_percent
                    test_frame[temp_name + ' Quadrant 3 (%)'] = q3_percent
                    test_frame[temp_name + ' Quadrant 4 (%)'] = q4_percent
                    test_frame[temp_name + ' 0% Capacity'] = zero_cap
                    test_frame[temp_name + ' 100% Capacity'] = hund_cap


            combined_frame = pd.concat([combined_frame, parameter], axis=1)

        combined = pd.concat([test_frame, combined_frame], axis=1)
        return combined, rack_names

    def aggregate_data(self):
        df_type = self.check_df_type()

        if df_type != True:
            return(logging.error('Insert Dataframe'))

        quadrants, rack_names = self.quads_and_cap()
        data_values = self.cycle_count_runtime()
        result = pd.concat([data_values, quadrants], axis=1)

        return result,rack_names