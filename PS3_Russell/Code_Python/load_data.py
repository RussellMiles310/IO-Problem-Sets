# -*- coding: utf-8 -*-
"""
Load data
"""

import autograd.numpy as np
import pandas as pd

#Load the data
def load_data(model):
    
    filename = "../PS3_data_changedtoxlsx.xlsx"
    df0 = pd.read_excel(filename)
    #Remove missing materials columns
    df = df0[['year', 'firm_id', 'X03', 'X04', 'X05', 'X16', 'X40', 'X43', 'X44', 'X45', 'X49']]
    #new_names = ["year", "firm_id", "obs", "ly", "s01", "s02", "lc", "ll", "lm"]
    new_names = ["t", "firm_id", "y_gross", "s01", "s02", "s13", "k", "l", "m", 'py', 'pm']
    df.columns = new_names
    #Drop missing materials data
    df=df[df['m']!=0]
    #Keep industry 1 only
    df=df[df['s13']==1]
    
    if model == "ACF":
        #Creating value-added y
        df['y'] = np.log(np.exp(df['y_gross'] + df['py']) - np.exp(df['m'] + df['pm']))
    elif model == "GNR":
        #in GNR, we simply use gross y
        df['y'] = df['y_gross']
    else: 
        print("Please enter the string ACF or GNR" )

    #Creating lagged variables
    df = df.sort_values(by=['firm_id', 't'])
    df['kprev'] = df.groupby('firm_id')['k'].shift(1)
    df['lprev'] = df.groupby('firm_id')['l'].shift(1)
    df['mprev'] = df.groupby('firm_id')['m'].shift(1)
    
    return df