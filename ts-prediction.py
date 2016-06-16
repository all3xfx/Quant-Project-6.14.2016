"""
Created on Tues June 14 2016

Author - Douglas Jacobson
Python Version - 2.7.11
This program is to be run as a stand alone module and will perform a variety
of machine learning algorithms and attempt to predict a time series based on
related time series.  

"""
import os
from dateutil import parser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.stats.api import ols


def read_CSV(filename):
    df = pd.read_csv(filename)
    return df
    
    
def df_cleaner(dataframe):
    '''
    Cleans data.  None to clean for now.
    '''
    dfNew = dataframe
    #dfNew = dataframe.fillna(0.0)
    return dfNew
    
    
def build_lagged_dataframe(dataframe, ts2, lagged=1):
    tsNew = dataframe[ts2].shift(lagged)
    dataframe[ts2] = tsNew
    return dataframe
    
    
def build_signs_dataframe(dataframe, ts2):
    columns = ['date', 'S1', ts2, 'signs']
    df = pd.DataFrame(columns=columns)
    df['date'] = dataframe['date']
    df['S1'] = dataframe['S1']
    df[ts2] = dataframe[ts2]
        
    return df
 

def time_series_plot(dataSeries):
    pass


def scatter_plot(df, symbol2):
    '''
    Creates scatter plot with two time series.  
    '''
    T=np.arctan2(df['S1'], df[symbol2])    
    
    plt.ylabel('Series S1')
    plt.xlabel('%s' % symbol2)
    plt.title('%s and %s Price Scatterplot' % ('S1', symbol2))
    plt.scatter(df['S1'], df[symbol2], s=25,c=T,alpha=0.7)
    
    
    
    #T=np.arctan2(ts1, ts2)
    
    #plt.scatter(df[ts1], df[ts2],s=25,c=T,alpha=0.7)
    
    #Best Fit line
    
    x = df[symbol2]
    y = df['S1']
    m,b=np.polyfit(x,y,1)
    plt.plot(x, m*x + b, '-')
    plt.show()
    
def regression_analysis(df):
    res = ols(y=df['S1'], x=df[['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']])
    return res

if __name__ == "__main__":
    # Set data directory as needed
    path = str(os.path.dirname(os.path.realpath(__file__)))+'/data/'
    filePathString = path + 'stock_returns_base150.csv'
    
    # Read and clean CSV Data if needed
    df = read_CSV(filePathString)
    dfCleaned = df_cleaner(df)
    
    #What Symbol
    symbol = 'S6'
    lags = 4
    
    #Create lagged dataframe
    dfLagged = build_lagged_dataframe(dfCleaned, symbol, lags)
    res = regression_analysis(dfLagged)
    print(res)
    #Data frame with only the data and signs
    dfSigns = build_signs_dataframe(dfLagged, symbol)
    
    scatter_plot(dfSigns, symbol)
    
    