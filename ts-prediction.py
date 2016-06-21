"""
Created on Tues June 14 2016

Author - Douglas Jacobson
Python Version - 2.7.11
This program is to be run as a stand alone module and will perform a variety
of machine learning algorithms and attempt to predict a time series based on
related time series.  

"""
import os


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from pandas.stats.api import ols


def read_CSV(filename):
    '''
    Reads in a CSV.  
    Arguements:
    filename - The name of the desired file to read in.
    '''
    df = pd.read_csv(filename)
    return df
    
    
def df_cleaner(dataframe):
    '''
    Cleans data.  None to clean for this project.
    '''
    dfNew = dataframe
    return dfNew
    
    
def build_lagged_dataframe(dataframe, lagged=1):
    '''
    Builds a lagged dataframe
    arguements
    dataframe - the dataframe to alter
    lagged (default is 1) -  how many trading days to lag the data
    '''
    dataframe['S1c'] = dataframe['S1']
    tsNew = dataframe['S1'].shift(lagged)
    dataframe['S1'] = tsNew
    return dataframe
    

def scatter_plot(df, symbol2):
    '''
    Creates scatter plot with two time series.  
    arguements
    df - the dataframe to read in
    symbol2 - The symbol to plot against S1
    '''
    T=np.arctan2(df['S1'], df[symbol2])    
    
    plt.ylabel('Series S1')
    plt.xlabel('%s' % symbol2)
    plt.title('%s and %s Price Scatterplot' % ('S1', symbol2))
    plt.scatter(df['S1'], df[symbol2], s=25,c=T,alpha=0.7)
    
    #Best Fit line
    x = df[symbol2]
    y = df['S1']
    m,b=np.polyfit(x,y,1)
    plt.plot(x, m*x + b, '-')
    plt.show()
    
def regression_analysis(df, symbol):
    '''
    Performs the OLS regression analysis of the data.
    arguements
    df-  Dataframe with data
    symbol -  can be a symbol or array of symbols as the predictors
    '''
    res = ols(y=df['S1'], x=df[symbol])
    print(res)
    return res
    
    
    
def ols_trainTest(X,y, start_test):
    '''
    Tests the resudual square eror of the models using training 
    and test data. Also provides scatter plots.
    arguements-
    X - Predictors
    y- response data
    start_test - where to start the testing.  We did a 50/50 split
    Note- The scatter plots must be manually changed within the function.
    '''
    X_test = X[X.index >= start_test][:25] 
    X_train = X[y.index < start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test][:25]
    
    model = sm.OLS(y_train, X_train).fit()
    y_predict = model.predict(X_test)

    # Print Coefficients and Mean square error
    #print('Coefficients: \n', model.coef)
    print("Residual sum of squares: %f"% 
            (np.mean(y_predict-y_test) ** 2))
    

    # Plot results
    plt.scatter(X_test['S6'], y_test,  color='black', label='S6')
    #plt.scatter(X_test['S3'], y_test,  color='green', label='S3')
    #plt.scatter(X_test['S5'], y_test,  color='purple', label='S5')
    plt.title("OLS Prediction of S1")
    plt.xlabel("Stock Returns - S6")
    plt.ylabel("Stock Returns Response - S1")
    plt.legend()
    plt.grid()
    fit = np.polyfit(X_train['S6'], y_train, deg=1)
    plt.plot(X_train['S6'],fit[0]*X_train['S6']+fit[1], color='red')
    #fit1 = np.polyfit(X_train['S3'], y_train, deg=1)
    #plt.plot(X_train['S3'],fit1[0]*X_train['S3']+fit1[1], color='blue')
    #fit2 = np.polyfit(X_train['S5'], y_train, deg=1)
    #plt.plot(X_train['S5'],fit2[0]*X_train['S5']+fit2[1], color='grey')
    
    plt.show()
    

def ols_prediction_newData(X,y, path): 
    '''
    This function takes all 50 data points from the predictor set
    and predicts S1.  It also saves the data to a CSV file 
    arguements
    X - Predictor data
    y- response data to train model
    path -where to save the CSV file
    '''
    X_train = X[y.index < 50]
    y_train = y[y.index < 50]
    X_pred = X[X.index >= 50][:50]
    # Train the model
    model = sm.OLS(y_train, X_train).fit()
    y_predict = model.predict(X_pred)
    
    index = pd.date_range('8/11/2014', '10/21/2014', freq='B')
    index = index.drop(pd.Timestamp('2014-09-01 00:00:00'))
    index = index.drop(pd.Timestamp('2014-10-13 00:00:00'))
    
    df = pd.DataFrame(index=index)
    df['value'] = y_predict
    filePathString = path + 'predictions.csv'
    df.to_csv(filePathString)
    print(df)
    
    
 
def lag_analysis(lags, dataframe):
    '''
    Performs the lag analysis for all stock symbols and produces
    a summary report of the analysis.
    arguements
    lags - number of lagged trading days ( 0 is one trading day etc)
    dataframe - cleaned stock data
    '''
    symbols = ['S2', 'S3', 'S4', 'S5', 'S6'
                         , 'S7', 'S8', 'S9', 'S10', 'S1c']
    dfLagged = build_lagged_dataframe(dataframe, lags)
    
    print('Number of lags used are %f' %lags)
    for symbol in enumerate(symbols):
        print(symbol[1] + ' analysis \n')
        regression_analysis(dfLagged, symbol[1])
    

def trend_plot(df, symbol):
    '''
    Plots the cumulative trends of the desired stock symbol
    arguements
    df - dataframe with data
    symbol - Desired symbol to plot
    '''
    Xts = dfCleaned[symbol][:50]
    xn = pd.DataFrame(index=range(50), columns=['value'])
    for index, value in enumerate(Xts):
        if index == 0:
            xn['value'][index] = 1+1*value/100
        else:
            xn['value'][index] = xn['value'][index-1]+xn['value'][index-1]*value/100
            
    plt = xn[['value']].plot(figsize=(7,5))
    

if __name__ == "__main__":
    # Set data directory as needed
    path = str(os.path.dirname(os.path.realpath(__file__)))+'/data/'
    filePathString = path + 'stock_returns_base150.csv'
    
    # Read and clean CSV Data if needed
    df = read_CSV(filePathString)
    dfCleaned = df_cleaner(df)
    
    #### INPUT lag input #####
    lags = 0

    #Lag analysis -  Change the lags value to get results of all
    # stocks and the lienar regression analysis summary
    lag_analysis(lags, dfCleaned)
    
    
    # Stock analysis
    lag_analysis(0 ,dfCleaned)
    
    
    
    # Multivariate Analysis
    #### INPUT, change symbolMVOLS to check new stocks ####
    symbolsMVOLS = [ 'S6']
    res = regression_analysis(dfCleaned, symbolsMVOLS)
    print(res)
    
    #Training and testing of model
    #Define training and test sizes -  50/50 split
    start_test=25
    X = dfCleaned[['S6']]
    y = dfCleaned['S1']
    ols_trainTest(X, y, start_test)
    ols_prediction_newData(X,y, path)
    
    
    
    #Determining trends
    trend_plot(dfCleaned, 'S6')