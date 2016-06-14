"""
Created on Tues June 14 2016

Author - Douglas Jacobson
Python Version - 2.7.11
This program is to be run as a stand alone module and will perform a variety
of machine learning algorithms and attempt to predict a time series based on
related time series.  

"""
import os

import numpy as np
import pandas as pd


def read_CSV(filename):
    df = pd.read_csv(filename)
    return df

if __name__ == "__main__":
    path = str(os.path.dirname(os.path.realpath(__file__)))+'/data/'
    filePathString = path + 'stock_returns_base150.numbers'
    df = read_CSV(filePathString)