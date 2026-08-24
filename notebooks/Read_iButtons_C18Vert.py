#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 19:05:14 2023
Read Vertical C18 profile, deployed from 7/21/2023 to 7/22/2023, at 1 minute intervals
@author: mira
"""


import glob
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats


#path = '/Users/mira/Documents/UW/JIRP/DATA/iButtons/TESTS/'
path = '/Users/mira/Documents/UW/JIRP/DATA/iButtons/C18_Vert/'

files = glob.glob(path+'*.csv')


fig, (ax0, ax1) = plt.subplots(nrows = 1, ncols = 2, figsize=(12,6))

###############################
#read and plot the data, original and resampled (choose the interval and take the mean over that interval here)
###############################

# enter the elevations that correspond in order that "files" variable shows in the console below
# Vertical profile reads them in as AA, RR< SS, FF, II, JJ, DD< KK, QQ< CC

hgt = [0.4, 1.8, 0.6, 1.6, 2, 1, 1.2, 1.4, 0.8, 0.2]  # enter the elevations in order of how the data is read in. 

for ii, file in enumerate(files[0:]):
    
    df = pd.read_csv(file, header=13, encoding='cp1252', index_col=0,
                     names=['Unit'+str(ii+1), 'Value'])
    df.index = pd.to_datetime(df.index, format='%m/%d/%y %I:%M:%S %p')
    
    df = df.loc[datetime(2023, 7, 21, 16, 00, 0):datetime(2023, 7, 22, 16, 00, 0)]
    
    df_resamp = df.resample('10Min').mean()  # set the interval i want to resample and mean
    
    ax0.plot(df.Value,'o-')
    ax0.set_title('Original Data')
    ax0.set_ylabel('Temp (C)')
    
    ax1.plot(df_resamp, 'o-')
    ax1.set_title('Resamp Data (5min)')
    ax1.set_ylabel('Temp (C)')


fig, (ax0) = plt.subplots(nrows = 1, ncols = 1, figsize=(6,6))

# save the mean temps
#

for ii, file in enumerate(files[0:]):
    df = pd.read_csv(file, header=13, encoding='cp1252', index_col=0,
                     names=['Unit'+str(ii+1), 'Value'])
    df.index = pd.to_datetime(df.index, format='%m/%d/%y %I:%M:%S %p')
    
    df = df.loc[datetime(2023, 7, 21, 16, 00, 0):datetime(2023, 7, 22, 16, 00, 0)]
    
    #df_resamp = df.resample('60Min').mean()  # set the interval i want to resample and mean
    df_resamp = df.resample('20H').mean()  # set the interval i want to resample and mean

    print(df_resamp.mean())
    ax0.plot(df_resamp.mean(), hgt[ii] , 'x')  # this takes the mean of the entire time series and plots the elevation vs temp 
    ax0.set_title('Temp vs Height')
    ax0.set_xlabel('Mean Temp')
    ax0.set_ylabel('Height Above Surface (m)')
    
