#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 12:12:45 2022

@author: jabadgeley
"""
import glob
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

path = '/Volumes/LACIE SHARE/iButton Data/freezer_variance_test_07012022/'
files = glob.glob(path+'*.csv')

dfm = pd.read_csv(files[0], header=13, encoding='cp1252', index_col=0, 
                  names=['Unit0', 'Value0'])
dfm.index = pd.to_datetime(dfm.index, format='%m/%d/%y %I:%M:%S %p') 
if dfm['Unit0'][0] == 'C':
    dfm['Value0'] = dfm['Value0'] * (9/5) + 32
dfm = pd.DataFrame(dfm['Value0'])

for ii, file in enumerate(files[1:]):
    if (ii+1 == 4) | (ii+1 == 6) | (ii+1 == 10) | (ii+1 == 12):
        continue
    df = pd.read_csv(file, header=13, encoding='cp1252', index_col=0,
                     names=['Unit'+str(ii+1), 'Value'+str(ii+1)])
    df.index = pd.to_datetime(df.index, format='%m/%d/%y %I:%M:%S %p')
    if df['Unit'+str(ii+1)][0] == 'C':
        df['Value'+str(ii+1)] = df['Value'+str(ii+1)] * (9/5) + 32
    dfm = dfm.join(df['Value'+str(ii+1)])
    #datetime.strptime(df['Date/Time'][0], '%m/%d/%y %I:%M:%S %p')
    
fig = plt.figure('one')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#plt.gca().xaxis.set_major_locator(mdates.DayLocator())
for jj in range(len(files)):
    plt.plot(dfm.index, dfm['Value'+str(jj)], 'o')
plt.gcf().autofmt_xdate()
plt.xlabel('Time (hour:min)')
plt.ylabel(r'temperature ($^{\circ}$F)')
#plt.ylim([29,33])
plt.xlim([datetime(2022, 7, 1, 20, 30, 0), datetime(2022, 7, 1, 21, 15, 0)])

# Plot overall variance
fig = plt.figure('two')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.plot(dfm.index, dfm.var(axis=1, skipna=True))
plt.gcf().autofmt_xdate()
plt.xlabel('Time (hour:min)')
plt.ylabel(r'temperature ($^{\circ}$F)')
plt.ylim([0,2])

# Look at freezer variance
df_fv = dfm.loc[datetime(2022, 7, 1, 20, 40, 0):datetime(2022, 7, 1, 20, 55, 0)]
df_fv.var(axis=1, skipna=False)