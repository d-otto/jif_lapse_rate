# -*- coding: utf-8 -*-
"""
data.py

Description.

Author: drotto
Created: 6/13/24 @ 16:39
Project: jif_lapse_rate
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm

#%%

def clean_hobo_pendants(ps: list[Path] | Path, dir_out: Path):
    '''Reads data exported from HOBOware and HOBOconnect and outputs it as netcdf'''
    
    
    # make sure ps is always a list
    if isinstance(ps, list) is False:
        ps = [ps]
    
    
    for p in ps:
        df = pd.read_csv(p)
        if df.columns[0] == "#":
            df = df.drop(columns="#")
        
        # Check if it is a new or old pendant
        # Currently, all old pendant files have a space instead of a dash in "Date-Time"
        if df.columns[0] == 'Date Time':  # if old pendant
            
            sensor_generation = "old"
            # get s/n
            sn = p.name.split('.')[0]
            tz = 'local'  # no timezone info
            
            # clean the col names just in case 
            df.columns = (
                df.columns.str.lower()
                .str.strip()
                .str.replace('°', '')
                .str.replace(r'\(.*\)', '')
                .str.split(' ', n=1, expand=False)
                .str.get(0)
            )
            print(df.columns)
            col_map = {
                "date"  : "datetime",
                f"#{sn}"      : "intensity_lux",
                "temp": "temp_c"
            }
            df = df.loc[:, col_map.keys()]
            df.columns = df.columns.map(col_map)
            print(df)
            
        else: # if new pendant
            
            sensor_generation = "new"
            # get s/n
            sn = p.name.split(' ')[0].strip()
    
            # get tz
            tz = (df.columns
            .str.split(" ")
            .str.get(-1)
            .str.replace(r'[\(\)]', '', regex=True)[0]
            )
            # clean the col names just in case 
            df.columns = (
                df.columns.str.lower()
                .str.replace('°', '')
                # .str.replace(r'\(.*\)', '')
                .str.split(' ', n=1, expand=False)
                .str.get(0)
            )
    
            col_map = {
                f"date-time" : "datetime",
                "light"      : "intensity_lux",
                "temperature": "temp_c"
            }
            df = df.loc[:, col_map.keys()]
            df.columns = df.columns.map(col_map)
            
    
        # convert to datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.loc[:, ['temp_c', 'intensity_lux']] = df.loc[:, ['temp_c', 'intensity_lux']].apply(pd.to_numeric,
                                                                                              errors='coerce')
        # Convert the cleaned DataFrame to a xarray Dataset
        ds = xr.Dataset.from_dataframe(df.set_index('datetime'))
        attr_dict = {
            "sensor_type": "hobo pendant",
            "sensor_generation": sensor_generation,
            "sensor_id": sn,
            "tz": tz,
        }
        #ds.expand_dims(dim={'sensor_id': sn})
        ds.coords['sensor_id'] = sn
        ds['temp_c'].attrs = attr_dict
        ds['intensity_lux'].attrs = attr_dict
        
        # Output to netcdf
        fname = f"{sn}_{df.datetime.iloc[0].strftime('%Y%m%dT%H%M')}_{df.datetime.iloc[-1].strftime('%Y%m%dT%H%M')}.nc"
        pout = dir_path / fname
        ds.to_netcdf(pout)
    
        return None

#%%

def read_hobo_pendants(ps: list[Path], sensor_as_var=False):     
    def round_to_minute(d):
        try:
            d['datetime'] = d['datetime'].dt.round('min')
            d = d.drop_duplicates('datetime')
        except: # not sure why this would be the case but it might be for good reason!
            pass
        return d
    
    
    # if sensor_as_var:
    #     mfds = {}
    #     for p in tqdm(ps):
    #         ds = xr.open_dataset(p, decode_times=True)
    #         ds = round_to_minute(ds)
    #         ds = ds.swap_dims({'sensor_id', })
    # 
    # 
    #         if mfds is None:
    #             mfds = {k:v for k, v in ds.items()}  # give each data var a key in mfds
    #         else:
    #             # concat each data var to msds using [k]
    #             mfds = {k:xr.concat([mfds[k], v], dim='sensor_id', combine_attrs="drop") for k,v in ds.items()}
    
    # else:    
    mfds = None
    for p in tqdm(ps):
        ds = xr.open_dataset(p, decode_times=True)
        ds = round_to_minute(ds)
        # if sensor_as_var:
        #     ds = ds.expand_dims("sensor_id", create_index_for_new_dim=False)
        #     das = {k: v for k, v in ds.items()}
        #     das = {k: v.swap_dims({"sensor_id": k}).unstack('sensor_id') for k, v in das.items()}
        #     if mfds is None:
        #         mfds = das  # give each data var a key in mfds
        #     else:
        #         # concat each data var to msds using [k]
        #         mfds = {k: xr.concat([mfds[k], v], dim=k, combine_attrs="drop") for k, v in ds.items()}
        # else:
        if mfds is None:
            mfds = ds
        else:
            mfds = xr.concat([mfds, ds], dim='sensor_id', combine_attrs="drop")
        
    return mfds
