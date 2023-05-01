# %%
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
#import xarray as xr
from netCDF4 import Dataset
import sys
import os
import glob
from wrf import getvar, ALL_TIMES
import pandas as pd
# %%
time_list = []
wind_list = []
time_list2 = []
wind_list2 = []
wind_diff = []
wind_diff_list = []

sdir='/home/015911532/thomas_fire/concat_d02/'
sdir2='/home/015911532/thomas_fire/concat_d02/'
sfile='concat_d02_SVM_new'
sfile2='wrfout_d02_all'

#list wrf output files
#for sfile in os.listdir(sdir):
#    print(sfile)

#for sfile in os.listdir(sdir):
#    print(sfile)
# %%
nc_fid=Dataset(sdir+sfile,'r')

lats = nc_fid.variables['XLAT'][0,:,:]
lons = nc_fid.variables['XLONG'][0,:,:]
time = getvar(nc_fid, "times", ALL_TIMES, meta=False)
u10 = nc_fid.variables['U10']
v10 = nc_fid.variables['V10']
wspd = np.sqrt(np.array(u10)**2 + np.array(v10)**2)
#print(len(time))
# %%
for itime in range(len(time)):
    # u10 = nc_fid.variables['U10']
    # v10 = nc_fid.variables['V10']
    # wspd = np.sqrt(np.array(u10)**2 + np.array(v10)**2)
    #print(len(u10))
    #rainnc = nc_fid.variables['RAINNC'][:]
    #wind=u10
    #print(len(wind))

    fig = plt.figure(1, figsize=(12, 8))

    #for i in range(1):
    time_list.append(time[itime])
    wind_list.append(wspd[itime])
        #plt.plot(time[i], wind.max())
        #print(wind.max())
#print(time_list)
#print(wind_list)

#list wrf output files
#for sfile2 in os.listdir(sdir2):
#    print(sfile2)

#for sfile2 in os.listdir(sdir2):
#print(sfile2)
# %%
nc_fid2=Dataset(sdir2+sfile2,'r')

lats2 = nc_fid2.variables['XLAT'][0,:,:]
lons2 = nc_fid2.variables['XLONG'][0,:,:]
time2 = getvar(nc_fid2, "times", ALL_TIMES, meta=False)
u10_2 = nc_fid2.variables['U10']
v10_2 = nc_fid2.variables['V10']
wspd2 = np.sqrt(np.array(u10_2)**2 + np.array(v10_2)**2)
wind_diff = np.array(wspd) - np.array(wspd2)
#print(time)
# %%
for itime2 in range(len(time) - 1):
    # u10_2 = nc_fid2.variables['U10']
    # v10_2 = nc_fid2.variables['V10']
    # wspd2 = np.sqrt(np.array(u10_2)**2 + np.array(v10_2)**2)
    #rainnc = nc_fid.variables['RAINNC'][:]
    #wind2=u10_2

    fig2 = plt.figure(1, figsize=(12, 8))

    #for a in range(1):
    time_list2.append(str(pd.to_datetime(time2[itime2]).date()))
    wind_list2.append(wind_diff[itime2].max())
    #df = pd.DataFrame(time_list2, wind_list2, columns=["time","data"])

    # wind_diff.append(np.array(wind_list[itime2]) - np.array(wind_list2[itime2]))

    # wind_diff_list.append(wind_diff[itime2].max())

#wind_diff = np.array(wind_list) - np.array(wind_list2)
# %%
#df_col = np.resize(df,[77,4])
#df = {'date': [time_list2], 'wind': [wind_list2]}
arr = [np.transpose([np.array(time_list2)]), np.transpose([np.array(wind_list2)])]
df = pd.DataFrame(wind_list2, columns=["data"])
df['date'] = np.array(time_list2)


df.boxplot(by='date')
#plt.plot(time_list2, wind_list2, 'b')
plt.suptitle('')
plt.title('Max Fire-Induced Wind Speed Over Time \nfor the Thomas Fire (Fire - No Fire)')
plt.xlabel('Time (UTC)')
plt.ylabel('Wind Speed (m/s)')
plt.grid()
#plt.legend(['Fire','No Fire'])

plt.show()
# %%