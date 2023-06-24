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
from metpy.calc import brunt_vaisala_frequency

time_list = []
#wind_list = []
time_list2 = []
#wind_list2 = []
# z_list = []
# z_list2 = []
# N_list = []
# N2_list = []
g = 9.81
rho = 1.2
cp = 1.005
max_height = 8847
fr_list = []
fr_list2 = []

sdir= '/adata/jhaw/thomas_fire_runs/d02/'
sdir2='/adata/jhaw/thomas_fire_runs/d02_nofire/'
#sfile='wrfout_d02_2017-12-04_18-08_00'
#sfile2='wrfout_d02_2017-12-04_18-08_00'

#list wrf output files
#for sfile in os.listdir(sdir):
#    print(sfile)
# %%
for sfile in os.listdir(sdir):
    print(sfile)

    nc_fid=Dataset(sdir+sfile,'r')

    lats = nc_fid.variables['XLAT'][0,:,:]
    lons = nc_fid.variables['XLONG'][0,:,:]
    time = getvar(nc_fid, "times", ALL_TIMES, meta=False)
    u10 = nc_fid.variables['U10']
    v10 = nc_fid.variables['V10']
    wspd = np.sqrt(np.array(u10)**2 + np.array(v10)**2)
    z = getvar(nc_fid, "z", units='m', timeidx=ALL_TIMES)
    t = getvar(nc_fid, "temp", units='K', timeidx=ALL_TIMES)#.variables['T'][itime] + 300
    flineint = nc_fid.variables['F_LINEINT']
    ros = nc_fid.variables['F_ROS']
    #N = brunt_vaisala_frequency(z,t)
#print(len(time))

    for itime in range(len(time)):

        #fig = plt.figure(1, figsize=(12, 8))

        #for i in range(1):
        time_list.append(time[itime])
        #wind_list.append(wspd[itime].max())
        #z_list.append(z[itime].min())
        wspd_t = np.array(wspd[itime])
        fline_t = np.array(flineint[itime])
        ros_t = np.array(ros[itime])
        t_t = np.array(t[itime])
        #N_t = np.array(N[itime])
        #N_list.append(N_t[N_t > 0].min())
        #fr = wspd_t/max_height/N_t
        #fr = fr[np.isfinite(fr)]
        #fr_list.append(np.nanmax(fr))
        #fr_list.append([((x)/(max_height)) for x in wind_list]/(np.array(N_list)))#(np.sqrt(g*np.array(z_list)))

            #plt.plot(time[i], wind.max())
            #print(wind.max())
    #print(time_list)
    #print(wind_list)
    wspd_mean = np.mean(wspd_t)
    fline_mean = np.mean(fline_t)
    ros_mean = np.mean(ros_t)
    t_mean = np.mean(t_t)

    #list wrf output files
    #for sfile2 in os.listdir(sdir2):
    #    print(sfile2)
# %%
# for sfile2 in os.listdir(sdir2):
#     print(sfile2)

#     nc_fid2=Dataset(sdir2+sfile2,'r')

#     lats2 = nc_fid2.variables['XLAT'][0,:,:]
#     lons2 = nc_fid2.variables['XLONG'][0,:,:]
#     time2 = getvar(nc_fid2, "times", ALL_TIMES, meta=False)
#     u10_2 = nc_fid2.variables['U10']
#     v10_2 = nc_fid2.variables['V10']
#     wspd2 = np.sqrt(np.array(u10_2)**2 + np.array(v10_2)**2)
#     z1 = getvar(nc_fid2, "z", units='m', timeidx=ALL_TIMES)
#     t1 = getvar(nc_fid2, "temp", units='K', timeidx=ALL_TIMES)#.variables['T'][itime] + 300
#     flineint2 = nc_fid2.variables['FLINEINT']
#     ros2 = nc_fid2.variables['F_ROS']
#     #N2 = brunt_vaisala_frequency(z1,t1)
#     #print(time)

#     for itime2 in range(len(time2)):
        
#         #fig2 = plt.figure(1, figsize=(12, 8))

#         #for a in range(1):
#         time_list2.append(time2[itime2])
#         #wind_list2.append(wspd2[itime2].max())
#         #z_list2.append(z1[itime2].min())
#         wspd_t2 = np.array(wspd2[itime2])
#         fline_t2 = np.array(flineint2[itime2])
#         ros_t2 = np.array(ros2[itime2])
#         t_t2 = np.array(t1[itime2])
#         #N_t = np.array(N2[itime2])
#         #N_list.append(N_t[N_t > 0].min())
#         #fr = wspd_t/max_height/N_t
#         #fr = fr[np.isfinite(fr)]
#         #fr_list2.append(np.nanmax(fr))
#         #N2_list.append(N_t[N_t > 0].min())

# wspd_mean2 = np.mean(wspd_t2)
# fline_mean2 = np.mean(fline_t2)
# ros_mean2 = np.mean(ros_t2)
# t_mean2 = np.mean(t_t2)
# %%
#fig = plt.figure(1, figsize=(12, 8))

fr_fire = (2 * g * fline_mean/1000)/(rho * cp * t_mean * (wspd_mean - ros_mean)**3)
#fr_nf = (2 * g * (fline_mean2/1000))/(rho * cp * t_mean2 * (wspd_mean2 - ros_mean2)^3)
#fr_fire = [((x)/(2697)) for x in wind_list]/(np.array(N_list))#(np.sqrt(g*np.array(z_list)))
#fr_nf = [((x2)/(2697)) for x2 in wind_list2]/(np.array(N2_list))#(np.sqrt(g*np.array(z_list2)))

print('Byrams Convection Number for the fire run is:', fr_fire)
#print('Byrams Convection Number for the no fire run is:', fr_nf)

# plt.plot(time_list, fr_list, 'r')
# plt.plot(time_list2, fr_list2, 'b')
# plt.title('Convective Froude Number Over Time for the Thomas Fire')
# plt.xlabel('Time (UTC)')
# plt.ylabel('Froude Number')
# plt.grid()
# plt.legend(['Fire','No Fire'])

# plt.show()
# %%