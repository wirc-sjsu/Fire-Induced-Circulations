# %%
#  %%
print("Importing Libraries")
import matplotlib.pyplot as plt
import numpy as np
#import xarray as xr
from netCDF4 import Dataset
import pandas as pd
import glob
from wrf import getvar, ALL_TIMES, interplevel
# %%
print("Creating empty lists and file paths")
time_list = []
wind_list = []
time_list2 = []
wind_list2 = []

sdir = sorted(glob.glob('/home/akochanski/wrfxpy/wksp/Thomas/wrf_SVM/wrfout_d02_2017-12-0[4567]*'))
sdir2 = sorted(glob.glob('/home/015911532/thomas_fire/d02_nofire/wrfout_d02_2017-12-0[4567]*'))
#sdir='/home/jhaw/research/thomas_fire/d02/'
#sdir2='/home/jhaw/research/thomas_fire/d02_nofire/'
#sfile='wrfout_d02_2017-12-04_18-08_00'
#sfile2='wrfout_d02_2017-12-04_18-08_00'

#list wrf output files
#for sfile in os.listdir(sdir):
#    print(sfile)
# %%
print("Looping over files")
for f in range(len(sdir)):
    sfile = sorted((sdir))[f]
    print(sfile)

    nc_fid=Dataset(sdir[f],'r')

    time = getvar(nc_fid, "times", ALL_TIMES, meta=False)
    frac = nc_fid.variables['GRNHFX'][:]
    mask = frac > 0
    u10 = nc_fid.variables['U10']
    v10 = nc_fid.variables['V10']
    wspd = np.sqrt(np.array(u10)**2 + np.array(v10)**2)
    wspd_max = [wspd[i][m].max() if len(wspd[i][m]) > 0 else np.nan for i,m in enumerate(mask)]

    # for itime in range(len(time)):
    #     #u10 = nc_fid.variables['U10']
    #     #v10 = nc_fid.variables['V10']
    #     #wspd = np.sqrt(np.array(u10)**2 + np.array(v10)**2)
    #     time_list.append(time[itime])
    #     wind_list.append(pres_max[itime])

    sfile2 = sorted((sdir2))[f]
    print(sfile2)

    nc_fid2=Dataset(sdir2[f],'r')

    lats2 = nc_fid2.variables['XLAT'][0,:,:]
    lons2 = nc_fid2.variables['XLONG'][0,:,:]
    time2 = getvar(nc_fid2, "times", ALL_TIMES, meta=False)
    frac2 = nc_fid2.variables['AVG_FUEL_FRAC'][:]
    u10_2 = nc_fid2.variables['U10']
    v10_2 = nc_fid2.variables['V10']
    wspd2 = np.sqrt(np.array(u10_2)**2 + np.array(v10_2)**2)
    #wspd2_max = wspd2.max(axis=(1,2))
    wspd2_max = [wspd2[i][m].max() if len(wspd2[i][m]) > 0 else np.nan for i,m in enumerate(mask)]
    if len(time) != len(time2):
        mask = [mask[0]]
        time_list += [time[0]]
        time_list2 += [time2[0]]
    else:
        time_list += list(time)
        time_list2 += list(time2)
    wind_max = [wspd[i][m].max() if len(wspd[i][m]) > 0 else np.nan for i,m in enumerate(mask)]
    wind2_max = [wspd2[i][m].max() if len(wspd2[i][m]) > 0 else np.nan for i,m in enumerate(mask)]
    wind_diff = np.array(wind_max) - np.array(wind2_max)
    #pres_list += list(pres_max)
    wind_list2 += list(wind_diff)

    # for itime in range(len(time)):
    #     #frac2 = nc_fid2.variables['AVG_FUEL_FRAC'][:]
    #     #u10_2 = nc_fid2.variables['U10']
    #     #v10_2 = nc_fid2.variables['V10']
    #     #wspd2 = np.sqrt(np.array(u10_2)**2 + np.array(v10_2)**2)
    #     #wspd2_max = wspd2.max(axis=(1,2))

    #     time_list2.append(time2[itime])
    #     wind_list2.append(pres2_max[itime])
    # #print(time)
# %%
# fig2 = plt.figure(1, figsize=(12, 8))

# #plt.plot(time_list, pres_list, 'r')
# df.boxplot(column=['data'])
# plt.title('Max Wind Speed Difference Around Burned Area (AVG_FUEL_FRAC < 1) \nOver Time for the Thomas Fire (Fire vs. No Fire)')
# plt.xlabel('Time (UTC)')
# plt.ylabel('Wind Speed (m/s)')
# plt.grid()
# #plt.legend(['Fire','No Fire'])

# plt.show()
# %%
fig2 = plt.figure(1, figsize=(12, 8))
# df_col = []
# for i in range(3):
#     df_col[:,i] = df[i*78:i*78+77]
time_list2 = [str(x.astype('datetime64[D]')) for x in time_list2]
arr = [np.transpose([np.array(time_list2)]), np.transpose([np.array(wind_list2)])]
df = pd.DataFrame(wind_list2, columns=["data"])
#df_col = np.resize(df,[77,4])
df['date'] = np.array(time_list2)
#df_col = pd.DataFrame(df_col,columns = ['12/04', '12/05', '12/06', '12/07'])
# df_col = df.resize(78,4)
#plt.plot(timeiiiiiiii_list, pres_list, 'r')
df.boxplot(by='date')
plt.suptitle('')
plt.title('Max Fire-Induced Wind Speed Around Active Fire (GRNHFX > 0) \nfor the Thomas Fire (Fire vs. No Fire)')
plt.xlabel('Time (UTC)')
plt.ylabel('Wind Speed (m/s)')
plt.grid()
#plt.legend(['Fire','No Fire'])

plt.show()
# %%