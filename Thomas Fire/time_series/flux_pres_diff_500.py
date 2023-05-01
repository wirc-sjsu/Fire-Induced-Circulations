#  %%
print("Importing Libraries")
import matplotlib.pyplot as plt
import numpy as np
#import xarray as xr
from netCDF4 import Dataset
import os
import glob
from wrf import getvar, ALL_TIMES, interplevel
# %%
print("Creating empty lists and file paths")
time_list = []
pres_list = []
time_list2 = []
pres_list2 = []

sdir = sorted(glob.glob('/adata/jhaw/thomas_fire_runs/d02/wrfout_d02_2017-12-0[4567]*'))
sdir2 = sorted(glob.glob('/adata/jhaw/thomas_fire_runs/d02_nofire/wrfout_d02_2017-12-0[4567]*'))
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
    pres = getvar(nc_fid, "pressure", timeidx=ALL_TIMES)
    z = getvar(nc_fid, "z", units="m", timeidx=ALL_TIMES)
    #slp = getvar(nc_fid, "slp", timeidx=ALL_TIMES)
    ht_surf = interplevel(z, pres, 500).data

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
    pres2 = getvar(nc_fid2, "pressure", timeidx=ALL_TIMES)
    z1 = getvar(nc_fid2, "z", units="m", timeidx=ALL_TIMES)
    #slp1 = getvar(nc_fid2, "slp", timeidx=ALL_TIMES)
    ht_surf1 = interplevel(z1, pres2, 500).data
    if len(time) != len(time2):
        mask = [mask[0]]
        time_list += [time[0]]
        time_list2 += [time2[0]]
    else:
        time_list += list(time)
        time_list2 += list(time2)
    pres_diff = np.array(ht_surf) - np.array(ht_surf1)
    pres_max = [pres_diff[i][m].min() if len(ht_surf[i][m]) > 0 else np.nan for i,m in enumerate(mask)]
    #pres2_max = [ht_surf1[i][m].max() if len(ht_surf1[i][m]) > 0 else np.nan for i,m in enumerate(mask)]
    #pres_diff = np.array(pres_max) - np.array(pres2_max)
    #pres_list += list(pres_max)
    pres_list2 += list(pres_max)

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
fig2 = plt.figure(1, figsize=(12, 8))

#plt.plot(time_list, pres_list, 'r')
plt.plot(time_list2, pres_list2, 'r')
plt.title('Min 500 hPa Geopotential Height Difference Around Active Fire (GRNHFX > 0) \nOver Time for the Thomas Fire (Fire vs. No Fire)')
plt.xlabel('Time (UTC)')
plt.ylabel('Geopotential Height (m)')
plt.grid()
#plt.legend(['Fire','No Fire'])

plt.show()
# %%
