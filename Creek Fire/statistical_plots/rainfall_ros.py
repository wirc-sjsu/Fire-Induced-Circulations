# %%
print("Importing Libraries")
import matplotlib.pyplot as plt
import numpy as np
#import xarray as xr
from netCDF4 import Dataset
import os
from wrf import getvar, ALL_TIMES
from scipy.interpolate import griddata
#import matlab.engine
# %%
sdir='/home/015911532/creek_fire/d03/'
sfile='wrfout_d03_2020-09-12_00:15:00'
sdir1='/home/015911532/creek_fire/d03_nofire/'
sfile1='wrfout_d03_2020-09-12_00:15:00'

nc_f=Dataset(sdir+sfile,'r')
nc_nf=Dataset(sdir1+sfile1,'r') 
# %%
#time = getvar(nc_f, "times", ALL_TIMES, meta=False)
frac = nc_f.variables['GRNHFX'][:]
mask = frac > 0
rain = nc_f.variables['RAINNC'][:]
u10 = nc_f.variables['U10']
v10 = nc_f.variables['V10']
wspd = np.sqrt(np.array(u10)**2 + np.array(v10)**2)

#time1 = getvar(nc_nf, "times", ALL_TIMES, meta=False)
#frac1 = nc_nf.variables['GRNHFX'][:]
#mask = frac > 0
rain1 = nc_nf.variables['RAINNC'][:]
u10_1 = nc_nf.variables['U10']
v10_1 = nc_nf.variables['V10']
wspd1 = np.sqrt(np.array(u10_1)**2 + np.array(v10_1)**2)
long = nc_nf.variables['XLONG'][0]
fxlat = nc_nf.variables['FXLAT'][0,:-20,:-20]
fxlong = nc_nf.variables['FXLONG'][0,:-20,:-20]
lat = nc_nf.variables['XLAT'][0]
fuel_cat = nc_nf.variables['NFUEL_CAT'][0,:-20,:-20]
dzdxf = nc_nf.variables['DZDXF'][0,:-20,:-20]
dzdyf = nc_nf.variables['DZDYF'][0,:-20,:-20]
slope = np.sqrt(np.array(dzdxf)**2 + np.array(dzdyf)**2)
# %%
rain_max = [(np.argmax(rain[i][m]),rain[i][m].max()) if len(rain[i][m]) > 0 else np.nan for i,m in enumerate(mask)]
tmax = np.argmax(np.array([r[1] for r in rain_max]))
idx = rain_max[tmax][0]
x = long[mask[tmax]][idx]
y = lat[mask[tmax]][idx]

wspd_interp = wspd[tmax][mask[tmax]][idx]
wspd1_interp = wspd1[tmax][mask[tmax]][idx]
fuel_interp = griddata(np.c_[fxlong.ravel(),fxlat.ravel()],fuel_cat.ravel(),(x,y),method='nearest')
slope_interp = griddata(np.c_[fxlong.ravel(),fxlat.ravel()],slope.ravel(),(x,y))

#rain2_max = [np.argmax(rain1[i][m]) if len(rain1[i][m]) > 0 else np.nan for i,m in enumerate(mask)]
#wspd_max = [wspd[i][m].max() if len(wspd[i][m]) > 0 else np.nan for i,m in enumerate(mask)]
#wspd_max = [wspd[i][m].max() if len(wspd[i][m]) > 0 else np.nan for i,m in enumerate(mask)]

# %%
# eng = matlab.engine.start_matlab()
# eng.fuels()
# eng.fire_ros(fuel,speed,tanphi)
# eng.quit()