# %%
from datetime import datetime
from os import times
from tkinter import N
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
#import xarray as xr
from netCDF4 import Dataset
#import sys
#import os
import glob
from wrf import (getvar, interplevel, to_np, latlon_coords, get_cartopy,
                 ALL_TIMES)
# %%
fire_files = sorted(glob.glob('/home/015911532/creek_fire/d02/wrfout_d02_2020-09*'))
no_fire_files = sorted(glob.glob('/home/015911532/creek_fire/d02_nofire/wrfout_d02_2020-09*'))
fire_line_files = sorted(glob.glob('/home/015911532/creek_fire/d03/wrfout_d03_2020-09*'))
time_d03 = [(f,t,k) for f in fire_line_files for k,t in enumerate(getvar(Dataset(f), "times", ALL_TIMES, meta=False))]
# %%
lamCon = ccrs.LambertConformal(central_longitude=-120,
                               central_latitude=38,
                               standard_parallels=(30, 60))
platCar = ccrs.PlateCarree()
states = cfeature.NaturalEarthFeature(category='cultural', 
                             scale='50m', 
                             facecolor='none',
                             name='admin_1_states_provinces')
# %%
# loop through all d01 files
for idx in range(len(fire_files)):
    print("Processing index ", idx+1, "/", len(fire_files))
    # open d01 fire/no fire files
    nc_f = Dataset(fire_files[idx])
    nc_nf = Dataset(no_fire_files[idx])
    #nc_nf = Dataset(no_fire_files[idx])
    # getting all timesteps in d02
    time_d01 = getvar(nc_f, "times", timeidx=ALL_TIMES, meta=False)
    time_d01_nf = getvar(nc_nf, "times", timeidx=ALL_TIMES, meta=False)
    t_cut = min(len(time_d01),len(time_d01_nf))
    time_d01 = time_d01[:t_cut]
    # get variables from d01 fire/no fire files
    #p = getvar(nc_f, "pressure", timeidx=ALL_TIMES)
    #p1 = getvar(nc_nf, "pressure", timeidx=ALL_TIMES)
    #z = getvar(nc_f, "z", units="dm", timeidx=ALL_TIMES)
    #z1 = getvar(nc_nf, "z", units="dm", timeidx=ALL_TIMES)
    #slp = getvar(nc_f, "slp", timeidx=ALL_TIMES)
    #slp1 = getvar(nc_nf, "slp", timeidx=ALL_TIMES)
    rainc = nc_f.variables['RAINC'][:t_cut]
    rainnc = nc_f.variables['RAINNC'][:t_cut]
    rainc_nf = nc_nf.variables['RAINC'][:t_cut]
    rainnc_nf = nc_nf.variables['RAINNC'][:t_cut]
    rain=(rainc[:,:,:]+rainnc[:,:,:]) - (rainc_nf[:,:,:]+rainnc_nf[:,:,:])
    #ctt = getvar(nc_f, "cloudfrac", timeidx=ALL_TIMES)
    # Get all the surface pressures interpolated by geopotential height
    #ht_850 = interplevel(p, z, 715)
    #ht_850_2 = interplevel(p1, z1, 500)
    # get coordinates and map projection from d01 
    #lats, lons = latlon_coords(ht_850)
    #cart_proj = get_cartopy(ht_850)

    lamCon = ccrs.LambertConformal(central_longitude=-120,
                               central_latitude=38,
                               standard_parallels=(30, 60))
    platCar = ccrs.PlateCarree()

    # read in d01 topo file
    im = plt.imread('/home/015911532/creek_fire/topo.png')
    for tidx in range(len(time_d01)):
        # time in d01
        t = time_d01[tidx]
        # computing temporal distance between time in d01 vs all times in d02
        t_dist = [abs(i[1]-t) for i in time_d03]
        # find the closest time in d02 compared to d01
        k = np.argmin(t_dist)
        # open d02 which has the closest time by minimizing the temporal distance
        nc_fire = Dataset(time_d03[k][0])

        if idx == 0:
            lats1 = nc_fire.variables['XLAT'][0,:,:]
            lons1 = nc_fire.variables['XLONG'][0,:,:]
        
        # read AVG_FUEL_FRAC from file nc_fire at the index found by minimizing the temporal distance
        fire = nc_fire.variables['AVG_FUEL_FRAC'][time_d03[k][2]]
        fig = plt.figure(1, figsize=(12, 8))
        ax = plt.subplot(111,projection=lamCon)
        plt.imshow(im, extent=[lons1.min(),lons1.max(),lats1.min(),lats1.max()],transform=lamCon,alpha=0.5)
        #plt.contour(lons, lats, wspd, np.arange(-0.7,0.7,0.02), alpha=0, cmap=get_cmap("jet"), antialiased=True)
        #try:
        plt.contour(lons1, lats1, fire, colors="black", linewidths=0.8, antialiased=True)
        
        levels = np.linspace(0, 4.5, 100)
        qc = to_np(rain[tidx])
        qc[qc == 0] = np.nan
        plt.contourf(to_np(lons1), to_np(lats1), qc, levels=levels, transform=lamCon, alpha=0.7, cmap='viridis_r')
        ax.set_xticks(np.linspace(lons1.min(),lons1.max(),8))
        ax.set_yticks(np.linspace(lats1.min(),lats1.max(),6))
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.grid(True)
        plt.title("WRF Fire-Induced Accumulated Precipitation for Creek Fire "+str(time_d01[tidx])[:19])
        cbar = plt.colorbar(ax=ax,shrink=0.95)
        cbar.set_label('Accumulated Precipitation (mm)', rotation=270)
        fig.savefig('fig_'+str(idx)+"_"+str(tidx)+'_surf'+'.png',bbox_inches='tight',dpi=150)
        plt.close(fig)
# %%
