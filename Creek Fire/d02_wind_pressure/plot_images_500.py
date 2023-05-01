# %%
print("Importing packages")
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
print("Getting all needed files")
fire_files = sorted(glob.glob('/home/015911532/creek_fire/d02/wrfout_d02_2020-09*'))
no_fire_files = sorted(glob.glob('/home/015911532/creek_fire/d02_nofire/wrfout_d02_2020-09*'))
fire_line_files = sorted(glob.glob('/home/015911532/creek_fire/d03/wrfout_d03_2020-09*'))
time_d03 = [(f,t,k) for f in fire_line_files for k,t in enumerate(getvar(Dataset(f), "times", ALL_TIMES, meta=False))]
# %%
print("Getting map projections")
lamCon = ccrs.LambertConformal(central_longitude=-120,
                               central_latitude=38,
                               standard_parallels=(30, 60))
platCar = ccrs.PlateCarree()
states = cfeature.NaturalEarthFeature(category='cultural', 
                             scale='50m', 
                             facecolor='none',
                             name='admin_1_states_provinces')
ref = 10
# %%
print("Looping through all files")
# loop through all d01 files
for idx in range(0, len(fire_files)):
    print("Processing index ", idx+1, "/", len(fire_files))
    # open d01 fire/no fire files
    nc_f = Dataset(fire_files[idx])
    nc_nf = Dataset(no_fire_files[idx])
    # getting all timesteps in d01
    time_d01 = getvar(nc_f, "times", timeidx=ALL_TIMES, meta=False)
    # get variables from d01 fire/no fire files
    units_z = "m"
    p = getvar(nc_f, "pressure", timeidx=ALL_TIMES)
    p1 = getvar(nc_nf, "pressure", timeidx=ALL_TIMES)
    z = getvar(nc_f, "z", units=units_z, timeidx=ALL_TIMES)
    z1 = getvar(nc_nf, "z", units=units_z, timeidx=ALL_TIMES)
    #slp = getvar(nc_f, "slp", timeidx=ALL_TIMES)
    #slp1 = getvar(nc_nf, "slp", timeidx=ALL_TIMES)
    u10,v10 = getvar(nc_f, "uvmet10", timeidx=ALL_TIMES, units = "m/s")
    u10_1,v10_1 = getvar(nc_nf, "uvmet10", timeidx=ALL_TIMES, units = "m/s")
    # Get all the surface pressures interpolated by geopotential height
    level = 500
    ht_850 = interplevel(z, p, level)
    ht_850_2 = interplevel(z1, p1, level)
    # get coordinates and map projection from d01 
    lats, lons = latlon_coords(ht_850)
    cart_proj = get_cartopy(ht_850)
    # read in d01 topo file
    im = plt.imread('/home/015911532/creek_fire/topo_d02.png')
    for tidx in range(len(time_d01)):
        # time in d01
        t = time_d01[tidx]
        # computing temporal distance between time in d01 vs all times in d02
        t_dist = [abs(i[1]-t) for i in time_d03]
        # find the closest time in d02 compared to d01
        k = np.argmin(t_dist)
        # open d02 which has the closest time by minimizing the temporal distance
        nc_fire = Dataset(time_d03[k][0])

        ##if idx == 0:
        lats1 = nc_fire.variables['XLAT'][0,:,:]
        lons1 = nc_fire.variables['XLONG'][0,:,:]
        
        # read AVG_FUEL_FRAC from file nc_fire at the index found by minimizing the temporal distance
        fire = nc_fire.variables['AVG_FUEL_FRAC'][time_d03[k][2]]
        fig = plt.figure(1, figsize=(12, 8))
        ax = plt.subplot(111,projection=cart_proj)
        plt.imshow(im, extent=[lons.min(),lons.max(),lats.min(),lats.max()],transform=cart_proj,alpha=0.5)
        #plt.contour(lons, lats, wspd, np.arange(-0.7,0.7,0.02), alpha=0, cmap=get_cmap("jet"), antialiased=True)
        #try:
        plt.contour(lons1, lats1, fire, colors="black", linewidths=0.8, antialiased=True)
        #except: 
        #    print("Warning: No fire in timestep ", time_d01[tidx]) 
        try:
            #levels = np.arange(-3.0, 3.0, 0.02)
            contours = plt.pcolormesh(to_np(lons), to_np(lats), to_np(ht_850[tidx] - ht_850_2[tidx]), vmin=-3.0, vmax=3.0, cmap=get_cmap("RdBu_r"), transform=cart_proj, linestyles='solid', alpha=0.5)
            cbar = plt.colorbar(ax=ax,shrink=0.95)
            cbar.set_label('Geopotential Height Difference (m)', rotation=270)
            #plt.clabel(contours, inline=1, fontsize=10, fmt='%1.3f', colors='black')
        except: 
            print("Warning: No pressure contours in timestep ", time_d01[tidx])
        #X,Y = np.meshgrid(lons,lats) 
        u = u10[tidx]
        v = v10[tidx]
        u1 = u10_1[tidx]
        v1 = v10_1[tidx] 
        plt.quiver(lons[::ref,::ref], lats[::ref,::ref], u[::ref,::ref], v[::ref,::ref], pivot='middle', transform=cart_proj, color='r')
        plt.quiver(lons[::ref,::ref], lats[::ref,::ref], u1[::ref,::ref], v1[::ref,::ref], pivot='middle', transform=cart_proj, color='b')
        ax.set_xticks(np.linspace(lons.min(),lons.max(),8))
        ax.set_yticks(np.linspace(lats.min(),lats.max(),6))
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.grid(True)
        plt.title("WRF 500 hPa Geopotential Height Difference and Wind Field \nWith Creek Fire (Fire - No Fire) "+str(time_d01[tidx])[:19])
        fig.savefig('fig_'+str(idx)+"_"+str(tidx)+'_500'+'.png',bbox_inches='tight',dpi=150)
        plt.close(fig)

        if len(u10) != len(u10_1):
            break