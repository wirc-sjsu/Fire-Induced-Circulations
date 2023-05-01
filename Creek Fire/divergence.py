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
import metpy.calc as mpc
from metpy.units import units
# %%
print("Getting all needed files")
fire_files = sorted(glob.glob('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-*'))
no_fire_files = sorted(glob.glob('/home/015911532/creek_fire/d03_nofire/wrfout_d03_2020-09-*'))
fire_line_files = sorted(glob.glob('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-*'))
time_d02 = [(f,t,k) for f in fire_line_files for k,t in enumerate(getvar(Dataset(f), "times", ALL_TIMES, meta=False))]
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
# loop through all d01 files
print("Looping through all files")
for idx in range(len(fire_files)):
    print("Processing index ", idx+1, "/", len(fire_files))
    # open d01 fire/no fire files
    nc_f = Dataset(fire_files[idx])
    nc_nf = Dataset(no_fire_files[idx])
    # getting all timesteps in d01
    time_d01 = getvar(nc_f, "times", timeidx=ALL_TIMES, meta=False)
    # get variables from d01 fire/no fire files
    #p = getvar(nc_f, "pressure", timeidx=ALL_TIMES)
    #p1 = getvar(nc_nf, "pressure", timeidx=ALL_TIMES)
    #z = getvar(nc_f, "z", units="m", timeidx=ALL_TIMES)
    #z1 = getvar(nc_nf, "z", units="m", timeidx=ALL_TIMES)
    #slp = getvar(nc_f, "slp", timeidx=ALL_TIMES)
    #slp1 = getvar(nc_nf, "slp", timeidx=ALL_TIMES)
    #u10,v10 = getvar(nc_f, "uvmet10", timeidx=ALL_TIMES, units = "m/s")
    #u10_1,v10_1 = getvar(nc_nf, "uvmet10", timeidx=ALL_TIMES, units = "m/s")
    # Get all the surface pressures interpolated by geopotential height
    #ht_850 = interplevel(p, z, slp)
    #ht_850_2 = interplevel(p1, z1, slp1)
    # get coordinates and map projection from d01 
    #lats, lons = latlon_coords(ht_850)
    #cart_proj = get_cartopy(ht_850)
    # read in d01 topo file
    im = plt.imread('/home/015911532/creek_fire/topo.png')
    for tidx in range(len(time_d01)):
        # time in d01
        t = time_d01[tidx]
        # computing temporal distance between time in d01 vs all times in d02
        t_dist = [abs(i[1]-t) for i in time_d02]
        # find the closest time in d02 compared to d01
        k = np.argmin(t_dist)
        # open d02 which has the closest time by minimizing the temporal distance
        nc_fire = Dataset(time_d02[k][0])

        #if idx == 0:
        lats = nc_fire.variables['XLAT'][0,:,:]
        lons = nc_fire.variables['XLONG'][0,:,:]
        
        # read AVG_FUEL_FRAC from file nc_fire at the index found by minimizing the temporal distance
        u10 = nc_f.variables['U10'][tidx]
        v10 = nc_f.variables['V10'][tidx]
        #wspd = np.sqrt(u10**2 + v10**2)
        u10_2 = nc_nf.variables['U10'][tidx]
        v10_2 = nc_nf.variables['V10'][tidx]
        #wspd2 = np.sqrt(u10_2**2 + v10_2**2)
        #wspd_tot = np.array(wspd) - np.array(wspd2)
        #u_tot = np.array(u10) - np.array(u10_2)
        #v_tot = np.array(v10) - np.array(v10_2) 
        div = mpc.divergence(u10, v10, dx = 10 * units.m, dy = 10 * units.m)
        #div2 = mpc.divergence(u10_2, v10_2, dx = 10 * units.m, dy = 10 * units.m)
        #div_tot = np.array(div) - np.array(div2)
        fire = nc_fire.variables['AVG_FUEL_FRAC'][tidx]
        fig = plt.figure(1, figsize=(12, 8))
        ax = plt.subplot(111, projection=lamCon)
        plt.imshow(im, extent=[lons.min(),lons.max(),lats.min(),lats.max()],transform=lamCon,alpha=0.5)
        #plt.contour(lons, lats, wspd, np.arange(-0.7,0.7,0.02), alpha=0, cmap=get_cmap("jet"), antialiased=True)
        #try:
        plt.contour(lons, lats, fire, colors="black", linewidths=0.8, antialiased=True)
        #except: 
        #    print("Warning: No fire in timestep ", time_d01[tidx]) 
        try:
            #levels = np.arange(-20, 20, 1)
            #plt.contour(lons, lats, wspd, np.arange(-20,20,5), colors="black", linewidths=0.4, alpha=0.7, antialiased=True)
            #plt.contourf(to_np(lons), to_np(lats), wspd_tot, levels=levels, cmap=get_cmap("RdBu_r"), transform=lamCon, linestyles='solid', alpha=0.7)
            #plt.plot([-119.45,-118.80],[34.24,34.65],color='black',linewidth=3,label='a).')
            #plt.plot([-118.97,-119.52],[34.30,34.50],color='black',linewidth=3,label='b).')
            contours = plt.pcolormesh(to_np(lons), to_np(lats), -(div), vmin=-0.3, vmax=0.3, cmap=get_cmap("RdBu_r"), transform=lamCon, linestyles='solid', alpha=0.5)
            cbar = plt.colorbar(ax=ax,shrink=0.95)
            cbar.set_label('Horizontal Convergence (1/m)', rotation=270)
        except: 
            print("Warning: No divergence contours in timestep ", time_d01[tidx])
        #u = u10[tidx]
        #v = v10[tidx]
        #u1 = u10_2[tidx]
        #v1 = v10_2[tidx] 
        ax.streamplot(lons[::ref,::ref], lats[::ref,::ref], u10[::ref,::ref], v10[::ref,::ref], density = 2, transform=lamCon, color='black')
        #ax.quiver(lons[::ref,::ref], lats[::ref,::ref], u1[::ref,::ref], v1[::ref,::ref], pivot='middle', transform=cart_proj, color='b')
        ax.set_xticks(np.linspace(lons.min(),lons.max(),8))
        ax.set_yticks(np.linspace(lats.min(),lats.max(),6))
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.grid(True)
        plt.title("WRF 10-meter Horizontal Convergence With Creek Fire "+str(time_d01[tidx])[:19])
        #plt.clabel(contours, inline=1, fontsize=10, fmt='%1.3f',colors='black')
        fig.savefig('fig_'+str(idx)+"_"+str(tidx)+'_surf'+'.png',bbox_inches='tight',dpi=150)
        plt.close(fig)

        if len(u10) != len(u10_2):
            break
# %%
