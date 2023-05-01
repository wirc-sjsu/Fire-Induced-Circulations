# %%
print("Importing Libraries")
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import sys
import os
import glob
from wrf import getvar, ALL_TIMES

matplotlib.use('Agg')
# %%
print("Getting all needed files")
fire_files = sorted(glob.glob('/home/015911532/creek_fire/d03/wrfout_d03_2020-09*'))
#fire_line_files = sorted(glob.glob('/home/015911532/creek_fire/d03/wrfout_d03_2020-09*'))
#list wrf output files
#for sfile in os.listdir(sdir):
#    print(sfile)

######################################
#Get data
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
for idx in range(len(fire_files)):
    print("Processing index ", idx+1, "/", len(fire_files))
    # open d01 fire/no fire files
    nc_f = Dataset(fire_files[idx])
    m,n = nc_f.variables['XLONG'][0,:,:].shape
    fm,fn = nc_f.variables['FXLONG'][0,:,:].shape
    fm = int(fm-fm//(m+1))
    fn = int(fn-fn//(n+1))
    # get variables from d01 fire/no fire files
    fmc = nc_f.variables['FMC_G'][:,:fm,:fn]    # Get all the surface pressures interpolated by geopotential height
    lats = nc_f.variables['FXLAT'][0,:fm,:fn]
    lons = nc_f.variables['FXLONG'][0,:fm,:fn]
    fire = 1 - nc_f.variables['FIRE_AREA'][:,:fm,:fn]
    time = getvar(nc_f, "times", ALL_TIMES, meta=False)
    # read in d01 topo file
    im = plt.imread('/home/015911532/creek_fire/topo.png')
    for tidx in range(len(time)):
        
        # lats_2 = np.linspace(lats.min(),lats.max(),3920)
        # lons_2 = np.linspace(lons.min(),lons.max(),3920)
        # lats_3 = np.array(np.meshgrid(lats_2, lats_2))
        # lons_3 = np.array(np.meshgrid(lons_2, lons_2))

    #Plot
        fig = plt.figure(1, figsize=(12, 8))
        ax = plt.subplot(111, projection=lamCon)
        plt.imshow(im, extent=[lons.min(),lons.max(),lats.min(),lats.max()], alpha = 0.5)

        try:
        #    plt.contourf(lons, lats, fmc, np.arange(0,1,100), cmap=get_cmap("Reds"), alpha=0.7, antialiased=True)
            plt.contourf(lons, lats, fmc[tidx,:,:]*100, np.linspace(0,32,20), cmap=get_cmap("gist_earth_r"), alpha=0.7, antialiased=True)      # use jet_r for FMC_GC
            cbar = plt.colorbar(ax=ax,shrink=0.95)
            cbar.set_label('Fuel Moisture Content (%)', rotation=270)
        except:
            print("Warning: No fmc contours in timestep ", time[tidx])
        plt.contour(lons, lats, fire[tidx,:,:], colors="black", linewidths=0.8, antialiased=True)
        
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
        ax.add_feature(cfeature.STATES.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    #ax.gridlines(color="black",linestyle="dotted",draw_labels=True)
        ax.set_xticks(np.linspace(lons.min(),lons.max(),8))
        ax.set_yticks(np.linspace(lats.min(),lats.max(),6))
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.grid(True)
    #ax.set_xlim(lons.min(), lons.max())
    #ax.set_ylim(lats.min(), lats.max())

        plt.title("WRF Fuel Moisture Content With Creek Fire "+str(time[tidx])[:19])

        fig.savefig('fig_'+str(idx)+"_"+str(tidx)+'_surf'+'.png',bbox_inches='tight',dpi=150)
        plt.close(fig)
# %%