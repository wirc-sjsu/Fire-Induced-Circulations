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
    level = 629
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
        plt.title("WRF Surface Geopotential Height Difference and Wind Field \nWith Creek Fire (Fire - No Fire) "+str(time_d01[tidx])[:19])
        fig.savefig('fig_'+str(idx)+"_"+str(tidx)+'_surf'+'.png',bbox_inches='tight',dpi=150)
        plt.close(fig)

        if len(u10) != len(u10_1):
            break
# %%
# from datetime import datetime
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap
# import numpy as np
# #import xarray as xr
# from netCDF4 import Dataset
# import sys
# import os
# import glob
# from wrf import (getvar, interplevel, to_np, latlon_coords, get_cartopy,
#                  cartopy_xlim, cartopy_ylim)

# nc_fid=Dataset('/home/015911532/creek_fire/d02/wrfout_d02_2020-09-05_00:00:00')
# nc_fid2=Dataset('/home/015911532/creek_fire/d02_nofire/wrfout_d02_2020-09-05_00:00:00')
# nc_fid3=Dataset('/home/015911532/creek_fire/d02/wrfout_d02_2020-09-06_00:00:00')
# nc_fid4=Dataset('/home/015911532/creek_fire/d02_nofire/wrfout_d02_2020-09-06_00:00:00')
# nc_fid5=Dataset('/home/015911532/creek_fire/d02/wrfout_d02_2020-09-07_00:00:00')
# nc_fid6=Dataset('/home/015911532/creek_fire/d02_nofire/wrfout_d02_2020-09-07_00:00:00')
# nc_fid7=Dataset('/home/015911532/creek_fire/d02/wrfout_d02_2020-09-08_00:00:00')
# nc_fid8=Dataset('/home/015911532/creek_fire/d02_nofire/wrfout_d02_2020-09-08_00:00:00')
# nc_fid9=Dataset('/home/015911532/creek_fire/d02/wrfout_d02_2020-09-08_00:30:00')
# nc_fid10=Dataset('/home/015911532/creek_fire/d02_nofire/wrfout_d02_2020-09-08_00:30:00')
# nc_fid11=Dataset('/home/015911532/creek_fire/d02/wrfout_d02_2020-09-09_00:30:00')
# nc_fid12=Dataset('/home/015911532/creek_fire/d02_nofire/wrfout_d02_2020-09-09_00:30:00')
# nc_fid13=Dataset('/home/015911532/creek_fire/d02/wrfout_d02_2020-09-10_00:30:00')
# nc_fid14=Dataset('/home/015911532/creek_fire/d02_nofire/wrfout_d02_2020-09-10_00:30:00')
# nc_fid15=Dataset('/home/015911532/creek_fire/d02/wrfout_d02_2020-09-11_00:30:00')
# nc_fid16=Dataset('/home/015911532/creek_fire/d02_nofire/wrfout_d02_2020-09-11_00:30:00')
# nc_fid17=Dataset('/home/015911532/creek_fire/d02/wrfout_d02_2020-09-12_00:30:00')
# nc_fid18=Dataset('/home/015911532/creek_fire/d02_nofire/wrfout_d02_2020-09-12_00:30:00')
# nc_fid19=Dataset('/home/015911532/creek_fire/d02/wrfout_d02_2020-09-13_00:30:00')
# nc_fid20=Dataset('/home/015911532/creek_fire/d02_nofire/wrfout_d02_2020-09-13_00:30:00')
# nc_fid21=Dataset('/home/015911532/creek_fire/d02/wrfout_d02_2020-09-14_00:30:00')
# nc_fid22=Dataset('/home/015911532/creek_fire/d02_nofire/wrfout_d02_2020-09-14_00:30:00')
# nc_fire=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-05_00:00:00')
# nc_fire1=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-06_00:00:00')
# nc_fire2=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-07_00:00:00')
# nc_fire3=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-08_00:00:00')
# nc_fire4=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-08_00:15:00')
# nc_fire5=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-09_00:15:00')
# nc_fire6=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-10_00:15:00')
# nc_fire7=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-11_00:15:00')
# nc_fire8=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-12_00:15:00')
# nc_fire9=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-13_00:15:00')
# nc_fire10=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-14_00:15:00')
# sfile='wrfout_d03_2020_09-05_00:00:00'
# sfile1='wrfout_d03_2020_09-06_00:00:00'
# sfile2='wrfout_d03_2020_09-07_00:00:00'
# sfile3='wrfout_d03_2020_09-08_00:00:00'
# sfile4='wrfout_d03_2020_09-08_00:30:00'
# sfile5='wrfout_d03_2020_09-09_00:30:00'
# sfile6='wrfout_d03_2020_09-10_00:30:00'
# sfile7='wrfout_d03_2020_09-11_00:30:00'
# sfile8='wrfout_d03_2020_09-12_00:30:00'
# sfile9='wrfout_d03_2020_09-13_00:30:00'
# sfile10='wrfout_d03_2020_09-14_00:30:00'
# #print(nc_fid.variables.keys())
# # #data.coords
# lats1 = nc_fire.variables['XLAT'][0,:,:]
# lons1 = nc_fire.variables['XLONG'][0,:,:]
# #print(np.shape(lats))
# #print(lons.min())
# #print(lons.max())
# #print(lats.min())
# #print(lats.max())
# u10 = nc_fid.variables['U10'][:] - nc_fid2.variables['U10'][:]
# u10_1 = nc_fid3.variables['U10'][:] - nc_fid4.variables['U10'][:]
# u10_2 = nc_fid5.variables['U10'][:] - nc_fid6.variables['U10'][:]
# u10_3 = nc_fid7.variables['U10'][0,:,:] - nc_fid8.variables['U10'][0,:,:]
# u10_4 = nc_fid9.variables['U10'][:] - nc_fid10.variables['U10'][:]
# u10_5 = nc_fid11.variables['U10'][:] - nc_fid12.variables['U10'][:]
# u10_6 = nc_fid13.variables['U10'][:] - nc_fid14.variables['U10'][:]
# u10_7 = nc_fid15.variables['U10'][:] - nc_fid16.variables['U10'][:]
# u10_8 = nc_fid17.variables['U10'][:] - nc_fid18.variables['U10'][:]
# u10_9 = nc_fid19.variables['U10'][:] - nc_fid20.variables['U10'][:]
# u10_10 = nc_fid21.variables['U10'][:] - nc_fid22.variables['U10'][:]
# v10 = nc_fid.variables['V10'][:] - nc_fid2.variables['V10'][:]
# v10_1 = nc_fid3.variables['V10'][:] - nc_fid4.variables['V10'][:]
# v10_2 = nc_fid5.variables['V10'][:] - nc_fid6.variables['V10'][:]
# v10_3 = nc_fid7.variables['V10'][0,:,:] - nc_fid8.variables['V10'][0,:,:]
# v10_4 = nc_fid9.variables['V10'][:] - nc_fid10.variables['V10'][:]
# v10_5 = nc_fid11.variables['V10'][:] - nc_fid12.variables['V10'][:]
# v10_6 = nc_fid13.variables['V10'][:] - nc_fid14.variables['V10'][:]
# v10_7 = nc_fid15.variables['V10'][:] - nc_fid16.variables['V10'][:]
# v10_8 = nc_fid17.variables['V10'][:] - nc_fid18.variables['V10'][:]
# v10_9 = nc_fid19.variables['V10'][:] - nc_fid20.variables['V10'][:]
# v10_10 = nc_fid21.variables['V10'][:] - nc_fid22.variables['V10'][:]
# wind0=u10[-1,:,:]
# wind=v10[-1,:,:]
# wind05=u10_1[-1,:,:]
# wind1=v10_1[-1,:,:]
# wind15=u10_2[-1,:,:]
# wind2=v10_2[-1,:,:]
# wind25=u10_3[:,:]
# wind3=v10_3[:,:]
# wind35=u10_4[-1,:,:]
# wind4=v10_4[-1,:,:]
# wind45=u10_5[-1,:,:]
# wind5=v10_5[-1,:,:]
# wind55=u10_6[-1,:,:]
# wind6=v10_6[-1,:,:]
# wind65=u10_7[-1,:,:]
# wind7=v10_7[-1,:,:]
# wind75=u10_8[-1,:,:]
# wind8=v10_8[-1,:,:]
# wind85=u10_9[-1,:,:]
# wind9=v10_9[-1,:,:]
# wind95=u10_10[-1,:,:]
# wind10=v10_10[-1,:,:]
# wspd = np.sqrt(np.array(wind0)**2 + np.array(wind)**2)
# wspd1 = np.sqrt(np.array(wind05)**2 + np.array(wind1)**2)
# wspd2 = np.sqrt(np.array(wind15)**2 + np.array(wind2)**2)
# wspd3 = np.sqrt(np.array(wind25)**2 + np.array(wind3)**2)
# wspd4 = np.sqrt(np.array(wind35)**2 + np.array(wind4)**2)
# wspd5 = np.sqrt(np.array(wind45)**2 + np.array(wind5)**2)
# wspd6 = np.sqrt(np.array(wind55)**2 + np.array(wind6)**2)
# wspd7 = np.sqrt(np.array(wind65)**2 + np.array(wind7)**2)
# wspd8 = np.sqrt(np.array(wind75)**2 + np.array(wind8)**2)
# wspd9 = np.sqrt(np.array(wind85)**2 + np.array(wind9)**2)
# wspd10 = np.sqrt(np.array(wind95)**2 + np.array(wind10)**2)
# fire = nc_fire.variables['AVG_FUEL_FRAC'][-1]
# #print(np.shape(fire))
# fire1 = nc_fire.variables['AVG_FUEL_FRAC'][-1]
# fire2 = nc_fire1.variables['AVG_FUEL_FRAC'][-1]
# fire3 = nc_fire2.variables['AVG_FUEL_FRAC'][-1]
# fire4 = nc_fire3.variables['AVG_FUEL_FRAC'][-1]
# fire5 = nc_fire4.variables['AVG_FUEL_FRAC'][-1]
# fire6 = nc_fire5.variables['AVG_FUEL_FRAC'][-1]
# fire7 = nc_fire6.variables['AVG_FUEL_FRAC'][-1]
# fire8 = nc_fire7.variables['AVG_FUEL_FRAC'][-1]
# fire9 = nc_fire8.variables['AVG_FUEL_FRAC'][-1]
# fire10 = nc_fire9.variables['AVG_FUEL_FRAC'][-1]
# fire11 = nc_fire10.variables['AVG_FUEL_FRAC'][-1]
# p = getvar(nc_fid, "pressure")
# p1 = getvar(nc_fid2, "pressure")
# p2 = getvar(nc_fid3, "pressure")
# p3 = getvar(nc_fid4, "pressure")
# p4 = getvar(nc_fid5, "pressure")
# p5 = getvar(nc_fid6, "pressure")
# p6 = getvar(nc_fid7, "pressure")
# p7 = getvar(nc_fid8, "pressure")
# p8 = getvar(nc_fid9, "pressure")
# p9 = getvar(nc_fid10, "pressure")
# p10 = getvar(nc_fid11, "pressure")
# p11 = getvar(nc_fid12, "pressure")
# p12 = getvar(nc_fid13, "pressure")
# p13 = getvar(nc_fid14, "pressure")
# p14 = getvar(nc_fid15, "pressure")
# p15 = getvar(nc_fid16, "pressure")
# p16 = getvar(nc_fid17, "pressure")
# p17 = getvar(nc_fid18, "pressure")
# p18 = getvar(nc_fid19, "pressure")
# p19 = getvar(nc_fid20, "pressure")
# p20 = getvar(nc_fid21, "pressure")
# p21 = getvar(nc_fid22, "pressure")
# z = getvar(nc_fid, "z", units="dm")
# z1 = getvar(nc_fid2, "z", units="dm")
# z2 = getvar(nc_fid3, "z", units="dm")
# z3 = getvar(nc_fid4, "z", units="dm")
# z4 = getvar(nc_fid5, "z", units="dm")
# z5 = getvar(nc_fid6, "z", units="dm")
# z6 = getvar(nc_fid7, "z", units="dm")
# z7 = getvar(nc_fid8, "z", units="dm")
# z8 = getvar(nc_fid9, "z", units="dm")
# z9 = getvar(nc_fid10, "z", units="dm")
# z10 = getvar(nc_fid11, "z", units="dm")
# z11 = getvar(nc_fid12, "z", units="dm")
# z12 = getvar(nc_fid13, "z", units="dm")
# z13 = getvar(nc_fid14, "z", units="dm")
# z14 = getvar(nc_fid15, "z", units="dm")
# z15 = getvar(nc_fid16, "z", units="dm")
# z16 = getvar(nc_fid17, "z", units="dm")
# z17 = getvar(nc_fid18, "z", units="dm")
# z18 = getvar(nc_fid19, "z", units="dm")
# z19 = getvar(nc_fid20, "z", units="dm")
# z20 = getvar(nc_fid21, "z", units="dm")
# z21 = getvar(nc_fid22, "z", units="dm")
# slp = getvar(nc_fid, "slp")
# slp1 = getvar(nc_fid2, "slp")
# slp2 = getvar(nc_fid3, "slp")
# slp3 = getvar(nc_fid4, "slp")
# slp4 = getvar(nc_fid5, "slp")
# slp5 = getvar(nc_fid6, "slp")
# slp6 = getvar(nc_fid7, "slp")
# slp7 = getvar(nc_fid8, "slp")
# slp8 = getvar(nc_fid9, "slp")
# slp9 = getvar(nc_fid10, "slp")
# slp10 = getvar(nc_fid11, "slp")
# slp11 = getvar(nc_fid12, "slp")
# slp12 = getvar(nc_fid13, "slp")
# slp13 = getvar(nc_fid14, "slp")
# slp14 = getvar(nc_fid15, "slp")
# slp15 = getvar(nc_fid16, "slp")
# slp16 = getvar(nc_fid17, "slp")
# slp17 = getvar(nc_fid18, "slp")
# slp18 = getvar(nc_fid19, "slp")
# slp19 = getvar(nc_fid20, "slp")
# slp20 = getvar(nc_fid21, "slp")
# slp21 = getvar(nc_fid22, "slp")
# lamCon = ccrs.LambertConformal(central_longitude=-120,
#                                central_latitude=38,
#                                standard_parallels=(30, 60))
# platCar = ccrs.PlateCarree()
# states = cfeature.NaturalEarthFeature(category='cultural', 
#                              scale='50m', 
#                              facecolor='none',
#                              name='admin_1_states_provinces')
# # #Plot
# im = plt.imread('/home/015911532/creek_fire/topo_d02.png')
# ht_850 = interplevel(p, z, slp)
# ht_850_2 = interplevel(p1, z1, slp1) 
# lats, lons = latlon_coords(ht_850)
# # print(lons.min())
# # print(lons.max())
# # print(lats.min())
# # print(lats.max())
# cart_proj = get_cartopy(ht_850)
# fig = plt.figure(1, figsize=(12, 8))
# ax = plt.subplot(111,projection=cart_proj)
# plt.imshow(im, extent=[lons.min(),lons.max(),lats.min(),lats.max()],transform=cart_proj,alpha=0.5)
# #ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
# #ax.add_feature(cfeature.STATES.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
# #plt.contour(lons, lats, wspd, np.arange(0,10,1), colors="black", alpha=0.7, linewidths=0.4, antialiased=True)
# plt.contour(lons, lats, wspd, np.arange(-0.8,0.8,0.02), alpha=0, cmap=get_cmap("jet"), antialiased=True)
# #sm = np.array(np.arange(-0.7,0.7,0.02))
# cbar = plt.colorbar(ax=ax,shrink=0.95)
# cbar.set_label('Geopotential Height Difference (hPa)', rotation=270)
# plt.contour(lons1, lats1, fire, colors="black", linewidths=0.8, antialiased=True)
# levels = np.arange(-0.8, 0.8, 0.02)
# contours = plt.contour(to_np(lons), to_np(lats), to_np(ht_850 - ht_850_2), levels=levels, cmap=get_cmap("jet"), transform=cart_proj, linestyles='solid', alpha=0.7)
# #contours2 = plt.contour(to_np(lons), to_np(lats), to_np(ht_850_2), levels=levels, colors="blue", transform=cart_proj)
# # labels = ['Geopotential Height Difference']
# # #labels2 = ['No Fire']
# # for i in range(len(labels)):
# #     contours.collections[i].set_label(labels[i])
# #     #contours2.collections[i].set_label(labels2[i])
# # plt.legend(loc='upper right')
# plt.clabel(contours, inline=1, fontsize=10, fmt='%1.3f')
# #plt.clabel(contours2, inline=1, fontsize=10, fmt="%i")
# ax.set_xticks(np.linspace(lons.min(),lons.max(),8))
# ax.set_yticks(np.linspace(lats.min(),lats.max(),6))
# #ax.set_xlim(cartopy_xlim(ht_850))
# #ax.set_ylim(cartopy_ylim(ht_850))
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# plt.grid(True)
# plt.title("WRF Surface Geopotential Height Difference With Creek Fire (Fire - No Fire) "+sfile[11:24])
# fig.savefig(sfile[0:24]+'_surf'+'.png',bbox_inches='tight',dpi=150)
# plt.close(fig)

# %%
