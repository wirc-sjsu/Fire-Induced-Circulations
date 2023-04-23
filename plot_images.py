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
from wrf import getvar, ALL_TIMES, to_np
# %%
nc_fid=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-05_00:00:00')
nc_fid2=Dataset('/home/015911532/creek_fire/d03_nofire/wrfout_d03_2020-09-05_00:00:00')
nc_fid3=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-06_00:00:00')
nc_fid4=Dataset('/home/015911532/creek_fire/d03_nofire/wrfout_d03_2020-09-06_00:00:00')
nc_fid5=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-07_00:00:00')
nc_fid6=Dataset('/home/015911532/creek_fire/d03_nofire/wrfout_d03_2020-09-07_00:00:00')
#nc_fid7=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-08_00:00:00')
#nc_fid8=Dataset('/home/015911532/creek_fire/d03_nofire/wrfout_d03_2020-09-08_00:00:00')
nc_fid9=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-08_00:15:00')
nc_fid10=Dataset('/home/015911532/creek_fire/d03_nofire/wrfout_d03_2020-09-08_00:15:00')
nc_fid11=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-09_00:15:00')
nc_fid12=Dataset('/home/015911532/creek_fire/d03_nofire/wrfout_d03_2020-09-09_00:15:00')
nc_fid13=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-10_00:15:00')
nc_fid14=Dataset('/home/015911532/creek_fire/d03_nofire/wrfout_d03_2020-09-10_00:15:00')
nc_fid15=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-11_00:15:00')
nc_fid16=Dataset('/home/015911532/creek_fire/d03_nofire/wrfout_d03_2020-09-11_00:15:00')
nc_fid17=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-12_00:15:00')
nc_fid18=Dataset('/home/015911532/creek_fire/d03_nofire/wrfout_d03_2020-09-12_00:15:00')
nc_fid19=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-13_00:15:00')
nc_fid20=Dataset('/home/015911532/creek_fire/d03_nofire/wrfout_d03_2020-09-13_00:15:00')
nc_fid21=Dataset('/home/015911532/creek_fire/d03/wrfout_d03_2020-09-14_00:15:00')
nc_fid22=Dataset('/home/015911532/creek_fire/d03_nofire/wrfout_d03_2020-09-14_00:15:00')
sfile='wrfout_d03_2020_09-05_00:00:00'
sfile1='wrfout_d03_2020_09-06_00:00:00'
sfile2='wrfout_d03_2020_09-07_00:00:00'
sfile3='wrfout_d03_2020_09-08_00:00:00'
sfile4='wrfout_d03_2020_09-08_00:15:00'
sfile5='wrfout_d03_2020_09-09_00:15:00'
sfile6='wrfout_d03_2020_09-10_00:15:00'
sfile7='wrfout_d03_2020_09-11_00:15:00'
sfile8='wrfout_d03_2020_09-12_00:15:00'
sfile9='wrfout_d03_2020_09-13_00:15:00'
sfile10='wrfout_d03_2020_09-14_00:15:00'
# %%
# #data.coords
lats = nc_fid.variables['XLAT'][0,:,:]
lons = nc_fid.variables['XLONG'][0,:,:]
lamCon = ccrs.LambertConformal(central_longitude=-120,
                               central_latitude=38,
                               standard_parallels=(30, 60))
platCar = ccrs.PlateCarree()
# %%
time = getvar(nc_fid, "times", ALL_TIMES, meta=False)
time1 = getvar(nc_fid3, "times", ALL_TIMES, meta=False)
time2 = getvar(nc_fid5, "times", ALL_TIMES, meta=False)
#time3 = getvar(nc_fid7, "times", ALL_TIMES, meta=False)
time4 = getvar(nc_fid9, "times", ALL_TIMES, meta=False)
time5 = getvar(nc_fid11, "times", ALL_TIMES, meta=False)
time6 = getvar(nc_fid13, "times", ALL_TIMES, meta=False)
time7 = getvar(nc_fid15, "times", ALL_TIMES, meta=False)
time8 = getvar(nc_fid17, "times", ALL_TIMES, meta=False)
time9 = getvar(nc_fid19, "times", ALL_TIMES, meta=False)
time10 = getvar(nc_fid21, "times", ALL_TIMES, meta=False)
# %%
# #Plot
im = plt.imread('/home/015911532/creek_fire/topo.png')
for itime in range(len(time)):
    u10 = nc_fid.variables['U10'][itime]
    u10_2 = nc_fid2.variables['U10'][itime]
    v10 = nc_fid.variables['V10'][itime]
    v10_2 = nc_fid2.variables['V10'][itime]
    wspd = np.sqrt(np.array(u10)**2 + np.array(v10)**2)
    wspd2 = np.sqrt(np.array(u10_2)**2 + np.array(v10_2)**2)
    wspd_tot = np.array(wspd) - np.array(wspd2)
    fire = nc_fid.variables['AVG_FUEL_FRAC'][itime]
    fig = plt.figure(1, figsize=(12, 8))
    ax = plt.subplot(111, projection=lamCon)
    plt.imshow(im, extent=[lons.min(),lons.max(),lats.min(),lats.max()])
    #plt.contour(lons, lats, wspd_tot, np.arange(-18,18,1), colors="black", alpha=0.7, linewidths=0.4, antialiased=True)
    contours = plt.pcolormesh(to_np(lons), to_np(lats), wspd_tot, vmin=-4.0, vmax=4.0, cmap=get_cmap("RdBu_r"), transform=lamCon, linestyles='solid', alpha=0.5)
    cbar = plt.colorbar(ax=ax,shrink=0.95)
    cbar.set_label('Wind Speed (m/s)', rotation=270)
    plt.contour(lons, lats, fire, colors="black", linewidths=0.8, antialiased=True)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.add_feature(cfeature.STATES.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.set_xticks(np.linspace(lons.min(),lons.max(),8))
    ax.set_yticks(np.linspace(lats.min(),lats.max(),6))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.grid(True)
    plt.title("WRF Fire-Induced Wind Speed With Creek Fire (Fire - No Fire) "+str(time[itime])[:19])
    fig.savefig('fig_'+str(itime)+'.png',bbox_inches='tight',dpi=150)
    plt.close(fig)
# %%
# #Plot
im = plt.imread('/home/015911532/creek_fire/topo.png')
for itime2 in range(len(time1)):
    u10 = nc_fid3.variables['U10'][itime2]
    u10_2 = nc_fid4.variables['U10'][itime2]
    v10 = nc_fid3.variables['V10'][itime2]
    v10_2 = nc_fid4.variables['V10'][itime2]
    wspd = np.sqrt(np.array(u10)**2 + np.array(v10)**2)
    wspd2 = np.sqrt(np.array(u10_2)**2 + np.array(v10_2)**2)
    wspd_tot = np.array(wspd) - np.array(wspd2)
    fire = nc_fid3.variables['AVG_FUEL_FRAC'][itime2]
    fig = plt.figure(1, figsize=(12, 8))
    ax = plt.subplot(111, projection=lamCon)
    plt.imshow(im, extent=[lons.min(),lons.max(),lats.min(),lats.max()])
    #plt.contour(lons, lats, wspd_tot, np.arange(-18,18,1), colors="black", alpha=0.7, linewidths=0.4, antialiased=True)
    contours = plt.pcolormesh(to_np(lons), to_np(lats), wspd_tot, vmin=-4.0, vmax=4.0, cmap=get_cmap("RdBu_r"), transform=lamCon, linestyles='solid', alpha=0.5)    
    cbar = plt.colorbar(ax=ax,shrink=0.95)
    cbar.set_label('Wind Speed (m/s)', rotation=270)
    plt.contour(lons, lats, fire, colors="black", linewidths=0.8, antialiased=True)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.add_feature(cfeature.STATES.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.set_xticks(np.linspace(lons.min(),lons.max(),8))
    ax.set_yticks(np.linspace(lats.min(),lats.max(),6))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.grid(True)
    plt.title("WRF Fire-Induced Wind Speed With Creek Fire (Fire - No Fire) "+str(time1[itime2])[:19])
    fig.savefig('gig_'+str(itime2)+'.png',bbox_inches='tight',dpi=150)
    plt.close(fig)
# %%
# # #Plot
im = plt.imread('/home/015911532/creek_fire/topo.png')
for itime3 in range(len(time2)):
    u10 = nc_fid5.variables['U10'][itime3]
    u10_2 = nc_fid6.variables['U10'][itime3]
    v10 = nc_fid5.variables['V10'][itime3]
    v10_2 = nc_fid6.variables['V10'][itime3]
    wspd = np.sqrt(np.array(u10)**2 + np.array(v10)**2)
    wspd2 = np.sqrt(np.array(u10_2)**2 + np.array(v10_2)**2)
    wspd_tot = np.array(wspd) - np.array(wspd2)
    fire = nc_fid5.variables['AVG_FUEL_FRAC'][itime3]
    fig = plt.figure(1, figsize=(12, 8))
    ax = plt.subplot(111, projection=lamCon)
    plt.imshow(im, extent=[lons.min(),lons.max(),lats.min(),lats.max()])
    #plt.contour(lons, lats, wspd_tot, np.arange(-18,18,1), colors="black", alpha=0.7, linewidths=0.4, antialiased=True)
    contours = plt.pcolormesh(to_np(lons), to_np(lats), wspd_tot, vmin=-4.0, vmax=4.0, cmap=get_cmap("RdBu_r"), transform=lamCon, linestyles='solid', alpha=0.5)
    cbar = plt.colorbar(ax=ax,shrink=0.95)
    cbar.set_label('Wind Speed (m/s)', rotation=270)
    plt.contour(lons, lats, fire, colors="black", linewidths=0.8, antialiased=True)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.add_feature(cfeature.STATES.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.set_xticks(np.linspace(lons.min(),lons.max(),8))
    ax.set_yticks(np.linspace(lats.min(),lats.max(),6))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.grid(True)
    plt.title("WRF Fire-Induced Wind Speed With Creek Fire (Fire - No Fire) "+str(time2[itime3])[:19])
    fig.savefig('hig_'+str(itime3)+'.png',bbox_inches='tight',dpi=150)
    plt.close(fig)
# %%
# #Plot
# im = plt.imread('/home/015911532/creek_fire/topo.png')
# for itime4 in range(64):
#     u10 = nc_fid7.variables['U10'][itime4]
#     u10_2 = nc_fid8.variables['U10'][itime4]
#     v10 = nc_fid7.variables['V10'][itime4]
#     v10_2 = nc_fid8.variables['V10'][itime4]
#     wspd = np.sqrt(np.array(u10)**2 + np.array(v10)**2)
#     wspd2 = np.sqrt(np.array(u10_2)**2 + np.array(v10_2)**2)
#     wspd_tot = np.array(wspd) - np.array(wspd2)
#     fire = nc_fid7.variables['AVG_FUEL_FRAC'][itime4]
#     fig = plt.figure(1, figsize=(12, 8))
#     ax = plt.subplot(111, projection=lamCon)
#     plt.imshow(im, extent=[lons.min(),lons.max(),lats.min(),lats.max()])
#     #plt.contour(lons, lats, wspd_tot, np.arange(-18,18,1), colors="black", alpha=0.7, linewidths=0.4, antialiased=True)
#     contours = plt.pcolormesh(to_np(lons), to_np(lats), wspd_tot, vmin=-4.0, vmax=4.0, cmap=get_cmap("RdBu_r"), transform=lamCon, linestyles='solid', alpha=0.5)    
#     cbar = plt.colorbar(ax=ax,shrink=0.95)
#     cbar.set_label('Wind Speed (m/s)', rotation=270)
#     plt.contour(lons, lats, fire, colors="black", linewidths=0.8, antialiased=True)
#     ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
#     ax.add_feature(cfeature.STATES.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
#     ax.set_xticks(np.linspace(lons.min(),lons.max(),8))
#     ax.set_yticks(np.linspace(lats.min(),lats.max(),6))
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
#     plt.grid(True)
#     plt.title("WRF Fire-Induced Wind Speed With Creek Fire (Fire - No Fire) "+str(time3[itime4])[:19])
#     fig.savefig('iig_'+str(itime4)+'.png',bbox_inches='tight',dpi=150)
#     plt.close(fig)
# %%
# #Plot
im = plt.imread('/home/015911532/creek_fire/topo.png')
for itime5 in range(len(time4)):
    u10 = nc_fid9.variables['U10'][itime5]
    u10_2 = nc_fid10.variables['U10'][itime5]
    v10 = nc_fid9.variables['V10'][itime5]
    v10_2 = nc_fid10.variables['V10'][itime5]
    wspd = np.sqrt(np.array(u10)**2 + np.array(v10)**2)
    wspd2 = np.sqrt(np.array(u10_2)**2 + np.array(v10_2)**2)
    wspd_tot = np.array(wspd) - np.array(wspd2)
    fire = nc_fid9.variables['AVG_FUEL_FRAC'][itime5]
    fig = plt.figure(1, figsize=(12, 8))
    ax = plt.subplot(111, projection=lamCon)
    plt.imshow(im, extent=[lons.min(),lons.max(),lats.min(),lats.max()])
    #plt.contour(lons, lats, wspd_tot, np.arange(-18,18,1), colors="black", alpha=0.7, linewidths=0.4, antialiased=True)
    contours = plt.pcolormesh(to_np(lons), to_np(lats), wspd_tot, vmin=-4.0, vmax=4.0, cmap=get_cmap("RdBu_r"), transform=lamCon, linestyles='solid', alpha=0.5)    
    cbar = plt.colorbar(ax=ax,shrink=0.95)
    cbar.set_label('Wind Speed (m/s)', rotation=270)
    plt.contour(lons, lats, fire, colors="black", linewidths=0.8, antialiased=True)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.add_feature(cfeature.STATES.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.set_xticks(np.linspace(lons.min(),lons.max(),8))
    ax.set_yticks(np.linspace(lats.min(),lats.max(),6))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.grid(True)
    plt.title("WRF Fire-Induced Wind Speed With Creek Fire (Fire - No Fire) "+str(time4[itime5])[:19])
    fig.savefig('jig_'+str(itime5)+'.png',bbox_inches='tight',dpi=150)
    plt.close(fig)
# %%
# #Plot
im = plt.imread('/home/015911532/creek_fire/topo.png')
for itime6 in range(len(time5)):
    u10 = nc_fid11.variables['U10'][itime6]
    u10_2 = nc_fid12.variables['U10'][itime6]
    v10 = nc_fid11.variables['V10'][itime6]
    v10_2 = nc_fid12.variables['V10'][itime6]
    wspd = np.sqrt(np.array(u10)**2 + np.array(v10)**2)
    wspd2 = np.sqrt(np.array(u10_2)**2 + np.array(v10_2)**2)
    wspd_tot = np.array(wspd) - np.array(wspd2)
    fire = nc_fid11.variables['AVG_FUEL_FRAC'][itime6]
    fig = plt.figure(1, figsize=(12, 8))
    ax = plt.subplot(111, projection=lamCon)
    plt.imshow(im, extent=[lons.min(),lons.max(),lats.min(),lats.max()])
    #plt.contour(lons, lats, wspd_tot, np.arange(-18,18,1), colors="black", alpha=0.7, linewidths=0.4, antialiased=True)
    contours = plt.pcolormesh(to_np(lons), to_np(lats), wspd_tot, vmin=-4.0, vmax=4.0, cmap=get_cmap("RdBu_r"), transform=lamCon, linestyles='solid', alpha=0.5)    
    cbar = plt.colorbar(ax=ax,shrink=0.95)
    cbar.set_label('Wind Speed (m/s)', rotation=270)
    plt.contour(lons, lats, fire, colors="black", linewidths=0.8, antialiased=True)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.add_feature(cfeature.STATES.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.set_xticks(np.linspace(lons.min(),lons.max(),8))
    ax.set_yticks(np.linspace(lats.min(),lats.max(),6))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.grid(True)
    plt.title("WRF Fire-Induced Wind Speed With Creek Fire (Fire - No Fire) "+str(time5[itime6])[:19])
    fig.savefig('kig_'+str(itime6)+'.png',bbox_inches='tight',dpi=150)
    plt.close(fig)
# %%
# #Plot
im = plt.imread('/home/015911532/creek_fire/topo.png')
for itime7 in range(len(time6)):
    u10 = nc_fid13.variables['U10'][itime7]
    u10_2 = nc_fid14.variables['U10'][itime7]
    v10 = nc_fid13.variables['V10'][itime7]
    v10_2 = nc_fid14.variables['V10'][itime7]
    wspd = np.sqrt(np.array(u10)**2 + np.array(v10)**2)
    wspd2 = np.sqrt(np.array(u10_2)**2 + np.array(v10_2)**2)
    wspd_tot = np.array(wspd) - np.array(wspd2)
    fire = nc_fid13.variables['AVG_FUEL_FRAC'][itime7]
    fig = plt.figure(1, figsize=(12, 8))
    ax = plt.subplot(111, projection=lamCon)
    plt.imshow(im, extent=[lons.min(),lons.max(),lats.min(),lats.max()])
    #plt.contour(lons, lats, wspd_tot, np.arange(-18,18,1), colors="black", alpha=0.7, linewidths=0.4, antialiased=True)
    contours = plt.pcolormesh(to_np(lons), to_np(lats), wspd_tot, vmin=-4.0, vmax=4.0, cmap=get_cmap("RdBu_r"), transform=lamCon, linestyles='solid', alpha=0.5)    
    cbar = plt.colorbar(ax=ax,shrink=0.95)
    cbar.set_label('Wind Speed (m/s)', rotation=270)
    plt.contour(lons, lats, fire, colors="black", linewidths=0.8, antialiased=True)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.add_feature(cfeature.STATES.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.set_xticks(np.linspace(lons.min(),lons.max(),8))
    ax.set_yticks(np.linspace(lats.min(),lats.max(),6))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.grid(True)
    plt.title("WRF Fire-Induced Wind Speed With Creek Fire (Fire - No Fire) "+str(time6[itime7])[:19])
    fig.savefig('lig_'+str(itime7)+'.png',bbox_inches='tight',dpi=150)
    plt.close(fig)
# %%
# #Plot
im = plt.imread('/home/015911532/creek_fire/topo.png')
for itime8 in range(len(time7)):
    u10 = nc_fid15.variables['U10'][itime8]
    u10_2 = nc_fid16.variables['U10'][itime8]
    v10 = nc_fid15.variables['V10'][itime8]
    v10_2 = nc_fid16.variables['V10'][itime8]
    wspd = np.sqrt(np.array(u10)**2 + np.array(v10)**2)
    wspd2 = np.sqrt(np.array(u10_2)**2 + np.array(v10_2)**2)
    wspd_tot = np.array(wspd) - np.array(wspd2)
    fire = nc_fid15.variables['AVG_FUEL_FRAC'][itime8]
    fig = plt.figure(1, figsize=(12, 8))
    ax = plt.subplot(111, projection=lamCon)
    plt.imshow(im, extent=[lons.min(),lons.max(),lats.min(),lats.max()])
    #plt.contour(lons, lats, wspd_tot, np.arange(-18,18,1), colors="black", alpha=0.7, linewidths=0.4, antialiased=True)
    contours = plt.pcolormesh(to_np(lons), to_np(lats), wspd_tot, vmin=-4.0, vmax=4.0, cmap=get_cmap("RdBu_r"), transform=lamCon, linestyles='solid', alpha=0.5)    
    cbar = plt.colorbar(ax=ax,shrink=0.95)
    cbar.set_label('Wind Speed (m/s)', rotation=270)
    plt.contour(lons, lats, fire, colors="black", linewidths=0.8, antialiased=True)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.add_feature(cfeature.STATES.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.set_xticks(np.linspace(lons.min(),lons.max(),8))
    ax.set_yticks(np.linspace(lats.min(),lats.max(),6))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.grid(True)
    plt.title("WRF Fire-Induced Wind Speed With Creek Fire (Fire - No Fire) "+str(time7[itime8])[:19])
    fig.savefig('mig_'+str(itime8)+'.png',bbox_inches='tight',dpi=150)
    plt.close(fig)
# %%
# #Plot
im = plt.imread('/home/015911532/creek_fire/topo.png')
for itime9 in range(len(time8)):
    u10 = nc_fid17.variables['U10'][itime9]
    u10_2 = nc_fid18.variables['U10'][itime9]
    v10 = nc_fid17.variables['V10'][itime9]
    v10_2 = nc_fid18.variables['V10'][itime9]
    wspd = np.sqrt(np.array(u10)**2 + np.array(v10)**2)
    wspd2 = np.sqrt(np.array(u10_2)**2 + np.array(v10_2)**2)
    wspd_tot = np.array(wspd) - np.array(wspd2)
    fire = nc_fid17.variables['AVG_FUEL_FRAC'][itime9]
    fig = plt.figure(1, figsize=(12, 8))
    ax = plt.subplot(111, projection=lamCon)
    plt.imshow(im, extent=[lons.min(),lons.max(),lats.min(),lats.max()])
    #plt.contour(lons, lats, wspd_tot, np.arange(-18,18,1), colors="black", alpha=0.7, linewidths=0.4, antialiased=True)
    contours = plt.pcolormesh(to_np(lons), to_np(lats), wspd_tot, vmin=-4.0, vmax=4.0, cmap=get_cmap("RdBu_r"), transform=lamCon, linestyles='solid', alpha=0.5)    
    cbar = plt.colorbar(ax=ax,shrink=0.95)
    cbar.set_label('Wind Speed (m/s)', rotation=270)
    plt.contour(lons, lats, fire, colors="black", linewidths=0.8, antialiased=True)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.add_feature(cfeature.STATES.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.set_xticks(np.linspace(lons.min(),lons.max(),8))
    ax.set_yticks(np.linspace(lats.min(),lats.max(),6))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.grid(True)
    plt.title("WRF Fire-Induced Wind Speed With Creek Fire (Fire - No Fire) "+str(time8[itime9])[:19])
    fig.savefig('nig_'+str(itime9)+'.png',bbox_inches='tight',dpi=150)
    plt.close(fig)
# %%
# #Plot
im = plt.imread('/home/015911532/creek_fire/topo.png')
for itime10 in range(len(time9)):
    u10 = nc_fid19.variables['U10'][itime10]
    u10_2 = nc_fid20.variables['U10'][itime10]
    v10 = nc_fid19.variables['V10'][itime10]
    v10_2 = nc_fid20.variables['V10'][itime10]
    wspd = np.sqrt(np.array(u10)**2 + np.array(v10)**2)
    wspd2 = np.sqrt(np.array(u10_2)**2 + np.array(v10_2)**2)
    wspd_tot = np.array(wspd) - np.array(wspd2)
    fire = nc_fid19.variables['AVG_FUEL_FRAC'][itime10]
    fig = plt.figure(1, figsize=(12, 8))
    ax = plt.subplot(111, projection=lamCon)
    plt.imshow(im, extent=[lons.min(),lons.max(),lats.min(),lats.max()])
    #plt.contour(lons, lats, wspd_tot, np.arange(-18,18,1), colors="black", alpha=0.7, linewidths=0.4, antialiased=True)
    contours = plt.pcolormesh(to_np(lons), to_np(lats), wspd_tot, vmin=-4.0, vmax=4.0, cmap=get_cmap("RdBu_r"), transform=lamCon, linestyles='solid', alpha=0.5)    
    cbar = plt.colorbar(ax=ax,shrink=0.95)
    cbar.set_label('Wind Speed (m/s)', rotation=270)
    plt.contour(lons, lats, fire, colors="black", linewidths=0.8, antialiased=True)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.add_feature(cfeature.STATES.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.set_xticks(np.linspace(lons.min(),lons.max(),8))
    ax.set_yticks(np.linspace(lats.min(),lats.max(),6))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.grid(True)
    plt.title("WRF Fire-Induced Wind Speed With Creek Fire (Fire - No Fire) "+str(time9[itime10])[:19])
    fig.savefig('oig_'+str(itime10)+'.png',bbox_inches='tight',dpi=150)
    plt.close(fig)
# %%
# #Plot
im = plt.imread('/home/015911532/creek_fire/topo.png')
for itime11 in range(len(time10)):
    u10 = nc_fid21.variables['U10'][itime11]
    u10_2 = nc_fid22.variables['U10'][itime11]
    v10 = nc_fid21.variables['V10'][itime11]
    v10_2 = nc_fid22.variables['V10'][itime11]
    wspd = np.sqrt(np.array(u10)**2 + np.array(v10)**2)
    wspd2 = np.sqrt(np.array(u10_2)**2 + np.array(v10_2)**2)
    wspd_tot = np.array(wspd) - np.array(wspd2)
    fire = nc_fid21.variables['AVG_FUEL_FRAC'][itime11]
    fig = plt.figure(1, figsize=(12, 8))
    ax = plt.subplot(111, projection=lamCon)
    plt.imshow(im, extent=[lons.min(),lons.max(),lats.min(),lats.max()])
    #plt.contour(lons, lats, wspd_tot, np.arange(-18,18,1), colors="black", alpha=0.7, linewidths=0.4, antialiased=True)
    contours = plt.pcolormesh(to_np(lons), to_np(lats), wspd_tot, vmin=-4.0, vmax=4.0, cmap=get_cmap("RdBu_r"), transform=lamCon, linestyles='solid', alpha=0.5)    
    cbar = plt.colorbar(ax=ax,shrink=0.95)
    cbar.set_label('Wind Speed (m/s)', rotation=270)
    plt.contour(lons, lats, fire, colors="black", linewidths=0.8, antialiased=True)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.add_feature(cfeature.STATES.with_scale('50m'),linewidth=1.5,edgecolor=['#616060'])
    ax.set_xticks(np.linspace(lons.min(),lons.max(),8))
    ax.set_yticks(np.linspace(lats.min(),lats.max(),6))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.grid(True)
    plt.title("WRF Fire-Induced Wind Speed With Creek Fire (Fire - No Fire) "+str(time10[itime11])[:19])
    fig.savefig('pig_'+str(itime11)+'.png',bbox_inches='tight',dpi=150)
    plt.close(fig)
# %%

