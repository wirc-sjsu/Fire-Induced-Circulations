# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
from netCDF4 import Dataset
import xarray as xr
import os
import wrf
import glob

from wrf import to_np, getvar, CoordPair, vertcross, ALL_TIMES

matplotlib.use('Agg')
# %%
fire_files = sorted(glob.glob('/adata/jhaw/thomas_fire_runs/d02/wrfout_d02_2017-12-0[4567]*'))
no_fire_files = sorted(glob.glob('/adata/jhaw/thomas_fire_runs/d02_nofire/wrfout_d02_2017-12-0[4567]*'))
fire_line_files = sorted(glob.glob('/adata/jhaw/thomas_fire_runs/d02/wrfout_d02_2017-12-0[4567]*'))

temp = []
temp2 = []

start_point = CoordPair(lat=34.23, lon=-119.17)
end_point = CoordPair(lat=34.51, lon=-119.35)
# %%
######################################
#Get data

#    print('here')
for idx in range(len(fire_files)):
    print("Processing index ", idx+1, "/", len(fire_files))
    ncfile=Dataset(fire_files[idx])
    ncfile_nf=Dataset(no_fire_files[idx])
    #nc_fid=xr.open_dataset(sdir+sfile)
    ncfile_t=Dataset(fire_files[idx])
    #nc_fid_t=xr.open_dataset(sdir2+sfile2)
    time = getvar(ncfile, "times", timeidx=ALL_TIMES, meta=False)
    #print(len(time))
# Create the start point and end point for the cross section

    for tidx in range(len(time)):
        #print(len(time))
        z = getvar(ncfile_t,'z')[:-13,:,:]
        #print(np.shape(z))
        u = ncfile.variables['U'][tidx]
        #print(np.shape(u))
        v = ncfile.variables['V'][tidx]
        w = ncfile.variables['W'][tidx]
        fire_ = ncfile.variables['GRNHFX'][tidx]
        u1 = ncfile_nf.variables['U'][tidx]
        #print(np.shape(u))
        v1 = ncfile_nf.variables['V'][tidx]
        w1 = ncfile_nf.variables['W'][tidx]
        #print(np.shape(v))
        u = u[:-13,:,:-1]
        v = v[:-13,:-1,:]
        w = w[:-14,:,:]
        u1 = u1[:-13,:,:-1]
        v1 = v1[:-13,:-1,:]
        w1 = w1[:-14,:,:]
        wspd = np.sqrt(np.array(u)**2 + np.array(v)**2 + np.array(w)**2)
        wspd1 = np.sqrt(np.array(u1)**2 + np.array(v1)**2 + np.array(w1)**2)
        wspd_tot = np.array(wspd) - np.array(wspd1)
        #print(np.shape(wspd))
        t = ncfile_t.variables['T'][tidx]
        temp.append(t)
        #print(np.shape(t))
        #print(np.shape(temp))
        temp_diff = np.array(temp) + 300
        #print(np.shape(temp_diff))
        # Compute the vertical cross-section interpolation.  Also, include the
        # lat/lon points along the cross-section.
        wspd_cross = vertcross(xr.DataArray(wspd_tot), z, wrfin=ncfile, start_point=start_point,
                            end_point=end_point, latlon=True, meta=True)
        temp_cross = vertcross(xr.DataArray(temp_diff)[-1,:-13,:,:], z, wrfin=ncfile_t, start_point=start_point,
                            end_point=end_point, latlon=True, meta=True)
        fire_line = wrf.interpline(xr.DataArray(fire_), wrfin = ncfile, 
                           start_point = start_point, end_point = end_point, latlon=True, meta=True)

        # Create the figure
        #fig = plt.figure(figsize=(12,6))
        #ax = plt.axes()
        fig, (ax, ax2) = plt.subplots(2, figsize = (25, 18), gridspec_kw={'height_ratios': [8, 1]})

        # Make the contour plot
        wspd_contours = ax.contourf(to_np(wspd_cross), np.arange(-30,30,0.5), cmap=get_cmap("jet"))
        temp_contours = ax.contour(to_np(temp_cross), np.arange(285,340,5), colors="white")
        ax2.plot(fire_line / 1000, color = 'blue', label = 'Ground Heat Flux (KW/m^2)')
        ax.clabel(temp_contours)

        # Add the color bar
        #plt.colorbar(wspd_contours, ax=ax)
        cbar_ax = fig.add_axes([0.82, 0.15, 0.04, 0.7])
        pg = plt.colorbar(wspd_contours, cax=cbar_ax)
        pg.set_label('Wind Speed (m/s)', fontsize = 18, fontweight = 'bold')
        pg.ax.tick_params(labelsize=15)
        plt.subplots_adjust(right=0.8)

        # Set the x-ticks to use latitude and longitude labels.
        coord_pairs = to_np(wspd_cross.coords["xy_loc"])
        x_ticks = np.arange(coord_pairs.shape[0])
        #_labels = [pair.latlon_str(fmt="{:.2f}, {:.2f}")
        #            for pair in to_np(coord_pairs)]
        x_labels = ['34.23, -119.17', '34.31, -119.22', '34.39, -119.27', '34.47, -119.33']
        ax2.set_xticks(x_ticks[::20])
        ax2.set_xticklabels(x_labels, rotation=45, fontsize=15)
        ax2.set_xlabel('Latitude, Longitude', fontsize = 18, fontweight = 'bold')
        ax2.set_ylabel('KW/m^2', fontsize = 18, fontweight = 'bold')
        ax2.grid()
        ax2.legend(prop={'size':20})
        ax2.set_ylim(0, 25)
        #ax2.set_xlim(0, x_ticks[-1] * 10)
        ax2.set_title('Ground Heat Flux (KW/m^2)', fontsize = 20, fontweight = 'bold')

        # Set the y-ticks to be height.
        vert_vals = to_np(wspd_cross.coords["vertical"])
        v_ticks = np.arange(vert_vals.shape[0])
        ax.set_yticks(v_ticks[::20])
        ax.set_yticklabels(vert_vals[::20], fontsize=15)
        ax.set_xticks(x_ticks[::20])
        #ax.set_xlim(0, x_ticks[-1])
        ax.set_xticklabels([])

        # Set the x-axis and  y-axis labels
        #ax.set_xlabel("Latitude, Longitude", fontsize=12)
        ax.set_ylabel("Height (m)", fontsize=18, fontweight='bold')

        ax.set_title("WRF Fire-Induced Wind Speed With Thomas Fire "+str(time[tidx])[:19], fontsize = 20, fontweight = 'bold')

        fig.savefig('fig_'+str(idx)+"_"+str(tidx)+'_vert'+'.png',bbox_inches='tight',dpi=150)
        plt.close(fig)
# %%