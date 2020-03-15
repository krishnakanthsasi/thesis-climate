# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 18:28:53 2019

@author: krish
"""

# Soil Moisture analysis
# Volumetric water layer  1

# All necessary modules


#Imported Libraries

#%matplotlib qt
import yaml
yaml.warnings({'YAMLLoadWarning': False})
import pickle
# the default loader is deprecated, I dont know how to change the default loader
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
import pandas as pd
from matplotlib import colors
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import matplotlib as mlp
from scipy import stats
from matplotlib.colors import DivergingNorm

# READING NETCDF FILES
# the whole dataset contains weekly information of years from 1966 til 2018
def read_data(): 
    """
    Reads the nc file 
    and returns the 2 dataset for Soilmoisture volumetric water layer 1 analysis.
    1 - Soil moisture value for each point in Northern Hemisphere from 25N to 90N
    2 - zonal soil moisture value in Northern Hemisphere 25N to 90N
    """
    print("NETCDF files being read")
    years = np.arange(1979, 2019)
    file_path = r"C:\Users\krish\Documents\Data_era5\input_raw"
    
    
    # 2m temperature

    soil_moisture_1_datasets = []
    for year in years:
        whole_file_path = file_path +r"\wl1\wl1_%d.nc"%(year)
        dataset = xr.open_dataset(whole_file_path, chunks = {'time':10})
        dataset = dataset.resample(time='D').mean(dim='time')
        dataset = dataset.resample(time='M').mean(dim='time')
        soil_moisture_1_datasets.append(dataset)
    print("Raw data being processed for analysis")
    
    dataset_wl1 = xr.concat(soil_moisture_1_datasets, 'time', data_vars = 'minimal')
    
    
    # Looking at only 25-90N northern hemisphere.
    
    dataset_wl1_map = dataset_wl1.sel(latitude = slice(90,25))
    
    # Looking at zonal soil moisture - 1 values.
    dataset_wl1_zonal = dataset_wl1.sel(latitude = slice(90,25)).mean(dim = 'longitude')    
    
    return dataset_wl1_map, dataset_wl1_zonal


# MAKING CLIMATOLOGY PLOTS
    
    
def climatology(analysis, forcmp=False, alsozonal=False):
    """
    Mean and standard deviation plots are made for soil moisture - 1 for the months of June, July
    and August between years 1979-2018
    """
    
    print("Climatology analysis running")
    #MEANS
    if analysis == "mean":
        June_sw1 = np.mean(dataset_wl1_map.swvl1[0::3,:,:].values,axis=0)
        July_sw1 = np.mean(dataset_wl1_map.swvl1[1::3,:,:].values,axis=0)
        August_sw1 = np.mean(dataset_wl1_map.swvl1[2::3,:,:].values,axis=0)
        
        #Plotting
        if forcmp == True:
            return July_sw1, August_sw1
        
        w_max, w_min, delta = 1, 0, 0.1
        colorbar_title = "in m^3 m^-3"  # unit described on era5 m^3m^-3
        june_title = "Northern Hemisphere Soil Water Volumetric Level 1 - Climatology map for June (1979-2018)"
        july_title = "Northern Hemisphere Soil Water Volumetric Level 1 - Climatology map for July (1979-2018)"
        august_title = "Northern Hemisphere Soil Water Volumetric Level 1 - Climatology map for August (1979-2018)"
        makePlots([June_sw1, July_sw1, August_sw1],[june_title,july_title,august_title],w_min,w_max,delta, colorbar_title,res=2, color="Blues")    
        if alsozonal == True:
            #Plots zonal maps
            zonalmaps([July_sw1, August_sw1], title = 'Mean Northern Hemisphere Soil Water Volumetric Level 1 (1979-2018) for July and August')
            return
    #STANDARD DEVIATION
    elif analysis == "std":
        June_sw1 = np.std(dataset_wl1_map.swvl1[0::3,:,:].values,axis=0)
        July_sw1 = np.std(dataset_wl1_map.swvl1[1::3,:,:].values,axis=0)
        August_sw1 = np.std(dataset_wl1_map.swvl1[2::3,:,:].values,axis=0)

# =============================================================================
        # Helps with the range of the colorbar
#         print(np.amax(June_sw1), np.amin(June_sw1))
#         print(np.amax(July_sw1), np.amin(July_sw1))
#         print(np.amax(August_sw1), np.amin(August_sw1))
# =============================================================================
        
        #Plotting
        w_max, w_min, delta = 0.15, 0, 0.025
        colorbar_title = "in m^3 m^-3"  # unit described on era5 m^3m^-3
        june_title = "Northern Hemisphere Soil Water Volumetric Level 1 - Standard deviation Map for June (1979-2018)"
        july_title = "Northern Hemisphere Soil Water Volumetric Level 1 - Standard deviation Map for July (1979-2018)"
        august_title = "Northern Hemisphere Soil Water Volumetric Level 1 - Standard deviation Map for August (1979-2018)"
        makePlots([June_sw1, July_sw1, August_sw1],[june_title,july_title,august_title],w_min,w_max,delta, colorbar_title,res=2, color="Blues")    
        
    return


# Making zonal maps

def zonalmaps(months, title, subtitles=['July', 'August'], allin=False):
    """
    Input:months is a list containing all the datasets in lat/lon format and value(SM) color coded
    Output:Again a list of months containing datasets in zonal mean/lat format
    """
    months_zonal_means = []
    if allin == False:
        
        for month in months:
            # take zonal mean for mean month
            #print(month.shape)
            newmonth = np.flip(np.mean(month,axis=1))
            print(newmonth.shape)
            months_zonal_means.append(newmonth)
        xaxis = np.linspace(25.5,90,87)
        
        #plot months_zonal_means
        sns.set()
        plt.figure()
        plt.suptitle("Zonal Maps for %s" % title)
        i = 0
        
        for month in months_zonal_means:
            i += 1
            plt.subplot(2,1,i)
            plt.title(subtitles[i-1])
            plt.plot(xaxis, month)
            plt.ylabel("Soil Water Volumetric Level 1(in m^3 m^-3)")#, fontsize=16)
        plt.xlabel("Latitude("+u"\N{DEGREE SIGN}"+"N)")#, fontsize=16)
        plt.show()
    
    else:
        
        # (dj-j, sj-j, nj-j, dj-a, sj-a, nj-a)
        for month in months:
            # take zonal mean for mean month
            #print(month.shape)
            newmonth = np.flip(np.mean(month,axis=1))
            print(newmonth.shape)
            months_zonal_means.append(newmonth)
        xaxis = np.linspace(25.5,90,87)
        #plot months_zonal_means
        sns.set()
        plt.figure()
        plt.title("Zonal Maps for %s" % title)
        i = 0
        #plt.title(subtitles[i-1])
          
        for month in months_zonal_means:
            i += 1
            r = (i-1)%3
            d = int((i-1)/3)
            if d==0:
                if r == 0:
                    print("here")
                    plt.plot(xaxis, month, 'g-', label=subtitles[i-1])
                elif r == 1:
                    plt.plot(xaxis, month, 'g--', label=subtitles[i-1])
                else:
                    plt.plot(xaxis, month, 'g*', label=subtitles[i-1])
            else:
                if r == 0:
                    plt.plot(xaxis, month, 'b-', label=subtitles[i-1])
                elif r == 1:
                    plt.plot(xaxis, month, 'b--', label=subtitles[i-1])
                else:
                    plt.plot(xaxis, month, 'b*', label=subtitles[i-1])
        
            plt.ylabel("Soil Water Volumetric Level 1(in m^3 m^-3)")#, fontsize=16)
            
        plt.xlabel("Latitude("+u"\N{DEGREE SIGN}"+"N)")#, fontsize=16)
        
        plt.legend(fontsize=9)
        plt.show()

# Plotting Function
def makePlots(datasets, titles, vmin, vmax, delta, colorbar_title, split = 3, sup_title = [],res=2, color='seismic'):
    """
    datas should be of length 3, arranged in the order of dataset for June, July and August
    Useful for generating climatology, trend maps, anomaly maps and composites 
    """
    plt.figure()
    if len(sup_title)!=0:
        plt.suptitle(sup_title, fontsize=18)
        y=1.08
    else:
        y=1
    subplots = len(datasets)
    
    ticks = np.arange(vmin, vmax+delta, delta)
    colors = plt.cm.get_cmap(color, len(ticks)-1)
    parallels = np.arange(25,90,10.)
    meridians = np.arange(0.,360.,30.)
    for i in np.arange(subplots):
        plt.subplot(split,int(subplots/split),i+1)
        plt.title(titles[i], y=y, fontsize=17)
        
        m = Basemap(projection='cyl',llcrnrlat=25,urcrnrlat=90,\
                llcrnrlon=0,urcrnrlon=360,resolution='c')
        m.imshow(datasets[i], norm=DivergingNorm(0.05), origin ='upper', vmin = vmin, vmax = vmax, cmap=colors)
        
        m.drawcoastlines()
#        t_s = ticks[0::res]
#        if t_s[-1]==ticks[-1]:
#            cbar = m.colorbar(ticks=t_s)
#        else:
#            cbar = m.colorbar(ticks=ticks[1:-1:res])
#        cbar.ax.set_title(colorbar_title, fontsize=14)
        
        m.drawparallels(parallels, labels = [1,0,0,0])
        m.drawmeridians(meridians, labels = [0,0,0,1])
        m.drawmapboundary(fill_color='white')
        
    plt.subplots_adjust(left = 0.1, bottom=0.1, right=0.8, top=0.9, hspace = 0.3)
    cax = plt.axes([0.83, 0.1, 0.01, 0.8])
    t_s = ticks[0::res]
    if t_s[-1]==ticks[-1]:
        cbar = plt.colorbar(cax=cax, orientation = "vertical", ticks=t_s)
    else:
        cbar = plt.colorbar(cax=cax, orientation = "vertical", ticks=ticks[1:-1:res])
    cbar.ax.set_ylabel(colorbar_title, fontsize=14)
    
    plt.show()
    return

# MAKING TREND MAPS PLOTS
        
def trendmaps():
    """
    Plots the trend of soil moisture - 1 in months of June, July and August (1979-2018) 
    """
    
    print("Plotting Trend maps")

    years = np.arange(0, 40) 
    slopes_june_sw1 = np.zeros([87, 480])
    intercepts_june_sw1 = np.zeros([87, 480])
    slopes_july_sw1 = np.zeros([87, 480])
    intercepts_july_sw1 = np.zeros([87, 480])
    slopes_august_sw1 = np.zeros([87, 480])
    intercepts_august_sw1 = np.zeros([87, 480])
    
    for row in np.arange(87):
        june_values = []
        july_values = []
        august_values = []
        w_array_june = dataset_wl1_map.swvl1[0:120:3,row,:].values
        w_array_july = dataset_wl1_map.swvl1[1:120:3,row,:].values
        w_array_august = dataset_wl1_map.swvl1[2:120:3,row,:].values
        print(row)
        for col in np.arange(480):
            
            
            june_values.append(w_array_june[:,col])
            july_values.append(w_array_july[:,col])
            august_values.append(w_array_august[:,col])
            
            m, b = np.polyfit(years, june_values[col], 1)
            slopes_june_sw1[row, col] = m
            intercepts_june_sw1[row, col] = b
            
            m, b = np.polyfit(years, july_values[col], 1)
            slopes_july_sw1[row, col] = m
            intercepts_july_sw1[row, col] = b
            
            m, b = np.polyfit(years, august_values[col], 1)
            slopes_august_sw1[row, col] = m
            intercepts_august_sw1[row, col] = b
        
# =============================================================================
#     # Determining the range of colorbar        
#     print(np.amax(slopes_june_sw1),np.amin(slopes_june_sw1))
#     print(np.amax(slopes_july_sw1),np.amin(slopes_july_sw1))
#     print(np.amax(slopes_august_sw1),np.amin(slopes_august_sw1))
#     print(w_array_june.shape, len(june_values[2]))
# =============================================================================
    
    #Plotting
    w_max, w_min,delta = 0.0105, -0.0105, 0.001
    colorbar_title = "change per annum(in m^3 m^-3)"
    june_title = "Northern Hemisphere Soil Water Volumetric Level 1 - Trend map for June (1979-2018)"
    july_title = "Northern Hemisphere Soil Water Volumetric Level 1 - Trend map for July (1979-2018)"
    august_title = "Northern Hemisphere Soil Water Volumetric Level 1 - Trend map for August (1979-2018)"
    makePlots([slopes_june_sw1, slopes_july_sw1, slopes_august_sw1],[june_title,july_title,august_title],w_min,w_max,delta,colorbar_title,res=3,color="seismic")  


# MAKING ANOMALY PLOTS
def anomalymaps():
    """
    Divides the dataset into two subsets, and finds the difference in mean soil moisture - 1 
    between the datasets.
    Is used to cross check with respect to the trendmap, if the observed variation is close 
    to estimated trend.
    """
    print("Plotting Anomaly maps")
    June_sw1_b = np.mean(dataset_wl1_map.swvl1[0:20*3:3,:,:].values, axis=0)
    July_sw1_b = np.mean(dataset_wl1_map.swvl1[1:1+20*3:3,:,:].values, axis=0)
    August_sw1_b = np.mean(dataset_wl1_map.swvl1[2:2+20*3:3,:,:].values, axis=0)
    June_sw1_l = np.mean(dataset_wl1_map.swvl1[20*3::3,:,:].values, axis=0)
    July_sw1_l = np.mean(dataset_wl1_map.swvl1[1+20*3::3,:,:].values, axis=0)
    August_sw1_l = np.mean(dataset_wl1_map.swvl1[2+20*3::3,:,:].values, axis=0)
    
    June_anomaly = June_sw1_l-June_sw1_b
    July_anomaly = July_sw1_l-July_sw1_b
    August_anomaly = August_sw1_l-August_sw1_b
    
# =============================================================================
#     # Helps determine the range
#     print(np.amax(June_anomaly),np.amin(June_anomaly))
#     print(np.amax(July_anomaly),np.amin(July_anomaly))
#     print(np.amax(August_anomaly),np.amin(August_anomaly))
# =============================================================================

    #Plotting
    w_max, w_min, delta = 0.25, -0.25,0.02
    colorbar_title = "in m^3 m^-3"
    plot_caption = "Difference between the mean soil water volumetric level 1 of 1979-1998 and 1999-2018"
    june_title = "Northern Hemisphere Soil Water Volumetric Level 1 - Anomaly map for June (1979-2018)"
    july_title = "Northern Hemisphere Soil Water Volumetric Level 1 - Anomaly map for July (1979-2018)"
    august_title = "Northern Hemisphere Soil Water Volumetric Level 1 - Anomaly map for August (1979-2018)"
    makePlots([June_anomaly, July_anomaly, August_anomaly],[june_title,july_title,august_title],w_min,w_max,delta,colorbar_title,res=5)
    return
    
    
    
# MAKING COMPOSITE PLOTS
def compositemaps(analysis):
    """
    Mean and standard deviation plots are made for soil moisture - 1 for the months of July
    and August for following subset of years
    Years with single-jet domination in July
    Years with double-jet domination in July 
    Years with no domination in July
    
    Years with single-jet domination in August
    Years with double-jet domination in August 
    Years with no domination in August
    
    are being run as seperate subfunctions
    """
    
    print("Composite maps is being plot")
    
    # And now July and August composites for soil moisture - 1
    # Next step is to use seggregated years (c1 dominated, c3 dominated, neither) to find average plots for each variable 
    # within these particular years
    # And now its monthly data
    # Taking into consideration the zonal means
    
# =============================================================================
#     Not relevant anymore - lines
#     June_jc1t, July_jc1t, August_jc1t = np.zeros((87,480)), np.zeros((87,480)), np.zeros((87,480))
#     June_jc3t, July_jc3t, August_jc3t = np.zeros((87,480)), np.zeros((87,480)), np.zeros((87,480))
#     June_jnt, July_jnt, August_jnt = np.zeros((87,480)), np.zeros((87,480)), np.zeros((87,480))
#     
#     June_ac1t, July_ac1t, August_ac1t = np.zeros((87,480)), np.zeros((87,480)), np.zeros((87,480))
#     June_ac3t, July_ac3t, August_ac3t = np.zeros((87,480)), np.zeros((87,480)), np.zeros((87,480))
#     June_ant, July_ant, August_ant = np.zeros((87,480)), np.zeros((87,480)), np.zeros((87,480))
#     
#     
#     # Timeseries variables for comparison
#     July_jc1t_timeseries, August_jc1t_timeseries = [],[]
#     July_jc3t_timeseries, August_jc3t_timeseries = [],[]
#     July_jnt_timeseries, August_jnt_timeseries = [],[]
#     
#     July_ac1t_timeseries, August_ac1t_timeseries = [],[]
#     July_ac3t_timeseries, August_ac3t_timeseries = [],[]
#     July_ant_timeseries, August_ant_timeseries = [],[]
#     
# =============================================================================
    
    if analysis == 'mean':    
        # july_c1_dominated    
        year = np.multiply(np.subtract(july_c1_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_jc1sw1 = np.mean(dataset_wl1_map.swvl1[indx_july,:,:].values, axis = 0) 
        August_jc1sw1 = np.mean(dataset_wl1_map.swvl1[indx_august,:,:].values, axis = 0)
                
        # july_c3_dominated:
        year = np.multiply(np.subtract(july_c3_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_jc3sw1 = np.mean(dataset_wl1_map.swvl1[indx_july,:,:].values, axis = 0) 
        August_jc3sw1 = np.mean(dataset_wl1_map.swvl1[indx_august,:,:].values, axis = 0)
        
        # july_neither_dominated:
        year = np.multiply(np.subtract(july_neither_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_jnsw1 = np.mean(dataset_wl1_map.swvl1[indx_july,:,:].values, axis = 0) 
        August_jnsw1 = np.mean(dataset_wl1_map.swvl1[indx_august,:,:].values, axis = 0)
        
        # august_c1_dominated:
        year = np.multiply(np.subtract(august_c1_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_ac1sw1 = np.mean(dataset_wl1_map.swvl1[indx_july,:,:].values, axis = 0) 
        August_ac1sw1 = np.mean(dataset_wl1_map.swvl1[indx_august,:,:].values, axis = 0)
        
        # august_c3_dominated:
        year = np.multiply(np.subtract(august_c3_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_ac3sw1 = np.mean(dataset_wl1_map.swvl1[indx_july,:,:].values, axis = 0) 
        August_ac3sw1 = np.mean(dataset_wl1_map.swvl1[indx_august,:,:].values, axis = 0)
        
        # august_neither_dominated:
        year = np.multiply(np.subtract(august_neither_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_answ1 = np.mean(dataset_wl1_map.swvl1[indx_july,:,:].values, axis = 0) 
        August_answ1 = np.mean(dataset_wl1_map.swvl1[indx_august,:,:].values, axis = 0)    
        
    # =============================================================================
    #     #Helps determine the range of colorbar
    #     
    #     print("July-C1")
    #     print(np.amax(July_jc1sw1-July_sw1),np.amin(July_jc1sw1-July_sw1))
    #     print(np.amax(August_jc1sw1-August_sw1),np.amin(August_jc1sw1-August_sw1))
    #     print("July-C3")
    #     print(np.amax(July_jc3sw1-July_sw1),np.amin(July_jc3sw1-July_sw1))
    #     print(np.amax(August_jc3sw1-August_sw1),np.amin(August_jc3sw1-August_sw1))
    #     print("July-Neither")
    #     print(np.amax(July_jnsw1-July_sw1),np.amin(July_jnsw1-July_sw1))
    #     print(np.amax(August_jnsw1-August_sw1),np.amin(August_jnsw1-August_sw1))
    #     
    #     print("August-C1")
    #     print(np.amax(July_ac1sw1-July_sw1),np.amin(July_ac1sw1-July_sw1))
    #     print(np.amax(August_ac1sw1-August_sw1),np.amin(August_ac1sw1-August_sw1))
    #     print("August-C3")
    #     print(np.amax(July_ac3sw1-July_sw1),np.amin(July_ac3sw1-July_sw1))
    #     print(np.amax(August_ac3sw1-August_sw1),np.amin(August_ac3sw1-August_sw1))
    #     print("August-Neither")
    #     print(np.amax(July_answ1-July_sw1),np.amin(July_answ1-July_sw1))
    #     print(np.amax(August_answ1-August_sw1),np.amin(August_answ1-August_sw1))
    #     
    # =============================================================================
        # Plotting parameters
        July_sw1, August_sw1 = climatology('mean', forcmp=True)
        w_max, w_min, delta = 0.085, -0.085, 0.01
        colorbar_title = "in m^3 m^-3"
        
        # more Plotting parameters
        jul_aug_title = "Mean Soil Water Volumetric Layer 1 during specific years relative to Mean Soil Water Volumetric Layer 1(1979-2018) for July and August"
        july_jc1 = "July of years with domination of double jet"
        august_jc1 = "August of years with domination of double jet in July"
        
        july_jc3 = "July of years with domination of single jet"
        august_jc3 = "August of years with domination of single jet in July"
        
        july_jn = "July of years with domination of neither jet"
        august_jn = "August of years with domination of neither jet in July"
        
        july_ac1 = "July of years with domination of double jet in August"
        august_ac1 = "August of years with domination of double jet"
        
        july_ac3 = "July of years with domination of single jet in August"
        august_ac3 = "August of years with domination of single jet"
        
        july_an = "July of years with domination of neither jet in August"
        august_an = "August of years with domination of neither jet"
    
        #Plotting
        #C1-Dominant
        #makePlots([July_jc1sw1-July_sw1, August_jc1sw1-August_sw1],[july_jc1,august_jc1],w_min,w_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,res=1)
        #makePlots([July_ac1sw1-July_sw1, August_ac1sw1-August_sw1],[july_ac1,august_ac1],w_min,w_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,res=1)
        
        #zonal maps 
        zonalmaps([July_jc1sw1-July_sw1, July_jc3sw1-July_sw1, July_jnsw1-July_sw1, July_ac1sw1-July_sw1, July_ac3sw1-July_sw1, July_answ1-July_sw1],
                  title="Mean Soil Water Volumetric Layer 1 during specific years relative to Mean Soil Water Volumetric Layer 1(1979-2018) for July",\
                  subtitles=[july_jc1,july_jc3,july_jn,july_ac1,july_ac3,july_an], allin=True)
        zonalmaps([August_jc1sw1-August_sw1, August_jc3sw1-August_sw1, August_jnsw1-August_sw1, August_ac1sw1-August_sw1, August_ac3sw1-August_sw1, August_answ1-August_sw1],
                  title="Mean Soil Water Volumetric Layer 1 during specific years relative to Mean Soil Water Volumetric Layer 1(1979-2018) for August",\
                  subtitles=[august_jc1,august_jc3,august_jn,august_ac1,august_ac3,august_an], allin=True)
        #C3-Dominant
        #makePlots([July_jc3sw1-July_sw1, August_jc3sw1-August_sw1],[july_jc3,august_jc3],w_min,w_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,res=1)
        #makePlots([July_ac3sw1-July_sw1, August_ac3sw1-August_sw1],[july_ac3,august_ac3],w_min,w_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,res=1)
        
        #zonal maps 
        #zonalmaps([July_jc3sw1, July_sw1, August_jc3sw1, August_sw1], title="Mean Soil Water Volumetric Layer 1 during specific years relative to Mean Soil Water Volumetric Layer 1(1979-2018) for July and August",fill=True)
        #zonalmaps([July_ac3sw1, July_sw1, August_ac3sw1, August_sw1], title="Mean Soil Water Volumetric Layer 1 during specific years relative to Mean Soil Water Volumetric Layer 1(1979-2018) for July and August",fill=True)
        
        #Neither-Dominant
        #makePlots([July_jnsw1-July_sw1, August_jnsw1-August_sw1],[july_jn,august_jn],w_min,w_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,res=1)
        #makePlots([July_answ1-July_sw1, August_answ1-August_sw1],[july_an,august_an],w_min,w_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,res=1)
    
        #zonal maps 
        #zonalmaps([July_jnsw1, July_sw1, August_jnsw1, August_sw1], title="Mean Soil Water Volumetric Layer 1 during specific years relative to Mean Soil Water Volumetric Layer 1(1979-2018) for July and August",fill=True)
        #zonalmaps([July_answ1, July_sw1, August_answ1, August_sw1], title="Mean Soil Water Volumetric Layer 1 during specific years relative to Mean Soil Water Volumetric Layer 1(1979-2018) for July and August",fill=True)
        
    elif analysis == 'std':
        # july_c1_dominated    
        year = np.multiply(np.subtract(july_c1_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_jc1sw1 = np.std(dataset_wl1_map.swvl1[indx_july,:,:].values, axis = 0) 
        August_jc1sw1 = np.std(dataset_wl1_map.swvl1[indx_august,:,:].values, axis = 0)
                
        # july_c3_dominated:
        year = np.multiply(np.subtract(july_c3_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_jc3sw1 = np.std(dataset_wl1_map.swvl1[indx_july,:,:].values, axis = 0) 
        August_jc3sw1 = np.std(dataset_wl1_map.swvl1[indx_august,:,:].values, axis = 0)
        
        # july_neither_dominated:
        year = np.multiply(np.subtract(july_neither_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_jnsw1 = np.std(dataset_wl1_map.swvl1[indx_july,:,:].values, axis = 0) 
        August_jnsw1 = np.std(dataset_wl1_map.swvl1[indx_august,:,:].values, axis = 0)
        
        # august_c1_dominated:
        year = np.multiply(np.subtract(august_c1_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_ac1sw1 = np.std(dataset_wl1_map.swvl1[indx_july,:,:].values, axis = 0) 
        August_ac1sw1 = np.std(dataset_wl1_map.swvl1[indx_august,:,:].values, axis = 0)
        
        # august_c3_dominated:
        year = np.multiply(np.subtract(august_c3_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_ac3sw1 = np.std(dataset_wl1_map.swvl1[indx_july,:,:].values, axis = 0) 
        August_ac3sw1 = np.std(dataset_wl1_map.swvl1[indx_august,:,:].values, axis = 0)
        
        # august_neither_dominated:
        year = np.multiply(np.subtract(august_neither_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_answ1 = np.std(dataset_wl1_map.swvl1[indx_july,:,:].values, axis = 0) 
        August_answ1 = np.std(dataset_wl1_map.swvl1[indx_august,:,:].values, axis = 0)    
        
    # =============================================================================
    #     #Helps determine the range of colorbar
    #     
#        print("July-C1")
#        print(np.amax(July_jc1sw1),np.amin(July_jc1sw1))
#        print(np.amax(August_jc1sw1),np.amin(August_jc1sw1))
#        print("July-C3")
#        print(np.amax(July_jc3sw1),np.amin(July_jc3sw1))
#        print(np.amax(August_jc3sw1),np.amin(August_jc3sw1))
#        print("July-Neither")
#        print(np.amax(July_jnsw1),np.amin(July_jnsw1))
#        print(np.amax(August_jnsw1),np.amin(August_jnsw1))
#         
#        print("August-C1")
#        print(np.amax(July_ac1sw1),np.amin(July_ac1sw1))
#        print(np.amax(August_ac1sw1),np.amin(August_ac1sw1))
#        print("August-C3")
#        print(np.amax(July_ac3sw1),np.amin(July_ac3sw1))
#        print(np.amax(August_ac3sw1),np.amin(August_ac3sw1))
#        print("August-Neither")
#        print(np.amax(July_answ1),np.amin(July_answ1))
#        print(np.amax(August_answ1),np.amin(August_answ1))
         
    # =============================================================================
#        # Plotting parameters
        w_max, w_min, delta = 0.16, 0, 0.02
        colorbar_title = "in m^3 m^-3"
        
        # more Plotting parameters
        jul_aug_title = "Standard deviation of Soil Water Volumetric Layer 1 during specific years for July and August"
        july_jc1 = "July of years with domination of double jet"
        august_jc1 = "August of years with domination of double jet in July"
        
        july_jc3 = "July of years with domination of single jet"
        august_jc3 = "August of years with domination of single jet in July"
        
        july_jn = "July of years with domination of neither jet"
        august_jn = "August of years with domination of neither jet in July"
        
        july_ac1 = "July of years with domination of double jet in August"
        august_ac1 = "August of years with domination of double jet"
        
        july_ac3 = "July of years with domination of single jet in August"
        august_ac3 = "August of years with domination of single jet"
        
        july_an = "July of years with domination of neither jet in August"
        august_an = "August of years with domination of neither jet"
    
        #Plotting
        #C1-Dominant
        makePlots([July_jc1sw1, August_jc1sw1],[july_jc1,august_jc1],w_min,w_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,res=1,color="Blues")
        makePlots([July_ac1sw1, August_ac1sw1],[july_ac1,august_ac1],w_min,w_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,res=1,color="Blues")
        
        #C3-Dominant
        makePlots([July_jc3sw1, August_jc3sw1],[july_jc3,august_jc3],w_min,w_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,res=1,color="Blues")
        makePlots([July_ac3sw1, August_ac3sw1],[july_ac3,august_ac3],w_min,w_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,res=1,color="Blues")
        
        #Neither-Dominant
        makePlots([July_jnsw1, August_jnsw1],[july_jn,august_jn],w_min,w_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,res=1,color="Blues")
        makePlots([July_answ1, August_answ1],[july_an,august_an],w_min,w_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,res=1,color="Blues")

        
    return 


def p_value_maps():
    """
    This function in intended to produce a p value map comparing July of C1 dominated July
    """
    # creating the comparison datasets
    
        # july_c1_dominated:
    year = np.multiply(np.subtract(july_c1_dominated,1979),3)
    indx_july = np.add(1, year)
    indx_august = np.add(2, year)
    July_jc1sw1 = abs(np.round(dataset_wl1_map.swvl1[indx_july,:,:].values,4)) 
    August_jc1sw1 = abs(np.round(dataset_wl1_map.swvl1[indx_august,:,:].values,4))
    
    
                
        # july_c3_dominated:
    year = np.multiply(np.subtract(july_c3_dominated,1979),3)
    indx_july = np.add(1, year)
    indx_august = np.add(2, year)
    July_jc3sw1 = abs(np.round(dataset_wl1_map.swvl1[indx_july,:,:].values,4)) 
    August_jc3sw1 = abs(np.round(dataset_wl1_map.swvl1[indx_august,:,:].values,4))
    
    
        # july_neither_dominated:
    year = np.multiply(np.subtract(july_neither_dominated,1979),3)
    indx_july = np.add(1, year)
    indx_august = np.add(2, year)
    July_jnsw1 = abs(np.round(dataset_wl1_map.swvl1[indx_july,:,:].values,4)) 
    August_jnsw1 = abs(np.round(dataset_wl1_map.swvl1[indx_august,:,:].values,4))
     
        # august_c1_dominated:
    year = np.multiply(np.subtract(august_c1_dominated,1979),3)
    indx_july = np.add(1, year)
    indx_august = np.add(2, year)
    July_ac1sw1 = abs(np.round(dataset_wl1_map.swvl1[indx_july,:,:].values,4)) 
    August_ac1sw1 = abs(np.round(dataset_wl1_map.swvl1[indx_august,:,:].values,4))
                
        # august_c3_dominated:
    year = np.multiply(np.subtract(august_c3_dominated,1979),3)
    indx_july = np.add(1, year)
    indx_august = np.add(2, year)
    July_ac3sw1 = abs(np.round(dataset_wl1_map.swvl1[indx_july,:,:].values,4)) 
    August_ac3sw1 = abs(np.round(dataset_wl1_map.swvl1[indx_august,:,:].values,4))
        
        # august_neither_dominated:
    year = np.multiply(np.subtract(august_neither_dominated,1979),3)
    indx_july = np.add(1, year)
    indx_august = np.add(2, year)
    July_answ1 = abs(np.round(dataset_wl1_map.swvl1[indx_july,:,:].values,4)) 
    August_answ1 = abs(np.round(dataset_wl1_map.swvl1[indx_august,:,:].values,4))
    
    # DJ-J vs SJ-J
    
    July_jc1c3 = run_studenttest3(July_jc1sw1, July_jc3sw1)
    August_jc1c3 = run_studenttest3(August_jc1sw1, August_jc3sw1)
    
    # SJ-J vs NJ-J
    
    July_jc3cn = run_studenttest3(July_jc3sw1, July_jnsw1)
    August_jc3cn = run_studenttest3(August_jc3sw1, August_jnsw1)
    
    # NJ-J vs DJ-J    
    
    July_jcnc1 = run_studenttest3(July_jnsw1, July_jc1sw1)
    August_jcnc1 = run_studenttest3(August_jnsw1, August_jc1sw1)
    
    
    # DJ-A vs SJ-A
    
    July_ac1c3 = run_studenttest3(July_ac1sw1, July_ac3sw1)
    August_ac1c3 = run_studenttest3(August_ac1sw1, August_ac3sw1)
    
    # SJ-A vs NJ-A
    
    July_ac3cn = run_studenttest3(July_ac3sw1, July_answ1)
    August_ac3cn = run_studenttest3(August_ac3sw1, August_answ1)
    
    # NJ-A vs DJ-A    
    
    July_acnc1 = run_studenttest3(July_answ1, July_ac1sw1)
    August_acnc1 = run_studenttest3(August_answ1, August_ac1sw1)
    
    #return [July_jc1c3, July_jc1sw1, July_jc3sw1],[August_jc1c3, August_jc1sw1, August_jc3sw1]
    # Plotting parameters
    p_max, p_min, delta = 1, 0, 0.005
    colorbar_title = " "
    
    # more Plotting parameters
    jul_aug_title_DJSJ_J = "P value map from Student t-test (Years with domination of double jet in July vs years with domination of single jet in July)"
    jul_aug_title_SJNJ_J = "P value map from Student t-test (Years with domination of single jet in July vs years with domination of neither jet in July)"
    jul_aug_title_NJDJ_J = "P value map from Student t-test (Years with domination of neither jet in July vs years with domination of double jet in July)"
    jul_aug_title_DJSJ_A = "P value map from Student t-test (Years with domination of double jet in August vs years with domination of single jet in August)"
    jul_aug_title_SJNJ_A = "P value map from Student t-test (Years with domination of single jet in August vs years with domination of neither jet in August)"
    jul_aug_title_NJDJ_A = "P value map from Student t-test (Years with domination of neither jet in August vs years with domination of double jet in August)"
    
    subtitles = ["July", "August"]
    
    #Plotting
    #Comparison of years in July-Domination
    makePlots([July_jc1c3, August_jc1c3],subtitles,p_min,p_max,delta,colorbar_title,split=2,res=8,sup_title=jul_aug_title_DJSJ_J)#,color='Blues')
    makePlots([July_jc3cn, August_jc3cn],subtitles,p_min,p_max,delta,colorbar_title,split=2,res=8,sup_title=jul_aug_title_SJNJ_J)
    makePlots([July_jcnc1, August_jcnc1],subtitles,p_min,p_max,delta,colorbar_title,split=2,res=8,sup_title=jul_aug_title_NJDJ_J)
    
    #Comparison of years in August-Domination
    makePlots([July_ac1c3, August_ac1c3],subtitles,p_min,p_max,delta,colorbar_title,split=2,res=8,sup_title=jul_aug_title_DJSJ_A)
    makePlots([July_ac3cn, August_ac3cn],subtitles,p_min,p_max,delta,colorbar_title,split=2,res=8,sup_title=jul_aug_title_SJNJ_A)
    makePlots([July_acnc1, August_acnc1],subtitles,p_min,p_max,delta,colorbar_title,split=2,res=8,sup_title=jul_aug_title_NJDJ_A)
        
    
def run_studenttest1(dataset_1, dataset_2):
    if len(dataset_1) > len(dataset_2):
        indice = random.randint(0,14)
        m_dataset_1 = np.delete(dataset_1,indice,0)
        t, p = stats.ttest_rel(m_dataset_1, dataset_2, axis=0)
    elif len(dataset_1) < len(dataset_2):
        indice = random.randint(0,14)
        m_dataset_2 = np.delete(dataset_2,indice,0)
        t, p = stats.ttest_rel(dataset_1, m_dataset_2, axis=0)
    else:
        t, p = stats.ttest_rel(dataset_1, dataset_2, axis=0)
    return p

def run_studenttest2(dataset_1, dataset_2):
    t, p = stats.ttest_ind(dataset_1, dataset_2, axis = 0, equal_var = True)
    return p

def run_studenttest3(dataset_1, dataset_2):
    mean1 = np.mean(dataset_1, axis=0)
    mean2 = np.mean(dataset_2, axis=0)
    std1 = np.std(dataset_1, axis=0)
    std2 = np.std(dataset_1, axis=0)
    nobs1 = len(dataset_1)
    nobs2 = len(dataset_2)
    t, p = stats.ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2, equal_var=False)
    return p



    
if __name__ == "__main__":
    print("Soil Moisture Analysis")
    dataset_wl1_map, dataset_wl1_zonal = read_data()
    
    #climatology("mean", alsozonal=True)
    #climatology("std")
    #anomalymaps()
    #trendmaps()
    with open('cluster_years.pickle', 'rb') as f:
        july_c3_dominated, july_c1_dominated, july_neither_dominated, august_c3_dominated, august_c1_dominated, august_neither_dominated = pickle.load(f)
    #compositemaps('mean')
    #compositemaps('std')
    p_value_maps()
    
#    Julys, Augusts, all_months = dataset4test()
#    
#    results = np.round(run_ttest(all_months), 3)
#    print("T value   P Value")
#    print(results[0])
#    print(results[1])
#    print(results[2])
#    print(results[3])
#    print(results[4])
#    print(results[5])
# =============================================================================
#     print(Julys[0].shape, Augusts[0].shape, all_months[0].shape)
#     
#     plt.figure()
#     plt.title("SM Julys of DJ-July dom")
#     sns.distplot(Julys[0], hist=False)
#     plt.figure()
#     plt.title("SM Augusts of SJ-July dom")
#     sns.distplot(Julys[1], hist=False)
#     plt.figure()
#     plt.title("SM Julys and Augusts of DJ-July dom")
#     sns.distplot(all_months[0])
#     plt.show()
# =============================================================================
    
    #print(np.isnan(Mays).any())
    #print(np.isnan(Junes).any())
    #print(np.isnan(Julys).any())
    #print(np.isnan(Augusts).any())
#    variables = [Julys, Augusts]
#    months_list = ["July", "August"]
#    results = run_ttest(variables, months_list)
#    print(results)
#    
#    indx = 0
#    ttest_results = []
#    #anova_results = []
#    for var in variables:
#        print("Statistical Test for %s" %(months_list[indx]))
#        res = run_ttest(var)
#        ttest_results.append(res)
##        res = run_anova(var)
##        anova_results.append(res)
#        indx += 1