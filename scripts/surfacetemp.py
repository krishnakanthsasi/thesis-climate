# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:50:00 2019

@author: krish
"""

# Surface temperature analysis

# All necessary modules


#Imported Libraries

#%matplotlib qt
import yaml
yaml.warnings({'YAMLLoadWarning': False})
# the default loader is deprecated, I dont know how to change the default loader
import pickle
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
    and returns the 2 dataset for surface temperature analysis.
    1 - surface temperature value for each point in Northern Hemisphere from 25N to 90N
    2 - zonal surface temperature value in Northern Hemisphere 25N to 90N
    """
    print("NETCDF files being read")
    years = np.arange(1979, 2019)
    file_path = r"C:\Users\krish\Documents\Data_era5\input_raw"
    
    
    temperature_datasets = []
    for year in years:
        whole_file_path = file_path +r"\2t\2t_%d.nc"%(year)
        dataset = xr.open_dataset(whole_file_path, chunks = {'time':10})
        dataset = dataset.resample(time='D').mean(dim='time')
        dataset = dataset.resample(time='M').mean(dim='time')
        temperature_datasets.append(dataset)
    
    print("Raw data being processed for analysis")
    dataset_2t = xr.concat(temperature_datasets, 'time', data_vars = 'minimal')
    # Looking at only 25-90N northern hemisphere.
    dataset_2t_map = dataset_2t.sel(latitude = slice(90,25))
    
    # Looking at zonal surface temperature values.
    dataset_2t_zonal = dataset_2t_map.mean(dim = 'longitude')
    
    return dataset_2t_map, dataset_2t_zonal


# MAKING CLIMATOLOGY PLOTS
    
    
def climatology(analysis, forcmp=False , alsozonal=False):
    """
    Mean and standard deviation plots are made for surface temperature for the months of June, July
    and August between years 1979-2018
    """
    
    print("Climatology analysis running")
    #MEANS
    if analysis == "mean":
        June_t = np.mean(dataset_2t_map.t2m[0::3,:,:].values,axis=0)
        July_t = np.mean(dataset_2t_map.t2m[1::3,:,:].values,axis=0)
        August_t = np.mean(dataset_2t_map.t2m[2::3,:,:].values,axis=0)
        
        if forcmp == True:
            return July_t, August_t
        
        #Plotting
        t_max, t_min, delta = 315, 255, 2.5
        colorbar_title = "in K"
        june_title = "Northern Hemisphere Surface Temperature - Climatology Map for June (1979-2018)"
        july_title = "Northern Hemisphere Surface Temperature - Climatology Map for July (1979-2018)"
        august_title = "Northern Hemisphere Surface Temperature - Climatology Map for August (1979-2018)"
        makePlots([June_t, July_t, August_t],[june_title,july_title,august_title],t_min,t_max,delta,colorbar_title)
        if alsozonal == True:
            #Plots zonal maps
            zonalmaps([July_t, August_t], title = 'Mean Northern Hemisphere Surface Temperature(1979-2018) for July and August')
            return
            
    #STANDARD DEVIATION
    elif analysis == "std":
        June_t = np.std(dataset_2t_map.t2m[0::3,:,:].values,axis=0)
        July_t = np.std(dataset_2t_map.t2m[1::3,:,:].values,axis=0)
        August_t = np.std(dataset_2t_map.t2m[2::3,:,:].values,axis=0)

        print(np.amax(June_t), np.amin(June_t))
        print(np.amax(July_t), np.amin(July_t))
        print(np.amax(August_t), np.amin(August_t))
        
        #Plotting
        t_max, t_min, delta = 4, 0, 0.05
        colorbar_title = "in K"
        june_title = "Northern Hemisphere Surface Temperature - Standard deviation Map for June (1979-2018)"
        july_title = "Northern Hemisphere Surface Temperature - Standard deviation Map for July (1979-2018)"
        august_title = "Northern Hemisphere Surface Temperature - Standard deviation Map for August (1979-2018)"
        makePlots([June_t, July_t, August_t],[june_title,july_title,august_title],t_min,t_max,delta,colorbar_title)


# Making zonal maps

def zonalmaps(months, title, subtitles=['July', 'August'], fill=False, allin=False):
    """
    Input:months is a list containing all the datasets in lat/lon format and value(ST ) color coded
    Output:Again a list of months containing datasets in zonal mean/lat format
    """
    months_zonal_means = []
    
    if fill==False and allin==False:
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
            plt.ylabel("Temperature 2m from Surface(in K)")#, fontsize=16)
        plt.xlabel("Latitude("+u"\N{DEGREE SIGN}"+"N)")#, fontsize=16)
        
        plt.show()
    
    elif fill==True and allin==False:
        #plot months_zonal_means
        months_zonal_means_2 = []
        i=0
        for month in months:
        # take zonal mean for mean month
        #print(month.shape)
            newmonth = np.flip(np.mean(month,axis=1))
            print(newmonth.shape)
            if i%2==0:        
                months_zonal_means.append(newmonth) #composite
            else:
                months_zonal_means_2.append(newmonth) #climatology
            i+=1
            
        xaxis = np.linspace(25.5,90,87)
        sns.set()
        plt.figure()
        plt.suptitle("Zonal Maps for %s" % title)
        i = 0
        
        for month in months_zonal_means:
            i += 1
            plt.subplot(2,1,i)
            plt.title(subtitles[i-1])
            plt.plot(xaxis, month, '--r', label='Year with specific jet state')
            plt.plot(xaxis, months_zonal_means_2[i-1], '-b', label='Climatology')
            plt.fill_between(xaxis, month, months_zonal_means_2[i-1], where=months_zonal_means_2[i-1] >= month, facecolor='lightskyblue', interpolate=True)
            plt.fill_between(xaxis, month, months_zonal_means_2[i-1], where=months_zonal_means_2[i-1] <= month, facecolor='indianred', interpolate=True)
            plt.ylabel("Temperature 2m from Surface(in K)")#, fontsize=16)
            plt.legend()
        plt.xlabel("Latitude("+u"\N{DEGREE SIGN}"+"N)")#, fontsize=16)
        plt.show()
    
    elif fill==False and allin==True:
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
        
            plt.ylabel("Temperature 2m from Surface(in K)")#, fontsize=16)
            
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
    #[left, bottom, width, height]    
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
    Plots the trend of surface temperature in months of June, July and August (1979-2018) 
    """
    
    print("Plotting Trend maps")

    years = np.arange(0, 40) 
    slopes_june_t = np.zeros([87, 480])
    intercepts_june_t = np.zeros([87, 480])
    slopes_july_t = np.zeros([87, 480])
    intercepts_july_t = np.zeros([87, 480])
    slopes_august_t = np.zeros([87, 480])
    intercepts_august_t = np.zeros([87, 480])
    
    for row in np.arange(87):
        june_values = []
        july_values = []
        august_values = []
        w_array_june = dataset_2t_map.t2m[0:120:3,row,:].values
        w_array_july = dataset_2t_map.t2m[1:120:3,row,:].values
        w_array_august = dataset_2t_map.t2m[2:120:3,row,:].values
        print(row)
        for col in np.arange(480):
            
            
            june_values.append(w_array_june[:,col])
            july_values.append(w_array_july[:,col])
            august_values.append(w_array_august[:,col])
            
            m, b = np.polyfit(years, june_values[col], 1)
            slopes_june_t[row, col] = m
            intercepts_june_t[row, col] = b
            
            m, b = np.polyfit(years, july_values[col], 1)
            slopes_july_t[row, col] = m
            intercepts_july_t[row, col] = b
            
            m, b = np.polyfit(years, august_values[col], 1)
            slopes_august_t[row, col] = m
            intercepts_august_t[row, col] = b
        
# =============================================================================
# Tells us the range for the colorbar
#     print(np.amax(slopes_june_t*40),np.amin(slopes_june_t*40))
#     print(np.amax(slopes_july_t*40),np.amin(slopes_july_t*40))
#     print(np.amax(slopes_august_t*40),np.amin(slopes_august_t*40))
#     print(w_array_june.shape, len(june_values[2]))
# =============================================================================
    
    #Plotting
    t_max, t_min,delta = 0.2, -0.2, 0.0125
    colorbar_title = "change per annum(in K)"
    june_title = "Northern Hemisphere Surface Temperature - Trend map for June (1979-2018)"
    july_title = "Northern Hemisphere Surface Temperature - Trend map for July (1979-2018)"
    august_title = "Northern Hemisphere Surface Temperature - Trend map for August (1979-2018)"
    makePlots([slopes_june_t, slopes_july_t, slopes_august_t],[june_title,july_title,august_title],t_min,t_max,delta,colorbar_title,res=4)
        


# MAKING ANOMALY PLOTS
def anomalymaps():
    """
    Divides the dataset into two subsets, and finds the difference in mean surface temperature 
    between the datasets.
    Is used to cross check with respect to the trendmap, if the observed variation is close 
    to estimated trend.
    """
    print("Plotting Anomaly maps")
    June_t_b = np.mean(dataset_2t_map.t2m[0:20*3:3,:,:].values, axis=0)
    July_t_b = np.mean(dataset_2t_map.t2m[1:1+20*3:3,:,:].values, axis=0)
    August_t_b = np.mean(dataset_2t_map.t2m[2:2+20*3:3,:,:].values, axis=0)
    June_t_l = np.mean(dataset_2t_map.t2m[20*3::3,:,:].values, axis=0)
    July_t_l = np.mean(dataset_2t_map.t2m[1+20*3::3,:,:].values, axis=0)
    August_t_l = np.mean(dataset_2t_map.t2m[2+20*3::3,:,:].values, axis=0)
    
    June_anomaly = June_t_l-June_t_b
    July_anomaly = July_t_l-July_t_b
    August_anomaly = August_t_l-August_t_b

    #Plotting
    t_max, t_min,delta = 4.5, -4.5, 0.25
    colorbar_title = "in K"
    june_title = "Northern Hemisphere Surface Temperature - Anomaly map for June (1979-2018)"
    july_title = "Northern Hemisphere Surface Temperature - Anomaly map for July (1979-2018)"
    august_title = "Northern Hemisphere Surface Temperature - Anomaly map for August (1979-2018)"
    makePlots([June_anomaly, July_anomaly, August_anomaly],[june_title,july_title,august_title],t_min,t_max,delta,colorbar_title,res=4)
    return
    
    
    
# MAKING COMPOSITE PLOTS
def compositemaps(analysis):
    """
    Mean and standard deviation plots are made for surface temperature for the months of July
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
    
    # And now July and August composites for Surface Temperature
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
        July_jc1t = np.mean(dataset_2t_map.t2m[indx_july,:,:].values, axis = 0) 
        August_jc1t = np.mean(dataset_2t_map.t2m[indx_august,:,:].values, axis = 0)
                
        # july_c3_dominated:
        year = np.multiply(np.subtract(july_c3_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_jc3t = np.mean(dataset_2t_map.t2m[indx_july,:,:].values, axis = 0) 
        August_jc3t = np.mean(dataset_2t_map.t2m[indx_august,:,:].values, axis = 0)
        
        # july_neither_dominated:
        year = np.multiply(np.subtract(july_neither_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_jnt = np.mean(dataset_2t_map.t2m[indx_july,:,:].values, axis = 0) 
        August_jnt = np.mean(dataset_2t_map.t2m[indx_august,:,:].values, axis = 0)
        
        # august_c1_dominated:
        year = np.multiply(np.subtract(august_c1_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_ac1t = np.mean(dataset_2t_map.t2m[indx_july,:,:].values, axis = 0) 
        August_ac1t = np.mean(dataset_2t_map.t2m[indx_august,:,:].values, axis = 0)
        
        # august_c3_dominated:
        year = np.multiply(np.subtract(august_c3_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_ac3t = np.mean(dataset_2t_map.t2m[indx_july,:,:].values, axis = 0) 
        August_ac3t = np.mean(dataset_2t_map.t2m[indx_august,:,:].values, axis = 0)
        
        # august_neither_dominated:
        year = np.multiply(np.subtract(august_neither_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_ant = np.mean(dataset_2t_map.t2m[indx_july,:,:].values, axis = 0) 
        August_ant = np.mean(dataset_2t_map.t2m[indx_august,:,:].values, axis = 0)    
        
        # =============================================================================
        #     Used for getting the range of colorbars    
        #     print("July-C1")
        #     print(np.amax(July_jc1t-July_t),np.amin(July_jc1t-July_t))
        #     print(np.amax(August_jc1t-August_t),np.amin(August_jc1t-August_t))
        #     print("July-C3")
        #     print(np.amax(July_jc3t-July_t),np.amin(July_jc3t-July_t))
        #     print(np.amax(August_jc3t-August_t),np.amin(August_jc3t-August_t))
        #     print("July-Neither")
        #     print(np.amax(July_jnt-July_t),np.amin(July_jnt-July_t))
        #     print(np.amax(August_jnt-August_t),np.amin(August_jnt-August_t))
        #     
        #     print("August-C1")
        #     print(np.amax(July_ac1t-July_t),np.amin(July_ac1t-July_t))
        #     print(np.amax(August_ac1t-August_t),np.amin(August_ac1t-August_t))
        #     print("August-C3")
        #     print(np.amax(July_ac3t-July_t),np.amin(July_ac3t-July_t))
        #     print(np.amax(August_ac3t-August_t),np.amin(August_ac3t-August_t))
        #     print("August-Neither")
        #     print(np.amax(July_ant-July_t),np.amin(July_ant-July_t))
        #     print(np.amax(August_ant-August_t),np.amin(August_ant-August_t))
        # 
        # =============================================================================
        
        
        # Plotting parameters
        July_t, August_t = climatology('mean', forcmp=True)
        t_max, t_min, delta = 2.5, -2.5, 0.25
        colorbar_title = "in K"
        
        # more Plotting parameters
        jul_aug_title = "Mean Surface Temperature during specific years relative to Mean Surface Temperature(1979-2018) for July and August"
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
#        makePlots([July_jc1t-July_t, August_jc1t-August_t],[july_jc1,august_jc1],t_min,t_max,delta,colorbar_title,split=2,sup_title=jul_aug_title)
#        makePlots([July_ac1t-July_t, August_ac1t-August_t],[july_ac1,august_ac1],t_min,t_max,delta,colorbar_title,split=2,sup_title=jul_aug_title)
        
        #zonal maps 
#        zonalmaps([July_jc1t, July_t, August_jc1t, August_t], title="Mean Surface Temperature during specific years relative to Mean Surface Temperature(1979-2018) for July and August",fill=True)
#        zonalmaps([July_ac1t, July_t, August_ac1t, August_t], title="Mean Surface Temperature during specific years relative to Mean Surface Temperature(1979-2018) for July and August",fill=True)
        #all Julys
        zonalmaps([July_jc1t-July_t, July_jc3t-July_t, July_jnt-July_t, July_ac1t-July_t, July_ac3t-July_t, July_ant-July_t], 
                  title="Mean Surface Temperature during specific years relative to Mean Surface Temperature(1979-2018) for July", \
                  subtitles=[july_jc1,july_jc3,july_jn,july_ac1,july_ac3,july_an],fill=False, allin=True)
        
        #all Augusts
        zonalmaps([August_jc1t-August_t, August_jc3t-August_t, August_jnt-August_t, August_ac1t-August_t, August_ac3t-August_t, August_ant-August_t], 
                  title="Mean Surface Temperature during specific years relative to Mean Surface Temperature(1979-2018) for August", \
                  subtitles=[august_jc1,august_jc3,august_jn,august_ac1,august_ac3,august_an],fill=False, allin=True)
        
        #C3-Dominant
#        makePlots([July_jc3t-July_t, August_jc3t-August_t],[july_jc3,august_jc3],t_min,t_max,delta,colorbar_title,split=2,sup_title=jul_aug_title)
#        makePlots([July_ac3t-July_t, August_ac3t-August_t],[july_ac3,august_ac3],t_min,t_max,delta,colorbar_title,split=2,sup_title=jul_aug_title)
        
        #zonal maps 
#        zonalmaps([July_jc3t, July_t, August_jc3t, August_t], title="Mean Surface Temperature during specific years relative to Mean Surface Temperature(1979-2018) for July and August",fill=True)
#        zonalmaps([July_ac3t, July_t, August_ac3t, August_t], title="Mean Surface Temperature during specific years relative to Mean Surface Temperature(1979-2018) for July and August",fill=True)
        
        #Neither-Dominant
#        makePlots([July_jnt-July_t, August_jnt-August_t],[july_jn,august_jn],t_min,t_max,delta,colorbar_title,split=2,sup_title=jul_aug_title)
#        makePlots([July_ant-July_t, August_ant-August_t],[july_an,august_an],t_min,t_max,delta,colorbar_title,split=2,sup_title=jul_aug_title)
        
        #zonal maps 
#        zonalmaps([July_jnt, July_t, August_jnt, August_t], title="Mean Surface Temperature during specific years relative to Mean Surface Temperature(1979-2018) for July and August",fill=True)
#        zonalmaps([July_ant, July_t, August_ant, August_t], title="Mean Surface Temperature during specific years relative to Mean Surface Temperature(1979-2018) for July and August",fill=True)
        
       
#        all in one
        
#        zonalmaps([July_jc1t-July_t, August_jc1t-August_t, July_ac1t-July_t, August_ac1t-August_t, July_jc3t-July_t, August_jc3t-August_t, July_ac3t-July_t, August_ac3t-August_t, July_jnt-July_t, August_jnt-August_t, July_ant-July_t, August_ant-August_t], 
#                  title="Mean Surface Temperature during specific years relative to Mean Surface Temperature(1979-2018) for July and August", \
#                  subtitles=[july_jc1,august_jc1,july_ac1,august_ac1,july_jc3,august_jc3,july_ac3,august_ac3,july_jn,august_jn,july_an,august_an],fill=False, allin=True)
        
        #flipped
#        zonalmaps([July_jc1t-July_t, August_jc1t-August_t, July_ac1t-July_t, August_ac1t-August_t, July_jc3t-July_t, August_jc3t-August_t, July_ac3t-July_t, August_ac3t-August_t, July_jnt-July_t, August_jnt-August_t, July_ant-July_t, August_ant-August_t], 
#                  title="Mean Surface Temperature during specific years relative to Mean Surface Temperature(1979-2018) for July and August", \
#                  subtitles=[july_jc1,august_jc1,july_ac1,august_ac1,july_jc3,august_jc3,july_ac3,august_ac3,july_jn,august_jn,july_an,august_an],fill=False, allin=True, flip=True)
    elif analysis == 'std':
        # july_c1_dominated    
        year = np.multiply(np.subtract(july_c1_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_jc1t = np.std(dataset_2t_map.t2m[indx_july,:,:].values, axis = 0) 
        August_jc1t = np.std(dataset_2t_map.t2m[indx_august,:,:].values, axis = 0)
                
        # july_c3_dominated:
        year = np.multiply(np.subtract(july_c3_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_jc3t = np.std(dataset_2t_map.t2m[indx_july,:,:].values, axis = 0) 
        August_jc3t = np.std(dataset_2t_map.t2m[indx_august,:,:].values, axis = 0)
        
        # july_neither_dominated:
        year = np.multiply(np.subtract(july_neither_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_jnt = np.std(dataset_2t_map.t2m[indx_july,:,:].values, axis = 0) 
        August_jnt = np.std(dataset_2t_map.t2m[indx_august,:,:].values, axis = 0)
        
        # august_c1_dominated:
        year = np.multiply(np.subtract(august_c1_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_ac1t = np.std(dataset_2t_map.t2m[indx_july,:,:].values, axis = 0) 
        August_ac1t = np.std(dataset_2t_map.t2m[indx_august,:,:].values, axis = 0)
        
        # august_c3_dominated:
        year = np.multiply(np.subtract(august_c3_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_ac3t = np.std(dataset_2t_map.t2m[indx_july,:,:].values, axis = 0) 
        August_ac3t = np.std(dataset_2t_map.t2m[indx_august,:,:].values, axis = 0)
        
        # august_neither_dominated:
        year = np.multiply(np.subtract(august_neither_dominated,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July_ant = np.std(dataset_2t_map.t2m[indx_july,:,:].values, axis = 0) 
        August_ant = np.std(dataset_2t_map.t2m[indx_august,:,:].values, axis = 0)    
        
        # =============================================================================
        #     Used for getting the range of colorbars    
#        print("July-C1")
#        print(np.amax(July_jc1t),np.amin(July_jc1t))
#        print(np.amax(August_jc1t),np.amin(August_jc1t))
#        print("July-C3")
#        print(np.amax(July_jc3t),np.amin(July_jc3t))
#        print(np.amax(August_jc3t),np.amin(August_jc3t))
#        print("July-Neither")
#        print(np.amax(July_jnt),np.amin(July_jnt))
#        print(np.amax(August_jnt),np.amin(August_jnt))
#         
#        print("August-C1")
#        print(np.amax(July_ac1t),np.amin(July_ac1t))
#        print(np.amax(August_ac1t),np.amin(August_ac1t))
#        print("August-C3")
#        print(np.amax(July_ac3t),np.amin(July_ac3t))
#        print(np.amax(August_ac3t),np.amin(August_ac3t))
#        print("August-Neither")
#        print(np.amax(July_ant),np.amin(July_ant))
#        print(np.amax(August_ant),np.amin(August_ant))
#     
        # =============================================================================
        
        
        # Plotting parameters
        
        t_max, t_min, delta = 4.5, 0, 0.25
        colorbar_title = "in K"
        
        # more Plotting parameters
        jul_aug_title = "Standard deviation of Surface Temperature during specific years for July and August"
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
        makePlots([July_jc1t, August_jc1t],[july_jc1,august_jc1],t_min,t_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,color="Blues")
        makePlots([July_ac1t, August_ac1t],[july_ac1,august_ac1],t_min,t_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,color="Blues")
        
        #C3-Dominant
        makePlots([July_jc3t, August_jc3t],[july_jc3,august_jc3],t_min,t_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,color="Blues")
        makePlots([July_ac3t, August_ac3t],[july_ac3,august_ac3],t_min,t_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,color="Blues")
        
        #Neither-Dominant
        makePlots([July_jnt, August_jnt],[july_jn,august_jn],t_min,t_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,color="Blues")
        makePlots([July_ant, August_ant],[july_an,august_an],t_min,t_max,delta,colorbar_title,split=2,sup_title=jul_aug_title,color="Blues")

    return 



# RUNNING STATISTICAL TEST
    
def dataset4test():
    """
    Creates datasets of variables with each index representing the value from a different group
    Index order:
    [july_c1_dominated, july_c3_dominated, july_neither_dominated, 
    august_c1_dominated, august_c3_dominated, august_neither_dominated]
    
    """
    Julys, Augusts = [], []
    all_months_grps = []
    groups = [july_c1_dominated, july_c3_dominated, july_neither_dominated, august_c1_dominated, august_c3_dominated, august_neither_dominated]
    for group in groups:
        all_months = []
        year = np.multiply(np.subtract(group,1979),3)
        indx_july = np.add(1, year)
        indx_august = np.add(2, year)
        July = dataset_2t_map.t2m[indx_july,:,:].values 
        July = np.ndarray.flatten(July)
        all_months.append(July)
        August = dataset_2t_map.t2m[indx_august,:,:].values
        August = np.ndarray.flatten(August)
        all_months.append(August)
        all_months = np.ndarray.flatten(np.asarray(all_months))
        Julys.append(July), Augusts.append(August)
        all_months_grps.append(all_months)
    return np.asarray(Julys), np.asarray(Augusts), np.asarray(all_months_grps)


def p_value_maps():
    """
    This function in intended to produce a p value map comparing July of C1 dominated July
    """
    # creating the comparison datasets
    
        # july_c1_dominated:
    year = np.multiply(np.subtract(july_c1_dominated,1979),3)
    indx_july = np.add(1, year)
    indx_august = np.add(2, year)
    July_jc1t = dataset_2t_map.t2m[indx_july,:,:].values 
    August_jc1t = dataset_2t_map.t2m[indx_august,:,:].values
                
        # july_c3_dominated:
    year = np.multiply(np.subtract(july_c3_dominated,1979),3)
    indx_july = np.add(1, year)
    indx_august = np.add(2, year)
    July_jc3t = dataset_2t_map.t2m[indx_july,:,:].values 
    August_jc3t = dataset_2t_map.t2m[indx_august,:,:].values
    
    
        # july_neither_dominated:
    year = np.multiply(np.subtract(july_neither_dominated,1979),3)
    indx_july = np.add(1, year)
    indx_august = np.add(2, year)
    July_jnt = dataset_2t_map.t2m[indx_july,:,:].values 
    August_jnt = dataset_2t_map.t2m[indx_august,:,:].values
     
        # august_c1_dominated:
    year = np.multiply(np.subtract(august_c1_dominated,1979),3)
    indx_july = np.add(1, year)
    indx_august = np.add(2, year)
    July_ac1t = dataset_2t_map.t2m[indx_july,:,:].values 
    August_ac1t = dataset_2t_map.t2m[indx_august,:,:].values
                
        # august_c3_dominated:
    year = np.multiply(np.subtract(august_c3_dominated,1979),3)
    indx_july = np.add(1, year)
    indx_august = np.add(2, year)
    July_ac3t = dataset_2t_map.t2m[indx_july,:,:].values 
    August_ac3t = dataset_2t_map.t2m[indx_august,:,:].values
        
        # august_neither_dominated:
    year = np.multiply(np.subtract(august_neither_dominated,1979),3)
    indx_july = np.add(1, year)
    indx_august = np.add(2, year)
    July_ant = dataset_2t_map.t2m[indx_july,:,:].values 
    August_ant = dataset_2t_map.t2m[indx_august,:,:].values
    
    # DJ-J vs SJ-J
    
    July_jc1c3 = run_studenttest3(July_jc1t, July_jc3t)
    August_jc1c3 = run_studenttest3(August_jc1t, August_jc3t)
    
    # SJ-J vs NJ-J
    
    July_jc3cn = run_studenttest3(July_jc3t, July_jnt)
    August_jc3cn = run_studenttest3(August_jc3t, August_jnt)
    
    # NJ-J vs DJ-J    
    
    July_jcnc1 = run_studenttest3(July_jnt, July_jc1t)
    August_jcnc1 = run_studenttest3(August_jnt, August_jc1t)
    
    
    # DJ-A vs SJ-A
    
    July_ac1c3 = run_studenttest3(July_ac1t, July_ac3t)
    August_ac1c3 = run_studenttest3(August_ac1t, August_ac3t)
    
    # SJ-A vs NJ-A
    
    July_ac3cn = run_studenttest3(July_ac3t, July_ant)
    August_ac3cn = run_studenttest3(August_ac3t, August_ant)
    
    # NJ-A vs DJ-A    
    
    July_acnc1 = run_studenttest3(July_ant, July_ac1t)
    August_acnc1 = run_studenttest3(August_ant, August_ac1t)
    
    
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
    makePlots([July_jc1c3, August_jc1c3],subtitles,p_min,p_max,delta,colorbar_title,split=2,res=8,sup_title=jul_aug_title_DJSJ_J)#,color="Blues")
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
    t, p = stats.ttest_ind(dataset_1, dataset_2, axis = 0, equal_var = False)
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
    
    print("Surface Temperature Analysis")
    dataset_2t_map, dataset_2t_zonal = read_data()
    
    with open('cluster_years.pickle', 'rb') as f:
        july_c3_dominated, july_c1_dominated, july_neither_dominated, august_c3_dominated, august_c1_dominated, august_neither_dominated = pickle.load(f)
    
    #climatology("mean",alsozonal=True)
    #climatology("std")
    #trendmaps()
    #anomalymaps()
    #compositemaps('mean')
    #compositemaps('std')
    p_value_maps()
    #print(sample_dataset.shape)
    #print(p1.shape)
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
## =============================================================================
#     print(Julys.shape, Augusts.shape, all_months.shape)
#     print(Julys[0].shape, Augusts[0].shape, all_months[0].shape)
#     plt.figure()
#     plt.title("ST Julys of DJ-July dom")
#     sns.distplot(Julys[0])
#     plt.figure()
#     plt.title("ST Augusts of DJ-July dom")
#     sns.distplot(Augusts[0])
#     plt.figure()
#     plt.title("ST Julys and Augusts of DJ-July dom")
#     sns.distplot(all_months[0])
#     plt.show()
# =============================================================================
    #print(np.isnan(Mays).any())
    #print(np.isnan(Junes).any())
    #print(np.isnan(Julys).any())
    #print(np.isnan(Augusts).any())
#    variables = [Julys, Augusts]
#    months_list = ["July", "August"]
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