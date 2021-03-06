# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:23:41 2019

@author: krish
"""

# All necessary modules


import yaml
yaml.warnings({'YAMLLoadWarning': False})
import pickle
# the default loader is deprecated, I dont know how to change the default loader
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random
import math
from scipy import stats, signal
import seaborn as sns 
import pandas as pd
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import matplotlib as mlp
from matplotlib.colors import DivergingNorm

# SNOW COVER RELATED

# READING NETCDF FILES
# the whole dataset contains weekly information of years from 1966 til 2018
def read_data(): 
    """
    Reads the nc file and returns the relevant dataset for snowcover analysis.
    """
    print("NETCDF files being read")
    whole_dataset = r"C:\Users\krish\Desktop\Courses\Thesis\data\downloadtrials\snow\complete.nc"
    dataset = xr.open_dataset(whole_dataset, chunks = {'time':10}) #, 'row':10, 'col':10})
    
    # Narrowing the dataset to 1979-2018 spring-summer values(March-August)
    
    print("Raw data being processed for analysis")
    springs = []
    for year in np.arange(1979,2019):
        beginning = "%d-03-01" %(year)
        end = "%d-08-30" %(year)
        dataset_spring = dataset.sel(time=slice(beginning,end))
        
        # Dataset snowcover values resampled from Weekly data to Monthly data here
        springs.append(dataset_spring.resample(time='M').mean())
    


    all_springs = xr.concat(springs, 'time', data_vars = 'minimal')
    
    
    return all_springs

# UNDERLYING MAPS

def draw_map():
    
    m = Basemap(projection='laea', resolution=None, lat_0=90,lon_0=0,width = 25*1000*720,height =25*1000*720, boundinglat=0, round=False)
    #m.drawlsmask(land_color='gray',ocean_color='white', lakes=True)
    return m



# MAKING CLIMATOLOGY PLOTS
    
    
def climatology(analysis,forcmp=False):
    print("Climatology analysis running")
    #MEANS
    if analysis == "mean":
        March = np.mean(all_springs.snow_cover[0::6,:,:].values,axis=0)
        April = np.mean(all_springs.snow_cover[1::6,:,:].values,axis=0)
        May = np.mean(all_springs.snow_cover[2::6,:,:].values,axis=0)
        June = np.mean(all_springs.snow_cover[3::6,:,:].values,axis=0)
        July = np.mean(all_springs.snow_cover[4::6,:,:].values,axis=0)
        August = np.mean(all_springs.snow_cover[5::6,:,:].values,axis=0)
        title = "Northern Hemisphere Snowcover - Climatology Map"
        colorbar_title = "Frequency of the region being a snow covered land"
        vmin = 0
        vmax = 1
        vn = 9
    #STANDARD DEVIATION
    elif analysis == "std" and forcmp==False:
        March = np.std(all_springs.snow_cover[0::6,:,:].values,axis=0)
        April = np.std(all_springs.snow_cover[1::6,:,:].values,axis=0)
        May = np.std(all_springs.snow_cover[2::6,:,:].values,axis=0)
        June = np.std(all_springs.snow_cover[3::6,:,:].values,axis=0)
        July = np.std(all_springs.snow_cover[4::6,:,:].values,axis=0)
        August = np.std(all_springs.snow_cover[5::6,:,:].values,axis=0)
        title = "Northern Hemisphere Snowcover - Standard deviation Map"
        colorbar_title = "Standard deviation in frequency of the region being snow covered"
        vmin = 0
        vmax = 0.5
        vn = 9 

    months_clm = [March, April, May, June, July, August]
    if forcmp==True:
        return months_clm[2:]
    color = 'Blues'
    months_list = ["March", "April", "May", "June", "July", "August"]
    
    makePlots(months_clm, title, months_list, colorbar_title, vmin, vmax, vn, color, lndclr='firebrick')
    return

def makePlots(months, title, months_list, colorbar_title, vmin, vmax, vn, color, rnd=3, lndclr='gray'):
    """
    Plotting function for climatology, trend and anomaly maps
    """
    print("Plotting in progress")
    ticks = np.round(np.linspace(vmin,vmax,vn), rnd)
    
    color = plt.cm.get_cmap(color, vn-1)
    #print(len(color))
    color.set_bad(color='yellow')
    # plotting parameters
    parallels = np.arange(0,90,10.)
    meridians = np.arange(0.,360.,30.)
    
    indx = 0
    for month in months:
        plt.figure()
        mp = draw_map()
        plt.title("%s for %s (1979-2018)" %(title, months_list[indx]), y=1.08, fontsize=17)
        mp.imshow(month, origin = 'lower', norm=DivergingNorm(0), cmap = color, vmin = vmin, vmax = vmax)
        cbar = mp.colorbar(pad=0.6, ticks=ticks)
        cbar.ax.set_ylabel(colorbar_title, fontsize=14)
        #mp.drawlsmask(land_color='firebrick',ocean_color='aqua', lakes=True, alpha=0.3)
        mp.drawlsmask(land_color=lndclr,ocean_color='white', lakes=True, alpha=0.2)
        mp.drawparallels(parallels, labels = [0,0,0,0])
        mp.drawmeridians(meridians, labels = [1,1,1,1])
        indx += 1
    plt.show()

# MAKING TREND MAPS PLOTS
def makeCompositeplots(months, sup_title, subtitles, colorbar_title, vmin, vmax, vn, rnd, color):#, res=1):
    # Plotting the Composite Maps

    # July-c1 dominated years
    ticks = np.round(np.linspace(vmin,vmax,vn),rnd)
    color = plt.cm.get_cmap(color, vn-1)
    # plotting parameters
    parallels = np.arange(0,90,10.)
    meridians = np.arange(0.,360.,30.)
    plt.figure()
    plt.suptitle(sup_title, fontsize=18)
    i=0
    for month in months:
        mp=draw_map()
        plt.subplot(2,2,i+1)
        plt.title(subtitles[i], y=1.08, fontsize=17)
        mp.imshow(month, norm = DivergingNorm(0.05), vmax = vmax, vmin = vmin, origin = 'lower', cmap = color)
        #mp.drawcoastlines()
        mp.colorbar(pad=0.6, ticks = ticks)
        mp.drawlsmask(land_color='gray', ocean_color='white', lakes=True, alpha=0.1)
        mp.drawparallels(parallels, labels = [0,0,0,0])
        mp.drawmeridians(meridians, labels = [1,1,1,1])
        #mp.drawmapboundary(color='black')
        i+=1
        
    plt.subplots_adjust(left = 0.1, bottom=0.1, right=0.8, top=0.85, hspace = 0.3)
    #[left, bottom, width, height]
    cax = plt.axes([0.83, 0.1, 0.01, 0.75])
    res = vn
# =============================================================================
#    t_s = ticks[0::res]
#    if t_s[-1]==ticks[-1]:
#        cbar = plt.colorbar(cax=cax, orientation = "vertical", ticks=t_s)
#    else:
#        cbar = plt.colorbar(cax=cax, orientation = "vertical", ticks=ticks[1:-1:res])
# =============================================================================
    #cbar = plt.colorbar(cax=cax, orientation = "vertical", ticks=ticks[0::res])
#    cbar.ax.set_ylabel(colorbar_title, fontsize=14)
    plt.show()
    
    return
        
def trendmaps():
    """
    Plots the trend of snow cover in the months of June, July and August (1979-2018) 
    """
    
    print("Plotting Trend maps")


    # Execution of this block takes quite 
    years = np.arange(0, 40) 
    slopes_march = np.empty([720, 720])
    intercepts_march = np.empty([720, 720])
    slopes_april = np.empty([720, 720])
    intercepts_april = np.empty([720, 720])
    slopes_may = np.empty([720, 720])
    intercepts_may = np.empty([720, 720])
    slopes_june = np.empty([720, 720])
    intercepts_june = np.empty([720, 720])
    slopes_july = np.empty([720, 720])
    intercepts_july = np.empty([720, 720])
    slopes_august = np.empty([720, 720])
    intercepts_august = np.empty([720, 720])
    
    for row in np.arange(720):
        march_values = []
        april_values = []
        may_values = []
        june_values = []
        july_values = []
        august_values = []
        w_array_march = all_springs.snow_cover[0:240:6,row,:].values
        w_array_april = all_springs.snow_cover[1:240:6,row,:].values
        w_array_may = all_springs.snow_cover[2:240:6,row,:].values
        w_array_june = all_springs.snow_cover[3:240:6,row,:].values
        w_array_july = all_springs.snow_cover[4:240:6,row,:].values
        w_array_august = all_springs.snow_cover[5:240:6,row,:].values
        print(row)
        for col in np.arange(720):
            
            march_values.append(w_array_march[:,col])
            april_values.append(w_array_april[:,col])
            may_values.append(w_array_may[:,col])
            june_values.append(w_array_june[:,col])
            july_values.append(w_array_july[:,col])
            august_values.append(w_array_august[:,col])
            
            m, b = np.polyfit(years, march_values[col], 1)
            slopes_march[row, col] = m
            intercepts_march[row, col] = b
            
            m, b = np.polyfit(years, april_values[col], 1)
            slopes_april[row, col] = m
            intercepts_april[row, col] = b
            
            m, b = np.polyfit(years, may_values[col], 1)
            slopes_may[row, col] = m
            intercepts_may[row, col] = b
            
            m, b = np.polyfit(years, june_values[col], 1)
            slopes_june[row, col] = m
            intercepts_june[row, col] = b
            
            m, b = np.polyfit(years, july_values[col], 1)
            slopes_july[row, col] = m
            intercepts_july[row, col] = b
            
            m, b = np.polyfit(years, august_values[col], 1)
            slopes_august[row, col] = m
            intercepts_august[row, col] = b
            
# =============================================================================
    # Helps in determining the range of the colorbar
#     print(len(w_array_march),len(march_values[1]))
#     print(np.amax(slopes_march), np.amin(slopes_march), np.mean(slopes_march))
#     print(np.amax(slopes_april), np.amin(slopes_april), np.mean(slopes_april))
#     print(np.amax(slopes_may), np.amin(slopes_may), np.mean(slopes_may))
#     print(np.amax(slopes_june), np.amin(slopes_june), np.mean(slopes_june))
#     print(np.amax(slopes_july), np.amin(slopes_july), np.mean(slopes_july))
#     print(np.amax(slopes_august), np.amin(slopes_august), np.mean(slopes_august))
#             
# =============================================================================
    
    # Plots of trend maps
    # PLotting parameters
    
    months = [slopes_march, slopes_april, slopes_may, slopes_june, slopes_july, slopes_august]
    months_list = ["March", "April", "May", "June", "July", "August"]
    vmax = 0.04
    vmin = -0.04
    vn = 12
    rnd = 3
    color = 'seismic'
    title = "Northern Hemisphere Snowcover - Trend Map"
    colorbar_title = "Slope in variation of the region being snow covered"
    makePlots(months, title, months_list, colorbar_title, vmin, vmax, vn, color, rnd)
    
    return
    


# MAKING ANOMALY PLOTS
def anomalymaps():
    
    """
    Plots the trend of snow cover in the months of June, July and August (1979-2018) 
    """
    
    print("Plotting Anomaly maps")

    March_b = np.mean(all_springs.snow_cover[0:20*6:6,:,:].values, axis=0)
    April_b = np.mean(all_springs.snow_cover[1:20*6+1:6,:,:].values, axis=0)
    May_b = np.mean(all_springs.snow_cover[2:20*6+2:6,:,:].values, axis=0)
    June_b = np.mean(all_springs.snow_cover[3:20*6+3:6,:,:].values, axis=0)
    July_b = np.mean(all_springs.snow_cover[4:20*6+4:6,:,:].values, axis=0)
    August_b = np.mean(all_springs.snow_cover[5:20*6+5:6,:,:].values, axis=0)

    March_l = np.mean(all_springs.snow_cover[20*6::6,:,:].values, axis=0)
    April_l = np.mean(all_springs.snow_cover[20*6+1::6,:,:].values, axis=0)
    May_l = np.mean(all_springs.snow_cover[20*6+2::6,:,:].values, axis=0)
    June_l = np.mean(all_springs.snow_cover[20*6+3::6,:,:].values, axis=0)
    July_l = np.mean(all_springs.snow_cover[20*6+4::6,:,:].values, axis=0)
    August_l = np.mean(all_springs.snow_cover[20*6+5::6,:,:].values, axis=0)

    anomaly_march = (March_l - March_b)
    anomaly_april = (April_l - April_b)
    anomaly_may = (May_l - May_b)
    anomaly_june = (June_l - June_b)
    anomaly_july = (July_l - July_b)
    anomaly_august = (August_l - August_b)
    
    #Helps us in determining the range of the colorbar
#    print(np.amax(anomaly_march), np.amin(anomaly_march), np.mean(anomaly_march))
#    print(np.amax(anomaly_april), np.amin(anomaly_april), np.mean(anomaly_april))
#    print(np.amax(anomaly_may), np.amin(anomaly_may), np.mean(anomaly_may))
#    print(np.amax(anomaly_june), np.amin(anomaly_june), np.mean(anomaly_june))
#    print(np.amax(anomaly_july), np.amin(anomaly_july), np.mean(anomaly_july))
#    print(np.amax(anomaly_august), np.amin(anomaly_august), np.mean(anomaly_august))
    
    
    # Plots of anomaly maps
    # PLotting parameters
    
   #Waiting to put correct vmin and vmax values
    months = [anomaly_march, anomaly_april, anomaly_may, anomaly_june, anomaly_july, anomaly_august]
    months_list = ["March", "April", "May", "June", "July", "August"]
    vmax = 0.8
    vmin = -0.8
    vn = 12
    rnd = 3
    color = 'Spectral'
    title = "Northern Hemisphere Snowcover - Anomaly Map"
    colorbar_title = "Difference between the mean snowcover extent of 1979-1998 and 1999-2018"
    makePlots(months, title, months_list, colorbar_title, vmin, vmax, vn, color, rnd)
    
    return
    
# MAKING COMPOSITE PLOTS
def compositemaps(analysis):
    """
    Mean and standard deviation plots are made for snow cover extent for the months of May, June, 
    July and August for following subset of years
    Years with single-jet domination in July
    Years with double-jet domination in July 
    Years with no domination in July
    
    Years with single-jet domination in August
    Years with double-jet domination in August 
    Years with no domination in August
    
    are being run as seperate subfunctions
    """
    
    print("Composite maps is being plot")

    if analysis == 'mean':
        
        months_clm = climatology('mean',forcmp=True)
        print("Total months in composite =", len(months_clm))
        
        # july_c1_dominated    
        year = np.multiply(np.subtract(july_c1_dominated,1979),6)
        indx_may = np.add(2, year)
        indx_june = np.add(3, year)
        indx_july = np.add(4, year)
        indx_august = np.add(5, year)
        May_jc1 = np.mean(all_springs.snow_cover[indx_may,:,:].values, axis = 0) 
        June_jc1 = np.mean(all_springs.snow_cover[indx_june,:,:].values, axis = 0)
        July_jc1 = np.mean(all_springs.snow_cover[indx_july,:,:].values, axis = 0) 
        August_jc1 = np.mean(all_springs.snow_cover[indx_august,:,:].values, axis = 0)
        
        months = []
        months.append(May_jc1-months_clm[0])
        months.append(June_jc1-months_clm[1])
        months.append(July_jc1-months_clm[2])
        months.append(August_jc1-months_clm[3])
             
                
        may_jc1 = "May"#of years with domination of double jet in July"
        june_jc1 = "June" #of years with domination of double jet in July"
        july_jc1 = "July" #of years with domination of double jet"
        august_jc1 = "August"# of years with domination of double jet in July"
        subtitles = [may_jc1, june_jc1, july_jc1, august_jc1]
        sup_title = "Mean snow cover extent during years with domination of double jet in July relative to mean snow cover extent(1979-2018) for May, June July and August"

        colorbar_title = "Relative difference in frequency of the region being a snow covered land"
        vmax = 0.25
        vmin = -0.25
        vn = 10
        rnd = 3
        color = 'seismic'
        makeCompositeplots(months, sup_title, subtitles, colorbar_title, vmin, vmax, vn, rnd, color)


        
                
        # july_c3_dominated:
        year = np.multiply(np.subtract(july_c3_dominated,1979),6)
        indx_may = np.add(2, year)
        indx_june = np.add(3, year)
        indx_july = np.add(4, year)
        indx_august = np.add(5, year)
        May_jc3 = np.mean(all_springs.snow_cover[indx_may,:,:].values, axis = 0) 
        June_jc3 = np.mean(all_springs.snow_cover[indx_june,:,:].values, axis = 0)
        July_jc3 = np.mean(all_springs.snow_cover[indx_july,:,:].values, axis = 0) 
        August_jc3 = np.mean(all_springs.snow_cover[indx_august,:,:].values, axis = 0)
        
        months = []
        months.append(May_jc3-months_clm[0])
        months.append(June_jc3-months_clm[1])
        months.append(July_jc3-months_clm[2])
        months.append(August_jc3-months_clm[3])
        
        
        may_jc3 = "May" # of years with domination of single jet in July"
        june_jc3 = "June" # of years with domination of single jet in July"
        july_jc3 = "July" # of years with domination of single jet"
        august_jc3 = "August" # of years with domination of single jet in July"
        subtitles = [may_jc3, june_jc3, july_jc3, august_jc3]
        
        sup_title = "Mean snow cover extent during years with domination of single jet in July relative to mean snow cover extent(1979-2018) for May, June July and August"
        colorbar_title = "Relative difference in frequency of the region being a snow covered land"
        vmax = 0.25
        vmin = -0.25
        vn = 10
        rnd = 3
        color = 'seismic'
        makeCompositeplots(months, sup_title, subtitles, colorbar_title, vmin, vmax, vn, rnd, color)

        
        # july_neither_dominated:
        year = np.multiply(np.subtract(july_neither_dominated,1979),6)
        indx_may = np.add(2, year)
        indx_june = np.add(3, year)
        indx_july = np.add(4, year)
        indx_august = np.add(5, year)
        May_jn = np.mean(all_springs.snow_cover[indx_may,:,:].values, axis = 0) 
        June_jn = np.mean(all_springs.snow_cover[indx_june,:,:].values, axis = 0)
        July_jn = np.mean(all_springs.snow_cover[indx_july,:,:].values, axis = 0) 
        August_jn = np.mean(all_springs.snow_cover[indx_august,:,:].values, axis = 0)
        
        months = []
        months.append(May_jn-months_clm[0])
        months.append(June_jn-months_clm[1])
        months.append(July_jn-months_clm[2])
        months.append(August_jn-months_clm[3])
        
        may_jn = "May"# of years with domination of neither jet in July"
        june_jn = "June"# of years with domination of neither jet in July"
        july_jn = "July"# of years with domination of neither jet"
        august_jn = "August"# of years with domination of neither jet in July"
        subtitles = [may_jn, june_jn, july_jn, august_jn]
        sup_title = "Mean snow cover extent during years with domination of neither jet in July relative to mean snow cover extent(1979-2018) for May, June July and August"
        colorbar_title = "Relative difference in frequency of the region being a snow covered land"
        vmax = 0.25
        vmin = -0.25
        vn = 10
        rnd = 3
        color = 'seismic'
        makeCompositeplots(months, sup_title, subtitles, colorbar_title, vmin, vmax, vn, rnd, color)

        
        # august_c1_dominated:
        year = np.multiply(np.subtract(august_c1_dominated,1979),6)
        indx_may = np.add(2, year)
        indx_june = np.add(3, year)
        indx_july = np.add(4, year)
        indx_august = np.add(5, year)
        May_ac1 = np.mean(all_springs.snow_cover[indx_may,:,:].values, axis = 0) 
        June_ac1 = np.mean(all_springs.snow_cover[indx_june,:,:].values, axis = 0)
        July_ac1 = np.mean(all_springs.snow_cover[indx_july,:,:].values, axis = 0) 
        August_ac1 = np.mean(all_springs.snow_cover[indx_august,:,:].values, axis = 0)
        
        months = []
        months.append(May_ac1-months_clm[0])
        months.append(June_ac1-months_clm[1])
        months.append(July_ac1-months_clm[2])
        months.append(August_ac1-months_clm[3])
        
        may_ac1 = "May" #of years with domination of double jet in August"
        june_ac1 = "June" #of years with domination of double jet in August"
        july_ac1 = "July" #of years with domination of double jet in August"
        august_ac1 = "August" #of years with domination of double jet"
        subtitles = [may_ac1, june_ac1, july_ac1, august_ac1]
        
        sup_title = "Mean snow cover extent during years with domination of double jet in August relative to mean snow cover extent(1979-2018) for May, June July and August"
        colorbar_title = "Relative difference in frequency of the region being a snow covered land"
        vmax = 0.25
        vmin = -0.25
        vn = 10
        rnd = 3
        color = 'seismic'
        makeCompositeplots(months, sup_title, subtitles, colorbar_title, vmin, vmax, vn, rnd, color)

        
        # august_c3_dominated:
        year = np.multiply(np.subtract(august_c3_dominated,1979),6)
        indx_may = np.add(2, year)
        indx_june = np.add(3, year)
        indx_july = np.add(4, year)
        indx_august = np.add(5, year)
        May_ac3 = np.mean(all_springs.snow_cover[indx_may,:,:].values, axis = 0) 
        June_ac3 = np.mean(all_springs.snow_cover[indx_june,:,:].values, axis = 0)
        July_ac3 = np.mean(all_springs.snow_cover[indx_july,:,:].values, axis = 0) 
        August_ac3 = np.mean(all_springs.snow_cover[indx_august,:,:].values, axis = 0)
        
        months = []
        months.append(May_ac3-months_clm[0])
        months.append(June_ac3-months_clm[1])
        months.append(July_ac3-months_clm[2])
        months.append(August_ac3-months_clm[3])
        
        
        may_ac3 = "May"# of years with domination of double jet in August"
        june_ac3 = "June"# of years with domination of double jet in August"
        july_ac3 = "July"# of years with domination of single jet in August"
        august_ac3 = "August"# of years with domination of single jet"
        subtitles = [may_ac3, june_ac3, july_ac3, august_ac3]
        
        sup_title = "Mean snow cover extent during years with domination of single jet in August relative to mean snow cover extent(1979-2018) for May, June July and August"
        colorbar_title = "Relative difference in frequency of the region being a snow covered land"
        vmax = 0.25
        vmin = -0.25
        vn = 10
        rnd = 3
        color = 'seismic'
        makeCompositeplots(months, sup_title, subtitles, colorbar_title, vmin, vmax, vn, rnd, color)

        
        # august_neither_dominated:
        year = np.multiply(np.subtract(august_neither_dominated,1979),6)
        indx_may = np.add(2, year)
        indx_june = np.add(3, year)
        indx_july = np.add(4, year)
        indx_august = np.add(5, year)
        May_an = np.mean(all_springs.snow_cover[indx_may,:,:].values, axis = 0) 
        June_an = np.mean(all_springs.snow_cover[indx_june,:,:].values, axis = 0)
        July_an = np.mean(all_springs.snow_cover[indx_july,:,:].values, axis = 0) 
        August_an = np.mean(all_springs.snow_cover[indx_august,:,:].values, axis = 0)
        
        months = []
        months.append(May_an-months_clm[0])
        months.append(June_an-months_clm[1])
        months.append(July_an-months_clm[2])
        months.append(August_an-months_clm[3])
        
        
        may_an = "May"# of years with domination of neither jet in August"
        june_an = "June"# of years with domination of neither jet in August"
        july_an = "July"# of years with domination of neither jet in August"
        august_an = "August"# of years with domination of neither jet"
        subtitles = [may_an, june_an, july_an, august_an]
        sup_title = "Mean snow cover extent during years with domination of neither jet in August relative to mean snow cover extent(1979-2018) for May, June July and August"
        colorbar_title = "Relative difference in frequency of the region being a snow covered land"
        vmax = 0.25
        vmin = -0.25
        vn = 10
        rnd = 3
        color = 'seismic'
        makeCompositeplots(months, sup_title, subtitles, colorbar_title, vmin, vmax, vn, rnd, color)

        
        
        
    
    #STANDARD DEVIATION
    elif analysis == 'std':
        # july_c1_dominated    
        year = np.multiply(np.subtract(july_c1_dominated,1979),6)
        indx_may = np.add(2, year)
        indx_june = np.add(3, year)
        indx_july = np.add(4, year)
        indx_august = np.add(5, year)
        May_jc1 = np.std(all_springs.snow_cover[indx_may,:,:].values, axis = 0) 
        June_jc1 = np.std(all_springs.snow_cover[indx_june,:,:].values, axis = 0)
        July_jc1 = np.std(all_springs.snow_cover[indx_july,:,:].values, axis = 0) 
        August_jc1 = np.std(all_springs.snow_cover[indx_august,:,:].values, axis = 0)
        
        months = [May_jc1, June_jc1, July_jc1, August_jc1]
        may_jc1 = "May"# of years with domination of double jet in July"
        june_jc1 = "June"# of years with domination of double jet in July"
        july_jc1 = "July"# of years with domination of double jet"
        august_jc1 = "August"# of years with domination of double jet in July"
        subtitles = [may_jc1, june_jc1, july_jc1, august_jc1]
        
        sup_title = " Standard deviation of snow cover extent during years with domination of double jet in July for May, June July and August"
        colorbar_title = "Variation in frequency of the region being snow covered"
        vmax = 0.5
        vmin = 0
        vn = 10
        rnd = 3
        color = 'Blues'
        makeCompositeplots(months, sup_title, subtitles, colorbar_title, vmin, vmax, vn, rnd, color)
        
        # july_c3_dominated:
        year = np.multiply(np.subtract(july_c3_dominated,1979),6)
        indx_may = np.add(2, year)
        indx_june = np.add(3, year)
        indx_july = np.add(4, year)
        indx_august = np.add(5, year)
        May_jc3 = np.std(all_springs.snow_cover[indx_may,:,:].values, axis = 0) 
        June_jc3 = np.std(all_springs.snow_cover[indx_june,:,:].values, axis = 0)
        July_jc3 = np.std(all_springs.snow_cover[indx_july,:,:].values, axis = 0) 
        August_jc3 = np.std(all_springs.snow_cover[indx_august,:,:].values, axis = 0)
        
        months = [May_jc3, June_jc3, July_jc3, August_jc3]
        may_jc3 = "May"# of years with domination of single jet in July"
        june_jc3 = "June"# of years with domination of single jet in July"
        july_jc3 = "July"# of years with domination of single jet"
        august_jc3 = "August"# of years with domination of single jet in July"
        subtitles = [may_jc3, june_jc3, july_jc3, august_jc3]
        
        sup_title = " Standard deviation of snow cover extent during years with domination of single jet in July for May, June July and August"
        colorbar_title = "Variation in frequency of the region being snow covered"
        vmax = 0.5
        vmin = 0
        vn = 10
        rnd = 3
        color = 'Blues'
        makeCompositeplots(months, sup_title, subtitles, colorbar_title, vmin, vmax, vn, rnd, color)

        # july_neither_dominated:
        year = np.multiply(np.subtract(july_neither_dominated,1979),6)
        indx_may = np.add(2, year)
        indx_june = np.add(3, year)
        indx_july = np.add(4, year)
        indx_august = np.add(5, year)
        May_jn = np.std(all_springs.snow_cover[indx_may,:,:].values, axis = 0) 
        June_jn = np.std(all_springs.snow_cover[indx_june,:,:].values, axis = 0)
        July_jn = np.std(all_springs.snow_cover[indx_july,:,:].values, axis = 0) 
        August_jn = np.std(all_springs.snow_cover[indx_august,:,:].values, axis = 0)
        
        months = [May_jn, June_jn, July_jn, August_jn]
        may_jn = "May" #of years with domination of neither jet in July"
        june_jn = "June" #of years with domination of neither jet in July"
        july_jn = "July" #of years with domination of neither jet"
        august_jn = "August" #of years with domination of neither jet in July"
        subtitles = [may_jn, june_jn, july_jn, august_jn]
        
        sup_title = " Standard deviation of snow cover extent during years with domination of neither jet in July for May, June July and August"
        colorbar_title = "Variation in frequency of the region being snow covered"
        vmax = 0.5
        vmin = 0
        vn = 10
        rnd = 3
        color = 'Blues'
        makeCompositeplots(months, sup_title, subtitles, colorbar_title, vmin, vmax, vn, rnd, color)

        # august_c1_dominated:
        year = np.multiply(np.subtract(august_c1_dominated,1979),6)
        indx_may = np.add(2, year)
        indx_june = np.add(3, year)
        indx_july = np.add(4, year)
        indx_august = np.add(5, year)
        May_ac1 = np.std(all_springs.snow_cover[indx_may,:,:].values, axis = 0) 
        June_ac1 = np.std(all_springs.snow_cover[indx_june,:,:].values, axis = 0)
        July_ac1 = np.std(all_springs.snow_cover[indx_july,:,:].values, axis = 0) 
        August_ac1 = np.std(all_springs.snow_cover[indx_august,:,:].values, axis = 0)
        
        months = [May_ac1, June_ac1, July_ac1, August_ac1]
        may_ac1 = "May"# of years with domination of double jet in August"
        june_ac1 = "June"# of years with domination of double jet in August"
        july_ac1 = "July"# of years with domination of double jet in August"
        august_ac1 = "August"# of years with domination of double jet"
        subtitles = [may_ac1, june_ac1, july_ac1, august_ac1]
        
        sup_title = "Standard deviation of snow cover extent during years with domination of double jet in August for May, June July and August"
        colorbar_title = "Variation in frequency of the region being snow covered"
        vmax = 0.5
        vmin = 0
        vn = 10
        rnd = 3
        color = 'Blues'
        makeCompositeplots(months, sup_title, subtitles, colorbar_title, vmin, vmax, vn, rnd, color)

        
        # august_c3_dominated:
        year = np.multiply(np.subtract(august_c3_dominated,1979),6)
        indx_may = np.add(2, year)
        indx_june = np.add(3, year)
        indx_july = np.add(4, year)
        indx_august = np.add(5, year)
        May_ac3 = np.std(all_springs.snow_cover[indx_may,:,:].values, axis = 0) 
        June_ac3 = np.std(all_springs.snow_cover[indx_june,:,:].values, axis = 0)
        July_ac3 = np.std(all_springs.snow_cover[indx_july,:,:].values, axis = 0) 
        August_ac3 = np.std(all_springs.snow_cover[indx_august,:,:].values, axis = 0)
        
        months = [May_ac3, June_ac3, July_ac3, August_ac3]
        may_ac3 = "May"# of years with domination of single jet in August"
        june_ac3 = "June"# of years with domination of single jet in August"
        july_ac3 = "July"# of years with domination of single jet in August"
        august_ac3 = "August"# of years with domination of single jet"
        subtitles = [may_ac3, june_ac3, july_ac3, august_ac3]
        
        sup_title = "Standard deviation of snow cover extent during years with domination of single jet in August for May, June July and August"
        colorbar_title = "Variation in frequency of the region being snow covered"
        vmax = 0.5
        vmin = 0
        vn = 10
        rnd = 3
        color = 'Blues'
        makeCompositeplots(months, sup_title, subtitles, colorbar_title, vmin, vmax, vn, rnd, color)

        # august_neither_dominated:
        year = np.multiply(np.subtract(august_neither_dominated,1979),6)
        indx_may = np.add(2, year)
        indx_june = np.add(3, year)
        indx_july = np.add(4, year)
        indx_august = np.add(5, year)
        May_an = np.std(all_springs.snow_cover[indx_may,:,:].values, axis = 0) 
        June_an = np.std(all_springs.snow_cover[indx_june,:,:].values, axis = 0)
        July_an = np.std(all_springs.snow_cover[indx_july,:,:].values, axis = 0) 
        August_an = np.std(all_springs.snow_cover[indx_august,:,:].values, axis = 0)    
        
        months = [May_an, June_an, July_an, August_an]
        may_an = "May"# of years with domination of neither jet in August"
        june_an = "June"# of years with domination of neither jet in August"
        july_an = "July"# of years with domination of neither jet in August"
        august_an = "August"# of years with domination of neither jet"
        subtitles = [may_an, june_an, july_an, august_an]
             
                
        sup_title = "Standard deviation of snow cover extent during years with domination of neither jet in August for May, June July and August"
        colorbar_title = "Variation in frequency of the region being snow covered"
        vmax = 0.5
        vmin = 0
        vn = 10
        rnd = 3
        color = 'Blues'
        makeCompositeplots(months, sup_title, subtitles, colorbar_title, vmin, vmax, vn, rnd, color)
        return
    
    
def p_value_maps():
    """
    This function in intended to produce a p value map comparing July of C1 dominated July
    """
    # creating the comparison datasets
    
        # july_c1_dominated:
    year = np.multiply(np.subtract(july_c1_dominated,1979),6)
    indx_may = np.add(2, year)
    indx_june = np.add(3, year)
    indx_july = np.add(4, year)
    indx_august = np.add(5, year)
    May_jc1 = all_springs.snow_cover[indx_may,:,:].values 
    June_jc1 = all_springs.snow_cover[indx_june,:,:].values
    July_jc1 = all_springs.snow_cover[indx_july,:,:].values 
    August_jc1 = all_springs.snow_cover[indx_august,:,:].values
                
        # july_c3_dominated:
    year = np.multiply(np.subtract(july_c3_dominated,1979),6)
    indx_may = np.add(2, year)
    indx_june = np.add(3, year)
    indx_july = np.add(4, year)
    indx_august = np.add(5, year)
    May_jc3 = all_springs.snow_cover[indx_may,:,:].values 
    June_jc3 = all_springs.snow_cover[indx_june,:,:].values
    July_jc3 = all_springs.snow_cover[indx_july,:,:].values 
    August_jc3 = all_springs.snow_cover[indx_august,:,:].values
    
    
        # july_neither_dominated:
    year = np.multiply(np.subtract(july_neither_dominated,1979),6)
    indx_may = np.add(2, year)
    indx_june = np.add(3, year)
    indx_july = np.add(4, year)
    indx_august = np.add(5, year)
    May_jn = all_springs.snow_cover[indx_may,:,:].values 
    June_jn = all_springs.snow_cover[indx_june,:,:].values
    July_jn = all_springs.snow_cover[indx_july,:,:].values 
    August_jn = all_springs.snow_cover[indx_august,:,:].values
     
        # august_c1_dominated:
    year = np.multiply(np.subtract(august_c1_dominated,1979),6)
    indx_may = np.add(2, year)
    indx_june = np.add(3, year)
    indx_july = np.add(4, year)
    indx_august = np.add(5, year)
    May_ac1 = all_springs.snow_cover[indx_may,:,:].values 
    June_ac1 = all_springs.snow_cover[indx_june,:,:].values
    July_ac1 = all_springs.snow_cover[indx_july,:,:].values 
    August_ac1 = all_springs.snow_cover[indx_august,:,:].values
                
        # august_c3_dominated:
    year = np.multiply(np.subtract(august_c3_dominated,1979),6)
    indx_may = np.add(2, year)
    indx_june = np.add(3, year)
    indx_july = np.add(4, year)
    indx_august = np.add(5, year)
    May_ac3 = all_springs.snow_cover[indx_may,:,:].values 
    June_ac3 = all_springs.snow_cover[indx_june,:,:].values
    July_ac3 = all_springs.snow_cover[indx_july,:,:].values 
    August_ac3 = all_springs.snow_cover[indx_august,:,:].values
        
        # august_neither_dominated:
    year = np.multiply(np.subtract(august_neither_dominated,1979),6)
    indx_may = np.add(2, year)
    indx_june = np.add(3, year)
    indx_july = np.add(4, year)
    indx_august = np.add(5, year)
    May_an = all_springs.snow_cover[indx_may,:,:].values 
    June_an = all_springs.snow_cover[indx_june,:,:].values
    July_an = all_springs.snow_cover[indx_july,:,:].values 
    August_an = all_springs.snow_cover[indx_august,:,:].values
    
    # DJ-J vs SJ-J
    May_jc1c3 = run_studenttest1(May_jc1, May_jc3)
    June_jc1c3 = run_studenttest1(June_jc1, June_jc3)
    
    July_jc1c3 = run_studenttest1(July_jc1, July_jc3)
    August_jc1c3 = run_studenttest1(August_jc1, August_jc3)
    
    # SJ-J vs NJ-J
    May_jc3cn = run_studenttest1(May_jc3, May_jn)
    June_jc3cn = run_studenttest1(June_jc3, June_jn)
    
    July_jc3cn = run_studenttest1(July_jc3, July_jn)
    August_jc3cn = run_studenttest1(August_jc3, August_jn)
    
    # NJ-J vs DJ-J    
    May_jcnc1 = run_studenttest1(May_jn, May_jc1)
    June_jcnc1 = run_studenttest1(June_jn, June_jc1)
    
    July_jcnc1 = run_studenttest1(July_jn, July_jc1)
    August_jcnc1 = run_studenttest1(August_jn, August_jc1)
    
    
    # DJ-A vs SJ-A
    May_ac1c3 = run_studenttest1(May_ac1, May_ac3)
    June_ac1c3 = run_studenttest1(June_ac1, June_ac3)
    
    July_ac1c3 = run_studenttest1(July_ac1, July_ac3)
    August_ac1c3 = run_studenttest1(August_ac1, August_ac3)
    
    # SJ-A vs NJ-A
    May_ac3cn = run_studenttest1(May_ac3, May_an)
    June_ac3cn = run_studenttest1(June_ac3, June_an)
    
    July_ac3cn = run_studenttest1(July_ac3, July_an)
    August_ac3cn = run_studenttest1(August_ac3, August_an)
    
    # NJ-A vs DJ-A    
    May_acnc1 = run_studenttest1(May_an, May_ac1)
    June_acnc1 = run_studenttest1(June_an, June_ac1)
    
    July_acnc1 = run_studenttest1(July_an, July_ac1)
    August_acnc1 = run_studenttest1(August_an, August_ac1)
    
    
    # Plotting parameters
    p_max, p_min, delta = 1, 0, 0.005
    p_n = int(1/0.005)
    colorbar_title = " "
    color = 'seismic'
    rnd = 2
    # more Plotting parameters
    title_DJSJ_J = "P value map from Student t-test (Years with domination of double jet in July vs years with domination of single jet in July)"
    title_SJNJ_J = "P value map from Student t-test (Years with domination of single jet in July vs years with domination of neither jet in July)"
    title_NJDJ_J = "P value map from Student t-test (Years with domination of neither jet in July vs years with domination of double jet in July)"
    title_DJSJ_A = "P value map from Student t-test (Years with domination of double jet in August vs years with domination of single jet in August)"
    title_SJNJ_A = "P value map from Student t-test (Years with domination of single jet in August vs years with domination of neither jet in August)"
    title_NJDJ_A = "P value map from Student t-test (Years with domination of neither jet in August vs years with domination of double jet in August)"
    
    subtitles = ["May", "June", "July", "August"]
    
    #Plotting
    
    #Comparison of years in July-Domination
    makeCompositeplots([May_jc1c3, June_jc1c3, July_jc1c3, August_jc1c3],title_DJSJ_J,subtitles,colorbar_title,p_min,p_max,p_n,rnd,color,res=8)#,color="Blues")
    makeCompositeplots([May_jc3cn, June_jc3cn, July_jc3cn, August_jc3cn],title_SJNJ_J,subtitles,colorbar_title,p_min,p_max,p_n,rnd,color,res=8)
    makeCompositeplots([May_jcnc1, June_jcnc1, July_jcnc1, August_jcnc1],title_NJDJ_J,subtitles,colorbar_title,p_min,p_max,p_n,rnd,color,res=8)
    
    #Comparison of years in August-Domination
    makeCompositeplots([May_ac1c3, June_ac1c3, July_ac1c3, August_ac1c3],title_DJSJ_A,subtitles,colorbar_title,p_min,p_max,p_n,rnd,color,res=8)
    makeCompositeplots([May_ac3cn, June_ac3cn, July_ac3cn, August_ac3cn],title_SJNJ_A,subtitles,colorbar_title,p_min,p_max,p_n,rnd,color,res=8)
    makeCompositeplots([May_acnc1, June_acnc1, July_acnc1, August_acnc1],title_NJDJ_A,subtitles,colorbar_title,p_min,p_max,p_n,rnd,color,res=8)
        
    
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
    t, p = stats.ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2, equal_var=True)
    return p

    
if __name__ == "__main__":
    print("Snow Cover Analysis")
    all_springs = read_data()
#    climatology("mean")
    #climatology("std")
    #trendmaps()
    anomalymaps()
    with open('cluster_years.pickle', 'rb') as f:
        july_c3_dominated, july_c1_dominated, july_neither_dominated, august_c3_dominated, august_c1_dominated, august_neither_dominated = pickle.load(f)
    #compositemaps('mean')
    #compositemaps('std')
    
    #p_value_maps()       
    # =============================================================================
#     print(len(Mays[0]), Mays[0].shape)
#     print(len(all_months[0]), all_months[0].shape)
#     plt.figure()
#     plt.title("SNC Mays of DJ-July dom")
#     sns.distplot(Mays[0], hist=False)
#     plt.figure()
#     plt.title("SNC Junes of DJ-July dom")
#     sns.distplot(Junes[0], hist=False)
#     plt.figure()
#     plt.title("SNC Julys of DJ-July dom")
#     sns.distplot(Julys[0], hist=False)
#     plt.figure()
#     plt.title("SNC Augusts of DJ-July dom")
#     sns.distplot(Augusts[0], hist=False)
#     plt.figure()
#     plt.title("SNC Julys and Augusts of DJ-July dom")
#     sns.distplot(all_months[0], hist=False)
#     plt.show()
#     #print(np.isnan(Mays).any())
# =============================================================================
    #print(np.isnan(Junes).any())
    #print(np.isnan(Julys).any())
    #print(np.isnan(Augusts).any())
#    variables = [Mays, Junes, Julys, Augusts, all_months]
#    
#    months_list = ["May", "June", "July", "August", "All_months"]
#    indx = 0
#    ttest_results = []
#    #anova_results = []
#    for var in variables:
#        print("\nStatistical Test for %s \n\n" %(months_list[indx]))
#        res = run_ttest(var)
#        ttest_results.append(res)
##        res = run_anova(var)
##        anova_results.append(res)
#        indx += 1
        
    
    