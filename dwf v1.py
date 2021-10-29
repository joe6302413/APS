# -*- coding: utf-8 -*-
"""
Created on Thur Nov 19 2020

A script that reads in dwf.DAT and automatic analyse all spectrums.
The script include multiple parts:
    1. 
Please run the session you are looking for or following the flow.
This script is open for modification but the "apsmodule" is not.
Before modifying the module please consult with Yi-Chun

Note:
    1.
    For the filename I use my personal style of material_dwf.DAT. And this affects how the program truncate the name.
    To change the truncation length just use dwf.save_dwf_(stat_)csv(data,location,trunc=-n) for n word truncation.
    Again for my personall usage, the default folder direction is to my onedrive APS folder.

Last editing time: 03/12/2020
@author: Yi-Chun Chin    joe6302413@gmail.com
"""
#%% Packages and libraries
import matplotlib.pyplot as plt, tkinter as tk, tkinter.filedialog
from os.path import normpath,split
from os import getenv
from apsmodule import dwf, APS, calibrate
onedrive=getenv('OneDrive')
APSdir=normpath(onedrive+'\\Data\\APS')

#%% Clean filenames
filenames=[]

#%% Choose files
root=tk.Tk()
root.withdraw()
root.iconify()
# root.call('wm', 'attributes', '.', '-topmost', True)
filenames+=tk.filedialog.askopenfilenames(parent=root, initialdir=APSdir, title='Please select dwf files',filetypes=[('DAT','.DAT')])
root.destroy()
location=split(filenames[0])[0]

#%% Load files into data
plt.close('all')
data=[]
data+=dwf.import_from_files(filenames,trunc=-8)

#%% Choose ref APS,dwf and load it
root=tk.Tk()
root.withdraw()
root.iconify()
# root.call('wm', 'attributes', '.', '-topmost', True)
ref_APS_file=[tk.filedialog.askopenfilename(parent=root,initialdir=location, title='Please select ref APS files',filetypes=[('DAT','.DAT')])]
root.destroy()

root=tk.Tk()
root.withdraw()
root.iconify()
# root.call('wm', 'attributes', '.', '-topmost', True)
ref_dwf_file=[tk.filedialog.askopenfilename(parent=root,initialdir=location, title='Please select ref dwf files',filetypes=[('DAT','.DAT')])]
root.destroy()

#%% Load ref APS and ref dwf
[ref_APS]=APS.import_from_files(ref_APS_file,sqrt=True)
[ref_dwf]=dwf.import_from_files(ref_dwf_file)

#%% Analyze ref_APS and dwf_stat ref_dwf
ref_APS.analyze(fit_lower_bound=5,fit_upper_bound=50,smoothness=3,
                plot=True)
ref_dwf.dwf_stat(length=200)

#%% Load calibration object
cal=calibrate(ref_APS,ref_dwf)

#%% Calibrate dwf data
cal.cal(data)

#%% Calculate statistic data
for i in data:  i.dwf_stat(length=200)

#%% Save calibrated DWF into file
dwf.save_csv(data,location,filename='DWF')
dwf.save_dwf_stat_csv(data,location)