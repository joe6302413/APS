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
from os.path import split
from apsmodule import dwf, APS, calibrate

#%% Clean filenames
filenames=[]

#%% Choose files
root=tk.Tk()
root.withdraw()
filenames+=tkinter.filedialog.askopenfilenames(parent=root,initialdir='C:/Users/yc6017/OneDrive - Imperial College London/Data/APS', title='Please select dwf files',filetypes=[('DAT','.DAT')])
location=split(filenames[0])[0]

#%% Load files into data
plt.close('all')
data=[]
data+=dwf.import_from_files(filenames)

#%% Choose ref APS,dwf and load it
root=tk.Tk()
root.withdraw()
ref_APS_file=[tkinter.filedialog.askopenfilename(parent=root,initialdir=location, title='Please select ref APS files',filetypes=[('DAT','.DAT')])]
[ref_APS]=APS.import_from_files(ref_APS_file,sqrt=True)

root=tk.Tk()
root.withdraw()
ref_dwf_file=[tkinter.filedialog.askopenfilename(parent=root,initialdir=location, title='Please select ref dwf files',filetypes=[('DAT','.DAT')])]
[ref_dwf]=dwf.import_from_files(ref_dwf_file)

#%% Calibrate dwf data
cal=calibrate(ref_APS,ref_dwf)
cal.cal(data)

#%% Calculate statistic data
for i in data:  i.stat()

#%% Save calibrated DWF into file
dwf.save_dwf_csv(data,location)
dwf.save_dwf_stat_csv(data,location)