# -*- coding: utf-8 -*-
"""
Created on Fri Dec 3 2020

A script that reads in spv.DAT and automatic analyse all spectrums.
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
from apsmodule import spv
APSdir=normpath(getenv('OneDrive')+'\\Data\\APS') if getenv('OneDrive')!=None \
    else ''

#%% Clean spvfiles
spvfiles=[]

#%% Choose files
root=tk.Tk()
# root.withdraw()
# root.iconify()
# root.call('wm', 'attributes', '.', '-topmost', True)
spvfiles+=tk.filedialog.askopenfilenames(parent=root,initialdir=APSdir, title='Please select SPV files',filetypes=[('DAT','.DAT')])
root.destroy()
location=split(spvfiles[0])[0]

#%% Load files into spvdata
plt.close('all')
spvdata=[]
spvdata+=spv.import_from_files(spvfiles,timemap=(20,100,150,100,150),trunc=-8)

#%% Calibrate background SPV
for i in spvdata:  i.cal_background(plot=False)

#%% Plot SPV overlay
plt.figure('SPV overlay')
for i in spvdata: i.plot()

#%% Save SPV into csv
spv.save_csv(spvdata,location,filename='SPV')

#%% Normalized SPV
for i in spvdata: i.normalize(plot=False)

#%% Plot normalized SPV overlay
plt.figure('normalized SPV overlay')
for i in spvdata: i.norm_plot()

#%% Save normalized SPV into csv
spv.save_norm_spv_csv(spvdata,location)
