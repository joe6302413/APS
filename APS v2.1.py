# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 2020

A script that reads in APS.DAT and automatic analyse all spectrums.
The script include multiple parts:
    1. import packages and apsmodule
    2. tk choosing file section
    3. loading APS data into list of data for each element is an APS object
    4. analyzation (linear fit)
    5. overlay all the result
    6. writing back the result of substract APS and fit line in CSV
Please run the session you are looking for or following the flow.
This script is open for modification but the "apsmodule" is not.
Before modifying the module please consult with Yi-Chun

Note:
    1.
    For the filename I use my personal style of material_APS.DAT. And this affects how the program truncate the name.
    To change the truncation length just use APS.save_aps_(fit_)csv(data,location,trunc=-n) for n word truncation.
    Again for my personall usage, the default folder direction is to my onedrive APS folder.

Last editing time: 12/11/2020
@author: Yi-Chun Chin    joe6302413@gmail.com
"""
#%% packages and libraries
import matplotlib.pyplot as plt, tkinter as tk, tkinter.filedialog
from os.path import split
from apsmodule import APS

#%% clean filenames
filenames=[]

#%% choose files
root=tk.Tk()
root.withdraw()
filenames+=tkinter.filedialog.askopenfilenames(parent=root,initialdir='C:/Users/yc6017/OneDrive - Imperial College London/Data/APS', title='Please select APS files',filetypes=[('DAT','.DAT')])

#%% load files into data
plt.close('all')
data=[]
data+=APS.import_from_files(filenames)

#%% analyze data
plt.close('all')
for i in data:
    i.analyze(6,20)
    
#%% overlay all the data
fig=plt.figure(999)
for i in data: i.plot()
    
#%% Saving APS and APS fit and HOMO with error
location=split(filenames[0])[0]
APS.save_aps_csv(data,location)
APS.save_aps_fit_csv(data,location)
APS.save_homo_error_csv(data,location)

#%% analyze DOS
for i in data:
    i.DOS_analyze(bg=100,plot=False)

#%% smoothing DOS
_=[i.DOSsmooth(17,4,plot=True) for i in data]

#%% overlay all the DOS
plt.figure(1000)
for i in data: i.DOSplot()

#%% Saving DOS into csv
location=split(filenames[0])[0]
APS.save_DOS_csv(data,location)
