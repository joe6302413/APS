# -*- coding: utf-8 -*-
"""
Created on Thur Nov 19 2020

A script that reads in APS.DAT and automatic analyse all spectrums.
The script include multiple parts:
    1. 
Please run the session you are looking for or following the flow.
This script is open for modification but the "apsmodule" is not.
Before modifying the module please consult with Yi-Chun

Note:
    1.
    For the filename I use my personal style of material_APS.DAT. And this affects how the program truncate the name.
    To change the truncation length just use APS.save_aps_(fit_)csv(data,location,trunc=-n) for n word truncation.
    Again for my personall usage, the default folder direction is to my onedrive APS folder.

Last editing time: 19/11/2020
@author: Yi-Chun Chin    joe6302413@gmail.com
"""
#%% packages and libraries
import matplotlib.pyplot as plt, tkinter as tk, tkinter.filedialog
from os.path import split
from apsmodule import dwf
# from scipy.stats import linregress

#%% clean filenames
filenames=[]

#%% choose files
root=tk.Tk()
root.withdraw()
filenames+=tkinter.filedialog.askopenfilenames(parent=root,initialdir='C:/Users/yc6017/OneDrive - Imperial College London/Data/APS', title='Please select APS files',filetypes=[('DAT','.DAT')])

#%% load files into data
plt.close('all')
data=[]
data+=dwf.import_from_files(filenames)


