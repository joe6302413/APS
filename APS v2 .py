# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 2020

A script that reads in APS.DAT and automatic analyse all spectrums.
The script include multiple parts:
    1. definition of APS class and libraries
    2. data clean up and define section
    3. tk choosing file section
    4. loading APS data into list data for each element is an APS object
    5. analysation (linear fit)
    6. overlay all the result
    7. writing back the result of APS data in CSV
Please run the session you are looking for or following the flow.
The applicability of class APS is welcome to be implemented.
Before using it somewhere  please ask Yi-Chun for permission or consultation.

Note:
    For the filename I use my personal style of material_APS.DAT. And this affects how the program truncate the name. Please feel free to change that.
    Again for my personall usage, the default folder direction is to my onedrive APS folder. Please feel free to change that.
    These characters are not going to be fixed or changed for the triviality
Last editing time: 06/11/2020
@author: Yi-Chun Chin
"""
#%% define and libraries
import matplotlib.pyplot as plt, numpy as np, tkinter as tk, tkinter.filedialog, csv, os
from scipy.optimize import curve_fit, fmin
from scipy import integrate
from os.path import split
# from scipy.stats import linregress

plt.close('all')
def mofun(x,c,shift,MOenergy):
    return np.sum([np.exp(-(x-temp-shift)**2/2/c**2) for temp in MOenergy],axis=0)/c/(2*np.pi)**(1/2)
def apsfun(x,c,shift,scale,MOenergy):
    return np.cumsum([scale*integrate.quad(mofun,x[i-1],x[i],args=(c,shift,MOenergy))[0] if i!=0 else 0 for i in range(len(x))])
def count_in_range(x,baseline):
    return [1 if i<0.15 and i>-0.15 else 0 for i in x-baseline]

class APS:
    
    def __init__(self,energydata,APSdata,Name='no_name'):
        self.energydata=np.array(energydata)
        self.APSdata=np.array(APSdata)
        self.DOS=np.gradient(APSdata,energydata)
        self.name=Name
        
    def pick_range(self):
        plt.figure()
        plt.plot(self.energydata,self.APSdata,'o',label='experiment')
        plt.xlim(self.energydata[0],self.energydata[-1])
        _=plt.title('Pick the range for fitting (min&max)')
        [self.xmin,self.xmax]=np.array(plt.ginput(2))[:,0]
        plt.close()
        if self.xmax<self.xmin:
            self.xmax,self.xmin=self.xmin,self.xmax
        [self.minindex,self.maxindex]=[next(p for p,q in enumerate(self.energydata) if q>self.xmin),next(p for p,q in enumerate(self.energydata) if q>self.xmax)]
        
    def read_gaussian_MO(self,MOenergy):
        self.MOenergy=MOenergy
        
    def find_baseline(self,baseline=3,plot=True):
        baseline=fmin(lambda x: -np.sum(count_in_range(self.APSdata,x)),baseline,disp=False)
        self.baseline=baseline
        if plot==True:
            plt.figure()
            plt.plot(self.energydata,self.APSdata-self.baseline)
    def plot(self):
        plt.grid(True,which='both',axis='x')
        fig=plt.plot(self.energydata,self.APSdata-self.baseline,label=self.name[:-8])
        plt.axhline(y=0, color='k',ls='--')
        if hasattr(self,'lin_par'):
            plt.plot([self.homo,self.energydata[self.lin_stop_index]],[0,np.polyval(self.lin_par,self.energydata[self.lin_stop_index])],'--',c=fig[0]._color)
        if hasattr(self,'fit_par') and hasattr(self,'APSfit'):
            plt.plot(self.energydata,self.APSfit,label='fit: c=%1.4f, shift=%1.4f, scale=%2.1f' %tuple(self.par))
        plt.legend()
        
    def DOSplot(self):
        plt.grid(True,which='both',axis='x')
        if not hasattr(self,'baseline'):    self.find_baseline(plot=False)
        startindex=len(self.energydata)-next(i for i,j in enumerate(self.APSdata[::-1]-self.baseline) if j<0)-5
        _=plt.plot(self.energydata[startindex:],self.DOS[startindex:],label=self.name[:-8])
        plt.axhline(y=0, color='k',ls='--')
        plt.legend()

    def analyze(self, sig_lower_bound=3,sig_upper_bound=np.inf,smoothness=2):
        if smoothness==1:   gap=5
        elif smoothness==2: gap=7
        else: gap=10
        if not hasattr(self, 'baseline'):  self.find_baseline(plot=False)
        startindex=len(self.energydata)-next(i for i,j in enumerate(self.APSdata[::-1]-self.baseline) if j<sig_lower_bound)-1
        stopindex=len(self.energydata)-next(i for i,j in enumerate(self.APSdata[::-1]-self.baseline) if j<sig_upper_bound)
        self.homo_sig=np.inf
        for i,j in [[i,j] for i in range(startindex,stopindex) for j in range(i+gap,stopindex)]:
            [slope,intercept],[[var_slope,_],[_,var_intercept]]=np.polyfit(self.energydata[i:j],self.APSdata[i:j]-self.baseline,1,cov=True)
            homo_sig=np.sqrt(var_slope**2/slope**2+var_intercept**2/intercept**2)
            if homo_sig<self.homo_sig:  self.lin_start_index,self.lin_stop_index,self.lin_par,self.homo_sig=i,j,(slope,intercept),homo_sig
        self.homo=-self.lin_par[1]/self.lin_par[0]
        fig=plt.figure()
        ax=fig.gca()
        ax.grid(True,which='both',axis='x')
        plt.plot([self.homo,self.energydata[self.lin_stop_index]],[0,np.polyval(self.lin_par,self.energydata[self.lin_stop_index])],'--',label='linear fit')
        plt.plot(self.energydata,self.APSdata-self.baseline,label='APS data')
        plt.legend()
        plt.xlim([self.energydata[0],self.energydata[-1]])
        plt.ylim([-0.5,self.APSdata[-1]-self.baseline])
        plt.title(self.name[:-8])
        ax.text(.5, .95, 'HOMO=%1.2f\u00b1 %0.3f%%' %(self.homo,self.homo_sig), style='italic',bbox={'facecolor': 'yellow', 'alpha': 0.5},horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
        ax.axhline(y=0, color='k',ls='--')
        if self.lin_stop_index-self.lin_start_index==gap: print(self.name+' is using the minimum number of points\t')
        
    def APSfit(self,p0=[0.12,0.2,5],bounds=([0.1,-0.5,0.01],[0.5,0.5,1e4]),repick=True):
        self.p0=p0
        self.bounds=bounds
        if repick:
            self.pick_range()
        if not hasattr(self, 'MOenergy'):
            self.read_gaussian_MO(np.array(input("Input MOs from Gaussian:\n").split(),'float'))
        self.fit_par,_ = curve_fit(lambda x,c,shift,scale: apsfun(x,c,shift,scale,self.MOenergy),self.energydata[self.minindex:self.maxindex],self.APSdata[self.minindex:self.maxindex],p0=[0.12,0.2,5],bounds=([0.1,-0.5,0.01],[0.5,0.5,1e4]),absolute_sigma=True,ftol=1e-12)
        plt.figure()
        plt.plot(self.energydata,self.APSdata,'o',label='experiment')
        self.APSfit=apsfun(self.energydata,*self.par,self.MOenergy)
        plt.plot(self.energydata,self.APSfit,label='fit: c=%1.4f, shift=%1.4f, scale=%2.1f' %tuple(self.par))
        plt.xlabel('Energy (eV)')
        plt.ylabel('Photoemission^1/3 (a.u.)')
        plt.legend()
        plt.title('FitAPS')
        print('Broaden facter=%1.4f, shift=%1.4f, scale=%2.1f' %tuple(self.fit_par))

#%%clean up data
data=[]
#%% choose files
tk.Tk().withdraw()
filenames=tkinter.filedialog.askopenfilenames(initialdir='C:/Users/yc6017/OneDrive - Imperial College London/Data/APS', title='Please select APS files',filetypes=[('DAT','.DAT')])
os.chdir(split(filenames[0])[0])
#%% load files into data
for file in filenames:
    with open(file,newline='') as f:
        reader=csv.reader(f)
        numberoflines=sum([1 for i in open(file)])
        acceptlines=[i for i in range(2,numberoflines-20)]    
        save_index=[2,7] #index of saved column from raw data. 2 is energy and 7 is cuberoot. 6 is square-root.
        for i,j in enumerate(reader):
            if i==1: temp=np.array([[float(j[k]) for k in save_index if float(j[3])<1e4 ]])
            elif i in acceptlines: 
                if float(j[3])<1e4: temp=np.append(temp,[[float(j[k]) for k in save_index]],axis=0)
        data.append(APS(temp[:,0],temp[:,1],split(file)[1]))
#%% analyze data
for i in data:
    i.analyze()
#%% overlay all the data
fig=plt.figure(999)
for i in data: i.plot()

#%% writing csv for origin to read
origin_header=[['Energy','Photoemission\\+(1/3)'],['eV','a.u.']]
csv_data=[]
with open('APS.csv','w',newline='') as f:
    writer=csv.writer(f)
    writer.writerows([i*len(data) for i in origin_header])
    writer.writerows([[i for j in data for i in [None,j.name[:-8]]],[None]*2*len(data)])
    numberofrows=max([len(i.energydata) for i in data])
    for i in data:
        numberofmissrow=numberofrows-len(i.energydata)
        energydata=np.concatenate((i.energydata,[None]*numberofmissrow))
        APSdata=np.concatenate((i.APSdata-i.baseline,[None]*numberofmissrow))
        temp=np.column_stack((energydata,APSdata))
        csv_data=temp if len(csv_data)==0 else np.column_stack((csv_data,temp))
    writer.writerows(csv_data)

csv_data=[]
with open('APS_linear_regression.csv','w',newline='') as f:
    writer=csv.writer(f)
    writer.writerows([i*len(data) for i in origin_header])
    writer.writerows([[i for j in data for i in [None,j.name[:-8]+' HOMO %1.3f\u00b1 %0.3f%%' %(j.homo,j.homo_sig)]],[None]*2*len(data)])
    for i in data:
        temp=[[i.homo,0],[i.energydata[i.lin_stop_index],np.polyval(i.lin_par,i.energydata[i.lin_stop_index])]]
        csv_data=temp if len(csv_data)==0 else np.column_stack((csv_data,temp))
    writer.writerows(csv_data)



