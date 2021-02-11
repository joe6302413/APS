# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 2020

A small script that reads in MOenergy, APS data to fit by cumulative DOS formalism.

@author: Yi-Chun Chin
"""
#%% define and libraries
import matplotlib.pyplot as plt, numpy as np
from scipy.optimize import curve_fit
from scipy import integrate
plt.close('all')
def mofun(x,c,shift,MOenergy):
    return np.sum([np.exp(-(x-temp-shift)**2/2/c**2) for temp in MOenergy],axis=0)/c/(2*np.pi)**(1/2)
def apsfun(x,c,shift,scale):
    return np.cumsum([scale*integrate.quad(mofun,x[i-1],x[i],args=(c,shift,MOenergy))[0] if i!=0 else 0 for i in range(len(x))])
class APS:
    def __init__(self,energydata,APSdata):
        self.energydata=energydata
        self.APSdata=APSdata
        self.DOS=np.gradient(APSdata,energydata)
    def pick_fit_range(self):
        plt.figure()
        plt.plot(self.energydata,self.APSdata,'o',label='experiment')
        plt.xlim(self.energydata[0],self.energydata[-1])
        _=plt.title('Pick the range for fitting (min&max)')
        [self.xmin,self.xmax]=np.array(plt.ginput(2))[:,0]
        plt.close()
        if self.xmax<self.xmin:
            self.xmax,self.xmin=self.xmin,self.xmax
        [self.minindex,self.maxindex]=[next(p for p,q in enumerate(self.energydata) if q>self.xmin),next(p for p,q in enumerate(self.energydata) if q>self.xmax)-1]
#%% parameters read in
MOenergy=np.array(input("Input MOs from Gaussian:\n").split(),'float')

# energydata=np.array(input("Input APS energies:\n").split(),'float')

# APSdata=np.array(input("Input APS cube(square) root photoemission:\n").split(),'float')

data=APS(np.array(input("Input APS energies:\n").split(),'float'),np.array(input("Input APS cube(square) root photoemission:\n").split(),'float'))
#%% Picking fitting range
data.pick_fit_range()

#%% Fitting and plotting the final results
plt.close()
par,_ = curve_fit(apsfun,data.energydata[data.minindex:data.maxindex],data.APSdata[data.minindex:data.maxindex],p0=[0.12,0.2,5],bounds=([0.1,-0.5,0.01],[0.5,0.5,1e4]),absolute_sigma=True,ftol=1e-12)
plt.plot(data.energydata,data.APSdata,'o',label='experiment')
APSfit=apsfun(data.energydata,*par)
plt.plot(data.energydata,APSfit,label='fit: c=%1.4f, shift=%1.4f, scale=%2.1f' %tuple(par))
plt.xlabel('Energy (eV)')
plt.ylabel('Photoemission^1/3 (a.u.)')
plt.legend()
_=plt.title('FitAPS')
par