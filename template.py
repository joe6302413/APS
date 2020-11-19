# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:53:08 2020

@author: yc6017
"""
import matplotlib.pyplot as plt, numpy as np
def APSsim(c,e_range,energy,shift,xscale,yscale,dyscale):
    DOS=[(sum(np.exp(-(-energy+x)**2/2/c**2))/np.sqrt(2*np.pi)/c/len(energy)) for x in e_range]
    mo=np.array([(sum(np.exp(-(-energy+x)**2/2/0.007**2))) for x in e_range])
    x=e_range+shift
    n=next(p for p,q in enumerate(x) if q>xscale)
    cum=np.cumsum(DOS)
    cum*=np.array(yscale/cum[n])
    DOS*=np.array(dyscale/DOS[n])
    return x,DOS,cum,mo
#%% parameters
class APS:
    range=np.array([3.523,3.533,3.543,3.553,3.563,3.573,3.584,3.594,3.605,3.615,3.626,3.636,3.647,3.658,3.669,3.68,3.69,3.701,3.713,3.724,3.735,3.746,3.758,3.769,3.78,3.792,3.804,3.815,3.827,3.839,3.851,3.863,
3.875,3.887,3.899,3.912,3.924,3.937,3.949,3.962,3.974,3.987,4,4.013,4.026,4.039,4.052,4.066,4.079,4.092,4.106,4.12,4.133,4.147,4.161,4.175,4.189,4.203,4.218,4.232,4.247,4.261,4.276,4.291,
4.306,4.321,4.336,4.351,4.366,4.382,4.397,4.413,4.429,4.444,4.46,4.477,4.493,4.509,4.526,4.542,4.559,4.576,4.593,4.61,4.627,4.644,4.662,4.679,4.697,4.715,4.733,4.751,4.769,4.788,4.806,4.825,
4.844,4.863,4.882,4.901,4.921,4.94,4.96,4.98,5,5.02,5.041,5.061,5.082,5.103,5.124,5.145,5.167,5.188,5.21,5.232,5.254,5.277,5.299,5.322,5.345,5.368,5.391,5.415,5.439,5.463,5.487,5.511,
5.536,5.561,5.586,5.611,5.636,5.662,5.688,5.714,5.741,5.767,5.794,5.822,5.849,5.877,5.905,5.933,5.962,5.99,6.019,6.049,6.078,6.108,6.139,6.169,6.2,6.231,6.263,6.294,6.327,6.359,6.392,6.425,
6.458,6.492,6.526,6.561,6.596,6.631,6.667,6.703,6.739,6.776,6.813,6.851,6.889,6.927,6.966,7.006,7.045,7.086,7.126])
    
xmin,xmax=4,7.1
c=0.25
e_range=np.linspace(xmin,xmax,1000)
ymin=0
xscale,yscale=6.492,14.92541
dyscale=15.63
shift=0.25

energy=np.array(input("Input MOs from Gaussian:\n").split(),'float')
#%%
x,DOS,cum,MO=APSsim(c,e_range,energy,shift,xscale,yscale,dyscale)
APS.range=APS.range-shift
APS.x,APS.DOS,APS.cum,APS.MO=APSsim(c,APS.range,energy,shift,xscale,yscale,dyscale)

    #%% plotting
plt.plot(x,MO)
plt.title("MO")
axes = plt.gca()
axes.set_xlim([min(x),max(x)])
axes.set_ylim([min(MO),max(MO)])
plt.figure()
plt.plot(x,DOS)
plt.title("DOS")
axes = plt.gca()
axes.set_xlim([min(x),max(x)])
axes.set_ylim([min(DOS),max(DOS)])
plt.figure()
plt.plot(x,cum)
plt.title("acc DOS")
axes = plt.gca()
axes.set_xlim([min(x),max(x)])
ymax=max(cum)
axes.set_ylim([ymin,ymax])

#%%
#for shift in np.linspace(0,0.3,100):
#    e_range=np.linspace(xmin,xmax,1000)
#    ymin=0
#    xscale,yscale=7,24.86
##    shift=0.12
#    #%%
#    x,y=DOS(c,e_range,energy)
#    
#    #%%shifting energy
#    x+=shift
#    
#    
#    n,yscale=next(p for p,q in enumerate(e_range) if q>xscale),yscale
#    cum=np.cumsum(y)
#    cum=cum/cum[n]*yscale;
#    #%% plotting
#    
#    plt.plot(x,cum)
#    plt.title("acc DOS")
#    axes = plt.gca()
#    axes.set_xlim([min(x),max(x)])
#    ymax=max(cum)
#    axes.set_ylim([ymin,ymax])
#    plt.figure(num=None)
#    if out.shape[0]==0:
#        out=cum
#    elif out.shape[0]==1000:
#        out=np.append([out],[cum],axis=0)
#    else:
#        out=np.append(out,[cum],axis=0)
        
        
#cor=np.cumsum(y*x)
#cor=cor/cor[n]*yscale;
#plt.plot(x,cor)
#plt.title("cor acc DOS")
#axes = plt.gca()
#axes.set_xlim([xmin,xmax])
#ymax=max(cor)
#axes.set_ylim([0,max(cor)])