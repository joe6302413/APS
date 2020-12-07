# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:30:53 2020

The module for measurements done by APS04.
It includes a function for general csv saving in origin format.
The class APS is for handling APS signals, cube(sqr)-root v.s. energy.
The APS class has multiple methods for different physical quantities including HOMO level fit.

Note:
    1.
    

Last editing time: 12/11/2020
@author: Yi-Chun Chin   joe6302413@gmail.com
"""
#%% define and libraries


import matplotlib.pyplot as plt, csv
import numpy as np
from scipy.optimize import curve_fit, shgo
from scipy import integrate
from os.path import split,join
from scipy.signal import savgol_filter

__version__='1.0.2'

def save_csv_for_origin(data,location,filename=None,datanames=None,header=None):
    data_dim=len(data)
    assert [len(i) for i in data][1:]==[len(i) for i in data][:-1], 'number of data mismatch'
    assert len(header[0])==data_dim, 'header mismatch data dimension'
    numberofdata=len(data[0])
    data=[j for i in zip(*data) for j in i]
    maxlength=max(len(i) for i in data)
    data=np.transpose([np.append(i,[None]*(maxlength-len(i))) for i in data])
    if datanames==None:
        datanames=[['data'+str(i) for i in range(numberofdata) for j in range(data_dim)]]
    else:
        datanames=[[j for i in datanames for j in ([None]+[i]*(data_dim-1))]]
        # datanames=[[i for i in datanames for j in range(data_dim)]]
    if header==None:
        header=datanames+[[None]*numberofdata*data_dim]
    else:
        header=[i*numberofdata for i in header]
    with open(join(location,str(filename)+'.csv'),'w',newline='') as f:
        writer=csv.writer(f)
        writer.writerows(header)
        writer.writerows(datanames)
        writer.writerows(data)

class APS:
    def __init__(self,energydata,APSdata,Name='no_name',sqrt=False):
        self.energydata=np.array(energydata)
        self.APSdata=np.array(APSdata)
        self.DOS=np.gradient(APSdata,energydata)
        self.name=Name
        self.sqrt=sqrt
        
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
        
    def find_baseline(self,baseline_bounds=(0,5),plot=True):
        baseline_res=shgo(lambda x: -APS.mofun(x,0.3,self.APSdata),[baseline_bounds])
        self.baseline=baseline_res.x
        if plot==True:
            plt.figure()
            self.plot()
            
    def plot(self,trunc=-8):
        plt.grid(True,which='both',axis='x')
        if hasattr(self,'baseline'):
            fig=plt.plot(self.energydata,self.APSdata-self.baseline,label=self.name[:trunc])
        else:
            fig=plt.plot(self.energydata,self.APSdata,label=self.name[:trunc])
        plt.axhline(y=0, color='k',ls='--')
        if hasattr(self,'lin_par'):
            plt.plot([self.homo,self.energydata[self.lin_stop_index]],[0,np.polyval(self.lin_par,self.energydata[self.lin_stop_index])],'--',c=fig[0]._color)
        if hasattr(self,'fit_par') and hasattr(self,'APSfit'):
            plt.plot(self.energydata,self.APSfit,label='fit: c=%1.4f, shift=%1.4f, scale=%2.1f' %tuple(self.fit_par))
        # plt.xlim([self.energydata[0],self.energydata[-1]])
        # plt.ylim([-0.5,self.APSdata[-1]-self.baseline])
        plt.legend()
        plt.xlabel('Energy(eV)')
        if self.sqrt==False:
            plt.ylabel('Photoemission^(1/3) (a.u.)')
        else:
            plt.ylabel('Photoemission^(1/2) (a.u.)')
        plt.autoscale(enable=True,axis='both',tight=True)
            
    def DOSsmooth(self,*args,plot=False):
        self.DOS_origin=self.DOS
        self.DOS=savgol_filter(self.DOS,*args)
        if plot:
            plt.figure()
            self.DOSplot()
            plt.plot(self.energydata,self.DOS_origin,label='no smooth')
        
    def DOSplot(self,trunc=-8):
        plt.grid(True,which='both',axis='x')
        if not hasattr(self,'baseline'):    self.find_baseline(plot=False)
        startindex=len(self.energydata)-next(i for i,j in enumerate(self.APSdata[::-1]-self.baseline) if j<0)-5
        _=plt.plot(self.energydata[startindex:],self.DOS[startindex:],label=self.name[:trunc])
        plt.axhline(y=0, color='k',ls='--')
        plt.legend()
        plt.xlabel('Energy(eV)')
        plt.ylabel('DOS (a.u.)')

    def analyze(self, std_lower_bound=0.5,std_upper_bound=np.inf,smoothness=2,plot=True):
        if smoothness==1:   gap=5
        elif smoothness==2: gap=7
        else: gap=10
        if not hasattr(self, 'baseline'):  self.find_baseline(plot=False)
        startindex=len(self.energydata)-next(i for i,j in enumerate(self.APSdata[::-1]-self.baseline) if j<std_lower_bound)-1
        stopindex=len(self.energydata)-next(i for i,j in enumerate(self.APSdata[::-1]-self.baseline) if j<std_upper_bound)
        self.std_homo=np.inf
        for i,j in [[i,j] for i in range(startindex,stopindex) for j in range(i+gap,stopindex)]:
            [slope,intercept],[[var_slope,_],[_,var_intercept]]=np.polyfit(self.energydata[i:j],self.APSdata[i:j]-self.baseline,1,cov=True)
            std_homo=np.sqrt(var_slope**2/slope**2+var_intercept**2/intercept**2)
            if std_homo<self.std_homo:
                self.lin_start_index,self.lin_stop_index,self.lin_par,self.std_homo=i,j,(slope,intercept),std_homo
        if self.std_homo==np.inf:
            plt.figure()
            self.plot()
            raise Exception("Fitting fail!!! Rechoose fitting condition.")
        self.homo=-self.lin_par[1]/self.lin_par[0]
        if plot:
            fig=plt.figure()
            ax=fig.gca()
            self.plot()
            plt.title(self.name)
            plt.text(.5, .95, 'HOMO=%1.2f\u00b1 %0.3f%%' %(self.homo,100*self.std_homo), style='italic',bbox={'facecolor': 'yellow', 'alpha': 0.5},horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
        if self.lin_stop_index-self.lin_start_index==gap: print(self.name+' is using the minimum number of points\t')
            
    def APSfit(self,p0=[0.12,0.2,5],bounds=([0.1,-0.5,0.01],[0.5,0.5,1e4]),repick=True):
        self.p0=p0
        self.bounds=bounds
        if repick:
            self.pick_range()
        if not hasattr(self, 'MOenergy'):
            self.read_gaussian_MO(np.array(input("Input MOs from Gaussian:\n").split(),'float'))
        self.fit_par,_ = curve_fit(lambda x,c,scale,shift: self.apsfun(x,c,scale,self.MOenergy-shift),self.energydata[self.minindex:self.maxindex],self.APSdata[self.minindex:self.maxindex],p0=[0.12,5,0.2],bounds=([0.1,0.01,-0.5],[0.5,1e4,0.5]),absolute_sigma=True,ftol=1e-12)
        plt.figure()
        plt.plot(self.energydata,self.APSdata,'o',label='experiment')
        self.APSfit=self.apsfun(self.energydata,*self.fit_par[:-1],self.MOenergy-self.fit_par[-1])
        plt.plot(self.energydata,self.APSfit,label='fit: c=%1.4f, shift=%1.4f, scale=%2.1f' %tuple(self.fit_par))
        plt.xlabel('Energy (eV)')
        plt.ylabel('Photoemission^1/3 (a.u.)')
        plt.legend()
        plt.title('FitAPS')
        print('Broaden facter=%1.4f, shift=%1.4f, scale=%2.1f' %tuple(self.fit_par))
        
    @classmethod
    def import_from_files(cls,filenames,sqrt=False):
        data=[]
        save_index=[2,6] if sqrt else [2,7] #index of saved column from raw data. 2 is energy and 7 is cuberoot. 6 is square-root.
        for file in filenames:
            with open(file,newline='') as f:
                reader=csv.reader(f)
                for i,j in enumerate(f):
                    if j[:3]==' WF': stopindex=i
                f.seek(0)
                acceptlines=range(1,stopindex)
                temp=np.array([[float(j[save_index[0]]),float(j[save_index[1]])] for i,j in enumerate(reader) if i in acceptlines if float(j[3])<1e4])
            data.append(cls(temp[:,0],temp[:,1],split(file)[1],sqrt))
        return data
                    
    @staticmethod
    def save_aps_csv(data,location,trunc=-8,filename='APS',):
        datanames=[i.name[:trunc] for i in data]
        origin_header=[['Energy','Photoemission\\+(1/3)'],['eV','a.u.']] if all([i.sqrt==False for i in data]) else [['Energy','Photoemission\\+(1/2)'],['eV','a.u.']]
        x,y=[i.energydata for i in data],[i.APSdata-i.baseline for i in data]
        save_csv_for_origin((x,y),location,filename,datanames,origin_header)

    @staticmethod
    def save_aps_fit_csv(data,location,trunc=-8,filename='APS_linear_regression'):
        origin_header=[['Energy','Photoemission\\+(1/3)'],['eV','a.u.']] if all([i.sqrt==False for i in data]) else [['Energy','Photoemission\\+(1/2)'],['eV','a.u.']]
        datanames=[i.name[:trunc] for i in data]
        x=[np.array([i.homo,i.energydata[i.lin_stop_index]]) for i in data]
        y=[np.array([0,np.polyval(i.lin_par,i.energydata[i.lin_stop_index])]) for i in data]
        save_csv_for_origin((x,y),location,filename,datanames,origin_header)
    
    @staticmethod
    def save_homo_error_csv(data,location,trunc=-8,filename='APS_HOMO'):
        origin_header=[['Material','Energy','HOMO std'],[None,'eV','eV']]
        datanames=['HOMO']
        x=[[i.name[:trunc] for i in data]]
        y=[[-i.homo for i in data]]
        z=[[i.std_homo*i.homo for i in data]]
        save_csv_for_origin((x,y,z),location,filename,datanames,origin_header)
        
    @staticmethod
    def save_DOS_csv(data,location,trunc=-8,filename='DOS'):
        origin_header=[['Energy','DOS'],['eV','a.u.']]
        datanames=[i.name[:trunc] for i in data]
        x,y=[i.energydata for i in data],[i.DOS for i in data]
        save_csv_for_origin((x,y),location,filename,datanames,origin_header)
        
    @staticmethod
    def gaussian(x,c,center):
        return np.exp(-(x-center)**2/2/c**2)/c/(2*np.pi)**0.5
    
    @staticmethod
    def mofun(x,c,MOenergy):
        return np.sum([APS.gaussian(x,c,i) for i in MOenergy],axis=0)
    
    @staticmethod
    def apsfun(x,c,scale,MOenergy):
        return np.cumsum([scale*integrate.quad(APS.mofun,x[i-1],x[i],args=(c,MOenergy))[0] if i!=0 else 0 for i in range(len(x))])
    
class dwf:
    cal=False
    def __init__(self,time,CPDdata,name='no_name'):
        self.time=np.array(time)
        self.CPDdata=np.array(CPDdata)
        self.name=name
    
    def plot(self,trunc=-8):
        plt.grid(True,which='both',axis='both')
        if self.cal:
            plt.plot(self.time,self.CPDdata,label=self.name[:trunc])
            plt.ylabel('Fermi Level (eV)')
        else:
            plt.plot(self.time,self.CPDdata,label=self.name[:trunc])
            plt.ylabel('CPD (meV)')
        plt.legend()
        plt.xlabel('Time(s)')
        plt.autoscale(enable=True,axis='both',tight=True)
            
    def stat(self,length=200):
        stop_index=len(self.time)
        start_index=stop_index-next(i for i,j in enumerate(self.time[::-1]-self.time[-1]) if j<-length)
        self.average_CPD=np.average(self.CPDdata[start_index:stop_index])
        self.std_CPD=np.std(self.CPDdata[start_index:stop_index])
        self.length=length
            
    @classmethod
    def import_from_files(cls,filenames):
        data=[]
        save_index=[-3,2]
        for file in filenames:
            with open(file,newline='') as f:
                reader=csv.reader(f)
                for i,j in enumerate(reader):
                    if len(j)==1:
                        stopindex=i
                        break
                f.seek(0)
                acceptlines=range(1,stopindex)
                temp=np.array([[float(j[save_index[0]]),float(j[save_index[1]])] for i,j in enumerate(reader) if i in acceptlines])
            data.append(cls(temp[:,0],temp[:,1],split(file)[1]))
        return data
    
    @staticmethod
    def save_dwf_csv(data,location,trunc=-8,filename='DWF'):
        origin_header=[['Time','E\-(F)'],[None,'eV']] if all([i.cal for i in data]) else [['Time','CPD'],[None,'meV']]
        datanames=[i.name[:trunc] for i in data]
        x=[i.time for i in data]
        y=[i.CPDdata for i in data]
        save_csv_for_origin((x,y),location,filename,datanames,origin_header)
    
    @staticmethod
    def save_dwf_stat_csv(data,location,trunc=-8,filename='DWF_stat'):
        if not all([hasattr(i,'average_dwf') for i in data]):
            _=[i.stat() for i in data]
        if all([i.cal for i in data]):
            origin_header=[['Material','Energy','E\-(F) std'],[None,'eV','eV']]  
            datanames=['E\-(F)']
        else:
            origin_header=[['Material','Energy','CPD std'],[None,'eV','eV']]
            datanames=['CPD']
        x=[[i.name[:trunc] for i in data]]
        y=[[i.average_CPD for i in data]]
        z=[[i.std_CPD for i in data]]
        save_csv_for_origin((x,y,z),location,filename,datanames,origin_header)
        
class calibrate:
    def __init__(self,ref_APS,ref_dwf):
        if not hasattr(ref_APS,'homo'):
            ref_APS.analyze(plot=False)
        if not hasattr(ref_dwf,'average_CPD'):
            ref_dwf.stat()
        self.tip_dwf=-ref_APS.homo+ref_dwf.average_CPD/1000
    
    def cal(self,data):
        for i in data:
            i.CPDdata=-i.CPDdata/1000+self.tip_dwf
            i.cal=True
            i.stat()

class spv(dwf):
    spv_bg_cal=False
    def cal_background(self,bgtime):
        self.bgtime=bgtime
        self.spv_bg_cal=True
        stop_index=next(i for i,j in enumerate(self.time) if j>bgtime)
        self.bg_cpd=np.average(self.CPDdata[:stop_index])
        self.CPD_bg_cal=self.CPDdata-self.bg_cpd
        
    @staticmethod
    def save_SPV_csv(data,location,trunc=-8,filename='SPV'):
        origin_header=[['Time','SPV'],[None,'eV']] if all([i.cal for i in data]) else [['Time','SPV'],[None,'meV']]
        datanames=[i.name[:trunc] for i in data]
        x=[i.time for i in data]
        y=[i.CPD_bg_cal for i in data] if all([i.spv_bg_cal for i in data]) else [i.CPDdata for i in data]
        save_csv_for_origin((x,y),location,filename,datanames,origin_header)