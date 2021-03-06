# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:30:53 2020

The module for measurements done by APS04.
It includes a function for general csv saving in origin format.
The class APS is for handling APS signals, cube(sqr)-root v.s. energy.
The APS class has multiple methods for different physical quantities including HOMO level fit.
The DWF class handles all variations of CPD measurements.
Calibrate class takes into APS and DWF object for calibrating factor building.
Build DWF related measurements based on root class of DWF such as SPV.

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

__version__='1.1'

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

def find_overlap(x,y):
    '''
    Given 2D x=[x1,x2,...] and 2D y=[y1,y2,...] of each array xn or yn is a
    positive monotonic numpy array.
    Output x' and y' for which x' is a numpy array of the common x
    And y' is a 2D list such that y'=[y1',y2',...] with numpy array yn' 
    contains the value of yn with resspect to xn'
    '''
    x_min=np.max([i[0] for i in x])
    x_max=np.min([i[-1] for i in x])
    for i in range(len(x)):
        min_index=next(l for l,k in enumerate(x[i]) if k==x_min)
        max_index=next(l for l,k in enumerate(x[i]) if k==x_max)
        y[i]=y[i][min_index:max_index+1]
    x=x[-1][min_index:max_index+1]
    return x,y
    
def inv_gradient(x,g,y0=0):
    '''
    An inverse function of numpy.gradient.
    g=np.gradient(y,x) == y=inv_gradient(x,g,y0=y[0])
    
    Output numpy array y
    '''
    length=len(g)
    if length!=len(x):
        raise Exception('length of x and gradient should '+
                        'have match length')
    xdiff=np.diff(x)
    xdiff2=xdiff[:-1]+xdiff[1:]
    y=np.array([y0,g[0]*xdiff[0]+y0])
    s=g[0]
    for i in range(1,length-2):
        s=(g[i]-s*xdiff[i]/xdiff2[i-1])*xdiff2[i-1]/xdiff[i-1]
        y=np.append(y,y[i]+s*xdiff[i])
    y=np.append(y,y[-1]+g[-1]*xdiff[-1])
    return y

class APS:
    def __init__(self,energy,APSdata,sqrt=False,Name='no_name'):
        self.energy=np.array(energy)
        self.APSdata=np.array(APSdata)
        self.DOS=np.gradient(APSdata,energy)
        self.name=Name
        self.status={'sqrt': sqrt,'baseline':False,
                     'cutoff': False,'analyzed':False, 'DOS_analyzed': True,
                     'DOS smoothed': False}
        
    def __repr__(self):
        return self.name
    
    def __str__(self):
        summary='Name:\t'+self.name+'\n'
        summary+='\n'.join([i+':\t'+str(j) for i,j in self.status.items()])+'\n'
        if self.status['analyzed']==True:
            summary+='HOMO(eV):\t%1.2f\u00b1%1.2feV\n'%(
                self.homo,self.homo*self.std_homo)
        return summary
               
    # def pick_range(self):
    #     plt.figure()
    #     plt.plot(self.energy,self.APSdata,'o',label='experiment')
    #     plt.xlim(self.energy[0],self.energy[-1])
    #     _=plt.title('Pick the range for fitting (min&max)')
    #     [self.xmin,self.xmax]=np.array(plt.ginput(2))[:,0]
    #     plt.close()
    #     if self.xmax<self.xmin:
    #         self.xmax,self.xmin=self.xmin,self.xmax
    #     [self.minindex,self.maxindex]=[next(p for p,q in enumerate(self.energy) if q>self.xmin),next(p for p,q in enumerate(self.energy) if q>self.xmax)]
    
    def status_check(self):
        report=''
        if hasattr(self,'baseline'):
            if not self.status['baseline']==float(self.baseline):
                report+='Baseline is corrupted!\nRedo self.find_baseline'\
                    '(baseline_bounds) and self.find_cutoff()\n'
        elif self.status['baseline']:
            report+='Baseline is corrupted!\nRedo self.find_baseline'\
                    '(baseline_bounds) and self.find_cutoff()\n'
        if self.status['cutoff']!=hasattr(self,'cutoff_index'):
            report+='Cutoff is corrupted!\nRedo self.find_cutoff()\n'
        if self.status['analyzed']!=hasattr(self,'homo'):
            report+='Analaysis is corrupted!\nRedo self.analyze('\
                'fit_lower_bound,fit_upper_bound)\n'
        # if self.status['DOS_analyzed']!=hasattr(self,'DOS'):
        #     report+='DOS is corrupted!\nRedo self.DOS_analyze(bg)\n'
        if self.status['DOS smoothed']!=hasattr(self,'original_DOS'):
            report+='DOS smmoothing is corrupted!\nRedo self.DOSsmooth('\
                'pts,power)\n'
        if report=='':
            print(self.name+'\n--------\nCheck done and all statuses are '\
                  'good!\n')
        else:
            print(self.name+'\n--------\n'+report)
        
    def read_gaussian_MO(self,MOenergy):
        self.MOenergy=MOenergy
        
    def find_baseline(self,baseline_bounds=(1,5),plot=True):
        baseline_res=shgo(lambda x: -APS.mofun(x,0.3,self.APSdata),
                          [baseline_bounds],iters=2)
        self.baseline=baseline_res.x
        self.status['baseline']=float(self.baseline)
        if plot==True:
            plt.figure()
            self.plot()

    def find_cutoff(self):
        if not hasattr(self,'baseline'):
            self.find_baseline(plot=False)
            print('Automatic find baseline between (1,5) for '+self.name)
        index=next(len(self.APSdata)-i for i,j in enumerate(self.APSdata[::-1]
                                                 -self.baseline) if j<0)
        self.cutoff_index,self.cutoff_energy=index,self.energy[index]
        self.status['cutoff']=True
        
    def plot(self):
        plt.grid(True,which='both',axis='x')
        if hasattr(self,'baseline'):
            fig=plt.plot(self.energy,self.APSdata-self.baseline,
                         label=self.name)
        else:
            fig=plt.plot(self.energy,self.APSdata,label=self.name)
        plt.axhline(y=0, color='k',ls='--')
        if hasattr(self,'lin_par'):
            plt.plot([self.homo,self.energy[self.lin_stop_index]],
                     [0,np.polyval(self.lin_par,self.energy[
                         self.lin_stop_index])],'--',c=fig[0]._color)
        if hasattr(self,'fit_par') and hasattr(self,'APSfit'):
            plt.plot(self.energy,self.APSfit,label=
                     'fit: c=%1.4f, shift=%1.4f, scale=%2.1f' %tuple(self.fit_par))
        plt.legend()
        plt.xlabel('Energy (eV)')
        if self.status['sqrt']==False:
            plt.ylabel('Photoemission^(1/3)  (a.u.)')
        else:
            plt.ylabel('Photoemission^(1/2)  (a.u.)')
        plt.autoscale(enable=True,axis='both',tight=True)
            
    def DOSsmooth(self,*args,plot=False,**kwargs):
        if not hasattr(self,'DOS'):
            raise Exception("Calculate DOS with self.DOS_analyze()")
        if hasattr(self,'DOS_original'):
            self.DOS=self.DOS_original
        else:
            self.DOS_original=self.DOS
        self.DOS=savgol_filter(self.DOS,*args,**kwargs)
        self.status['DOS smoothed']=True
        if plot:
            plt.figure()
            plt.plot(self.energy,self.DOS_original,label='no smooth')
            self.DOSplot()
            plt.legend()
        
    def DOSplot(self):
        if not hasattr(self,'DOS'):
            raise Exception("Calculate DOS with self.DOS_analyze()")
        plt.grid(True,which='both',axis='x')
        if not hasattr(self,'cutoff_energy'):
            self.find_cutoff()
        # index=self.cutoff_index-5
        _=plt.plot(self.energy,self.DOS,'*-',label=self.name,
                   mfc='none')
        plt.axhline(y=0, color='k',ls='--')
        plt.autoscale(enable=True,axis='both',tight=True)
        plt.legend()
        plt.xlabel('Energy (eV)')
        plt.ylabel('DOS (a.u.)')

    # def DOS_analyze(self,bg=70,plot=False):
    #     if not hasattr(self,'baseline'):
    #         self.find_baseline(plot=False)
    #     if not hasattr(self,'cutoff_index'):
    #         self.find_cutoff()
    #     if self.status['sqrt']:
    #         Int_corr_raw=self.APSdata**2
    #     else:
    #         Int_corr_raw=self.APSdata**3
    #     bg_avg=np.average(Int_corr_raw[:self.cutoff_index])
    #     Int_corr_raw=Int_corr_raw-bg_avg+bg
    #     if self.status['sqrt']:
    #         self.DOS=np.gradient(Int_corr_raw**(1/2),self.energy)
    #     else:
    #         self.DOS=np.gradient(Int_corr_raw**(1/3),self.energy)
    #     if plot:
    #         plt.figure()
    #         self.DOSplot()
    #     self.status['DOS_analyzed']=True
        
    def analyze(self, fit_lower_bound=0.5,fit_upper_bound=np.inf,smoothness=2,
                plot=True):
        if smoothness==1:   gap=5
        elif smoothness==2: gap=7
        else: gap=10
        if not hasattr(self, 'baseline'):
            self.find_baseline(plot=False)
            print('Automatic find baseline between (1,5) for '+self.name)
        start=len(self.energy)-next(i for i,j in enumerate(
            self.APSdata[::-1]-self.baseline) if j<fit_lower_bound)-1
        stop=len(self.energy)-next(i for i,j in enumerate(
            self.APSdata[::-1]-self.baseline) if j<fit_upper_bound)
        self.std_homo=np.inf
        for i,j in [[i,j] for i in range(start,stop) 
                    for j in range(i+gap,stop)]:
            [slope,intercept],[[var_slope,_],[_,var_intercept]]=np.polyfit(
                self.energy[i:j],self.APSdata[i:j]-self.baseline,1,cov=True)
            std_homo=np.sqrt(var_slope/slope**2+var_intercept/intercept**2)
            if std_homo<self.std_homo:
                self.lin_start_index,self.lin_stop_index=i,j
                self.lin_par,self.std_homo=(slope,intercept),std_homo
        if self.std_homo==np.inf:
            plt.figure()
            self.plot()
            raise Exception("Fitting fail!!! Rechoose fitting condition.")
        self.homo=-self.lin_par[1]/self.lin_par[0]
        self.status['analyzed']=True
        if plot:
            fig=plt.figure()
            ax=fig.gca()
            self.plot()
            plt.title(self.name)
            plt.text(.5, .95, 'HOMO=%1.2f\u00b1 %0.3f%%' 
                     %(self.homo,100*self.std_homo), style='italic',
                     bbox={'facecolor': 'yellow', 'alpha': 0.5},
                     horizontalalignment='center',verticalalignment='center',
                     transform=ax.transAxes)
            ax.legend().remove()
        if self.lin_stop_index-self.lin_start_index==gap:
            print(self.name+' is using the minimum number of points\t')
    
    def DOSfit(self,p0):
        # if not hasattr(self,'DOS'):
        #     raise Exception("Calculate DOS with self.DOS_analyze()")
        self.pick_range()
        fit,_=curve_fit(lambda x,scale,c,center: scale*APS.gaussian(x,c,center),self.energy[self.minindex:self.maxindex],self.DOS[self.minindex:self.maxindex],p0)
        plt.figure()
        plt.plot(self.energy[self.minindex:self.maxindex],fit[0]*APS.gaussian(self.energy[self.minindex:self.maxindex],*fit[1:3]),label='fit')
        self.DOSplot()

    # def APSfit(self,p0=[0.12,0.2,5],bounds=([0.1,-0.5,0.01],[0.5,0.5,1e4]),repick=True):
    #     self.p0=p0
    #     self.bounds=bounds
    #     if repick:
    #         self.pick_range()
    #     if not hasattr(self, 'MOenergy'):
    #         self.read_gaussian_MO(np.array(input("Input MOs from Gaussian:\n").split(),'float'))
    #     self.fit_par,_ = curve_fit(lambda x,c,scale,shift: self.apsfun(x,c,scale,self.MOenergy-shift),self.energy[self.minindex:self.maxindex],self.APSdata[self.minindex:self.maxindex],p0=[0.12,5,0.2],bounds=([0.1,0.01,-0.5],[0.5,1e4,0.5]),absolute_sigma=True,ftol=1e-12)
    #     plt.figure()
    #     plt.plot(self.energy,self.APSdata,'o',label='experiment')
    #     self.APSfit=self.apsfun(self.energy,*self.fit_par[:-1],self.MOenergy-self.fit_par[-1])
    #     plt.plot(self.energy,self.APSfit,label='fit: c=%1.4f, shift=%1.4f, scale=%2.1f' %tuple(self.fit_par))
    #     plt.xlabel('Energy (eV)')
    #     plt.ylabel('Photoemission^1/3 (a.u.)')
    #     plt.legend()
    #     plt.title('FitAPS')
    #     print('Broaden facter=%1.4f, shift=%1.4f, scale=%2.1f' %tuple(self.fit_par))
        
    @classmethod
    def import_from_files(cls,filenames,sqrt=False,trunc=-4):
        data=[]
        save_index=[2,6] if sqrt else [2,7] 
        # index of saved column from raw data. 2 is energy and 7 is cuberoot. 
        #6 is square-root.
        for file in filenames:
            with open(file,newline='') as f:
                reader=csv.reader(f)
                for i,j in enumerate(reader):
                    try:
                        if not float(j[3])<1e4:
                            stopindex=i
                            break
                    except:
                        if j[0][:3]==' WF':
                            stopindex=i
                            break
                f.seek(0)
                acceptlines=range(1,stopindex)
                temp=np.array([[float(j[save_index[0]]),
                                float(j[save_index[1]])] 
                               for i,j in enumerate(reader)
                               if i in acceptlines])
            data.append(cls(temp[:,0],temp[:,1],sqrt,split(file)[1][:trunc]))
        return data
    
    @classmethod
    def APS_from_DOS(cls,energy,DOS,sqrt,Name='no_name'):
        APSdata=inv_gradient(energy,DOS)
        APS_obj=cls(energy,APSdata,sqrt,Name)
        APS_obj.find_baseline(plot=False)
        APS_obj.APSdata-=APS_obj.baseline
        return APS_obj

    @staticmethod
    def lc_DOS(data,coeff,cov,sqrt=False,Name='linear_combination'):
        '''
        linear combine the DOS of each data element with coeff.
        ----
        data=[data1,data2,...] where each element is an APS object
        coeff is a list or tuple of n-element [coeff1,coeff2,...]
        fmt has option 'd' (data) or 'o' (object).
        d will output (energy,DOS) and o will output APS_object
        **kwargs are sqrt and/or Name for APS object.
        '''
        assert len(data)==len(coeff), 'Dimension mismatch'
        assert all(data[i].status['sqrt']==data[i+1].status['sqrt'] 
                   for i in range(len(data)-1)),'data has to be the same sqrt type'
        # assert all(data[i].status['DOS_analyzed']==data[i+1].status['DOS_analyzed']
        #            for i in range(len(data)-1)),'DOS_analyze not yet done'
        energy,DOS=[i.energy for i in data],[i.DOS for i in data]
        energy,DOS=find_overlap(energy,DOS)
        DOS=np.dot(coeff,DOS)
        APS_obj=APS.APS_from_DOS(energy,DOS,sqrt,Name)
        APS_obj.lc_source=[i.name for i in data]
        APS_obj.lc_coeff,APS_obj.lc_cov=coeff,cov
        return APS_obj

    @staticmethod
    def lc_DOSfit(source,target,constrain=True):
        '''
        Linear combine multiple DOS from source to fit the DOS of target.
        source is a list of APS objects [APS1,APS2,...] to fit APS obj target.
        '''
        # assert all([hasattr(i,'DOS_analyzed') for i in [*source,target]]
        #            ),'DOS_analyze not yet done'
        if any([not hasattr(i,'cutoff_index') for i in [*source,target]]):
            _=[i.find_cutoff() for i in [*source,target]]
        cutoff_energy=[i.cutoff_energy for i in source]
        index=cutoff_energy.index(min(cutoff_energy))
        energy=[j.energy if i!=index else j.energy[j.cutoff_index:] for i,j
                in enumerate([*source,target])]
        DOS=[j.DOS if i!=index else j.DOS[j.cutoff_index:] for i,j in 
             enumerate([*source,target])]
        energy,DOS=find_overlap(energy,DOS)
        input_DOS=DOS[:-1]
        fit_DOS=DOS[-1]
        if constrain==1:
            fit,cov=curve_fit(lambda x,*c: np.dot((*c,1-sum(c)),x),input_DOS,
                              fit_DOS,p0=[1/len(input_DOS)]*(len(input_DOS)-1),
                          absolute_sigma=True,bounds=(0,np.inf))
            cov=np.diag(cov)
            return np.array([*fit,1-sum(fit)]),cov
        else:
            fit,cov=curve_fit(lambda x,*c: np.dot(c,x),input_DOS,
                              fit_DOS,p0=[1/len(input_DOS)]*len(input_DOS),
                          absolute_sigma=True,bounds=(0,np.inf))
            cov=np.diag(cov)
            return fit,cov
    
    @staticmethod
    def save_aps_csv(data,location,filename='APS'):
        datanames=[i.name for i in data]
        origin_header=[['Energy','Photoemission\\+(1/3)'],['eV','a.u.']] if all([i.status['sqrt']==False for i in data]) else [['Energy','Photoemission\\+(1/2)'],['eV','a.u.']]
        x,y=[i.energy for i in data],[i.APSdata-i.baseline if 
                                      hasattr(i,'baseline') else i.APSdata 
                                      for i in data]
        save_csv_for_origin((x,y),location,filename,datanames,origin_header)

    @staticmethod
    def save_aps_fit_csv(data,location,filename='APS_linear_regression'):
        assert all(i.status['analyzed'] for i in data), 'Input not yet analyzed'
        origin_header=[['Energy','Photoemission\\+(1/3)'],['eV','a.u.']] if all([i.status['sqrt']==False for i in data]) else [['Energy','Photoemission\\+(1/2)'],['eV','a.u.']]
        datanames=[i.name for i in data]
        x=[np.array([i.homo,i.energy[i.lin_stop_index]]) for i in data]
        y=[np.array([0,np.polyval(i.lin_par,i.energy[i.lin_stop_index])]) for i in data]
        save_csv_for_origin((x,y),location,filename,datanames,origin_header)
    
    @staticmethod
    def save_homo_error_csv(data,location,filename='APS_HOMO'):
        assert all(i.status['analyzed'] for i in data), 'Input not yet analyze'
        origin_header=[['Material','Energy','HOMO std'],[None,'eV','eV']]
        datanames=['HOMO']
        x=[[i.name for i in data]]
        y=[[-i.homo for i in data]]
        z=[[i.std_homo*i.homo for i in data]]
        save_csv_for_origin((x,y,z),location,filename,datanames,origin_header)
        
    @staticmethod
    def save_DOS_csv(data,location,filename='DOS'):
        assert all(i.status['DOS_analyzed'] for i in data), 'Input not yet DOS_analyzed'
        origin_header=[['Energy','DOS'],['eV','a.u.']]
        datanames=[i.name for i in data]
        x,y=[i.energy for i in data],[i.DOS for i in data]
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
    allowed_kwargs=[]
    def __init__(self,time,CPDdata,name='no_name',**kwargs):
        self.time=np.array(time)
        self.CPDdata=np.array(CPDdata)
        self.name=name
        self.data_type,self.data_unit='CPD','meV'
        try:
            self.__dict__.update((i,kwargs[i]) for i in self.allowed_kwargs)
        except KeyError:
            raise Exception('expect key words '+','.join(self.allowed_kwargs)
                            + ' missing')
        self.status={'calibrated':False, 'statistics':False}
        
    def __repr__(self):
        return self.name
    
    def __str__(self):
        summary='Name:\t'+self.name+'\n'
        summary+='\n'.join([i+':\t'+str(j) for i,j in self.status.items()])+'\n'
        if self.status['statistics']==True:
            summary+='Statistic region:\tlast '+str(self.length)+\
                's\naverage:\t%1.2f\u00b1%0.2feV\n'%(self.average_CPD
                                                     ,self.std_CPD)
        return summary
    
    def status_check(self):
        report=''
        if self.status['calibrated']!=(self.data_type=='Fermi level'):
            report+='Calibration is corrupted!\nRedo calibration\n'
        if self.status['statistics']!=hasattr(self,'average_CPD'):
            report+='Statistic is corrupted!\nRedo self.dwf_stat('\
                'length)\n'
        if report=='':
            print(self.name+'\n--------\nCheck done and all statuses are '\
                  'good!\n')
        else:
            print(self.name+'\n--------\n'+report)
            
    def plot(self):
        plt.grid(True,which='both',axis='both')
        plt.plot(self.time,self.CPDdata,label=self.name)
        plt.ylabel(self.data_type+' ('+self.data_unit+')')
        plt.legend()
        plt.xlabel('Time(s)')
        plt.autoscale(enable=True,axis='both',tight=True)
            
    def dwf_stat(self,length=200):
        start=next(i-1 for i,j in enumerate(self.time) if 
                   j>self.time[-1]-length)
        self.average_CPD=np.average(self.CPDdata[start:])
        self.std_CPD=np.std(self.CPDdata[start:])
        self.length=length
        self.status['statistics']=True
            
    @classmethod
    def import_from_files(cls,filenames,trunc=-4,**kwargs):
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
                temp=np.array([[float(j[save_index[0]]),
                                float(j[save_index[1]])] 
                               for i,j in enumerate(reader) 
                               if i in acceptlines])
            data.append(cls(temp[:,0],temp[:,1],split(file)[1][:trunc],**kwargs))
        return data
    
    @staticmethod
    def save_csv(data,location,filename='DWF'):
        assert all([data[i].__class__==data[i+1].__class__ for i in range(len(data)-1)]), 'Data are not the same class objects'
        origin_header=[['Time',data[0].data_type],['s',data[0].data_unit]]
        datanames=[i.name for i in data]
        x=[i.time for i in data]
        y=[i.CPDdata for i in data]
        save_csv_for_origin((x,y),location,filename,datanames,origin_header)
    
    @staticmethod
    def save_dwf_stat_csv(data,location,filename='DWF_stat'):
        assert all([data[i].__class__==data[i+1].__class__ for i in range(len(data)-1)]), 'Data are not the same class objects'
        if not all([i.status['statistics'] for i in data]):
            print('Use last 200sec data for statistic analysis')
            _=[i.dwf_stat() for i in data]
        origin_header=[['Material','Energy',data[0].data_type+' std'],[None,data[0].data_unit,data[0].data_unit]]  
        datanames=[data[0].data_type]
        x=[[i.name for i in data]]
        y=[[i.average_CPD for i in data]]
        z=[[i.std_CPD for i in data]]
        save_csv_for_origin((x,y,z),location,filename,datanames,origin_header)
        
class calibrate:
    def __init__(self,ref_APS,ref_dwf):
        assert hasattr(ref_APS,'homo'),'Analyze ref. APS by ref_APS.analyze()'
        assert hasattr(ref_dwf,'average_CPD'), \
            'Find average CPD by ref_dwf.dwf_stat()'
        self.tip_dwf=-ref_APS.homo+ref_dwf.average_CPD/1000
        self.name=(ref_APS.name,ref_dwf.name)
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name
    
    def cal(self,data):
        assert all([i.__class__.__name__=='dwf' for i in data]),\
            'Calibrate only applicable to CPD measurements'
        for i in data:
            i.CPDdata=-i.CPDdata/1000+self.tip_dwf
            i.data_type,i.data_unit='Fermi level','eV'
            i.status['calibrated']=True

class spv(dwf):
    allowed_kwargs=['timemap']
    def __init__(self,time,CPDdata,name='no_name',**kwargs):
        super().__init__(time,CPDdata,name=name,**kwargs)
        self.timeline=np.cumsum(self.timemap)
        self.timeline_index=[next(j-1 for j,k in enumerate(self.time) if k>i)
                             for i in self.timeline[:-1]]
        self.timeline_index.insert(0,0)
        self.timeline_index.append(len(self.time)-1)
        # self.bg_cal=False
        self.data_type,self.data_unit='raw SPV','meV'
        self.status={'background calibrated':False, 'normalized':False}
    
    def __str__(self):
        summary='Name:\t'+self.name+'\n'
        summary+='\n'.join([i+':\t'+str(j) for i,j in self.status.items()])+'\n'
        summary+='timemap:'+str(self.timemap)+'\n'
        return summary
    
    def status_check(self):
        report=''
        if self.status['background calibrated']!=(self.data_type=='SPV'):
            report+='Background calibration is corrupted!\nRedo self.cal_background()\n'
        if self.status['normalized']!=hasattr(self,'norm_spv'):
            report+='Noamlization is corrupted!\nRedo self.normalize('\
                'timezone,plot)\n'
        if report=='':
            print(self.name+'\n--------\nCheck done and all statuses are '\
                  'good!\n')
        else:
            print(self.name+'\n--------\n'+report)
            
    def cal_background(self,plot=False):
        self.bg_cpd=np.average(self.CPDdata[0:self.timeline_index[1]])
        self.CPDdata=self.CPDdata-self.bg_cpd
        self.data_type='SPV'
        self.status['background calibrated']=True
        if plot:
            plt.figure()
            self.plot()
        
    def normalize(self,timezone=1,plot=False):
        if not self.status['background calibrated']:
            self.cal_background()
        self.norm_zone=timezone
        scale_fac=max(abs(self.CPDdata[self.timeline_index[timezone]:self.timeline_index[timezone+1]]))
        self.norm_spv=self.CPDdata/scale_fac
        self.status['normalized']=True
        if plot:
            plt.figure()
            self.norm_plot()
    
    def plot(self):
        dwf.plot(self)
        self.plot_highlight()
        
    def norm_plot(self):
        assert hasattr(self,'norm_spv'),'Didn\'t noramlized yet'
        plt.grid(True,which='both',axis='both')
        plt.plot(self.time,self.norm_spv,label=self.name)
        plt.ylabel('normalized SPV (a.u.)')
        plt.legend()
        plt.xlabel('Time(s)')
        plt.autoscale(enable=True,axis='both',tight=True)
        self.plot_highlight()
        
    def plot_highlight(self):
        for i in range(len(self.timeline)//2):
            plt.axvspan(self.timeline[2*i],self.timeline[2*i+1],color='yellow',alpha=0.5)
    
    @staticmethod
    def save_norm_spv_csv(data,location,filename='Normalized_SPV'):
        assert all([data[i].__class__==data[i+1].__class__ for i in range(len(data)-1)]), 'Data are not the same class objects'
        if not all([hasattr(i,'norm_spv') for i in data]):
            print('Use first light on for normalization')
            _=[i.normalize() for i in data]
        origin_header=[['Time','Normalized SPV'],['s','a.u.']]
        datanames=[i.name for i in data]
        x=[i.time for i in data]
        y=[i.norm_spv for i in data]
        save_csv_for_origin((x,y),location,filename,datanames,origin_header)