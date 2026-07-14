# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:52:56 2026

@author: Paul

Script that generates bootstrapping data for robustness check of early warning
metrics in "Evaluating the skill of a geometric early warning for tipping in a
rapidly forced nonlinear system"
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import scipy.io as sp
from scipy import stats
import scipy

fontsize = 12
rc('font', **{'size' : fontsize})
rc('text', usetex=True)


def H(t,H0,H1,r1,r2,tpause,tstart):
    """
    Piecewise linear freshwater hosing
    
    Parameters
    ----------
    t : float64
        Time
    H0 : float64
        Initial freshwater hosing level.
    H1 : float64
        Final freshwater hosing level.
    r1 : float64
        Rate of forcing increase
    r2 : float64
        Rate of forcing decrease
    tpause : float64
        Plataeu period
    tstart : float64
        Start time of forcing

    Returns
    -------
    Hforcing : float64
        Piecewise linear freshwater hosing profile.
    """
    Hrampup = (H0 + r1*(t-tstart))
    Hrampdown = (H1 - r2*(t-(H1-H0)/r1-tpause-tstart))
    
    Hforcing = H0*(t<tstart) + Hrampup*(tstart<=t<(H1-H0)/r1+tstart) + (H0 + H1)*((H1-H0)/r1+tstart<=t<(H1-H0)/r1+tpause+tstart) +  Hrampdown*((H1-H0)/r1+tpause+tstart<=t<(r1+r2)*(H1-H0)/(r1*r2)+tpause+tstart) + H0*(t>=(r1+r2)*(H1-H0)/(r1*r2)+tpause+tstart)
    return(Hforcing)

def integrate(x, y):
    """
    Apply trapezoidal rule to integrate y with respect to x    
    Parameters
    ----------
    x : Array of float64
    y : Array of float64

    Returns
    -------
    sm : float64
          Integral of y with respect to x

    """
    sm = 0
    for i in range(1, len(x)):
        # Step spacing
        h = x[i] - x[i-1]
        # Trapezoidal rule
        sm += h * (y[i-1] + y[i]) / 2
 
    return sm


# Forcing parameters
H0 = 0                                      # Initial forcing
H1 = 0.38                                   # Final forcing
tup = 100                                   # Ramp up period
tpause = 300                                # Plataeu period
tdown = 200                                 # Ramp down period
tstart = 0                                  # Start of forcing

r1 = H1/tup                                 # Rate of forcing increase
r2 = H1/tdown                               # Rate of forcing decrease

# Thresholds for max FPR and min TPR
FPR_thr = 0.05
TPR_thr = 0.95

# Load in geometric early warning data
mat_contents = sp.loadmat('Monte_Carlo_edge_state_dist_'+str(H1).replace('.', 'p')+'_tstart'+str(tstart)+'_tup'+str(tup)+'_tpause'+str(tpause)+'_tdown'+str(tdown)+'.mat')

t = mat_contents['t2'][0]
Notip_traj = mat_contents['Notip_traj']
Tip_traj = mat_contents['Tip_traj']
Notip_mindx = mat_contents['Notip_mindx']
Tip_mindx = mat_contents['Tip_mindx']


wl = 200        # Window length return rate was calculated over
tau_wl = 100    # Time duration that Kendall tau is to be calculated over

# Load in return rate early warning data
mat2_contents = sp.loadmat('Monte_Carlo_EWS_wl'+str(wl)+'_'+str(H1).replace('.', 'p')+'_tstart'+str(tstart)+'_tup'+str(tup)+'_tpause'+str(tpause)+'_tdown'+str(tdown)+'.mat')

t2 = mat2_contents['t2'][0]
Decay2_notip = mat2_contents['Decay2_notip']
Decay2_tip = mat2_contents['Decay2_tip']

# Number of non-tipping cases (same as number of tipping cases) 
N = len(Notip_mindx[0,:])

# Array of thresholds to loop over and compare with scoring classifier to make prediction of tipping or not
SN_thresholds = np.linspace(0.0325,0.036,1001)
ST_thresholds = np.linspace(0.0355,0.039,1001)
Edge_thresholds = np.linspace(-0.0017,0.0018,1001)
EWS_thresholds = np.linspace(-1,1,1001)

np.random.seed(1)
B = 1000

Decay2_notip_boot = np.zeros((1201,N))
Decay2_tip_boot = np.zeros((1201,N))
Notip_traj_boot = np.zeros((2,1201,N))
Tip_traj_boot = np.zeros((2,1201,N))
Notip_mindx_boot = np.zeros((1200,N))
Tip_mindx_boot = np.zeros((1200,N))

SN_AUC = np.zeros((len(t)-1,B))
ST_AUC = np.zeros((len(t)-1,B))
Edge_AUC = np.zeros((len(t)-1,B))
EWS_AUC = np.zeros((len(t)-1,B))

for b in range(B):
    N_boot = np.random.choice(np.linspace(0,N-1,N),size=N,replace=True)
    
    print(b)
    
    idx = N_boot.astype(int)

    Decay2_notip_boot = Decay2_notip[:, idx]
    Decay2_tip_boot   = Decay2_tip[:, idx]
    
    Notip_traj_boot   = Notip_traj[:, :, idx]
    Tip_traj_boot     = Tip_traj[:, :, idx]
    
    Notip_mindx_boot  = Notip_mindx[:, idx]
    Tip_mindx_boot    = Tip_mindx[:, idx]
        
    # Initialise arrays
    tau_notip = np.zeros((len(t),len(Decay2_notip_boot[0,:])))
    tau_tip = np.zeros((len(t),len(Decay2_tip_boot[0,:])))
    
    
    Forcing_profile = np.zeros(len(t)-1)
    
    # Initialise counter
    count = 0
    
    for j in range(len(t)-1):
        
        Forcing_profile[j] = H(t[j],H0,H1,r1,r2,tpause,tstart)
        
        # Initialise arrays for false positive and true positive rates for square of decay rate
        SN_FPR = np.zeros(len(SN_thresholds))
        SN_TPR = np.zeros(len(SN_thresholds))
        ST_FPR = np.zeros(len(ST_thresholds))
        ST_TPR = np.zeros(len(ST_thresholds))
        Edge_FPR = np.zeros(len(Edge_thresholds))
        Edge_TPR = np.zeros(len(Edge_thresholds))
        EWS_FPR = np.zeros(len(Edge_thresholds))
        EWS_TPR = np.zeros(len(Edge_thresholds))
    
        notip_sorted = np.sort(Notip_traj_boot[0,j,:])
        tip_sorted   = np.sort(Tip_traj_boot[0,j,:])        
        SN_FPR = np.searchsorted(notip_sorted,SN_thresholds,side='right')/N        
        SN_TPR = np.searchsorted(tip_sorted,SN_thresholds,side='right')/N
        
        notip_sorted = np.sort(Notip_traj_boot[1,j,:])
        tip_sorted   = np.sort(Tip_traj_boot[1,j,:])        
        ST_FPR = np.searchsorted(notip_sorted,ST_thresholds,side='left')/N        
        ST_TPR = np.searchsorted(tip_sorted,ST_thresholds,side='left')/N
        
        notip_sorted = np.sort(Notip_mindx_boot[j,:])
        tip_sorted   = np.sort(Tip_mindx_boot[j,:])        
        Edge_FPR = np.searchsorted(notip_sorted,Edge_thresholds,side='right')/N        
        Edge_TPR = np.searchsorted(tip_sorted,Edge_thresholds,side='right')/N
        
        if j >= wl + tau_wl:
            
            tvec = np.arange(tau_wl)
            tvec = (tvec - tvec.mean()) / tvec.std()
            
            window_notip = Decay2_notip_boot[j-tau_wl:j, :]
            window_tip   = Decay2_tip_boot[j-tau_wl:j, :]
            
            Xn = window_notip - window_notip.mean(axis=0)
            Xn /= window_notip.std(axis=0)
            
            Xt = window_tip - window_tip.mean(axis=0)
            Xt /= window_tip.std(axis=0)
            
            tau_notip[j,:] = (tvec[:,None] * Xn).mean(axis=0)
            tau_tip[j,:]   = (tvec[:,None] * Xt).mean(axis=0)
                
            tau_notip_sorted = np.sort(tau_notip[j,:])
            tau_tip_sorted   = np.sort(tau_tip[j,:])            
            EWS_FPR = np.searchsorted(tau_notip_sorted,EWS_thresholds,side='right')/N
            EWS_TPR = np.searchsorted(tau_tip_sorted,EWS_thresholds,side='right')/N
        else:
            EWS_FPR = np.full(len(EWS_thresholds), np.nan)
            EWS_TPR = np.full(len(EWS_thresholds), np.nan)
    
        
        # Integrate area under ROC to get AUC   
        SN_AUC[j,b] = integrate(SN_FPR, SN_TPR)
        ST_AUC[j,b] = integrate(ST_FPR, ST_TPR)
        Edge_AUC[j,b] = integrate(Edge_FPR, Edge_TPR)
        EWS_AUC[j,b] = integrate(EWS_FPR, EWS_TPR)
    
    
scipy.io.savemat('AUC_bootstrap_wl'+str(wl)+'_'+str(H1).replace('.', 'p')+'_tstart'+str(tstart)+'_tup'+str(tup)+'_tpause'+str(tpause)+'_tdown'+str(tdown)+'.mat', {'SN_AUC_boot':SN_AUC, 'ST_AUC_boot':ST_AUC, 'Edge_AUC_boot':Edge_AUC, 'EWS_AUC_boot':EWS_AUC})




