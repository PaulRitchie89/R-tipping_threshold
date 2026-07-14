# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:49:17 2026

@author: Paul

Script that produces Fig 15 in "Evaluating the skill of a
geometric early warning for tipping in a rapidly forced nonlinear system"
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import scipy.io as sp
from scipy import stats

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
    
    Hforcing = H0*(t<tstart) + Hrampup*(tstart<=t<(H1-H0)/r1+tstart) + (H1)*((H1-H0)/r1+tstart<=t<(H1-H0)/r1+tpause+tstart) +  Hrampdown*((H1-H0)/r1+tpause+tstart<=t<(r1+r2)*(H1-H0)/(r1*r2)+tpause+tstart) + H0*(t>=(r1+r2)*(H1-H0)/(r1*r2)+tpause+tstart)
    
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
H0 = 0.37#0                                      # Initial forcing
H1 = 0.4#0.38                                   # Final forcing
tup = 100                                   # Ramp up period
tpause = 300                                # Plataeu period
tdown = 200                                 # Ramp down period
tstart = 0                                  # Start of forcing

r1 = (H1-H0)/tup                                 # Rate of forcing increase
r2 = (H1-H0)/tdown                               # Rate of forcing decrease

# Load in geometric early warning data
mat_contents = sp.loadmat('Monte_Carlo_edge_state_dist_'+str(H1).replace('.', 'p')+'_tstart'+str(tstart)+'_tup'+str(tup)+'_tpause'+str(tpause)+'_tdown'+str(tdown)+'_B_tipping.mat')

t = mat_contents['t2'][0]
Notip_traj = mat_contents['Notip_traj']
Tip_traj = mat_contents['Tip_traj']
Notip_mindx = mat_contents['Notip_mindx']
Tip_mindx = mat_contents['Tip_mindx']


wl = 200        # Window length return rate was calculated over
tau_wl = 100    # Time duration that Kendall tau is to be calculated over

# Load in return rate early warning data
mat2_contents = sp.loadmat('Monte_Carlo_EWS_wl'+str(wl)+'_'+str(H1).replace('.', 'p')+'_tstart'+str(tstart)+'_tup'+str(tup)+'_tpause'+str(tpause)+'_tdown'+str(tdown)+'_B_tipping.mat')

t2 = mat2_contents['t2'][0]
Decay2_notip = mat2_contents['Decay2_notip']
Decay2_tip = mat2_contents['Decay2_tip']

# Number of non-tipping cases (same as number of tipping cases) 
N = len(Notip_mindx[0,:])

# Array of thresholds to loop over and compare with scoring classifier to make prediction of tipping or not
SN_thresholds = np.linspace(0.0325,0.036,1001)
ST_thresholds = np.linspace(0.035,0.039,1001)#(0.0355,0.039,1001)
Edge_thresholds = np.linspace(-0.0017,0.0018,1001)
EWS_thresholds = np.linspace(-1,1,1001)

# Set fixed threshold levels in early warning metrics
SN_const_threshold = 0.034
ST_const_threshold = 0.0366
Edge_const_threshold = 0
EWS_const_threshold = 0

# Initialise arrays
tau_notip = np.zeros((len(t),len(Decay2_notip[0,:])))
tau_tip = np.zeros((len(t),len(Decay2_tip[0,:])))

SN_AUC = np.zeros(len(t)-1)
ST_AUC = np.zeros(len(t)-1)
Edge_AUC = np.zeros(len(t)-1)
EWS_AUC = np.zeros(len(t)-1)

SN_OptThr = np.zeros(len(t)-1)
ST_OptThr = np.zeros(len(t)-1)
Edge_OptThr = np.zeros(len(t)-1)
EWS_OptThr = np.zeros(len(t)-1)

Forcing_profile = np.zeros(len(t)-1)


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
    
    # If enough time has elapsed, calculate kendall tau on square of decay rate for all tipping & non-tipping trajectories
    if j >= wl+tau_wl:
        for k in range(len(Decay2_notip[0,:])):
            tau_notip[j,k],_ = stats.kendalltau(np.arange(tau_wl), Decay2_notip[j-tau_wl:j,k])
            tau_tip[j,k],_ = stats.kendalltau(np.arange(tau_wl), Decay2_tip[j-tau_wl:j,k])
    
    # Loop over different thresholds and calculate false positive and true positive 
    # rates for geometric EWS (comparing to score classifier) 
    for i in range(len(SN_thresholds)):
        SN_FPR[i] = np.sum(Notip_traj[0,j,:]<SN_thresholds[i])/N
        SN_TPR[i] = np.sum(Tip_traj[0,j,:]<SN_thresholds[i])/N
        
        ST_FPR[i] = np.sum(Notip_traj[1,j,:]>ST_thresholds[i])/N
        ST_TPR[i] = np.sum(Tip_traj[1,j,:]>ST_thresholds[i])/N
        
        Edge_FPR[i] = np.sum(Notip_mindx[j,:]<Edge_thresholds[i])/N
        Edge_TPR[i] = np.sum(Tip_mindx[j,:]<Edge_thresholds[i])/N
        
        # and after sufficient time square of decay rate
        if j >= wl+tau_wl:            
            EWS_FPR[i] = np.sum(tau_notip[j,:]<EWS_thresholds[i])/N
            EWS_TPR[i] = np.sum(tau_tip[j,:]<EWS_thresholds[i])/N
        else:
            EWS_FPR[i], EWS_TPR[i] = np.nan, np.nan
    

    
    # Integrate area under ROC to get AUC   
    SN_AUC[j] = integrate(SN_FPR, SN_TPR)
    ST_AUC[j] = integrate(ST_FPR[::-1], ST_TPR[::-1])
    Edge_AUC[j] = integrate(Edge_FPR, Edge_TPR)
    EWS_AUC[j] = integrate(EWS_FPR, EWS_TPR)
    
    
    ## Identify optimal threshold as threshold that gives closest value to (FPR,TPR) = (0,1) for SN
    minimum=min(np.sqrt(SN_FPR**2+(SN_TPR-1)**2))
    index=[idx for idx,val in enumerate(np.sqrt(SN_FPR**2+(SN_TPR-1)**2)) if val==minimum]
    if len(index) == 1:
        SN_idx = index[0]
    else:
        SN_idx = index[int(len(index)/2)]

    # Identify value of optimal threshold for SN
    SN_OptThr[j] = SN_thresholds[SN_idx]  
    
    
    ## Identify optimal threshold as threshold that gives closest value to (FPR,TPR) = (0,1) for ST
    minimum=min(np.sqrt(ST_FPR**2+(ST_TPR-1)**2))
    index=[idx for idx,val in enumerate(np.sqrt(ST_FPR**2+(ST_TPR-1)**2)) if val==minimum]
    if len(index) == 1:
        ST_idx = index[0]
    else:
        ST_idx = index[int(len(index)/2)]
    
    # Identify value of optimal threshold for ST
    ST_OptThr[j] = ST_thresholds[ST_idx]

    
    ## Identify optimal threshold as threshold that gives closest value to (FPR,TPR) = (0,1) for R-tipping threshold
    minimum=min(np.sqrt(Edge_FPR**2+(Edge_TPR-1)**2))
    index=[idx for idx,val in enumerate(np.sqrt(Edge_FPR**2+(Edge_TPR-1)**2)) if val==minimum]
    if len(index) == 1:
        Edge_idx = index[0]
    else:
        Edge_idx = index[int(len(index)/2)]
    
    # Identify value of optimal threshold of R-tipping indicator
    Edge_OptThr[j] = Edge_thresholds[Edge_idx]    
  

    # If AUC = 1 then make sure the optimal threshold does not change for future times
    if j>1:
        if SN_AUC[j] == 1:
            idx_SN_check = np.argmin(np.abs(SN_OptThr[j-1]-SN_thresholds))
            for k in range(idx_SN_check):
                if SN_FPR[idx_SN_check-k] == 0 and SN_TPR[idx_SN_check-k] == 1:
                    SN_OptThr[j] = SN_thresholds[idx_SN_check-k]
                    break
        if ST_AUC[j] == 1:
            idx_ST_check = np.argmin(np.abs(ST_OptThr[j-1]-ST_thresholds))
            for k in range(idx_ST_check):
                if ST_FPR[idx_ST_check-k] == 0 and ST_TPR[idx_ST_check-k] == 1:
                    ST_OptThr[j] = ST_thresholds[idx_ST_check-k]
                    break
        if Edge_AUC[j] == 1:
            idx_edge_check = np.argmin(np.abs(Edge_OptThr[j-1]-Edge_thresholds))
            for k in range(idx_edge_check):
                if Edge_FPR[idx_edge_check-k] == 0 and Edge_TPR[idx_edge_check-k] == 1:
                    Edge_OptThr[j] = Edge_thresholds[idx_edge_check-k]
                    break


# Plot Fig. 15
fig4,ax4 = plt.subplots(3,1,sharex=True)
ax4[0].plot(t[:-1],Forcing_profile,'k')
ax4[1].plot(t[:-1],SN_AUC,label='$S_N$')
ax4[1].plot(t[:-1],ST_AUC,label='$S_T$')
ax4[1].plot(t[:-1],Edge_AUC,label='R-tipping indicator')
ax4[1].plot(t[:-1],EWS_AUC,label=r'$\alpha^2$')
ax4[2].plot(t[1:-1],SN_OptThr[1:]-SN_OptThr[-1],label='$S_N$')
ax4[2].plot(t[1:-1],ST_OptThr[1:]-ST_OptThr[-1],label='$S_T$')
ax4[2].plot(t[1:-1],Edge_OptThr[1:]-Edge_OptThr[-1],label='R-tipping indicator')
ax4[0].set_xlim(0,1000)
ax4[0].set_ylabel('Freshwater hosing (Sv)')
ax4[1].set_ylabel('Area under ROC')
ax4[2].set_xlabel('Time (years)')
ax4[2].set_ylabel('Difference to final\noptimal threshold')
sns.despine()
ax4[1].legend(frameon=False,ncol=4,loc='lower center', bbox_to_anchor=(0.5, 2.15))
