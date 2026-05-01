# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 17:05:44 2025

@author: Paul

Script that produces Figs 4, 8, 9, 10, 11, 12 in "Evaluating the skill of a
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

# Set fixed threshold levels in early warning metrics
SN_const_threshold = 0.034
ST_const_threshold = 0.0366
Edge_const_threshold = 0
EWS_const_threshold = 0

# Initialise arrays
tau_notip = np.zeros((len(t),len(Decay2_notip[0,:])))
tau_tip = np.zeros((len(t),len(Decay2_tip[0,:])))

SN_const_FPR = np.zeros(len(t)-1)
SN_const_TPR = np.zeros(len(t)-1)
ST_const_FPR = np.zeros(len(t)-1)
ST_const_TPR = np.zeros(len(t)-1)
Edge_const_FPR = np.zeros(len(t)-1)
Edge_const_TPR = np.zeros(len(t)-1)
EWS_const_FPR = np.zeros(len(t)-1)
EWS_const_TPR = np.zeros(len(t)-1)

SN_AUC = np.zeros(len(t)-1)
ST_AUC = np.zeros(len(t)-1)
Edge_AUC = np.zeros(len(t)-1)
EWS_AUC = np.zeros(len(t)-1)

SN_OptThr = np.zeros(len(t)-1)
SN_Opt_FPR = np.zeros(len(t)-1)
SN_Opt_TPR = np.zeros(len(t)-1)
SN_minFPR = np.zeros(len(t)-1)
SN_maxTPR = np.zeros(len(t)-1)

ST_OptThr = np.zeros(len(t)-1)
ST_Opt_FPR = np.zeros(len(t)-1)
ST_Opt_TPR = np.zeros(len(t)-1)
ST_minFPR = np.zeros(len(t)-1)
ST_maxTPR = np.zeros(len(t)-1)

Edge_OptThr = np.zeros(len(t)-1)
Edge_Opt_FPR = np.zeros(len(t)-1)
Edge_Opt_TPR = np.zeros(len(t)-1)
Edge_minFPR = np.zeros(len(t)-1)
Edge_maxTPR = np.zeros(len(t)-1)

EWS_OptThr = np.zeros(len(t)-1)
EWS_Opt_FPR = np.zeros(len(t)-1)
EWS_Opt_TPR = np.zeros(len(t)-1)
EWS_minFPR = np.zeros(len(t)-1)
EWS_maxTPR = np.zeros(len(t)-1)


Forcing_profile = np.zeros(len(t)-1)

# Set time intervals to plot ROC curves
times = np.linspace(100+tstart,400+tstart,4)

# Initialise ROC figure (Fig. 9)
fig3,ax3 = plt.subplots(2,2,sharex=True,sharey=True)

ax3[0,0].set_xlim(0,1)
ax3[0,0].set_ylim(0,1)
ax3[1,0].set_xlabel('False positive rate')
ax3[1,1].set_xlabel('False positive rate')
ax3[0,0].set_ylabel('True positive rate')
ax3[1,0].set_ylabel('True positive rate')
sns.despine()

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
    
    # Same but now for a fixed threshold 
    SN_const_FPR[j] = np.sum(Notip_traj[0,j,:]<SN_const_threshold)/N
    SN_const_TPR[j] = np.sum(Tip_traj[0,j,:]<SN_const_threshold)/N
    
    ST_const_FPR[j] = np.sum(Notip_traj[1,j,:]>ST_const_threshold)/N
    ST_const_TPR[j] = np.sum(Tip_traj[1,j,:]>ST_const_threshold)/N
    
    Edge_const_FPR[j] = np.sum(Notip_mindx[j,:]<Edge_const_threshold)/N
    Edge_const_TPR[j] = np.sum(Tip_mindx[j,:]<Edge_const_threshold)/N
    
    if j >= wl+tau_wl:
        EWS_const_FPR[j] = np.sum(tau_notip[j,:]<EWS_const_threshold)/N
        EWS_const_TPR[j] = np.sum(tau_tip[j,:]<EWS_const_threshold)/N
    else:
        EWS_const_FPR[j], EWS_const_TPR[j] = np.nan, np.nan
    
    # Plot ROC curves at specified time intervals (Fig. 9)
    if t[j] in times:
        ax3[int(count/2),count%2].set_title('Time = '+str(int(t[j]))+' years')
        ax3[int(count/2),count%2].plot(SN_FPR,SN_TPR,label='$S_N$')
        ax3[int(count/2),count%2].plot(ST_FPR,ST_TPR,label='$S_T$')
        ax3[int(count/2),count%2].plot(Edge_FPR,Edge_TPR,label='R-tipping indicator')
        ax3[int(count/2),count%2].plot(EWS_FPR,EWS_TPR,label=r'$\alpha^2$')
        count += 1
    
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

    # Identify value of optimal threshold and FPR & TPR for optimal threshold in SN
    SN_OptThr[j] = SN_thresholds[SN_idx]
    SN_Opt_FPR[j] = SN_FPR[SN_idx]
    SN_Opt_TPR[j] = SN_TPR[SN_idx]
    
    # Calculate TPR/FPR for max error of the other type for SN
    SN_maxTPR[j] = SN_TPR[SN_FPR<=FPR_thr][np.argmax(SN_thresholds[SN_FPR<=FPR_thr])]
    SN_minFPR[j] = SN_FPR[SN_TPR>=TPR_thr][np.argmin(SN_thresholds[SN_TPR>=TPR_thr])]    
    
    
    ## Identify optimal threshold as threshold that gives closest value to (FPR,TPR) = (0,1) for ST
    minimum=min(np.sqrt(ST_FPR**2+(ST_TPR-1)**2))
    index=[idx for idx,val in enumerate(np.sqrt(ST_FPR**2+(ST_TPR-1)**2)) if val==minimum]
    if len(index) == 1:
        ST_idx = index[0]
    else:
        ST_idx = index[int(len(index)/2)]
    
    # Identify value of optimal threshold and FPR & TPR for optimal threshold in ST
    ST_OptThr[j] = ST_thresholds[ST_idx]
    ST_Opt_FPR[j] = ST_FPR[ST_idx]
    ST_Opt_TPR[j] = ST_TPR[ST_idx]
    
    # Calculate TPR/FPR for max error of the other type for ST
    ST_maxTPR[j] = ST_TPR[ST_FPR<=FPR_thr][np.argmin(ST_thresholds[ST_FPR<=FPR_thr])]
    ST_minFPR[j] = ST_FPR[ST_TPR>=TPR_thr][np.argmax(ST_thresholds[ST_TPR>=TPR_thr])]

    
    ## Identify optimal threshold as threshold that gives closest value to (FPR,TPR) = (0,1) for R-tipping threshold
    minimum=min(np.sqrt(Edge_FPR**2+(Edge_TPR-1)**2))
    index=[idx for idx,val in enumerate(np.sqrt(Edge_FPR**2+(Edge_TPR-1)**2)) if val==minimum]
    if len(index) == 1:
        Edge_idx = index[0]
    else:
        Edge_idx = index[int(len(index)/2)]
    
    # Identify value of optimal threshold and FPR & TPR for optimal threshold of R-tipping threshold
    Edge_OptThr[j] = Edge_thresholds[Edge_idx]
    Edge_Opt_FPR[j] = Edge_FPR[Edge_idx]
    Edge_Opt_TPR[j] = Edge_TPR[Edge_idx]
    
    # Calculate TPR/FPR for max error of the other type for R-tipping threshold
    Edge_maxTPR[j] = Edge_TPR[Edge_FPR<=FPR_thr][np.argmax(Edge_thresholds[Edge_FPR<=FPR_thr])]
    Edge_minFPR[j] = Edge_FPR[Edge_TPR>=TPR_thr][np.argmin(Edge_thresholds[Edge_TPR>=TPR_thr])]
    
    
    ## Same calculations for square of return rate provided enough time has elapsed
    if j >= wl+tau_wl:
        minimum=min(np.sqrt(EWS_FPR**2+(EWS_TPR-1)**2))
        index=[idx for idx,val in enumerate(np.sqrt(EWS_FPR**2+(EWS_TPR-1)**2)) if val==minimum]
        if len(index) == 1:
            EWS_idx = index[0]
        else:
            EWS_idx = index[int(len(index)/2)]
        EWS_OptThr[j] = EWS_thresholds[EWS_idx]
        EWS_Opt_FPR[j] = EWS_FPR[EWS_idx]
        EWS_Opt_TPR[j] = EWS_TPR[EWS_idx]
        EWS_maxTPR[j] = EWS_TPR[EWS_FPR<=FPR_thr][np.argmax(EWS_thresholds[EWS_FPR<=FPR_thr])]
        EWS_minFPR[j] = EWS_FPR[EWS_TPR>=TPR_thr][np.argmin(EWS_thresholds[EWS_TPR>=TPR_thr])]
    else:
        EWS_OptThr[j], EWS_Opt_FPR[j], EWS_Opt_TPR[j], EWS_maxTPR[j], EWS_minFPR[j] = np.nan, np.nan, np.nan, np.nan, np.nan
    
  

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
        if EWS_AUC[j] == 1:
            idx_edge_check = np.argmin(np.abs(EWS_OptThr[j-1]-EWS_thresholds))
            for k in range(idx_edge_check):
                if EWS_FPR[idx_edge_check-k] == 0 and EWS_TPR[idx_edge_check-k] == 1:
                    EWS_OptThr[j] = EWS_thresholds[idx_edge_check-k]
                    break



ax3[1,1].legend(frameon=False)

# Plot Fig. 4
fig,ax = plt.subplots(3,1,sharex=True)
ax[0].plot(t[:-1],Forcing_profile,'k')
ax[1].plot(t,Notip_traj[0,:,:],'b',alpha=0.1)
ax[1].plot(t,Tip_traj[0,:,:],'r',alpha=0.1)
ax[2].plot(t2,Decay2_notip,'b',alpha=0.1)
ax[2].plot(t2,Decay2_tip,'r',alpha=0.1)
ax[0].set_xlim(0,1000)
ax[0].set_ylabel('Freshwater hosing (Sv)')
ax[1].set_ylabel('$S_N$')
ax[2].set_xlabel('Time (years)')
ax[2].set_ylabel(r'$\alpha^2$')
sns.despine()

# Plot Fig. 8
fig,ax = plt.subplots(4,1,sharex=True)
ax[0].plot(t[:-1],Forcing_profile,'k')
ax[1].plot(t,Notip_traj[0,:,:],'b',alpha=0.1)
ax[1].plot(t,Tip_traj[0,:,:],'r',alpha=0.1)
ax[2].plot(t2,Notip_traj[1,:,:],'b',alpha=0.1)
ax[2].plot(t2,Tip_traj[1,:,:],'r',alpha=0.1)
ax[3].plot(t2[1:],Notip_mindx,'b',alpha=0.1)
ax[3].plot(t2[1:],Tip_mindx,'r',alpha=0.1)
ax[0].set_xlim(0,1000)
ax[0].set_ylabel('Freshwater\nhosing (Sv)')
ax[1].set_ylabel(r'$S_N$')
ax[2].set_ylabel(r'$S_T$')
ax[3].set_xlabel('Time (years)')
ax[3].set_ylabel('R-tipping indicator')
sns.despine()


ax[1].plot(t[1:-1],SN_OptThr[1:],'k')
ax[2].plot(t[1:-1],ST_OptThr[1:],'k')
ax[3].plot(t[1:-1],Edge_OptThr[1:],'k')
ax[1].plot([t[0],t[-1]],[SN_const_threshold,SN_const_threshold],'k--',alpha=0.5)
ax[2].plot([t[0],t[-1]],[ST_const_threshold,ST_const_threshold],'k--',alpha=0.5)
ax[3].plot([t[0],t[-1]],[Edge_const_threshold,Edge_const_threshold],'k--',alpha=0.5)

# Plot Fig. 10
fig4,ax4 = plt.subplots(3,1,sharex=True)
ax4[0].plot(t[:-1],Forcing_profile,'k')
ax4[1].plot(t[:-1],SN_AUC)
ax4[1].plot(t[:-1],ST_AUC)
ax4[1].plot(t[:-1],Edge_AUC)
ax4[1].plot(t[:-1],EWS_AUC,label=r'$\alpha^2$')
ax4[2].plot(t[1:-1],SN_OptThr[1:]-SN_OptThr[-1],label='$S_N$')
ax4[2].plot(t[1:-1],ST_OptThr[1:]-ST_OptThr[-1],label='$S_T$')
ax4[2].plot(t[1:-1],Edge_OptThr[1:]-Edge_OptThr[-1],label='R-tipping indicator')
ax4[0].set_xlim(0,1000)
ax4[0].set_ylabel('Freshwater hosing (Sv)')
ax4[1].set_ylabel('Area under ROC')
ax4[1].legend(frameon=False)
ax4[2].legend(frameon=False)
ax4[2].set_xlabel('Time (years)')
ax4[2].set_ylabel('Difference to final\noptimal threshold')
sns.despine()

# Plot Fig. 11 (left)
fig2,ax2 = plt.subplots(3,1,sharex=True)
ax2[0].plot(t[:-1],1-SN_Opt_FPR,'tab:blue',ls=':',label='Specificity')
ax2[0].plot(t[:-1],SN_Opt_TPR,'tab:blue',label='Sensitivity',alpha=0.2)
ax2[0].plot(t[:-1],1-ST_Opt_FPR,'tab:orange',ls=':')
ax2[0].plot(t[:-1],ST_Opt_TPR,'tab:orange',alpha=0.4)
ax2[0].plot(t[:-1],1-Edge_Opt_FPR,'tab:green',ls=':')
ax2[0].plot(t[:-1],Edge_Opt_TPR,'tab:green',alpha=0.4)
ax2[0].plot(t[:-1],1-EWS_Opt_FPR,'tab:red',ls=':')
ax2[0].plot(t[:-1],EWS_Opt_TPR,'tab:red',alpha=0.4)
ax2[0].set_ylabel('Specificity/Sensitivity')
ax2[0].legend(frameon=False)
SN_FOR = (1-SN_Opt_TPR)/(2-SN_Opt_TPR-SN_Opt_FPR)
ST_FOR = (1-ST_Opt_TPR)/(2-ST_Opt_TPR-ST_Opt_FPR)
Edge_FOR = (1-Edge_Opt_TPR)/(2-Edge_Opt_TPR-Edge_Opt_FPR)
EWS_FOR = (1-EWS_Opt_TPR)/(2-EWS_Opt_TPR-EWS_Opt_FPR)
SN_Informedness = SN_Opt_TPR-SN_Opt_FPR
ST_Informedness = ST_Opt_TPR-ST_Opt_FPR
Edge_Informedness = Edge_Opt_TPR-Edge_Opt_FPR
EWS_Informedness = EWS_Opt_TPR-EWS_Opt_FPR
ax2[1].plot(t[:-1],SN_FOR,label='Optimal $S_N$ threshold')
ax2[1].plot(t[:-1],ST_FOR,label='Optimal $S_T$ threshold')
ax2[1].plot(t[:-1],Edge_FOR,label='Optimal R-tipping indicator')
ax2[1].plot(t[:-1],EWS_FOR,label='Optimal tau threshold')
ax2[2].plot(t[:-1],SN_Informedness)
ax2[2].plot(t[:-1],ST_Informedness)
ax2[2].plot(t[:-1],Edge_Informedness)
ax2[2].plot(t[:-1],EWS_Informedness)
ax2[0].set_xlim(0,1000)
ax2[1].set_ylabel('False Omission Rate')
ax2[1].legend(frameon=False)
ax2[2].set_ylabel('Informedness')
ax2[2].set_xlabel('Time (years)')
sns.despine()

# Plot Fig. 11 (right)
fig2,ax2 = plt.subplots(3,1,sharex=True)
ax2[0].plot(t[:-1],1-SN_const_FPR,'tab:blue',ls=':',label='Specificity')
ax2[0].plot(t[:-1],SN_const_TPR,'tab:blue',label='Sensitivity',alpha=0.2)
ax2[0].plot(t[:-1],1-ST_const_FPR,'tab:orange',ls=':')
ax2[0].plot(t[:-1],ST_const_TPR,'tab:orange',alpha=0.4)
ax2[0].plot(t[:-1],1-Edge_const_FPR,'tab:green',ls=':')
ax2[0].plot(t[:-1],Edge_const_TPR,'tab:green',alpha=0.4)
ax2[0].plot(t[:-1],1-EWS_const_FPR,'tab:red',ls=':')
ax2[0].plot(t[:-1],EWS_const_TPR,'tab:red',alpha=0.4)
ax2[0].set_ylabel('Specificity/Sensitivity')
ax2[0].legend(frameon=False)
SN_FOR = (1-SN_const_TPR)/(2-SN_const_TPR-SN_const_FPR)
ST_FOR = (1-ST_const_TPR)/(2-ST_const_TPR-ST_const_FPR)
Edge_FOR = (1-Edge_const_TPR)/(2-Edge_const_TPR-Edge_const_FPR)
EWS_FOR = (1-EWS_const_TPR)/(2-EWS_const_TPR-EWS_const_FPR)
SN_Informedness = SN_const_TPR-SN_const_FPR
ST_Informedness = ST_const_TPR-ST_const_FPR
Edge_Informedness = Edge_const_TPR-Edge_const_FPR
EWS_Informedness = EWS_const_TPR-EWS_const_FPR
ax2[1].plot(t[:-1],SN_FOR,label='Fixed $S_N$ threshold')
ax2[1].plot(t[:-1],ST_FOR,label='Fixed $S_T$ threshold')
ax2[1].plot(t[:-1],Edge_FOR,label='Fixed R-tipping indicator')
ax2[1].plot(t[:-1],EWS_FOR,label='Fixed tau threshold')
ax2[2].plot(t[:-1],SN_Informedness)
ax2[2].plot(t[:-1],ST_Informedness)
ax2[2].plot(t[:-1],Edge_Informedness)
ax2[2].plot(t[:-1],EWS_Informedness)
ax2[0].set_xlim(0,1000)
ax2[1].set_ylabel('False Omission Rate')
ax2[1].legend(frameon=False)
ax2[2].set_ylabel('Informedness')
ax2[2].set_xlabel('Time (years)')
sns.despine()


# Plot Fig. 12
fig5,ax5 = plt.subplots(2,1,sharex=True)
ax5[0].plot([t[0],t[-1]],[TPR_thr,TPR_thr],'k',ls=':',label='Sensitivity')
ax5[0].plot(t[:-1],1-SN_minFPR,label='$S_N$ specificity')
ax5[0].plot(t[:-1],1-ST_minFPR,label='$S_T$ specificity')
ax5[0].plot(t[:-1],1-Edge_minFPR,label='R-tipping indicator specificity')
ax5[0].plot(t[:-1],1-EWS_minFPR,label='$\\alpha^2$ specificity')
ax5[1].plot([t[0],t[-1]],[1-FPR_thr,1-FPR_thr],'k',ls=':',label='Specificity')
ax5[1].plot(t[:-1],SN_maxTPR,label='$S_N$ sensitivity')
ax5[1].plot(t[:-1],ST_maxTPR,label='$S_T$ sensitivity')
ax5[1].plot(t[:-1],Edge_maxTPR,label='R-tipping indicator sensitivty')
ax5[1].plot(t[:-1],EWS_maxTPR,label='$\\alpha^2$ sensitivty')
ax5[0].set_ylabel('Specificity/Sensitivity')
ax5[0].legend(frameon=False)
ax5[1].set_ylabel('Specificity/Sensitivity')
ax5[1].legend(frameon=False)
ax5[1].set_xlabel('Time (years)')
ax5[0].set_xlim(0,1000)
sns.despine()

