# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:51:44 2026

@author: Paul

Script that produces right panel of Fig 14 in "Evaluating the skill of a
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


wl = [50,100,200,400]        # Window lengths return rate was calculated over
tau_wl = 100    # Time duration that Kendall tau is to be calculated over

# Load in return rate early warning data
mat2_contents = sp.loadmat('Monte_Carlo_EWS_'+str(H1).replace('.', 'p')+'_tstart'+str(tstart)+'_tup'+str(tup)+'_tpause'+str(tpause)+'_tdown'+str(tdown)+'_wl_sensitivity.mat')

t = mat2_contents['t2'][0]
Decay2_notip = mat2_contents['Decay2_notip']
Decay2_tip = mat2_contents['Decay2_tip']

# Number of non-tipping cases (same as number of tipping cases) 
N = len(Decay2_tip[0,:,0])

# Array of thresholds to loop over and compare with scoring classifier to make prediction of tipping or not
EWS_thresholds = np.linspace(-1,1,1001)

plt.figure()

for l in range(len(wl)):

    # Initialise arrays
    tau_notip = np.zeros((len(t),N))
    tau_tip = np.zeros((len(t),N))
    
    EWS_AUC = np.zeros(len(t)-1)    
    
    for j in range(len(t)-1):
        
        # Initialise arrays for false positive and true positive rates for square of decay rate
        EWS_FPR = np.zeros(len(EWS_thresholds))
        EWS_TPR = np.zeros(len(EWS_thresholds))
        
        # If enough time has elapsed, calculate kendall tau on square of decay rate for all tipping & non-tipping trajectories
        if j >= wl[l]+tau_wl:
            for k in range(len(Decay2_notip[0,:,l])):
                tau_notip[j,k],_ = stats.kendalltau(np.arange(tau_wl), Decay2_notip[j-tau_wl:j,k,l])
                tau_tip[j,k],_ = stats.kendalltau(np.arange(tau_wl), Decay2_tip[j-tau_wl:j,k,l])
        
        # Loop over different thresholds and calculate false positive and true positive 
        # rates for geometric EWS (comparing to score classifier) 
        for i in range(len(EWS_thresholds)):
            
            # and after sufficient time square of decay rate
            if j >= wl[l]+tau_wl:            
                EWS_FPR[i] = np.sum(tau_notip[j,:]<EWS_thresholds[i])/N
                EWS_TPR[i] = np.sum(tau_tip[j,:]<EWS_thresholds[i])/N
            else:
                EWS_FPR[i], EWS_TPR[i] = np.nan, np.nan
        
     
        # Integrate area under ROC to get AUC   
        EWS_AUC[j] = integrate(EWS_FPR, EWS_TPR)
    
    plt.plot(t[:-1],EWS_AUC)
    




