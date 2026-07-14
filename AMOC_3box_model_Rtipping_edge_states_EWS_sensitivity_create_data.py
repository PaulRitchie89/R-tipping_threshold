# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:43:13 2026

@author: Paul

Script to generate data for square of return rate for tipping and non-tipping
trajectories for a range of window lengths in "Evaluating the skill of a
geometric early warning for tipping in a rapidly forced nonlinear system"
"""

import numpy as np
import scipy
import scipy.io as sp
from scipy import stats
from scipy.signal import detrend

def calculate_EWS(x,x_det):
    """
    Calculation of generic early warning signals (lag-1 autocorrelation and 
    variance).

    Parameters
    ----------
    x : Array of float64
        An array of the state variable for the length of the sliding window.
    x_det : Array of float64
        An array of the determinstic state variable for the length of the
        sliding window.

    Returns
    -------
    Aut : float64
          Lag-1 autocorrelation.
    Var : float64
          Variance.

    """
    window = detrend(x, type='linear')#x-x_det        # Detrend window using deterministic solution
    
    # Calculate EWS
    Var = np.var(window)
    Aut,_ = stats.pearsonr(window[:-1],window[1:])
    
    return(Aut, Var)



S0 = 0.035  # Reference salinity

# Forcing parameters
H0 = 0                                      # Initial forcing
H1 = 0.38                                   # Final forcing
tup = 100                                   # Ramp up period
tpause = 300                                # Plataeu period (choose 300 (left fig panels) or 400 (right fig panels))
tdown = 200                                 # Ramp down period
tstart = 0                                  # Start of forcing

r1 = H1/tup                                 # Rate of forcing increase
r2 = H1/tdown                               # Rate of forcing decrease


# Load in Monte-Carlo simulations data
mat_contents = sp.loadmat('Monte_Carlo_simulations_'+str(H1).replace('.', 'p')+'_tstart'+str(tstart)+'_tup'+str(tup)+'_tpause'+str(tpause)+'_tdown'+str(tdown)+'.mat')
t = mat_contents['t'][0]
X = S0+mat_contents['X1']/100

# Number of Monte-Carlo simulations of each tipping and non-tipping
N = 100

# Index for non-tipping trajectories
idx = X[0,-1,:]>0.034

# Extract first N tipping and non-tipping trajectories
Notip_traj = X[:,:,idx][:,:,:N]
Tip_traj = X[:,:,~idx][:,:,:N]

n = len(Tip_traj[0,:,0])

# EWS parameters
wl = [50,100,200,400]                        # Range of time lengths for sliding window

Decay2_tip = np.zeros((n,N,len(wl)))
Decay2_notip = np.zeros((n,N,len(wl)))

for k in range(len(wl)):
    burn = 5                        # Burn time length to remove any transient behaviour
    wl_pts = wl[k]                     # Number of points in sliding window
    # n = len(Tip_traj[0,:,0])
    
    # Initialise EWS
    Aut_tip = np.zeros((n,N))
    Aut_notip = np.zeros((n,N))
    
    # Loop over all realisations
    for j in range(N):
        
        # Loop over time
        for i in range(n):
    
            # Calculate EWS
            if i>=wl_pts:
                Aut_notip[i,j], _ = calculate_EWS(Notip_traj[0,i-wl_pts:i,j], wl[k])
                Aut_tip[i,j], _ = calculate_EWS(Tip_traj[0,i-wl_pts:i,j], wl[k])
            else:
                Aut_notip[i,j], Aut_tip[i,j] = np.nan, np.nan
    
    # Calculate square of the decay rate
    Decay2_notip[:,:,k] = (-np.log(Aut_notip)/1)**2
    Decay2_tip[:,:,k] = (-np.log(Aut_tip)/1)**2

# Save data  
scipy.io.savemat('Monte_Carlo_EWS_'+str(H1).replace('.', 'p')+'_tstart'+str(tstart)+'_tup'+str(tup)+'_tpause'+str(tpause)+'_tdown'+str(tdown)+'_wl_sensitivity.mat', {'t2':t, 'Decay2_notip':Decay2_notip, 'Decay2_tip':Decay2_tip})
