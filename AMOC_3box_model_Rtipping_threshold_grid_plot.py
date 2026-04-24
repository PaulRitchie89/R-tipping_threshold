# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 13:44:39 2026

@author: Paul

Script that produces Fig 6 in "Evaluating the skill of a
geometric early warning for tipping in a rapidly forced nonlinear system"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.io as sp

fontsize = 12
rc('font', **{'size' : fontsize})
rc('text', usetex=True)


def BoxModel_2DH_IVP(t,x,H):

    """
    Reduction of Richard Wood's 5 box model to 3 box model for AMOC
    
    Parameters
    ----------
    t : float64
        Time
    x : float64
        Vector of state variables (SN, ST).
    H : float64
        Freshwater hosing acting as external forcing

    Returns
    -------
    z : float64
        RHS of ODE.
    """
    
    # Model parameters for 2 x CO2
    S0 = 0.035

    alpha = 0.12 # kg m^-3 C^-1
    beta = 790.0 # kg m^-3 
    Y = 3.15e7 # sec/year

    VN = 0.3683e17 # m^3
    VT = 0.5418e17 # m^3
    VS = 0.6097e17 # m^3
    VIP = 1.4860e17 # m^3
    VB = 9.9250e17 # m^3

    Ts = 7.919 # C
    T0 = 3.87 # C
    
    C = 4.4735e16 # m^3 (total sum of V*S)

    lamb = 1.62e7 # m^6kg^-1s^-1
    gamma = 0.36
    mu = 22e-8 # deg s m^-3
    
    KN = 1.762e6 # m^3s^-1
    KS = 1.872e6 # m^3s^-1

    FN = (0.486+0.1311*H)*1e6 # m^3s^-1
    FT = (-0.997+0.6961*H)*1e6 # m^3s^-1

    #====================================================

    SN = x[0]
    ST = x[1]
    SS = (0.034427-S0)*100
    SB = (0.034538-S0)*100
    SIP = 100*(C-(VN*SN+VT*ST+VS*SS+VB*SB)/100-S0*(VB+VN+VT+VIP+VS))/VIP

    q = lamb*(alpha*(Ts-T0)+beta*(SN-SS)/100)/(1+lamb*alpha*mu)
    aq = np.abs(q)

    z0p = (Y/VN)*(q*(ST-SN)+KN*(ST-SN)-100*FN*S0)
    z1p = (Y/VT)*(q*(gamma*SS+(1-gamma)*SIP-ST)+KS*(SS-ST)+KN*(SN-ST)-100*FT*S0)

    z0n = (Y/VN)*(aq*(SB-SN)+KN*(ST-SN)-100*FN*S0)
    z1n = (Y/VT)*(aq*(SN-ST)+KS*(SS-ST)+KN*(SN-ST)-100*FT*S0)
    
    z = np.zeros(2)

    z[0] = z0p*(q>=0)+z0n*(q<0)
    z[1] = z1p*(q>=0)+z1n*(q<0)

    return(z)
    

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

######################### Equilibria #########################################
    
S0 = 0.035

alpha = 0.12 # kg m^-3 C^-1
beta = 790.0 # kg m^-3 
Y = 3.15e7 # sec/year

VN = 0.3683e17 # m^3
VT = 0.5418e17 # m^3
VS = 0.6097e17 # m^3
VIP = 1.4860e17 # m^3
VB = 9.9250e17 # m^3

Ts = 7.919 # C
T0 = 3.87 # C

C = 4.4735e16 # m^3 (total sum of V*S)

lamb = 1.62e7 # m^6kg^-1s^-1
gamma = 0.36
mu = 22e-8 # deg s m^-3

KN = 1.762e6 # m^3s^-1
KS = 1.872e6 # m^3s^-1

#====================================================

SS = (0.034427-S0)*100
SB = (0.034538-S0)*100

FN1 = 0.1311E6
FN0 = 0.486E6
FT1 = 0.6961E6
FT0 = -0.997E6

SN = np.linspace(-0.3, 0.3, 1000001)

SIPbar = 100*(C-(VN*SN+VS*SS+VB*SB)/100-S0*(VB+VN+VT+VIP+VS))/VIP
q = lamb*(alpha*(Ts-T0)+beta*(SN-SS)/100)/(1+lamb*alpha*mu)

ST = (FN1*q*(gamma*SS + (1-gamma)*SIPbar) + FN1*KS*SS + FN1*KN*SN - S0*FT0*FN1*100 + FT1*q*SN + FT1*KN*SN + FT1*FN0*S0*100)/(FN1*q*(1-gamma)*VT/VIP + FN1*q + FN1*KS + FN1*KN + FT1*q + FT1*KN)
H2 = ((q*(ST-SN) + KN*(ST-SN))/(100*S0) - FN0)/FN1

H_sad1 = H2[np.where((q>=0)&(SN<-0.059)&(H2>=-0.05))]
SN_sad1 = SN[np.where((q>=0)&(SN<-0.059)&(H2>=-0.05))]
ST_sad1 = ST[np.where((q>=0)&(SN<-0.059)&(H2>=-0.05))]

# On state equilibrium
H_on = H2[np.where((q>=0)&(SN>-0.059)&(H2>=-0.05))]
SN_on = SN[np.where((q>=0)&(SN>-0.059)&(H2>=-0.05))]
ST_on = ST[np.where((q>=0)&(SN>-0.059)&(H2>=-0.05))]

aq = np.abs(q)

ST = (aq*SN + KS*SS + KN*SN - S0*FT0*100 - FT1*(aq*(SB-SN) - KN*SN)/FN1 + FT1*FN0*S0*100/FN1)/(aq + KS + KN + FT1*KN/FN1)
H2 = ((aq*(SB-SN) + KN*(ST-SN))/(100*S0) - FN0)/FN1

# Off state equilibrium
H_off = H2[np.where((q<0)&(SN<-0.15)&(H2>=-0.05))]
SN_off = SN[np.where((q<0)&(SN<-0.15)&(H2>=-0.05))]
ST_off = ST[np.where((q<0)&(SN<-0.15)&(H2>=-0.05))]

H_sad2 = H2[np.where((q<0)&(SN>-0.15)&(H2>=-0.05))]
SN_sad2 = SN[np.where((q<0)&(SN>-0.15)&(H2>=-0.05))]
ST_sad2 = ST[np.where((q<0)&(SN>-0.15)&(H2>=-0.05))]

# Saddle equilibrium
H_sad = np.concatenate((H_sad2,H_sad1))
SN_sad = np.concatenate((SN_sad2,SN_sad1))
ST_sad = np.concatenate((ST_sad2,ST_sad1))


##############################################################################

# Forcing parameters
H0 = 0                                      # Initial forcing
H1 = 0.38                                   # Final forcing
tup = 100                                   # Ramp up period
tpause = 300                                # Plataeu period (choose 300 (left fig panels) or 400 (right fig panels))
tdown = 200                                 # Ramp down period
tstart = 0                                  # Start of forcing

r1 = H1/tup                                 # Rate of forcing increase
r2 = H1/tdown                               # Rate of forcing decrease

# Load in gridded data of points in phase space that tip vs those that do not
mat_contents = sp.loadmat('Rtipping_threshold_grid_tstart'+str(tstart)+'_tup'+str(tup)+'_tpause'+str(tpause)+'_tdown'+str(tdown)+'_hires.mat')
t0_vals = mat_contents['t0_vals'][0]
SN_vals = mat_contents['SN_vals'][0]
ST_vals = mat_contents['ST_vals'][0]
Tip_idx = mat_contents['Tip_idx']

# Set max and min limits for SN and ST
SN_low = 0.033
SN_high = 0.04
ST_low = 0.033
ST_high = 0.045

# Time intervals before forcing starts to look at
t0_vals2 = np.linspace(0,-400,9)

# Indicies of on state and saddle at H0
ind1 = np.argmin(np.abs(H_sad-H0))
ind2 = np.argmin(np.abs(H_on-H0))

# Time settings for calculation of stable manifold(s) of saddle
tspan3 = [5000, -5000]
h3 = -1
t_basin_boundary = np.arange(tspan3[0],tspan3[1]-h3,h3)

# Initialise variables
X11 = np.zeros((2,len(t_basin_boundary)+1))
Y11 = np.zeros((2,len(t_basin_boundary)+1))

# Start just away from saddle   
X11[:,0] = [SN_sad[ind1],ST_sad[ind1]+0.0001]
Y11[:,0] = [SN_sad[ind1],ST_sad[ind1]-0.0001]

# Perform Forward-Euler backwards in time from saddle    
for k in range(len(t_basin_boundary)):
    X11[:,k+1] = X11[:,k] + h3*BoxModel_2DH_IVP(t_basin_boundary[k],X11[:,k],H_sad[ind1])
    Y11[:,k+1] = Y11[:,k] + h3*BoxModel_2DH_IVP(t_basin_boundary[k],Y11[:,k],H_sad[ind1])

# Plotting figure
fig, ax = plt.subplots(3,3,figsize=(7.5,6),sharex=True,sharey=True)

ax[0,0].set_xlim(0.033,0.04)
ax[0,0].set_ylim(0.033,0.045)

ax[2,0].set_xlabel(r'$S_N$')
ax[2,1].set_xlabel(r'$S_N$')
ax[2,2].set_xlabel(r'$S_N$')
ax[0,0].set_ylabel(r'$S_T$')
ax[1,0].set_ylabel(r'$S_T$')
ax[2,0].set_ylabel(r'$S_T$')

for j in range(len(t0_vals2)):
    
    idx = np.argmin(np.abs(t0_vals2[j]-t0_vals))

    ax[int(j/3),j%3].plot(S0+X11[0,:]/100, S0+X11[1,:]/100, 'k', alpha=0.2)
    ax[int(j/3),j%3].plot(S0+Y11[0,:]/100, S0+Y11[1,:]/100, 'k', alpha=0.2)  
    
    ax[int(j/3),j%3].plot(S0+SN_sad[ind1]/100, S0+ST_sad[ind1]/100, 'k', marker='X', ms = 4, alpha=0.2)
    ax[int(j/3),j%3].plot(S0+SN_on[ind2]/100, S0+ST_on[ind2]/100, 'tab:cyan', marker='o', ms = 4)
    ax[int(j/3),j%3].pcolor(SN_vals/100+S0,ST_vals/100+S0,Tip_idx[idx,:,:].T)
    
    ax[int(j/3),j%3].set_title("{:4n}".format(t0_vals2[j])+' years')

fig.tight_layout()









