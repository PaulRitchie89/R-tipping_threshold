# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 12:09:00 2025

@author: Paul

Script to plot time series and phase plane of reduced 3 box AMOC model
for Fig 2 in "Evaluating the skill of a geometric early warning
for tipping in a rapidly forced nonlinear system"
"""

#import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

fontsize = 12#10#
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
    
    # Positive AMOC strength equations
    z0p = (Y/VN)*(q*(ST-SN)+KN*(ST-SN)-100*FN*S0)
    z1p = (Y/VT)*(q*(gamma*SS+(1-gamma)*SIP-ST)+KS*(SS-ST)+KN*(SN-ST)-100*FT*S0)

    # Negative AMOC strength equations
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
        Max freshwater hosing level.
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

def Hlinear(t,H0,H1,r1,tstart):
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
    tstart : float64
        Start time of forcing

    Returns
    -------
    Hforcing : float64
        Piecewise linear freshwater hosing profile.
    """
    Hrampup = (H0 + r1*(t-tstart))
    
    Hforcing = H0*(t<tstart) + Hrampup*(tstart<=t<(H1-H0)/r1+tstart) +  H1*(t>=(H1-H0)/r1+tstart)
    return(Hforcing)


###################### Model parameters ####################################
    
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

######################### Equilibria #########################################

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

# Hopf bifurcation forcing level
Hhopf = 0.3887


##############################################################################

# Forcing parameters
H0 = 0#0.35                                      # Initial forcing
H1 = 0.38                                   # Final forcing
tup = 100#10                                   # Ramp up period
tuplinear = 1200
tpause = [300,400]#[2600,2900]#[337.3410965320802]#     # Plataeu period
tdown = 200#20                                 # Ramp down period
tstart = 0                                  # Start of forcing

r1 = (H1-H0)/tup                                 # Rate of forcing increase
r2 = (H1-H0)/tdown                               # Rate of forcing decrease
r1linear = (H1-H0)/tuplinear

# Index of on, off and saddle equilibria for zero hosing
ind11 = np.argmin(np.abs(H_sad-H0))
ind22 = np.argmin(np.abs(H_on-H0)) 
ind33 = np.argmin(np.abs(H_off-H0))

# Time parameters
tspan2 = [0, 10000]
h2 = 1
t2 = np.arange(tspan2[0],tspan2[1]+h2,h2)


# Initialise figure
fig2,ax2 = plt.subplots(2,1,sharex=True)
sns.despine()

ax2[0].set_ylabel('Freshwater hosing (Sv)')
ax2[1].set_ylabel('$S_N$')
ax2[1].set_xlabel('Time (years)')
ax2[0].set_xlim(0,2000)

fig, ax = plt.subplots(1,1)
ax.set_xlabel('$S_N$')
ax.set_ylabel('$S_T$')
sns.despine()

Forcing_profile = np.zeros(len(t2)-1)
Forcing_linear_profile = np.zeros(len(t2)-1)

ax2[0].plot(tspan2,[Hhopf,Hhopf],'k--')

# Loop over two plateau durations
for j in range(len(tpause)):
    
    # Initialise state variables
    X2 = np.zeros((2,len(t2)))
    X3 = np.zeros((2,len(t2)))
    
    # Start at on equilibrium
    X2[:,0] = [SN_on[ind22],ST_on[ind22]]
    X3[:,0] = [SN_on[ind22],ST_on[ind22]]
    
    # Perform Forward Euler
    for i in range(len(t2)-1):
        X2[:,i+1] = X2[:,i] + h2*BoxModel_2DH_IVP(t2[i],X2[:,i],H(t2[i],H0,H1,r1,r2,tpause[j],tstart))
        X3[:,i+1] = X3[:,i] + h2*BoxModel_2DH_IVP(t2[i],X3[:,i],Hlinear(t2[i],H0,H1,r1linear,tstart))
        Forcing_profile[i] = H(t2[i],H0,H1,r1,r2,tpause[j],tstart)
        Forcing_linear_profile[i] = Hlinear(t2[i],H0,H1,r1linear,tstart)
    
    # Plotting 
    ax2[0].plot(t2[:-1],Forcing_profile)
    ax2[1].plot(t2,S0+X2[0,:]/100)
    
    ax.plot(S0+X2[0,:]/100,S0+X2[1,:]/100)
    
    if j==0:
        ax2[0].plot(t2[:-1],Forcing_linear_profile,'tab:green')
        ax2[1].plot(t2,S0+X3[0,:]/100,'tab:green')
        ax.plot(S0+X3[0,:]/100,S0+X3[1,:]/100,'tab:green')
    
fig.tight_layout()
fig2.tight_layout()


ax.plot(S0+SN_on[ind22]/100,S0+ST_on[ind22]/100,'k.',ms=12)
ax.plot(S0+SN_off[ind33]/100,S0+ST_off[ind33]/100,'k.',ms=12)

ax.text(0.03305,0.0364,'\\textbf{OFF}')
ax.text(0.03515,0.0365,'\\textbf{ON}')


# Time settings for calculation of stable manifold(s) of saddle
tspan3 = [10000, -10000]
h3 = -1
t_basin_boundary = np.arange(tspan3[0],tspan3[1]-h3,h3)

# Initialise variables
X11 = np.zeros((2,len(t_basin_boundary)+1))
Y11 = np.zeros((2,len(t_basin_boundary)+1))
X22 = np.zeros((2,len(t_basin_boundary)+1))
Y22 = np.zeros((2,len(t_basin_boundary)+1))

# Indicies of on state and saddle at H0
ind1 = np.argmin(np.abs(H_sad-H0))
ind2 = np.argmin(np.abs(H_sad-H1))

# Start just away from saddle   
X11[:,0] = [SN_sad[ind1],ST_sad[ind1]+0.0001]
Y11[:,0] = [SN_sad[ind1],ST_sad[ind1]-0.0001]
X22[:,0] = [SN_sad[ind2],ST_sad[ind2]+0.000001]
Y22[:,0] = [SN_sad[ind2],ST_sad[ind2]-0.000001]

# Perform Forward-Euler backwards in time from saddle    
for k in range(len(t_basin_boundary)):
    X11[:,k+1] = X11[:,k] + h3*BoxModel_2DH_IVP(t_basin_boundary[k],X11[:,k],H_sad[ind1])
    Y11[:,k+1] = Y11[:,k] + h3*BoxModel_2DH_IVP(t_basin_boundary[k],Y11[:,k],H_sad[ind1])
    X22[:,k+1] = X22[:,k] + h3*BoxModel_2DH_IVP(t_basin_boundary[k],X22[:,k],H_sad[ind2])
    Y22[:,k+1] = Y22[:,k] + h3*BoxModel_2DH_IVP(t_basin_boundary[k],Y22[:,k],H_sad[ind2])

ax.plot(S0+X11[0,:]/100,S0+X11[1,:]/100,'k',zorder=-10)
ax.plot(S0+Y11[0,:]/100,S0+Y11[1,:]/100,'k',zorder=-10)

ax.plot(S0+X22[0,18500:]/100,S0+X22[1,18500:]/100,'k',alpha=0.3,zorder=-10)
ax.plot(S0+Y22[0,18500:]/100,S0+Y22[1,18500:]/100,'k',alpha=0.3,zorder=-10)

ind222 = np.argmin(np.abs(H_on-H0))
ind333 = np.argmin(np.abs(H_on-H1))

ax.plot(S0+SN_on[ind333:ind222]/100,S0+ST_on[ind333:ind222]/100,'k--',alpha=0.3,zorder=-10)
ax.plot(S0+SN_on[ind333]/100,S0+ST_on[ind333]/100,'k.',alpha=0.3,ms=12)

ax.set_xlim(0.0325,0.036)
ax.set_ylim(0.036,0.038)