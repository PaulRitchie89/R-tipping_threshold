# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 13:14:57 2025

@author: Paul

Script to generate R-tipping threshold data for "Evaluating the skill of a
geometric early warning for tipping in a rapidly forced nonlinear system"
"""

import numpy as np
import scipy


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

# Time settings for calculation of stable manifold(s) of saddle
tspan = [4000, -200]
h = -1
t = np.arange(tspan[0],tspan[1]+h,h)

# Indicies of on state and saddle at H0
ind11 = np.argmin(np.abs(H_sad-H0))
ind22 = np.argmin(np.abs(H_on-H0))

# Initialise variables
X22 = np.zeros((2,len(t)+1))
Y22 = np.zeros((2,len(t)+1))
  
# Start just away from saddle 
X22[:,0] = [SN_sad[ind11],ST_sad[ind11]+0.000001]
Y22[:,0] = [SN_sad[ind11],ST_sad[ind11]-0.000001]

# Perform Forward-Euler backwards in time from saddle 
for j in range(len(t)):
    X22[:,j+1] = X22[:,j] + h*BoxModel_2DH_IVP(t[j],X22[:,j],H_sad[ind11])
    Y22[:,j+1] = Y22[:,j] + h*BoxModel_2DH_IVP(t[j],Y22[:,j],H_sad[ind11])

Z22 = np.concatenate((X22,Y22),axis=1)

# Set max and min limits for SN and ST
SN_low = 0.02
SN_high = 0.04
ST_low = 0.02
ST_high = 0.045

# Number of points to dicretise along the R-tipping threshold
Nvals = 3200

# Identify data points of basin boundary (for zero/end of forcing) within max and min limits
Y22SN = Z22[0,(SN_low<S0+Z22[0,:]/100) & (S0+Z22[0,:]/100<=SN_high) & (ST_low<S0+Z22[1,:]/100) & (S0+Z22[1,:]/100<=ST_high)]
Y22ST = Z22[1,(SN_low<S0+Z22[0,:]/100) & (S0+Z22[0,:]/100<=SN_high) & (ST_low<S0+Z22[1,:]/100) & (S0+Z22[1,:]/100<=ST_high)]

# Evenly discretise points along basin boundary
dr = (np.diff(Y22SN)**2 + np.diff(Y22ST)**2)**.5 # segment lengths
r = np.zeros_like(Y22SN)
r[1:] = np.cumsum(dr) # integrate path
r_int = np.linspace(0, r.max(), Nvals) # regular spaced path
x_int = np.interp(r_int, r, Y22SN) # interpolate
y_int = np.interp(r_int, r, Y22ST)

# Time settings for calculation of R-tipping threshold
tspan = [1000, -200]
h = -1
t = np.arange(tspan[0],tspan[1]+h,h)

X1 = np.zeros((2,len(t),len(x_int)))
r_stats = np.zeros((2,len(t)-1))

x0 = [x_int,y_int]

X1[:,0,:] = x0

for i in range(len(t)-1):
    
    # Forward Euler step (backwards in time) for each point on R-tipping threshold
    for j in range(len(x_int)):
        X1[:,i+1,j] = X1[:,i,j] + h*BoxModel_2DH_IVP(t[i],X1[:,i,j],H(t[i],H0,H1,r1,r2,tpause,tstart))
    
    # Current points on R-tipping threshold within region of interest
    X1SN = X1[0,i+1,(SN_low<S0+X1[0,i+1,:]/100) & (S0+X1[0,i+1,:]/100<=SN_high) & (ST_low<S0+X1[1,i+1,:]/100) & (S0+X1[1,i+1,:]/100<=ST_high)]
    X1ST = X1[1,i+1,(SN_low<S0+X1[0,i+1,:]/100) & (S0+X1[0,i+1,:]/100<=SN_high) & (ST_low<S0+X1[1,i+1,:]/100) & (S0+X1[1,i+1,:]/100<=ST_high)]
    
    # Re-initialise evenly spaced points along R-tipping threshold 
    dr = (np.diff(X1SN)**2 + np.diff(X1ST)**2)**.5 # segment lengths
    r = np.zeros_like(X1SN)
    r_stats[:,i] = np.max(dr),len(X1SN)
    r[1:] = np.cumsum(dr) # integrate path
    r_int = np.linspace(0, r.max(), Nvals) # regular spaced path
    X1[0,i+1,:] = np.interp(r_int, r, X1SN) # interpolate
    X1[1,i+1,:] = np.interp(r_int, r, X1ST)
    
# Save data
scipy.io.savemat('Rtipping_edge_state_H'+str(H1).replace('.', 'p')+'_tstart'+str(tstart)+'_tup'+str(tup)+'_tpause'+str(tpause)+'_tdown'+str(tdown)+'.mat', {'t':t, 'X1':X1})