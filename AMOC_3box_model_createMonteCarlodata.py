# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 16:25:45 2025

@author: Paul

Script to generate Monte-Carlo simulations data for "Evaluating the skill of a
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

# Time settings for Monte-Carlo simulations
tspan = [-200, 1000]
h = 1
t = np.arange(tspan[0],tspan[1]+h,h)

# Indicies of on state and saddle at H0
ind11 = np.argmin(np.abs(H_sad-H0))
ind22 = np.argmin(np.abs(H_on-H0))

# Initialise seed
np.random.seed(1)

# Number of Monte-Carlo simulations to perform
N = 1000

# Noise realisations
randomv = np.random.normal(0, 1, size=(len(t),N))
randomv2 = np.random.normal(0, 1, size=(len(t),N))

# Noise parameters
sigma = 1E-4
A11 = 0.1263
A12 = -0.0869
A21 = 0
A22 = 0.1008

# Noise structure
A = np.array([[A11, A12], [A21, A22]])

# Initialise variable
X1 = np.zeros((2,len(t),N))

# Loop over Monte-Carlo simulations
for l in range(N):
    
    X1[:,0,l] = [SN_on[ind22],ST_on[ind22]]
    
    # Perform Forward Euler to calculate trajectory
    for i in range(len(t)-1):
        X1[:,i+1,l] = X1[:,i,l] + h*BoxModel_2DH_IVP(t[i],X1[:,i,l],H(t[i],H0,H1,r1,r2,tpause,tstart)) + np.sqrt(h*sigma)*np.matmul(A,np.array([[randomv[i,l]],[randomv2[i,l]]])).T
    

# Save data    
scipy.io.savemat('Monte_Carlo_simulations_'+str(H1).replace('.', 'p')+'_tstart'+str(tstart)+'_tup'+str(tup)+'_tpause'+str(tpause)+'_tdown'+str(tdown)+'.mat', {'t':t, 'X1':X1})