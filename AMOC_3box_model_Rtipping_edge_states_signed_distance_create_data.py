# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 10:21:05 2025

@author: Paul

Generate data to provide signed distance to R-tipping threshold for "Evaluating
the skill of a geometric early warning for tipping in a rapidly forced nonlinear system" 
"""

import numpy as np
import scipy
import scipy.io as sp
import time



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

# Load in R-tipping threshold data
mat_contents = sp.loadmat('Rtipping_edge_state_H'+str(H1).replace('.', 'p')+'_tstart'+str(tstart)+'_tup'+str(tup)+'_tpause'+str(tpause)+'_tdown'+str(tdown)+'.mat')
Edge_state = S0+mat_contents['X1'][:,::-1,:]/100

# Load in Monte-Carlo simulations data
mat_contents2 = sp.loadmat('Monte_Carlo_simulations_'+str(H1).replace('.', 'p')+'_tstart'+str(tstart)+'_tup'+str(tup)+'_tpause'+str(tpause)+'_tdown'+str(tdown)+'.mat')
t2 = mat_contents2['t'][0]
X2 = S0+mat_contents2['X1']/100

# Number of data points along R-tipping threshold
Nvals = len(Edge_state[0,0,:])

# Number of Monte-Carlo simulations of each tipping and non-tipping
N = 100

# Index for non-tipping trajectories
idx = X2[0,-1,:]>0.034

# Extract first N tipping and non-tipping trajectories
Notip_traj = X2[:,:,idx][:,:,:N]
Tip_traj = X2[:,:,~idx][:,:,:N]

# Initialise arrays
Notip_crossing_number = np.zeros((len(t2)-1,N))
Notip_mindx = np.zeros((len(t2)-1,N))

Tip_crossing_number = np.zeros((len(t2)-1,N))
Tip_mindx = np.zeros((len(t2)-1,N))

# Index of off state for H0 
ind33 = np.argmin(np.abs(H_off-H0))

# Off state coordinates       
SN_off_init = S0+SN_off[ind33]/100
ST_off_init = S0+ST_off[ind33]/100

# Construct linear line between off state and trajectories
Notip_m_line = (Notip_traj[1,:,:] - ST_off_init) / (Notip_traj[0,:,:] - SN_off_init)
Notip_b_line = ST_off_init - Notip_m_line * SN_off_init

Tip_m_line = (Tip_traj[1,:,:] - ST_off_init) / (Tip_traj[0,:,:] - SN_off_init)
Tip_b_line = ST_off_init - Tip_m_line * SN_off_init

x1, y1 = Edge_state[:,:,:-1]
x2, y2 = Edge_state[:,:,1:]

# Curve segment: get slope and intercept
m_curve = (y2 - y1) / (x2 - x1)
b_curve = y1 - m_curve * x1

Notip_x_start = np.minimum(SN_off_init,Notip_traj[0,:,:])
Notip_x_end = np.maximum(SN_off_init,Notip_traj[0,:,:])

Tip_x_start = np.minimum(SN_off_init,Tip_traj[0,:,:])
Tip_x_end = np.maximum(SN_off_init,Tip_traj[0,:,:])


for l in range(N):
    
    start = time.process_time()
    
    Notip_dx = np.sqrt((np.subtract(Edge_state[0,:-1,:].T,Notip_traj[0,:-1,l]))**2 + (np.subtract(Edge_state[1,:-1,:].T,Notip_traj[1,:-1,l]))**2)
    Notip_mindx[:,l] = np.min(Notip_dx,axis=0)
    
    Tip_dx = np.sqrt((np.subtract(Edge_state[0,:-1,:].T,Tip_traj[0,:-1,l]))**2 + (np.subtract(Edge_state[1,:-1,:].T,Tip_traj[1,:-1,l]))**2)
    Tip_mindx[:,l] = np.min(Tip_dx,axis=0)
    
    for i in range(len(t2)-1):       
    
        for j in range(Nvals - 1):            
            
            # Find intersection point x
            Notip_x_int = (Notip_b_line[i,l] - b_curve[i,j]) / (m_curve[i,j] - Notip_m_line[i,l])
            Notip_y_int = Notip_m_line[i,l] * Notip_x_int + Notip_b_line[i,l]
            
            # Find intersection point x
            Tip_x_int = (Tip_b_line[i,l] - b_curve[i,j]) / (m_curve[i,j] - Tip_m_line[i,l])
            Tip_y_int = Tip_m_line[i,l] * Tip_x_int + Tip_b_line[i,l]

            # Check if x_int is within segment bounds
            if min(x1[i,j], x2[i,j]) <= Notip_x_int <= max(x1[i,j], x2[i,j]) and Notip_x_start[i,l] <= Notip_x_int <= Notip_x_end[i,l]:
                Notip_crossing_number[i,l] += 1  
                
            if min(x1[i,j], x2[i,j]) <= Tip_x_int <= max(x1[i,j], x2[i,j]) and Tip_x_start[i,l] <= Tip_x_int <= Tip_x_end[i,l]:
                Tip_crossing_number[i,l] += 1 

        # If crossing number is not 1 then the system is outside the region enclosed by the R-tipping threshold
        # Therefore signed distance to R-tipping threshold is negative
        if Notip_crossing_number[i,l] != 1:
            Notip_mindx[i,l] = -Notip_mindx[i,l]
            
        if Tip_crossing_number[i,l] != 1:
            Tip_mindx[i,l] = -Tip_mindx[i,l]
            
    print(l,time.process_time() - start)

# Save data
scipy.io.savemat('Monte_Carlo_edge_state_dist_'+str(H1).replace('.', 'p')+'_tstart'+str(tstart)+'_tup'+str(tup)+'_tpause'+str(tpause)+'_tdown'+str(tdown)+'.mat', {'t2':t2, 'Notip_traj':Notip_traj, 'Tip_traj':Tip_traj, 'Notip_mindx':Notip_mindx, 'Tip_mindx':Tip_mindx})
