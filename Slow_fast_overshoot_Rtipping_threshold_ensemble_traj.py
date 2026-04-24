# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 09:17:48 2026

@author: pdlr201

Script to produce Fig 1 in "Evaluating the skill of a geometric early warning
for tipping in a rapidly forced nonlinear system"
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x, p, pmax):
    """
    Generic double fold bifurcation model.

    Parameters
    ----------
    x : float64
        State variable for system.
    p : float64
        External forcing parameter.
    pmax : float64
        Final level of forcing.

    Returns
    -------
    f : float64
        RHS of ODE.
    """    
    f = -(x**3)/3 + x - p*(pmax-p)    
    return (f)


def p(t, pmax, r):
    """
    Tanh ramp forcing for upstream system.

    Parameters
    ----------
    t : float64
        Time
    pmax : float64
        Final level of forcing.
    r : float64
        Rate parameter of ramp.

    Returns
    -------
    p : float64
        Ramp forcing profile.
    """   
    p = pmax*(np.tanh(r*t)+1)/2     
    return (p)


np.random.seed(seed=1)

# Time parameters
tstart = -200                           # Start time
tend = 200                              # End time
dt = 0.01                               # Spacing between time intervals
n = int((tend - tstart)/dt)             # Number of time intervals
t = np.linspace(tstart, tend, n+1)      # Time values

# Path parameters
pmax = 1.7                              # Final level of forcing
r1 = 0.4                                # Rate of fast forcing
r2 = 0.08                               # Rate of intermediate forcing
r3 = 0.015                              # Rate of slow forcing

# Number of realisations
N = 50

# Noise strength
sigma = np.sqrt(0.01)

# Initialise noise realisations
randomv1 = np.random.normal(0, 1, size=[n, N])
randomv2 = np.random.normal(0, 1, size=[n, N])
randomv3 = np.random.normal(0, 1, size=[n, N])


# Initialise variables 
x1 = np.zeros((n+1,N)) 
x2 = np.zeros((n+1,N))
x3 = np.zeros((n+1,N))    

x_det1 = np.zeros(n+1)     
x_det2 = np.zeros(n+1)
x_det3 = np.zeros(n+1)

y_det1 = np.zeros(n+1)     
y_det2 = np.zeros(n+1)
y_det3 = np.zeros(n+1)

# Define initial starting value
x1[0,:] = np.sqrt(3)
x2[0,:] = np.sqrt(3)
x3[0,:] = np.sqrt(3)

x_det1[0] = np.sqrt(3)
x_det2[0] = np.sqrt(3)
x_det3[0] = np.sqrt(3)


# Apply Forward Euler
for i in range(n):
    
    # Calculate pullback attractor for different forcing rates
    x_det1[i+1] = x_det1[i] + dt*f(x_det1[i], p(t[i], pmax, r1), pmax)
    x_det2[i+1] = x_det2[i] + dt*f(x_det2[i], p(t[i], pmax, r2), pmax)
    x_det3[i+1] = x_det3[i] + dt*f(x_det3[i], p(t[i], pmax, r3), pmax)
    
    # Calculate R-tipping threshold for different forcing rates
    y_det1[i+1] = y_det1[i] - dt*f(y_det1[i], p(t[-1-i], pmax, r1), pmax)
    y_det2[i+1] = y_det2[i] - dt*f(y_det2[i], p(t[-1-i], pmax, r2), pmax)
    y_det3[i+1] = y_det3[i] - dt*f(y_det3[i], p(t[-1-i], pmax, r3), pmax)

    # Loop over ensemble
    for j in range(N):
        
        x1[i+1,j] = x1[i,j] + dt*f(x1[i,j], p(t[i], pmax, r1), pmax) + np.sqrt(dt)*sigma*randomv1[i,j] 
        x2[i+1,j] = x2[i,j] + dt*f(x2[i,j], p(t[i], pmax, r2), pmax) + np.sqrt(dt)*sigma*randomv2[i,j]  
        x3[i+1,j] = x3[i,j] + dt*f(x3[i,j], p(t[i], pmax, r3), pmax) + np.sqrt(dt)*sigma*randomv3[i,j]  

# Determine tipping occurs if state variable is below zero
Tip_ind1 = x1[-1,:]<0
Tip_ind2 = x2[-1,:]<0
Tip_ind3 = x3[-1,:]<0

# Calculate quasi-static equilibria
xeq = np.linspace(-3,3,10000001)
peqplus = (pmax+np.sqrt(pmax**2-4*(-(xeq**3)/3+xeq)))/2
peqminus = (pmax-np.sqrt(pmax**2-4*(-(xeq**3)/3+xeq)))/2

# Plotting
fig, ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(10,3.5))

ax[0].set_xlim(0,pmax)
ax[0].set_ylim(-2.5,2.5)

if np.sum(Tip_ind1)>0:
    ax[2].plot(p(t, pmax, r1),x1[:,Tip_ind1],'r',alpha=0.1)
if np.sum(~Tip_ind1)>0:
    ax[2].plot(p(t, pmax, r1),x1[:,~Tip_ind1],'b',alpha=0.1)
if np.sum(Tip_ind2)>0:
    ax[1].plot(p(t, pmax, r2),x2[:,Tip_ind2],'r',alpha=0.1)
if np.sum(~Tip_ind2)>0:
    ax[1].plot(p(t, pmax, r2),x2[:,~Tip_ind2],'b',alpha=0.1)
if np.sum(Tip_ind3)>0:
    ax[0].plot(p(t, pmax, r3),x3[:,Tip_ind3],'r',alpha=0.1)
if np.sum(~Tip_ind3)>0:
    ax[0].plot(p(t, pmax, r3),x3[:,~Tip_ind3],'b',alpha=0.1)

for i in range(3):
    ax[i].plot(peqplus[(xeq<1)&(xeq>-1)],xeq[(xeq<1)&(xeq>-1)],'k--')
    ax[i].plot(peqplus[xeq<-1],xeq[xeq<-1],'k')
    ax[i].plot(peqplus[xeq>1],xeq[xeq>1],'k')
    ax[i].plot(peqminus[(xeq<1)&(xeq>-1)],xeq[(xeq<1)&(xeq>-1)],'k--')
    ax[i].plot(peqminus[xeq<-1],xeq[xeq<-1],'k')
    ax[i].plot(peqminus[xeq>1],xeq[xeq>1],'k')


ax[2].plot(p(t, pmax, r1),x_det1,c='tab:cyan',lw=2.5)
ax[2].plot(p(t, pmax, r1),y_det1[::-1],c='tab:orange',lw=2.5)
ax[2].fill_between(p(t, pmax, r1),y_det1[::-1],5,color='k',alpha=0.15)
ax[1].plot(p(t, pmax, r2),x_det2,c='tab:cyan',label='Pullback attractor',lw=2.5)
ax[1].plot(p(t, pmax, r2),y_det2[::-1],c='tab:orange',label='R-tipping threshold',lw=2.5)
ax[1].fill_between(p(t, pmax, r2),y_det2[::-1],5,color='k',alpha=0.15)
ax[0].plot(p(t, pmax, r3),x_det3,c='tab:cyan',lw=2.5)
ax[0].plot(p(t, pmax, r3),y_det3[::-1],c='tab:orange',lw=2.5)
ax[0].fill_between(p(t, pmax, r3),y_det3[::-1],5,color='k',alpha=0.15)

ax[0].set_xlabel(r'$p$')
ax[1].set_xlabel(r'$p$')
ax[2].set_xlabel(r'$p$')
ax[0].set_ylabel(r'$x$')

fig.tight_layout()