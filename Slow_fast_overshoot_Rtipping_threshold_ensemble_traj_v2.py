# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:52:05 2026

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

def f2(x, p, pmax):
    """
    Asymmetric double fold bifurcation model.

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
    f2 = -(x**3)/3 - 2*(x**2)/3 + 8*x/3 - p*(pmax-p)    
    return (f2)


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

# Path parameters for overshoot
pmax = 1.7                              # Final level of forcing
r1 = 0.4                                # Rate of fast forcing
r2 = 0.08                               # Rate of intermediate forcing
r3 = 0.015                              # Rate of slow forcing

# Path parameters for R-tipping
pmax2 = 5                              # Final level of forcing
rr1 = 0.9#0.22                                # Rate of fast forcing
rr2 = 0.45#0.185                               # Rate of intermediate forcing
rr3 = 0.2#0.05                              # Rate of slow forcing

# Number of realisations
N = 50

# Noise strength
sigma = np.sqrt(0.01)   # Overshoot example
sigma2 = np.sqrt(0.2)   # R-tipping example

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

# Initialise variables 
xx1 = np.zeros((n+1,N)) 
xx2 = np.zeros((n+1,N))
xx3 = np.zeros((n+1,N))    

xx_det1 = np.zeros(n+1)     
xx_det2 = np.zeros(n+1)
xx_det3 = np.zeros(n+1)

yy_det1 = np.zeros(n+1)     
yy_det2 = np.zeros(n+1)
yy_det3 = np.zeros(n+1)

# Define initial starting value
x1[0,:] = np.sqrt(3)
x2[0,:] = np.sqrt(3)
x3[0,:] = np.sqrt(3)

x_det1[0] = np.sqrt(3)
x_det2[0] = np.sqrt(3)
x_det3[0] = np.sqrt(3)

# Define initial starting value
xx1[0,:] = 2#np.sqrt(3)
xx2[0,:] = 2#np.sqrt(3)
xx3[0,:] = 2#np.sqrt(3)

xx_det1[0] = 2#np.sqrt(3)
xx_det2[0] = 2#np.sqrt(3)
xx_det3[0] = 2#np.sqrt(3)


# Apply Forward Euler
for i in range(n):
    
    # Calculate pullback attractor for different forcing rates in overshoot example
    x_det1[i+1] = x_det1[i] + dt*f(x_det1[i], p(t[i], pmax, r1), pmax)
    x_det2[i+1] = x_det2[i] + dt*f(x_det2[i], p(t[i], pmax, r2), pmax)
    x_det3[i+1] = x_det3[i] + dt*f(x_det3[i], p(t[i], pmax, r3), pmax)
    
    # Calculate R-tipping threshold for different forcing rates in overshoot example
    y_det1[i+1] = y_det1[i] - dt*f(y_det1[i], p(t[-1-i], pmax, r1), pmax)
    y_det2[i+1] = y_det2[i] - dt*f(y_det2[i], p(t[-1-i], pmax, r2), pmax)
    y_det3[i+1] = y_det3[i] - dt*f(y_det3[i], p(t[-1-i], pmax, r3), pmax)
    
    # Calculate pullback attractor for different forcing rates in R-tipping example
    xx_det1[i+1] = xx_det1[i] + dt*f2(xx_det1[i]- p(t[i], pmax2, rr1)*(pmax2-p(t[i], pmax2, rr1)), 0, 0)
    xx_det2[i+1] = xx_det2[i] + dt*f2(xx_det2[i]- p(t[i], pmax2, rr2)*(pmax2-p(t[i], pmax2, rr2)), 0, 0)
    xx_det3[i+1] = xx_det3[i] + dt*f2(xx_det3[i]- p(t[i], pmax2, rr3)*(pmax2-p(t[i], pmax2, rr3)), 0, 0)
    
    # Calculate R-tipping threshold for different forcing rates in R-tipping example
    yy_det1[i+1] = yy_det1[i] - dt*f2(yy_det1[i]- p(t[-1-i], pmax2, rr1)*(pmax2-p(t[-1-i], pmax2, rr1)), 0, 0)
    yy_det2[i+1] = yy_det2[i] - dt*f2(yy_det2[i]- p(t[-1-i], pmax2, rr2)*(pmax2-p(t[-1-i], pmax2, rr2)), 0, 0)
    yy_det3[i+1] = yy_det3[i] - dt*f2(yy_det3[i]- p(t[-1-i], pmax2, rr3)*(pmax2-p(t[-1-i], pmax2, rr3)), 0, 0)

    # Loop over ensemble
    for j in range(N):
        
        # Overshoot example
        x1[i+1,j] = x1[i,j] + dt*f(x1[i,j], p(t[i], pmax, r1), pmax) + np.sqrt(dt)*sigma*randomv1[i,j] 
        x2[i+1,j] = x2[i,j] + dt*f(x2[i,j], p(t[i], pmax, r2), pmax) + np.sqrt(dt)*sigma*randomv2[i,j]  
        x3[i+1,j] = x3[i,j] + dt*f(x3[i,j], p(t[i], pmax, r3), pmax) + np.sqrt(dt)*sigma*randomv3[i,j] 
        
        # R-tipping example
        xx1[i+1,j] = xx1[i,j] + dt*f2(xx1[i,j]- p(t[i], pmax2, rr1)*(pmax2-p(t[i], pmax2, rr1)), 0, 0) + np.sqrt(dt)*sigma2*randomv1[i,j] 
        xx2[i+1,j] = xx2[i,j] + dt*f2(xx2[i,j]- p(t[i], pmax2, rr2)*(pmax2-p(t[i], pmax2, rr2)), 0, 0) + np.sqrt(dt)*sigma2*randomv2[i,j]  
        xx3[i+1,j] = xx3[i,j] + dt*f2(xx3[i,j]- p(t[i], pmax2, rr3)*(pmax2-p(t[i], pmax2, rr3)), 0, 0) + np.sqrt(dt)*sigma2*randomv3[i,j]

# Determine tipping occurs if state variable is below zero in overshoot example
Tip_ind1 = x1[-1,:]<0
Tip_ind2 = x2[-1,:]<0
Tip_ind3 = x3[-1,:]<0

# Determine tipping occurs if state variable is below zero in R-tipping example
Tip_ind4 = xx1[-1,:]<0
Tip_ind5 = xx2[-1,:]<0
Tip_ind6 = xx3[-1,:]<0

# Calculate quasi-static equilibria in overshoot example
xeq = np.linspace(-3,3,10000001)
peqplus = (pmax+np.sqrt(pmax**2-4*(-(xeq**3)/3+xeq)))/2
peqminus = (pmax-np.sqrt(pmax**2-4*(-(xeq**3)/3+xeq)))/2

# Calculate quasi-static equilibria in R-tipping example
ppeq = np.linspace(0,pmax2,100001)
x0eq = ppeq*(pmax2-ppeq)
xeqplus = ppeq*(pmax2-ppeq)+2#np.sqrt(3)
xeqminus = ppeq*(pmax2-ppeq)-4#np.sqrt(3)

# Plotting
fig, ax = plt.subplots(2,3,figsize=(10,6))

ax[1,0].set_xlim(0,pmax)
ax[1,0].set_ylim(-2.5,2.5)
ax[1,1].set_xlim(0,pmax)
ax[1,1].set_ylim(-2.5,2.5)
ax[1,2].set_xlim(0,pmax)
ax[1,2].set_ylim(-2.5,2.5)

ax[0,0].set_xlim(0,pmax2)
ax[0,0].set_ylim(-5,9)
ax[0,1].set_xlim(0,pmax2)
ax[0,1].set_ylim(-5,9)
ax[0,2].set_xlim(0,pmax2)
ax[0,2].set_ylim(-5,9)

if np.sum(Tip_ind1)>0:
    ax[1,2].plot(p(t, pmax, r1),x1[:,Tip_ind1],'r',alpha=0.1)
if np.sum(~Tip_ind1)>0:
    ax[1,2].plot(p(t, pmax, r1),x1[:,~Tip_ind1],'b',alpha=0.1)
if np.sum(Tip_ind2)>0:
    ax[1,1].plot(p(t, pmax, r2),x2[:,Tip_ind2],'r',alpha=0.1)
if np.sum(~Tip_ind2)>0:
    ax[1,1].plot(p(t, pmax, r2),x2[:,~Tip_ind2],'b',alpha=0.1)
if np.sum(Tip_ind3)>0:
    ax[1,0].plot(p(t, pmax, r3),x3[:,Tip_ind3],'r',alpha=0.1)
if np.sum(~Tip_ind3)>0:
    ax[1,0].plot(p(t, pmax, r3),x3[:,~Tip_ind3],'b',alpha=0.1)

if np.sum(Tip_ind4)>0:
    ax[0,2].plot(p(t, pmax2, rr1),xx1[:,Tip_ind4],'r',alpha=0.1)
if np.sum(~Tip_ind4)>0:
    ax[0,2].plot(p(t, pmax2, rr1),xx1[:,~Tip_ind4],'b',alpha=0.1)
if np.sum(Tip_ind5)>0:
    ax[0,1].plot(p(t, pmax2, rr2),xx2[:,Tip_ind5],'r',alpha=0.1)
if np.sum(~Tip_ind5)>0:
    ax[0,1].plot(p(t, pmax2, rr2),xx2[:,~Tip_ind5],'b',alpha=0.1)
if np.sum(Tip_ind6)>0:
    ax[0,0].plot(p(t, pmax2, rr3),xx3[:,Tip_ind6],'r',alpha=0.1)
if np.sum(~Tip_ind6)>0:
    ax[0,0].plot(p(t, pmax2, rr3),xx3[:,~Tip_ind6],'b',alpha=0.1)

for i in range(3):
    ax[1,i].plot(peqplus[(xeq<1)&(xeq>-1)],xeq[(xeq<1)&(xeq>-1)],'k--')
    ax[1,i].plot(peqplus[xeq<-1],xeq[xeq<-1],'k')
    ax[1,i].plot(peqplus[xeq>1],xeq[xeq>1],'k')
    ax[1,i].plot(peqminus[(xeq<1)&(xeq>-1)],xeq[(xeq<1)&(xeq>-1)],'k--')
    ax[1,i].plot(peqminus[xeq<-1],xeq[xeq<-1],'k')
    ax[1,i].plot(peqminus[xeq>1],xeq[xeq>1],'k')
    
    ax[0,i].plot(ppeq,x0eq,'k--')
    ax[0,i].plot(ppeq,xeqplus,'k')
    ax[0,i].plot(ppeq,xeqminus,'k')


ax[1,2].plot(p(t, pmax, r1),x_det1,c='tab:cyan',lw=2.5)
ax[1,2].plot(p(t, pmax, r1),y_det1[::-1],c='tab:orange',lw=2.5)
ax[1,2].fill_between(p(t, pmax, r1),y_det1[::-1],5,color='k',alpha=0.15)
ax[1,1].plot(p(t, pmax, r2),x_det2,c='tab:cyan',label='Pullback attractor',lw=2.5)
ax[1,1].plot(p(t, pmax, r2),y_det2[::-1],c='tab:orange',label='R-tipping threshold',lw=2.5)
ax[1,1].fill_between(p(t, pmax, r2),y_det2[::-1],5,color='k',alpha=0.15)
ax[1,0].plot(p(t, pmax, r3),x_det3,c='tab:cyan',lw=2.5)
ax[1,0].plot(p(t, pmax, r3),y_det3[::-1],c='tab:orange',lw=2.5)
ax[1,0].fill_between(p(t, pmax, r3),y_det3[::-1],5,color='k',alpha=0.15)

ax[0,2].plot(p(t, pmax2, rr1),xx_det1,c='tab:cyan',lw=2.5)
ax[0,2].plot(p(t, pmax2, rr1),yy_det1[::-1],c='tab:orange',lw=2.5)
ax[0,2].fill_between(p(t, pmax2, rr1),yy_det1[::-1],10,color='k',alpha=0.15)
ax[0,1].plot(p(t, pmax2, rr2),xx_det2,c='tab:cyan',label='Pullback attractor',lw=2.5)
ax[0,1].plot(p(t, pmax2, rr2),yy_det2[::-1],c='tab:orange',label='R-tipping threshold',lw=2.5)
ax[0,1].fill_between(p(t, pmax2, rr2),yy_det2[::-1],10,color='k',alpha=0.15)
ax[0,0].plot(p(t, pmax2, rr3),xx_det3,c='tab:cyan',lw=2.5)
ax[0,0].plot(p(t, pmax2, rr3),yy_det3[::-1],c='tab:orange',lw=2.5)
ax[0,0].fill_between(p(t, pmax2, rr3),yy_det3[::-1],10,color='k',alpha=0.15)

ax[1,0].set_xlabel(r'$p$')
ax[1,1].set_xlabel(r'$p$')
ax[1,2].set_xlabel(r'$p$')
ax[0,0].set_ylabel(r'$x$')
ax[1,0].set_ylabel(r'$x$')

fig.tight_layout()