import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import sys
import scipy
from scipy import optimize

def rate_model(initial_vals, k, J_vals, V, t):
    '''
    Inputs:
       initial_vals is an array of values and each element corresponds to one group
       - initial_vals[0]: is the initial water fraction
       - initial_vals[1]: is the initial marbles fraction
       - initial_vals[2]: is the initial blobs fraction
       - initial_vals[3]: is the initial fractured fraction
       t is the time array in ms with the steps of dt
       k is an array of rates, 
       - k[0] is from marbles to blobs,
       - k[1] is from blobs to fractured,
       - k[2] is from marbles to fractured.
       J is an array with: len(1): a fixed nucleation rate in 1/m^3s^1, len(2): linear fitting parameters (m, b) for ln(J), len(t): one J value for each time points (TODO)
       V is the volume of the droplet in m^3*s/ms
    --------------------------------
    Outputs:
       fractions of different droplet shapes as a function of time
    '''
    water = np.zeros(len(t))
    marbles = np.zeros(len(t))
    blobs = np.zeros(len(t))
    fractured = np.zeros(len(t))
    dwdt = np.zeros(len(t))
    dmdt = np.zeros(len(t))
    dbdt = np.zeros(len(t))
    dcdt = np.zeros(len(t))
    dt = t[1] - t[0]
    for i in range(len(t)):
        if len(J_vals) == 1:
          J = J_vals[0]
        elif len(J_vals) == 2:
          J = np.exp(t[i]*J_vals[0]+J_vals[1])
        elif len(J_vals) == len(t):
          J = J_vals[i]
        else:
          print('unknown format of nucleation rate J:', J_vals)
          sys.exit(-1)
        if i == 0:
            water[i] = initial_vals[0]
            marbles[i] = initial_vals[1]
            blobs[i] = initial_vals[2]
            fractured[i] = initial_vals[3]
        else:
            water[i] = water[i-1] + dwdt[i-1]*dt
            marbles[i] = marbles[i-1] + dmdt[i-1]*dt
            blobs[i] = blobs[i-1] + dbdt[i-1]*dt
            fractured[i] = fractured[i-1] + dcdt[i-1]*dt
            dwdt[i] = -J*V*water[i] # water
            dmdt[i] = J*V*water[i] - (k[0] + k[2])*marbles[i] # marbles
            dbdt[i] = k[0]*marbles[i] - k[1]*blobs[i] # blobs
            dcdt[i] = k[2]*marbles[i] + k[1]*blobs[i] # fractured
    return(water, marbles, blobs, fractured)
