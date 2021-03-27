#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:09:38 2021

@author: jasperhajonides
"""

# cos convolve


def compute_asymmetry(data, angular_difference, min_deg = 30, max_deg = 60):
    """Compute distribution of x by phase-bins in the Instantaneous Frequency.

    Parameters
    ----------
    data : ndarray
        three-dimensional array of [trial repeats by classes by time points]. 
        Correct class is in position n_classes/2 
    angular_difference : ndarray
        Input vector in radians [-pi to pi] with the same length as trials 
    min_deg : float
         min angular distance cut-off point for trials to include
    max_deg : float
         max angular distance cut-off point for trials to include

    Returns
    -------
    shift : ndarray
        array containing asymmetry score for every time point.
    CW : ndarray
        array containing evidence for CW angular difference trials
    CCW : ndarray
        array containing evidence for CCW angular difference trials

    """
    
    n_trials, n_classes, n_tps = data.shape
    
    # compute evidence for all classes for trials CW from current angle
    CW = data[(angular_difference <= -min_deg/90*np.pi) & 
                (angular_difference > -max_deg/90*np.pi), :, :].mean(0)
    
    # compute evidence for all classes for trials CCW from current angle
    CCW =data[(angular_difference >= min_deg/90*np.pi) &
                (angular_difference < max_deg/90*np.pi), :, :].mean(0)

    # sum 
    shift = (CW[0:int(n_classes/2-1),:].mean(0) - 
             CW[int(n_classes/2):(n_classes-1), :].mean(0)) - 
            (CCW[0:int(n_classes/2-1), :].mean(0) - 
             CCW[int(n_classes/2):(n_classes-1), :].mean(0)) 

    return shift, CW, CCW
