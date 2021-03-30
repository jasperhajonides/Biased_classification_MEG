#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:47:31 2021

@author: jasperhajonides
"""
import numpy as np

def sliding_window(data, size_window=20):
    """ Reformat array so that time point t includes all information from
        features up to t-n where n is the size of the predefined window.

         Parameters
        ----------
        data : ndarray
            3-dimensional array of [trial repeats by features by time points].
            Demeans each feature within specified window.
        size_window : int
            number of time points to include in the sliding window
        Returns
        --------
        output : ndarray
            reshaped array where second dimension increased by size of
            size_window

        example:
            
        100, 61, 240 = data.shape

        data_out = sliding_window(data, size_window=5)

        100, 305, 240 = data_out.shape

    """

    n_obs, n_feature, n_time = data.shape
    if size_window > 1 or data.shape < 3 or n_time <= size_window:
        return data

    # predefine variables
    output = np.zeros((n_obs, n_feature*size_window, n_time))
    x_window = np.zeros((n_obs, n_feature, size_window))
    
    
    # loop over third dimension
    for time in range(size_window,n_time):
        
        #demean features within the sliding window
        mean_value = data[:, :, (time-(size_window-1)):(time+1)].mean(2)
        for count, val in enumerate(np.arange(-size_window+1, 1, 1)):
            print(count,val)
            x_window[:, :, count] = data[:, :, time+val] - mean_value
        print(x_window.shape)                        
            
        # reshape into trials by features*window_size
        output[:, :, time] = x_window.reshape(n_obs, n_feature*size_window)
    return output
 
    
 #%%
 
 x_try = X[0:100,0:20,0:6]
 
 X_out = sliding_window(x_try, size_window=5)
 
 print(X_out.shape)
 