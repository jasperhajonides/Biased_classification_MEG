#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:36:48 2021

@author: jasperhajonides
"""
#
def reject_outliers(data, m=2):
    """ Remove outliers"""
    return abs(data - np.mean(data)) < m * np.std(data)

def bin_array(y,nr_bins):
    """Takes circular array and digitises into n bins"""
    y = np.array(y)
    bins = np.arange(-math.pi-0.000001,math.pi - (2*math.pi/nr_bins),(2*math.pi/nr_bins))
    y_bin = np.digitize(pycircstat.cdiff(y,0), bins) 
    return y,y_bin


def matrix_vector_shift(matrix, vector, n_bins):

    """ Shift rows of a matrix by the amount of columns specified
		in the corresponding cell of the vector.

	e.g. M =0  1  0     V = 0 0 1 2     M_final =   0 1 0
			0  1  0									 0 1 0
			1  0  0									 0 1 0
			0  0  1								  	 0 1 0
            """
    row, col = matrix.shape
    matrix_shift = np.zeros((row, col))
    for row_id in range(0, row):
        matrix_shift[row_id, :] = np.roll(matrix[row_id, :], int(np.floor(n_bins/2)-vector[row_id]))
    return matrix_shift

def sliding_window(X, size_window=20):
    """ Reformat array so that time point t includes all information from 
        features up to t-n where n is the size of the predefined window.
        
         Parameters
        ----------
        X : ndarray
            3-dimensional array of [trial repeats by features by time points]. 
            Demeans each feature within specified window. 
        size_window : int
            in
            
    """
    
    if size_window > 1:
        # predefine variables 
        Xt = np.zeros((X.shape[0], X.shape[1]*size_window, X.shape[2]))
        Xsw = np.zeros((X.shape[0], X.shape[1], size_window))
        # loop over third dimension
        for tp in range(size_window,X.shape[2]):
            #demean features within the sliding window
            for count, s in enumerate(np.arange(size_window)-(size_window-1)):
                Xsw[:, :, count] = (X[:, :, tp+s] -
                                    X[:, :, (tp-(size_window-1)):(tp+1)].mean(2)
            # reshape into trials by features*window_size
            Xt[:, :, tp] = Xsw.reshape(Xsw.shape[0],
                                   Xsw.shape[1]*Xsw.shape[2])
        return Xt
    else:
        return X
    
#%%

%load_ext line_profiler insta