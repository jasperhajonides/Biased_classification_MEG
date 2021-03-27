#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:36:48 2021

@author: jasperhajonides
"""
#
def reject_outliers(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)

def bin_array(y,nr_bins):
    """Takes circular array and digitises into n bins"""
    y = np.array(y)
    bins = np.arange(-math.pi-0.000001,math.pi - (2*math.pi/nr_bins),(2*math.pi/nr_bins))
    y_bin = np.digitize(pycircstat.cdiff(y,0), bins) 
    return y,y_binr