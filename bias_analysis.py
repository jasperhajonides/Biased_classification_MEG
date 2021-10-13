#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:18:07 2021

@author: jasperhajonides

- change name of behavioural and stimulus file : --> ./BehaviouralData_StimulusInfo.pkl


"""

import math
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#from math import pi, sqrt, exp
import random
import pickle
from scipy import stats
import pycircstat
from temp_dec.decoding_functions import *

# =============================================================================
# paths  
projectloc = '/Users/jasperhajonides/Documents/Dphil/OSF'

# labels
subject_labels = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20']

#%% functions
def reject_outliers(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)

#%% run decoding 

steps = 1
nr_bins = 10
nsubs = 20
size_window = 30 
pca_var = .90 
is_classifier='LDA'
time_lim = [-0.4 , 0.9]
nr_tps = 260 
nbin = 100
pbin=1/nbin #

# initialise arrays
classifier_output = np.zeros((nr_tps, nsubs,11))
classifier_output_tuning = np.zeros((nr_bins, nr_tps, nsubs,4))
right = np.zeros((nr_bins, nr_tps,nsubs))
left = np.zeros((nr_bins, nr_tps,nsubs))
shift = np.zeros((nr_tps, nsubs))

# load in behavioural data first
if 'thetas_all' not in  locals():
    with open ('%s' %projectloc + '/BehaviouralData_StimulusInfo.pkl', 'rb') as fp:
            [thetas_all, stimulus_nrs, presented_angles, time] = pickle.load(fp)

# loop over subjects
for sb_count in range(nsubs):
    print(sb_count) 


    # read MEG data
    with open (projectloc + '/MEG_data/MEG_data_200Hz_%s' %subject_labels[sb_count], 'rb') as fp:
        [X] = pickle.load(fp)
    data = scipy.io.loadmat(projectloc + '/data/decoding_data/306ch_data_' + subject_labels[sb_count] + suffix)
    X,thetas,stimulus_nr,presented_angle,time= concatenate_data(data,sb_count,toi=[-0.4,3.5])
    X = prep_sensor_data(X,data,1,False)
    # time = data['time'][0].T
            

    
    # adam thompson. digital consultant
    
    
    
    