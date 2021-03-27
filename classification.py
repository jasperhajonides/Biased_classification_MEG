#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:36:05 2021

@author: jasperhajonides
"""

#%% import toolboxes
import sys
import math
import os
import scipy.io
import random
import pickle

from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pycircstat

#%% Custom functions

#%% paths and variables

#%% paths

subject_labels = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20']
projectloc = '/Users/jasperhajonides/Documents/EXP8_UpdProtec'
decodingloc = projectloc + '/data/decoding_data'
processed_loc = projectloc + '/MEG/data/preprocessed'

#%% functions



bin_array# func

#%% Classifier evidence for participants
opts = {}
opts['nsubs'] = 1   
overwrite = False
nr_tps = 260

classifier_output = np.zeros((nr_tps,opts['nsubs'],4))
shift = np.zeros((nr_tps,opts['nsubs']))
left, right = np.zeros((nr_tps,10,opts['nsubs'])), np.zeros((nr_tps,10,opts['nsubs']))


# import stimulus information for participants
if 'thetas_all' not in  locals():
    with open ('%s' %processed_loc + '/stimulus_info_for_decoding', 'rb') as fp:
            [thetas_all, stimulus_nrs, presented_angles, time] = pickle.load(fp)
            

       
            
for sb_count in range(19, opts['nsubs']):
    print('computing subject {}'.format(sb_count))
    dec_fname = '/Users/jasperhajonides/Documents/EXP8_UpdProtec/scores/S20_200hz_LongPreCuePeriod_TEMP_single_tp_time-0.4_to_0.9_LDA_windowsize_30_pca0.9_steps1_stim_all_chans_test_on_all'
    if  not(os.path.exists(dec_fname)) or overwrite == True:
    
        # read MEG data
        with open (processed_loc + '/neurophys/MEG_data_200Hz_%s' 
                   %subject_labels[sb_count], 'rb') as fp:
            X = pickle.load(fp)
       
        # read behaviour / orientations
        df_read = pd.read_csv(projectloc + '/data/behavioural/all_behavioural.csv')
        df_read = df_read.loc[(df_read['subject'] == sb_count), :]
    
        # include trials
        inx =   (df_read['stimulus_nr']<3) #& ~np.isnan(df_read['stim1'])  # & reject_outliers(abs(df_read['error']), m=2) & reject_outliers(df_read['RT'], m=3) # & ~np.isnan(df_read['stim2'])  & (abs(df_read['error']/math.pi*90)<math.pi/4) & (df_read['RT']<6) &  #
        X = X[inx, :, :]
    
        # select & set the target variable
        y, yb = bin_array(df_read['presented'][inx], 10) # define number of bins
     
        #select time
        time_lim = [-.4, 0.9] 
        X_all = X[:,:,(time >= time_lim[0]) & (time <= time_lim[1]) ] 
        time_ = time[(time >= time_lim[0]) & (time <= time_lim[1]) ] 
        
        # run decoding
        evidence = temporal_decoding(X_all, yb, time_,
                                        n_bins = 10,
                                        size_window = 30,
                                        n_folds = 10,
                                        classifier = 'LDA',
                                        use_pca = True,
                                        pca_components = 0.9,
                                        temporal_dynamics = True,
                                        demean = 'no') 
        
        # save classifier output  
        with open(dec_fname, 'wb') as fp:
            pickle.dump([evidence,time_], fp)
    else: 
        # load classifier output
        print('preloaded data imported')
        with open (dec_fname, 'rb') as fp:
            [evidence, time_] = pickle.load(fp)
            
    # read behaviour
    df_read = pd.read_csv(projectloc + '/data/behavioural/all_behavioural.csv')
    df_read = df_read.loc[(df_read['subject'] == sb_count), :]
    
    # select trials used for classification
    inx =  (df_read['stimulus_nr'] < 3 )
    # select a subset of trials for analysis
    inx2 =  (df_read['cue'][inx] == 2 )

    evidence['single_trial_evidence'] = evidence['single_trial_evidence'][inx2]


    # cross decoding
    for i, variable in enumerate(['stim1', 'prev_probe_ang']):
        # bin continuous variable into discrete classes
        y, evidence['y'] = bin_array(np.array(df_read[variable][inx][inx2]), 10)
        evidence = cos_convolve(evidence)
        classifier_output[:, sb_count, i] = evidence['cos_convolved']
    
    # compute shift in tuning curve
    y_diff = pycircstat.cdiff(df_read['stim1'],df_read['presented'])
    s, c, w = compute_asymmetry(evidence['single_trial_ev_centered'],
                                y_diff[inx][inx2], min_deg = 30, max_deg = 60)     
    shift[:, sb_count], left[:, :, sb_count], right[:, :, sb_count] = s, c, w
        
        
            
            
            
            