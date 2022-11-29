#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 19:53:51 2022

@author: jasperhajonides
"""

    
import os 
import pickle
import math
import pycircstat
import pandas as pd
import numpy as np
import scipy
from temp_dec.decoding_functions import * 


def reject_outliers(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)

def bin_array(y,nr_bins):
    """Takes circular array and digitises into n bins"""
    y = np.array(y)
    bins = np.arange(-math.pi-0.000001,math.pi - (2*math.pi/nr_bins),(2*math.pi/nr_bins))
    y_bin = np.digitize(pycircstat.cdiff(y,0), bins) 
    return y,y_bin

def tsplot(ax, data, time, color = 'k', linestyle = 'solid',legend='target',chance=0.0):
    x = time
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    se = (sd/np.sqrt(len(data)))
    cis = (est - se, est + se)
    
    ax.fill_between(x,cis[0],cis[1],alpha = 0.2, facecolor = color)
    ax.plot(x,est,color = color, linestyle = linestyle,label=legend)
    ax.margins(x=0)
    ax.hlines(chance,-.4,0.8,color='gray',linestyle='dashed')

    ax.set_xlabel('Times',fontsize=14)
    ax.set_ylabel('evidence',fontsize=14)  # Area Under the Curve
    ax.axvline(.0, color='k', linestyle='-')

def get_config(sb:int(),
               bias_type: str(),
               pca_variance: float() = .90,
               size_window: int() = 30,
               epoch: str() = 'med') -> dict:
    
    config = dict()
    # subject subject ID
    config['subject'] = 'S%02d' %(sb+1)
    config['sb'] = sb
    # identifier for this dataset
    config['suffix'] = '_200hz_LongPreCuePeriod_FINAL_single_tp_demeaned'
    
    # pca variance included in decodingßßß
    config['pca_var'] = pca_variance
    # size of the window for temporal decoding
    config['size_window'] = size_window
    # move forward in time increments of this integer. 1 means every tp is
    # included, 2 means ever other tp is skipped. Only use for speed.
    config['steps'] = 1
    # MEG channels to include:
    config['channels'] = 'all_chans'
    # bin orientation angles in this amount of bins
    config['nr_bins'] = 10
    # cross-validation in this many folds
    config['n_folds'] = 10
    # project directory
    config['projectloc'] = '/Users/jasperhajonides/Documents/EXP8_UpdProtec'

    if epoch == 'med':
        config['time_lim'] = [-.4, 0.9] 
        config['nr_tps'] = 260
        
        
    if bias_type == 'within_trial_protect':
        config['stimulus_nr_label'] = 'stimulus_2'
        config['stimulus_nr'] = [2]
        config['trial_subselection'] = dict({'cue':[1]})
        config['min_deg'] = 10
        config['max_dex'] = 50
        config['bias_source'] = 'stim1'
    elif bias_type == 'within_trial_update':
        config['stimulus_nr_label'] = 'stimulus_2'
        config['stimulus_nr'] = [2]
        config['trial_subselection'] = dict({'cue':[2]})
        config['min_deg'] = 10
        config['max_dex'] = 50           
        config['bias_source'] = 'stim1'                         
    elif bias_type == 'between_trial':
        config['stimulus_nr_label'] = 'sim_all'
        config['stimulus_nr'] = [1, 2]
        config['min_deg'] = 0
        config['max_dex'] = 60
        config['trial_subselection'] = dict({'trial_number': 
                                             np.arange(1,401)[(np.arange(
                                                 1,401)-1)%50 > 0]})
        config['bias_source'] = 'prev_probe_ang'

            
    config['save_file_name'] = (config['projectloc'] + '/scores/' +
                                config['subject'] + 
                                config['suffix']  + '_time'+
                                str(config['time_lim'][0]) +'_to_' + 
                                str(config['time_lim'][1]) +'_LDA_' +'windowsize_' + 
                                str(config['size_window']) + '_pca' + 
                                str(config['pca_var']) +'_steps' +
                                str(config['steps'])  + '_' + 
                                config['stimulus_nr_label'] + '_' + 
                                config['channels'] + '_' + 'test_on_all_trials') 

    return config
        
    
def decoding_function(config, overwrite):
    

    if  not(os.path.exists(config['save_file_name'])) or overwrite == True:
        print('running ' + config['subject'] )
        
        # read neural data
        with open (config['projectloc']  + 
                   '/data/decoding_data/%s_preprocessed_MEG_data' % config['subject'], 
                   'rb') as fp:
            [X,thetas,stimulus_nr,presented_angle,time] = pickle.load(fp)
        print('loaded data')
        # read behaviour
        df_read = pd.read_csv(config['projectloc']  + '/data/behavioural/all_behavioural_May2021.csv')
        df_read = df_read.loc[(df_read['subject'] == config['sb']) , :]
        
        
        # Get neural data and behavioural data for selected trials. 
        inx = df_read['stimulus_nr'].isin(config['stimulus_nr'])
        X_all = X[inx,:,:]
        y,yb = bin_array(df_read['presented'][inx],config['nr_bins'])
        
        #select time
        X_all = X_all[:,:,(time >= config['time_lim'][0]) & (time <= config['time_lim'][1]) ] 
        time_ = time[(time >= config['time_lim'][0]) & (time <= config['time_lim'][1]) ] #adjust the time vector
    
        # run decoding
        evidence = temporal_decoding(X_all, yb, size_window=config['size_window'], n_folds=config['n_folds'],
               classifier='LDA', pca_components=config['pca_var'], demean=False) 
        with open(config['save_file_name'], 'wb') as fp:
            pickle.dump([evidence,time_], fp)
    else:
        print('Load data for %s' %config['subject'])
        with open (config['save_file_name'], 'rb') as fp:
            [evidence,time_] = pickle.load(fp)
    return evidence, time
   
    
    