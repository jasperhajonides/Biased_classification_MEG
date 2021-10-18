#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:19:57 2020

@author: jasperhvdm
"""

#
#def startup_script:
#    
#      data = scipy.io.loadmat('/Users/jasperhvdm/Documents/DPhil/PROJECTS/EXP8_UpdProtec/data/decoding_data/306ch_data_' + subject_labels[sb_count] +suffix)
#      print('Preparing data...')
#      X,thetas,stimulus_nr,presented_angle,time = concatenate_data(data)
#      
#     presented_angles.append(presented_angle)
#     stimulus_nrs.append(stimulus_nr)
#     thetas_all.append(thetas)
#     
#     
#     return presented_angles, stimulus_nrs, thetas_all

import math
import numpy as np
import scipy.io
import pandas as pd
import pycircstat
from temp_dec import decoding_functions
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

from scipy.optimize import leastsq




def prep_sensor_data(X,data,smooth=10,baseline=True):
    """ To tidy up the main script we will do simple preprocessing
        of the MEG sensors here
        
        - select only MEG gradiometer and magnetometers
        - multiply magnetometers so that amplitude and variance is similar to 
            that of gradiometers
        - temporally smooth data
        - baseline MEG data"""
    
    meg_ch = scipy.io.loadmat('/Users/jasperhajonides/Documents/EXP8_UpdProtec/MEG/analyses/Channel_selection/posterior_electrodes.mat')
    #include right channels
    sens = meg_ch['sens'].T[0]
    sens_meg = sens[0:306]
    X = X[:,0:306,:]
    X[:,2::3,:] = X[:,2::3,:]*20
    # X = X[:,sens_meg==1,:]
#    X = X[:,309:311,:] #eyes
#         X_all = X_all[:,sens_meg[0]==1,:]
    
    ####   smooth it  #######
#    X_smooth = np.zeros((X.shape))
    if smooth > 1:
        print('smoothing')
        for sens in range(0,X.shape[1]):
            X[:,sens,:] = gaussian_filter1d(X[:,sens,:],smooth)
    ########################## 
    
    
    
    #remember to feed data from prev step
    if baseline == True:
        print('baseline each trial')
        for tp in range(X.shape[2]):
            X[:,:,tp] = X[:,:,tp] - X[:,:,30:60].mean(2) 
            
    return X

#    Y_df,lab =get_behaviour(thetas_all[sb_count],trial_selection,mapping_behaviour_to_MEG(sb_count),presented_angles[sb_count])

    
    
    

def concatenate_data(data,sb_count,toi=[-.4,3.5]):
    
    """ Function to take the data from the .mat file and concatenates data from stimulus 1 and stimulus 2.
    This gets rid of the jitter around stimulus 1"""
    thetas = data['y']
    X = data['X']


    # remove jitter from stim 1
    ntrls, nsens,ntps= X.shape
    jit = 2.1 - thetas[:,13]
    jit_tps = np.round(jit/0.005)
    X_stim1 = np.zeros((ntrls, nsens,ntps))
    for trl in range(0,X.shape[0]):
        X_cut = X[trl,:,int(jit_tps[trl]):ntps]
        X_stim1[trl,:,0:int(ntps-jit_tps[trl])] = X_cut

    #locked to stim 1
    X_stim1[((thetas[:,0]==1) & (thetas[:,16]==0)),:,:] = X_stim1[((thetas[:,0]==1) & (thetas[:,16]==0)),:,:] - np.nanmean(X_stim1[((thetas[:,0]==1) & (thetas[:,16]==0)),:,:],0)
    X_stim1[((thetas[:,0]==2) & (thetas[:,16]==0)),:,:] = X_stim1[((thetas[:,0]==2) & (thetas[:,16]==0)),:,:] - np.nanmean(X_stim1[((thetas[:,0]==2) & (thetas[:,16]==0)),:,:],0)
    X_stim1[((thetas[:,0]==1) & (thetas[:,16]==1)),:,:] = X_stim1[((thetas[:,0]==1) & (thetas[:,16]==1)),:,:] - np.nanmean(X_stim1[((thetas[:,0]==1) & (thetas[:,16]==1)),:,:],0)
    X_stim1[((thetas[:,0]==2) & (thetas[:,16]==1)),:,:] = X_stim1[((thetas[:,0]==2) & (thetas[:,16]==1)),:,:] - np.nanmean(X_stim1[((thetas[:,0]==2) & (thetas[:,16]==1)),:,:],0)
    
    #locked to stim 2
    X[((thetas[:,0]==1) & (thetas[:,16]==0)),:,:] = X[((thetas[:,0]==1) & (thetas[:,16]==0)),:,:] - np.nanmean(X[((thetas[:,0]==1) & (thetas[:,16]==0)),:,:],0)
    X[((thetas[:,0]==2) & (thetas[:,16]==0)),:,:] = X[((thetas[:,0]==2) & (thetas[:,16]==0)),:,:] - np.nanmean(X[((thetas[:,0]==2) & (thetas[:,16]==0)),:,:],0)
    X[((thetas[:,0]==1) & (thetas[:,16]==1)),:,:] = X[((thetas[:,0]==1) & (thetas[:,16]==1)),:,:] - np.nanmean(X[((thetas[:,0]==1) & (thetas[:,16]==1)),:,:],0)
    X[((thetas[:,0]==2) & (thetas[:,16]==1)),:,:] = X[((thetas[:,0]==2) & (thetas[:,16]==1)),:,:] - np.nanmean(X[((thetas[:,0]==2) & (thetas[:,16]==1)),:,:],0)
     
    
    
    #selection of trials where gratings are presented
    #stim1
    inx1 =  (thetas[:,16]==1) | ((thetas[:,0]==1) &(thetas[:,16]==0))
    X_stim1 = X_stim1[inx1,:,:]
    thetas1 = thetas[inx1,:]
    #stim2
    inx2 =  (thetas[:,16]==1) | ((thetas[:,0]==2) & (thetas[:,16]==0)) 
    X_stim2 = X[inx2,:,:]
    thetas2 = thetas[inx2,:]
    


    #cut out the right timepoints
    time_st1 = data['time'][0]+2.1
    time_st2 = data['time'][0]

    X_stim2_long = X_stim2
    X_stim1_long = X_stim1

    # X_stim1 = X_stim1[:,:,(time_st1 >toi[0]) & (time_st1 <toi[1])]
    # X_stim2 = X_stim2[:,:,(time_st2 >toi[0]) & (time_st2 <toi[1])]
    # time_out = time_st1[(time_st1 >-.4) & (time_st1 <1.25)]

    X_stim1 = X_stim1[:,:,(time_st1 >=-.4) & (time_st1 <=3.5)]
    X_stim2 = X_stim2[:,:,(time_st2 >= -.4) & (time_st2 <=3.5)]
    time_out = time_st1[(time_st1 >=-.4) & (time_st1 <=3.5)]

    # create new variable for stim 1 and stim 2
    stimulus_nr = np.append(np.ones(len(thetas1)),2*np.ones(len(thetas2)),axis=0)
    presented_angle = np.append(thetas1[:,1],thetas2[:,2],axis=0)
    
    # concatenate stim 1 and stim 2
    X_all = np.concatenate((X_stim1,X_stim2),axis=0)
    thetas_all = np.concatenate((thetas1,thetas2),axis=0)


    # remove first trials of runs
    
    y,lab=get_behaviour(thetas_all,stimulus_nr>0,mapping_behaviour_to_MEG(sb_count),presented_angle)
    
    #return all data but without the first trial in each run.
    return X_all[~np.isnan(y[:,2]),:,:], thetas_all[~np.isnan(y[:,2]),:], stimulus_nr[~np.isnan(y[:,2])], presented_angle[~np.isnan(y[:,2])],time_out



def join_together(all_out):
    """join tuning curves with offsets together"""
    repeats,bins,shifts = all_out.shape
    tuning = np.zeros((bins*shifts,repeats))
    if all_out.shape[2] == 8:
        tuning[0::8,:] = all_out[:,:,7].T
        tuning[1::8,:] = all_out[:,:,6].T
        tuning[2::8,:] = all_out[:,:,5].T
        tuning[3::8,:] = all_out[:,:,4].T
        tuning[4::8,:] = all_out[:,:,3].T
        tuning[5::8,:] = all_out[:,:,2].T
        tuning[6::8,:] = all_out[:,:,1].T
        tuning[7::8,:] = all_out[:,:,0].T
    elif all_out.shape[2] == 4:
        tuning[0::4,:] = all_out[:,:,3].T
        tuning[1::4,:] = all_out[:,:,2].T
        tuning[2::4,:] = all_out[:,:,1].T
        tuning[3::4,:] = all_out[:,:,0].T
    elif all_out.shape[2] == 20:
        tuning[0::shifts,:] = all_out[:,:,19].T
        tuning[1::shifts,:] = all_out[:,:,18].T
        tuning[2::shifts,:] = all_out[:,:,17].T
        tuning[3::shifts,:] = all_out[:,:,16].T
        tuning[4::shifts,:] = all_out[:,:,15].T
        tuning[5::shifts,:] = all_out[:,:,14].T
        tuning[6::shifts,:] = all_out[:,:,13].T
        tuning[7::shifts,:] = all_out[:,:,12].T
        tuning[8::shifts,:] = all_out[:,:,11].T
        tuning[9::shifts,:] = all_out[:,:,10].T
        tuning[10::shifts,:] = all_out[:,:,9].T
        tuning[11::shifts,:] = all_out[:,:,8].T
        tuning[12::shifts,:] = all_out[:,:,7].T
        tuning[13::shifts,:] = all_out[:,:,6].T
        tuning[14::shifts,:] = all_out[:,:,5].T
        tuning[15::shifts,:] = all_out[:,:,4].T
        tuning[16::shifts,:] = all_out[:,:,3].T
        tuning[17::shifts,:] = all_out[:,:,2].T
        tuning[18::shifts,:] = all_out[:,:,1].T
        tuning[19::shifts,:] = all_out[:,:,0].T
    elif all_out.shape[2] == 12:
        tuning[0::shifts,:] = all_out[:,:,0].T
        tuning[1::shifts,:] = all_out[:,:,1].T
        tuning[2::shifts,:] = all_out[:,:,2].T
        tuning[3::shifts,:] = all_out[:,:,3].T
        tuning[4::shifts,:] = all_out[:,:,4].T
        tuning[5::shifts,:] = all_out[:,:,5].T
        tuning[6::shifts,:] = all_out[:,:,6].T
        tuning[7::shifts,:] = all_out[:,:,7].T
        tuning[8::shifts,:] = all_out[:,:,8].T
        tuning[9::shifts,:] = all_out[:,:,9].T
        tuning[10::shifts,:] = all_out[:,:,10].T
        tuning[11::shifts,:] = all_out[:,:,11].T
      
    return tuning

def median_split(DAT,thetas,time,y_other=None,baseline=[-.2,0]):
    """median split from evidence on DAT[trials x time points] either based on
    itself or on y_other[trials x time points] within a certain timeframe"""
    
    indices = []
    out_DAT=[]
    ts= (time>=baseline[0]) & (time < baseline[1])

    split = y_other[:,ts].mean(1) >= np.median(y_other[:,ts].mean(1))
    
    indices.append([thetas[split==b,24] for b in range(0,2)])
    out_DAT.append([DAT[split==b,:] for b in range(0,2)])     

    return indices,out_DAT
        
    
    



def cosine_least_squares_Biasfit(output_bias):
    """Fit a sinusoidal function to tuning curve data to obtain model 
    fit Parameters

    Parameters
    ----------
    output_bias : struct
           containing the variable ['phase_conditions'] which is a 2d array

    Returns
    --------
    dictionary:
        data_fit : ndarray
                estimated cosine fit to data
        amplitude: ndarray
                highest value - mean of the cosine
       
        mean: ndarray
                mean value of cosine
    """
    
    # 2d array, [phase bins x time].
    # values indicate the centre of the tuning curve when a distractor with a
    # relative orientation is presented  [-0.5 pi to 0.5 pi]
    # e.g. distance between stimulus and distractor orientation is 0: phase shift = 0
    # e.g. distance between stimulus and distractor orientation is -.37: phase shift = -.08
    
    tuning = output_bias['phase'].T
    
    
    n_steps = tuning.shape[0]
    t = np.linspace(-np.pi+(0.5*2*np.pi/n_steps),np.pi-(0.5*2*np.pi/n_steps),n_steps)
    guess_mean = 0
    guess_std = np.std(tuning)#3*np.std(0.0033)/(2**0.5)/(2**0.5)
    guess_amp = 0.5

    # we'll use this to plot our first estimate. This might already be good enough for you
    data_first_guess = guess_std*np.sin(t) + guess_mean

    # if input is 2d array we loop over the second dimension
    if len(tuning.shape) > 1:
        print('Now fit sinus to biascurve')
        repeats = tuning.shape[1]
        #initialise parameter estimates
        est_amp = np.zeros(repeats)
        est_mean = np.zeros(repeats)
        data_fit = np.zeros((n_steps,repeats))

        for rep in range(0,tuning.shape[1]):
            optimize_func = lambda x: x[0]*np.sin(t) + x[1] - tuning[:,rep]
            ea, em = leastsq(optimize_func, [guess_amp, guess_mean])[0]
            est_amp[rep] = ea # if max amplitude is negative we also multiply phase shift by -1
            est_mean[rep] = em

            # recreate the fitted curve using the optimized parameters
            data_fit[:,rep] = ea*np.sin(t) + em
    else:
        optimize_func = lambda x: x[0]*np.sin(t) + x[1] - tuning
        est_amp, est_mean = leastsq(optimize_func, [guess_amp, guess_mean])[0]


        # recreate the fitted curve using the optimized parameters
        data_fit = est_amp*np.sin(est_freq*t) + est_mean


    output = {
    "data_fit": data_fit,
    "amplitude": est_amp,
    "mean" : est_mean
    }

    return output

def bin_array(y,nr_bins):
    """Takes circular array and digitises into n bins"""
    y = np.array(y)
    bins = np.arange(-math.pi-0.000001,math.pi - (2*math.pi/nr_bins),(2*math.pi/nr_bins))
    y_bin = np.digitize(pycircstat.cdiff(y,0), bins) 
    return y,y_bin
    


def mapping_behaviour_to_MEG(sb_count):
    """ mapping from MEG data to pull data from behavioural structure."""
    
    ma = np.zeros(20)
    ma[0] = 1
    ma[1] = 19
    ma[2] = 2
    ma[3] = 3
    ma[4] = 4
    ma[5] = 5
    ma[6] = 6
    ma[7] = 7
    ma[8] = 8
    ma[9] = 9
    ma[10] = 10
    ma[11] = 11
    ma[12] = 12
    ma[13] = 13
    ma[14] = 14
    ma[15] = 15
    ma[16] = 16
    ma[17] = 17
    ma[18] = 20
    ma[19] = 18
    
    return ma[sb_count]
    

def load_subject(sb_count,var,stimulus_nrs,presented_angles,thetas_all,stimulus=1):
    """
 0'stim1',
 1'stim2',
 2'prev_probe_ang'
 3'stim1_stim2_diff'
 4,'prev2_probe_ang',
 5'probed_ang',
 6'notprobed_ang',
 7'diff_stim_prev_pang',
 8'abs_diff_stim_prev_pang',
 9'cue',
 10'wmload',
 11'future_probe_ang']
     
 """

#    
#    mapping_behaviour_to_MEG = np.zeros(20)
#    mapping_behaviour_to_MEG[0] = 1
#    mapping_behaviour_to_MEG[1] = 19
#    mapping_behaviour_to_MEG[2] = 2
#    mapping_behaviour_to_MEG[3] = 3
#    mapping_behaviour_to_MEG[4] = 4
#    mapping_behaviour_to_MEG[5] = 5
#    mapping_behaviour_to_MEG[6] = 6
#    mapping_behaviour_to_MEG[7] = 7
#    mapping_behaviour_to_MEG[8] = 8
#    mapping_behaviour_to_MEG[9] = 9
#    mapping_behaviour_to_MEG[10] = 10
#    mapping_behaviour_to_MEG[11] = 11
#    mapping_behaviour_to_MEG[12] = 12
#    mapping_behaviour_to_MEG[13] = 13
#    mapping_behaviour_to_MEG[14] = 14
#    mapping_behaviour_to_MEG[15] = 15
#    mapping_behaviour_to_MEG[16] = 16
#    mapping_behaviour_to_MEG[17] = 17
#    mapping_behaviour_to_MEG[18] = 20
#    mapping_behaviour_to_MEG[19] = 18


    trial_selection = stimulus_nrs[sb_count]==1
    Y_df,lab =get_behaviour(thetas_all[sb_count],trial_selection,mapping_behaviour_to_MEG(sb_count),presented_angles[sb_count])

    df1 = pd.DataFrame()
    for i in range(Y_df.shape[1]):
        df1[lab[i]] = Y_df[:,i]
    df1['RT'] = thetas_all[sb_count][trial_selection,12]
    df1['stimulus_nr'] = 1
    df1['t1_offset'] = thetas_all[sb_count][trial_selection,13]
    df1['trial_number'] = thetas_all[sb_count][trial_selection,15]
    df1['usetrl'] = thetas_all[sb_count][trial_selection,14]
    df1['resp'] = thetas_all[sb_count][trial_selection,6]
    df1['presented'] = thetas_all[sb_count][trial_selection,1]
    
    
    trial_selection = stimulus_nrs[sb_count]==2
    Y_df,lab =get_behaviour(thetas_all[sb_count],trial_selection,mapping_behaviour_to_MEG(sb_count),presented_angles[sb_count])
    
    df2 = pd.DataFrame()
    for i in range(Y_df.shape[1]):
        df2[lab[i]] = Y_df[:,i]
    df2['RT'] = thetas_all[sb_count][trial_selection,12]
    df2['stimulus_nr'] = 2
    df2['t1_offset'] = thetas_all[sb_count][trial_selection,13]
    df2['trial_number'] = thetas_all[sb_count][trial_selection,15]
    df2['usetrl'] = thetas_all[sb_count][trial_selection,14]
    df2['resp'] = thetas_all[sb_count][trial_selection,6]
    df2['presented'] = thetas_all[sb_count][trial_selection,2]

    df_all = df1.append(df2,ignore_index=True)
    return df_all
    

def get_behaviour(thetas,trial_selection,sb,presented_angle):
    matlab_behav = scipy.io.loadmat('/Users/jasperhajonides/Documents/EXP8_UpdProtec/data/behavioural/all_behavioural_data_struct.mat')
    
    df = pd.DataFrame()
    df['stim1']=matlab_behav['x']['stim1'][0][0][:,0]
    df['stim2']=matlab_behav['x']['stim2'][0][0][:,0]
    df['prev_probe_ang']=matlab_behav['x']['prev_probe_ang'][0][0][:,0]
    df['error']=matlab_behav['x']['error'][0][0][:,0]
    df['subject']=matlab_behav['x']['subject'][0][0][:,0]
    df['MEGMRI']=matlab_behav['x']['MEGMRI'][0][0][:,0]
    df['blocktrial']=matlab_behav['x']['blocktrial'][0][0][:,0]
    df['wmload']=matlab_behav['x']['wmload'][0][0][:,0]
    df['cue']=matlab_behav['x']['cue'][0][0][:,0]
    df['prev2_probe_ang']=matlab_behav['x']['prev2_probe_ang'][0][0][:,0]
    df['probed_ang']=matlab_behav['x']['probed_ang'][0][0][:,0]
    df['notprobed_ang']=matlab_behav['x']['notprobed_ang'][0][0][:,0]
    df['future_probe_ang']=matlab_behav['x']['future_probe_ang'][0][0][:,0]

    iX = (df['subject']==sb)&(df['MEGMRI']==1)
    total_trls = np.sum(iX)
    y_df = np.zeros((total_trls,13))
    Y_labels = ['stim1','stim2','prev_probe_ang','stim1_stim2_diff','prev2_probe_ang','probed_ang','notprobed_ang','diff_stim_prev_pang','abs_diff_stim_prev_pang','cue','wmload','future_probe_ang','error']
    
    y_df[:,0] = np.array(df.loc[iX,'stim1'])/90*np.pi
    y_df[:,1] = np.array(df.loc[iX,'stim2'])/90*np.pi
    y_df[:,2] = np.array(df.loc[iX,'prev_probe_ang'])/90*np.pi
    y_df[:,3]= pycircstat.descriptive.cdiff(y_df[:,0],y_df[:,1]) #
    y_df[:,4]= np.array(df.loc[iX,'prev2_probe_ang'])/90*np.pi
    y_df[:,5]= np.array(df.loc[iX,'probed_ang'])/90*np.pi
    y_df[:,6]= np.array(df.loc[iX,'notprobed_ang'])/90*np.pi
    y_df[:,9]= np.array(df.loc[iX,'cue'])
    y_df[:,10]= np.array(df.loc[iX,'wmload'])
    y_df[:,11]= np.array(df.loc[iX,'future_probe_ang'])/90*np.pi
    y_df[:,12]= np.array(df.loc[iX,'error'])


    
    
    #0-400
    trial_indices = (np.arange(400)+1)
    
    #select MEG data indices
    trial_sel = thetas[trial_selection,24]
    Y_df = np.zeros((trial_sel.shape[0],13))*np.nan
    
    for var in range(Y_df.shape[1]):
        for i,trl in enumerate(trial_sel):
            Y_df[i,var] = y_df[trial_indices==trl,var]
    
    #these two following calculations involve the presented angles. 
    #hence we have to fist subselect the used trials above. and then
    #calculate these anglular differences
    
    # previous probe angle and presented angle on that trial (-pi to pi)
    Y_df[:,7] = pycircstat.descriptive.cdiff(Y_df[:,2],presented_angle[trial_selection])
    # previous probe angle and presented stimulus - taking the absolute (0-pi)
    Y_df[:,8] = abs(pycircstat.descriptive.cdiff(Y_df[:,2],presented_angle[trial_selection]))
    

    return Y_df,Y_labels


def temporal_crossdecoding(x_all, y, y2, time, n_bins=12, size_window=5,
                      n_folds=5, classifier='LDA', use_pca=False,
                      pca_components=.95, temporal_dynamics=True, 
                      demean='window',n_steps = 1):

    """
    Apply a multi-class classifier (amount of class is equal to n_bins)
    to each time point.

    The temporal_dynamics decoding takes a time course and uses a 
    sliding window approach to decode the feature of interest. We reshape 
    the window from trials x channels x time to trials x channels*time and 
    demean every channel


    Temporal_dynamics:
    (rows=features; columns=time)
    (n = window_size)

    	t  t+1  t+2     t+n		combined t until t+n
        1 	 1 	 1 ..	 1        1
        2 	 2 	 2 ..	 2        2
        3 	 3 	 3 ..	 3        3
        4 	 4 	 4 ..	 4 ---->  4
        5 	 5 	 5 ..	 5        5
        6 	 6 	 6 ..	 6        6
        7 	 7 	 7 .. 	 7        7
    							  1
								  2
								  3
								  4
								  5
								  6
								  7
								  1
								  ..
								  7

    Parameters
    ----------
    x_all : ndarray
            trials by channels by time
    y     : ndarray
            vector of labels of each trial
    time  : ndarray
            vector with time labels locked to the cue
    bins  : integer
            how many stimulus classes are present
    size_window :  integer
            size of the sliding window
    n_folds :  integer
            folds of cross validation
    classifier : string
            Classifier used for decoding
            options:
            - LDA: LinearDiscriminantAnalysis
            - LG: LogisticRegression
            - maha: nearest neighbours mahalanobis distance
            - GNB: Gaussian Naive Bayes
    use_pca     : bool
    		Apply PCA or not
    pca_components   : integer
            reduce features to N principal components,
            if N < 1 it indicates the % of explained variance
    temporal_dynamics : bool
            use sliding window (default is True),
            if false its just single time-point decoding

    Returns
    --------
    dictionary:
        accuracy : ndarray
                dimensions: time
                matrix containing class predictions for each time point.
        centered_prediction: ndarray
                dimensions: classes, time
                matrix containing evidence for each class for each time point.
        single_trial_evidence: ndarray
                dimensions: trials, classes, time
                evidence for each class, for each timepoint, for each trial
        cos_convolved: ndarray
                dimensions: time
                cosine convolved evidence for each timepoint.
    """

    #### do dimensions of the input data match?
    check_input_dim(x_all, y, time)

    # Get shape
    [n_trials, n_features, n_time] = x_all[:, :, :].shape

	#initialise variables
    # np.nan to avoid zeroes in resulting variable
    single_trial_evidence = np.zeros(([n_trials, n_bins, n_time, n_time])) * np.nan
    label_pred = np.zeros(([n_trials, n_time, n_time])) * np.nan
    accuracy = np.zeros((n_time,n_time)) * np.nan
    x_demeaned = np.zeros((n_trials, n_features, size_window)) * np.nan
    x_demeaned2 = np.zeros((n_trials, n_features, size_window)) * np.nan

    tp = -1
    for tpx in range((size_window-1), n_time,n_steps):
        tp+=1
        print('completed %s percent' %(tpx/n_time))
        if temporal_dynamics:
            #demean features within the sliding window if demean=='window'
            for count, s in enumerate(np.arange(size_window)-(size_window-1)):
                x_demeaned[:, :, count] = (x_all[:, :, tpx+s] -
                                           x_all[:, :, (tpx-(size_window-1)):(tpx+1)].mean(2)*
                                           ('window' in demean))
            # reshape into trials by features*time
            X = x_demeaned.reshape(x_demeaned.shape[0],
                                   x_demeaned.shape[1]*x_demeaned.shape[2])
        else:
            X = x_all[:, :, tpx]
        
        
        # reduce dimensionality
        if use_pca:
            pca = PCA(n_components=pca_components)
            X = pca.fit(X).transform(X)
            
                
        tp2 = -1
        for tp2x in range((size_window-1), n_time,n_steps):
            tp2 +=1
            # again, create sliding window
            if temporal_dynamics:
                #demean features within the sliding window if demean=='window'
                for count, s in enumerate(np.arange(size_window)-(size_window-1)):
                    x_demeaned2[:, :, count] = (x_all[:, :, tp2x+s] -
                                               x_all[:, :, (tp2x-(size_window-1)):(tp2x+1)].mean(2)*
                                               ('window' in demean))
                # reshape into trials by features*time
                X2 = x_demeaned2.reshape(x_demeaned2.shape[0],
                                       x_demeaned2.shape[1]*x_demeaned2.shape[2])
            else:
                X2 = x_all[:, :, tp2x]
                
            X2 = pca.transform(X2)
    
            #train test set
            rskf = RepeatedStratifiedKFold(n_splits=n_folds,
                                           n_repeats=1, random_state=42)
            for train_index, test_index in rskf.split(X, y):
                x_train, x_test = X[train_index], X2[test_index]
                y_train = y[train_index]
    
                #standardisation
                scaler = StandardScaler().fit(x_train)
                x_train = scaler.transform(x_train)
                x_test = scaler.transform(x_test)
    
                #initiate classifier
                clf = get_classifier(classifier, x_train)
    
                # train
                clf.fit(x_train, y_train)
    
                # test either binary or class probabilities
                single_trial_evidence[test_index, :, tp,tp2] = clf.predict_proba(x_test)
                label_pred[test_index, tp,tp2] = clf.predict(x_test)
    
            #compute accuracy score 
            accuracy[tp,tp2] = accuracy_score(y2, label_pred[:, tp,tp2])

    
    output = {"accuracy": accuracy,
        "single_trial_evidence": single_trial_evidence,
        "time": time,
        "y": y}

    return output
