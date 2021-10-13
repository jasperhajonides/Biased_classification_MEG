# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

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

#%% 
%load_ext autoreload
%autoreload 2

import sys
sys.path.append('/Users/jasperhajonides/Documents/scripts/DPhil_toolbox/')
from pc_neuro import *
sys.path.append('/Users/jasperhajonides/Documents/EXP8_UpdProtec/scripts/functions')
from UP_prep import *
sys.path.append('/Users/jasperhajonides/Documents/scripts/temp_dec/temp_dec')
from decoding_functions import *


#%% paths and variables

subject_labels = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20']
projectloc = '/Users/jasperhajonides/Documents/EXP8_UpdProtec'
decodingloc = projectloc + '/data/decoding_data'
prep_data_storage_loc = projectloc + '/MEG/data/preprocessed'

#imports
fitting_params = scipy.io.loadmat(projectloc + '/data/Fitting_params_Biases.mat')
post_s = scipy.io.loadmat(projectloc + '/MEG/analyses/Channel_selection/posterior_electrodes.mat')
neighb = scipy.io.loadmat(projectloc + '/data/channels/neighbours.mat')

#%%
# with open('%s' %prep_data_storage_loc + '/stimulus_info_for_decoding', 'wb') as fp:
#     pickle.dump([thetas_all,stimulus_nrs,presented_angles,time], fp)

def reject_outliers(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)
#%% run decoding
overwrite = False

steps=1
nr_bins = 10
nsubs = 20
size_window = 30 #30
pca_var = .90 #.87 #.95
is_classifier='LDA'

time_lim = [-0.4 , 0.9] #limits [-0.4 , 0.9]#[-0.4 , 3.5] 
nr_tps = 260 #260,681,781

nbin = 100
pbin=1/nbin #

#---------------
#
# steps=1
# nr_bins = 8
# nsubs = 20
# size_window = 1 #30
# pca_var = .95 #.87 #.95
# is_classifier='LDA'

# time_lim = [-.4, 0.9] #limits
# nr_tps = 260

nr_of_components = np.zeros((nr_tps,20))
classifier_output = np.zeros((nr_tps,nsubs,11))
classifier_output_tuning = np.zeros((nr_bins,nr_tps,nsubs,4))
classifier_output_tuning_distractor = np.zeros((nr_bins,nr_tps,nsubs))
distractor_tuning = np.zeros((2,nr_bins*1,nr_tps,nsubs,3))
distractor_tuning_smoothed = np.zeros((nr_bins*1,nr_tps,nbin,nsubs,3))

Distractor_bias = np.zeros((nr_bins*1,nr_tps,20))* np.nan
#Distractor_bias_conditions = np.zeros((nr_bins*4,2,nr_tps,20))* np.nan


phase_cosfit = np.zeros((2,nr_tps,nsubs,3))
amplitude_cosfit = np.zeros((2,nr_tps,nsubs,3))
#datafit_tuning = np.zeros((nr_bins*1,nr_tps,35,nsubs))

bias_ampl = np.zeros((nr_tps,nsubs))
bias_mean = np.zeros((nr_tps,nsubs))
bias_datafit = np.zeros((nbin,nr_tps,nsubs))

right = np.zeros((nr_bins,nr_tps,nsubs))
left = np.zeros((nr_bins,nr_tps,nsubs))
shift = np.zeros((nr_tps,nsubs))

if 'thetas_all' not in  locals():
    with open ('%s' %prep_data_storage_loc + '/stimulus_info_for_decoding', 'rb') as fp:
            [thetas_all,stimulus_nrs,presented_angles,time] = pickle.load(fp)


            

for sb_count in range(0,nsubs):
    print(sb_count) 

    # 'stim', 'prev_pang', 'not_presented_stim', 'future'
    stim_nr = 'stim'  #'stimulus_2' 'stimulus_1' 'stimulus_2_dec_stimulus_1' 'stimulus_1_dec_stimulus_2'  'probe' 'unprobe', 'prevPang_at_Stim2','prevPang_at_Stim1','difference_stim2_prevPang','difference_stim1_prevPang','difference_stim2_stim1'
    channels = 'all_chans' # 'posterior_chans' 'all_chans' 'eyes' 'all_chans_smoothed', 'not_posterior_chans' ,'all_chans_ALPHA'
    suffix = '_200hz_LongPreCuePeriod'  #_200hz_LongPreCuePeriod_Longpostcue
    Baseline = '_single_tp' #'_baselined', '', '_single_tp', '_no_baseline'
    train_test = 'test_on_all_trials'

    dec_fname = projectloc + '/scores/' + subject_labels[sb_count] + suffix + '_TEMP'+Baseline + '_time'+str(time_lim[0]) +'_to_' + str(time_lim[1]) +'_' + is_classifier + '_' +'windowsize_' + str(size_window) + '_pca' + str(pca_var) +'_steps' +str(steps)  + '_' + stim_nr + '_' + channels + '_'+ train_test 
    if  not(os.path.exists(dec_fname)) or overwrite == True:

        # read MEG data
        with open (prep_data_storage_loc + '/neurophys/MEG_data_200Hz_%s' %subject_labels[sb_count], 'rb') as fp:
            [X] = pickle.load(fp)
        data = scipy.io.loadmat(projectloc + '/data/decoding_data/306ch_data_' + subject_labels[sb_count] + suffix)
        X,thetas,stimulus_nr,presented_angle,time= concatenate_data(data,sb_count,toi=[-0.4,3.5])
        X = prep_sensor_data(X,data,1,False)
        # time = data['time'][0].T
            
        # select channels    
        # meg_ch = scipy.io.loadmat('/Users/jasperhajonides/Documents/EXP8_UpdProtec/MEG/analyses/Channel_selection/posterior_electrodes.mat')
        # sens = meg_ch['sens'].T[0][0:306]
        # X = X[:,sens==0,:]
        
        # read behaviour
        df_read = pd.read_csv(projectloc + '/data/behavioural/all_behavioural_May2021.csv')
        df_read = df_read.loc[(df_read['subject'] == sb_count), :]
        
        
        
        # include trials
        inx =   (df_read['stimulus_nr'] < 3) #& ~np.isnan(df_read['stim1'])  # & reject_outliers(abs(df_read['error']), m=2) & reject_outliers(df_read['RT'], m=3) # & ~np.isnan(df_read['stim2'])  & (abs(df_read['error']/math.pi*90)<math.pi/4) & (df_read['RT']<6) &  #
        X_all = X[inx,:,:]

        # select & set the target variable
        y,yb = bin_array(df_read['presented'][inx],nr_bins)
        
        
        #select time
        X_all = X_all[:,:,(time >= time_lim[0]) & (time <= time_lim[1]) ] #select time points ( >-.2 and <1.0)
        time_ = time[(time >= time_lim[0]) & (time <= time_lim[1]) ] #adjust the time vector
    
        # run decoding
        evidence = temporal_decoding(X_all[:,:,:], yb, size_window=size_window, n_folds=10,
               classifier=is_classifier, pca_components=.9, n_steps=1, demean=False)
        # evidence = temporal_decoding(X_all[:,:,0:50], yb,
        #                                 size_window=size_window,
        #                                 n_folds=10,
        #                                 classifier=is_classifier,
        #                                 pca_components=pca_var,
        #                                 n_steps=10,
        #                                 demean=False) 
        

        with open(dec_fname, 'wb') as fp:
            pickle.dump([evidence,time_], fp)
    else: 
        with open (dec_fname, 'rb') as fp:
            [evidence,time_] = pickle.load(fp)
#            
#                evidence['single_trial_evidence'] = empt
        print('pre-loaded data imported ')

    
    # read behaviour
    df_read = pd.read_csv(projectloc + '/data/behavioural/all_behavioural_May2021.csv')
    df_read = df_read.loc[(df_read['subject'] == sb_count) ,:]
    
    # include trials
    inx =   (df_read['stimulus_nr'] <3 ) #& reject_outliers(abs(df_read['error']), m=2) & reject_outliers(df_read['RT'], m=3) # & ~np.isnan(df_read['stim2'])  & (abs(df_read['error']/math.pi*90)<math.pi/4) & (df_read['RT']<6) &  #
    inx2 =  (df_read['stimulus_nr'][inx] < 3  )  & (df_read['prev_wmload'][inx] == 2) #& (df_read['usetrl'][inx] == 1) #& (df_read['prev_cue'][inx] == 2 ) #& ~np.isnan(df_read['stim1'][inx]) #   # # & (df_read['cue'][inx] == 2 ) #& reject_outliers(abs(df_read['error'][inx]), m=2) & reject_outliers(df_read['RT'][inx], m=3)## & reject_outliers(df_read['RT'][inx], m=3)   #& ~np.isnan(df_read['stim2'][inx]) # & (df_read['cue'][inx] == 1 ) & # ~np.isnan(df_read['stim1'][inx]) & (df_read['RT'][inx]<4) & (abs(df_read['error'][inx]/math.pi*90)<math.pi/8)



    # #convolve with cosine and obtain evidence
    # y, evidence['y'] = bin_array(np.array(df_read['stim1'])[inx][inx2],nr_bins)
    evidence['single_trial_evidence_store'] = evidence['single_trial_evidence']
    evidence['single_trial_evidence'] = evidence['single_trial_evidence'][inx2]
    # evidence = cos_convolve(evidence)
    # fitting evidence
    # classifier_output[:,sb_count,2] = evidence['cos_convolved']
    # classifier_output_tuning[:,:,sb_count,2] = evidence['centered_prediction']    
    
    #convolve with cosine and obtain evidence
    # y, evidence['y'] = bin_array(np.array(df_read['prev_non_probe_ang'])[inx][inx2],nr_bins)
    # evidence = cos_convolve(evidence)
    # #fitting evidence
    # classifier_output[:,sb_count,1] = evidence['cos_convolved']
    
    #convolve with cosine and obtain evidence
    # y, evidence['y'] = bin_array(np.array(df_read['prev_probe_ang'])[inx][inx2],nr_bins)
    # evidence = cos_convolve(evidence)
    # #fitting evidence
    # classifier_output[:,sb_count,2] = evidence['cos_convolved']
    
    
    # convolve with cosine and obtain evidence
    evidence['single_trial_evidence'] = evidence['single_trial_evidence_store']
    y, evidence['y'] = bin_array(np.array(df_read['presented'])[inx],nr_bins)
    evidence = cos_convolve(evidence)
    # #fitting evidence
    classifier_output[:,sb_count,0] = evidence['cos_convolved']
    classifier_output_tuning[:,:,sb_count,0] = evidence['centered_prediction']    
    
    # df_read.to_csv((projectloc + '/saved_data_forNick/behavioural_betweentrial_S%02d_June2021.csv' %sb_count))
    # scipy.io.savemat(projectloc + '/saved_data_forNick/LDA_betweentrial_S%02d_June2021.mat' %sb_count, {'data':evidence['single_trial_ev_centered']})

    df_read['diff_stim_prev_pang'] = pycircstat.cdiff(df_read['prev_non_probe_ang'],df_read['presented'])
    
    min_deg = 0 #2.903#2.8125 #10
    max_deg = 60 #58.06 #56.25 #50
    # right[:,:,sb_count] = evidence['single_trial_ev_centered'][(df_read['diff_stim_prev_pang'][inx] >= min_deg/90*np.pi) & (df_read['diff_stim_prev_pang'][inx] < max_deg/90*np.pi),:,:].mean(0)
    right[:,:,sb_count] = evidence['single_trial_ev_centered'][inx2][(df_read['diff_stim_prev_pang'][inx][inx2] >= min_deg/90*np.pi) & (df_read['diff_stim_prev_pang'][inx][inx2] < max_deg/90*np.pi),:,:].mean(0)
    # left[:,:,sb_count] = evidence['single_trial_ev_centered'][(df_read['diff_stim_prev_pang'][inx] <= -min_deg/90*np.pi) & (df_read['diff_stim_prev_pang'][inx] > -max_deg/90*np.pi),:,:].mean(0)
    left[:,:,sb_count] = evidence['single_trial_ev_centered'][inx2][(df_read['diff_stim_prev_pang'][inx][inx2] <= -min_deg/90*np.pi) & (df_read['diff_stim_prev_pang'][inx][inx2] > -max_deg/90*np.pi),:,:].mean(0)

    shift[:,sb_count] = (left[0:4,:,sb_count].mean(0) - left[5:9,:,sb_count].mean(0)) - (right[0:4,:,sb_count].mean(0) - right[5:9,:,sb_count].mean(0)) 

#%%   
# stim_prev = {}
# stim_prev['Crossdec_prev_probe_Protect'] = classifier_output[:,:,2]
# stim_prev['shift_prev_probe_Protect'] = shift

# stim_prev['Crossdec_prev_nonprobe_Protect'] =classifier_output[:,:,1]
# stim_prev['shift_prev_nonprobe_Protect'] = shift

# stim_prev = {}
# stim_prev['Crossdec_prev_probe_Update'] = classifier_output[:,:,2]
# stim_prev['Crossdec_prev_nonprobe_Update'] =classifier_output[:,:,1]
# stim_prev['shift_prev_nonprobe_Update'] = shift
# stim_prev['shift_prev_probe_Update'] = shift

# stim_prev['shift_prev_probe'] = shift
stim_prev['shift_prev_nonprobe'] = shift

with open('/Users/jasperhajonides/Documents/EXP8_UpdProtec/Figures/Figure_7_prev_uncued_cued/' + 'previous_trial_relevant_irrelevant', 'wb') as fp:
    pickle.dump([stim_prev,time_], fp)
    
    
fig = plt.figure(figsize=(10,10))
ax = plt.subplot(2,1,1)
tsplot(ax=ax, data=shift.T, time=time_, color = 'red', linestyle = 'solid',legend='Previous protect trial: non-probe angle',chance=0)
tsplot(ax=ax, data=stim_prev['Crossdec_prev_probe_Protect'].T, time=time_, color = 'red', linestyle = 'dotted',legend='Previous protect trial: probe angle',chance=0)
tsplot(ax=ax, data=stim_prev['Crossdec_prev_nonprobe_Update'].T, time=time_, color = 'blue', linestyle = 'solid',legend='Previous update trial: non-probe angle',chance=0)
tsplot(ax=ax, data=stim_prev['Crossdec_prev_probe_Update'].T, time=time_, color = 'blue', linestyle = 'dotted',legend='Previous update trial: probe angle',chance=0)
# stim_prev['shift_prev_nonprobe_Protect'] = shift
# stim_prev['shift_prev_nonprobe_Update'] = shift
tsplot(ax=ax, data=stim_prev['shift_prev_nonprobe_Update'].T, time=time_, color = 'blue', linestyle = 'dotted',legend='Previous probe angle',chance=0)
tsplot(ax=ax, data=stim_prev['shift_prev_nonprobe_Protect'].T, time=time_, color = 'red', linestyle = 'dotted',legend='Previous probe angle',chance=0)

 
    
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(1,1,1)
tsplot(ax=ax, data=shift[:,:].T, time=time_, color = 'blue', linestyle = 'solid',legend='shift',chance=0)

ax.secondary_yaxis('right',functions=(time_,classifier_output[:,:,1].mean(1)))

plt.tight_layout()
       
#%% bias
stim_serial = {}
stim_serial['shift_all'] = shift
stim_serial['left_all'] =left
stim_serial['right_all'] = right
stim_serial['classifier_output_all'] = classifier_output
stim_serial['classifier_output_tuning_all'] = classifier_output_tuning
stim_serial['shift_stim2'] = shift
stim_serial['left_stim2'] =left
stim_serial['right_astim2'] = right
stim_serial['classifier_output_stim2'] = classifier_output
stim_serial['shift_stim1'] = shift
stim_serial['left_stim1'] =left
stim_serial['right_astim1'] = right
stim_serial['classifier_output_stim1'] = classifier_output

# stim_serial['left_stim2_withstim1'] =left
# stim_serial['right_astim2_withstim1'] = right
# stim_serial['shift_stim2_withstim1'] = shift
# stim_serial['classifier_output_stim2_withstim1'] = classifier_output


with open('/Users/jasperhajonides/Documents/EXP8_UpdProtec/Figures/Figure_serial_bias/' + 'Bias_stim1_2_all_new_0to60deg', 'wb') as fp:
    pickle.dump([stim_serial,time_], fp)
   
      #%%                  
stim_2_bias['protect_shift_left'] = left
stim_2_bias['protect_shift_right'] = right
stim_2_bias['protect_shift'] = shift

stim_2_bias['update_shift_left'] = left
stim_2_bias['update_shift_right'] = right
stim_2_bias['update_shift'] = shift
stim2_bias = stim_2_bias

  #%%
from scipy.signal import savgol_filter
 
tmin = 130
tmax = 200

fig = plt.figure(figsize=(9,10))
ax = plt.subplot(2,2,1)

ax.scatter(np.arange(10),right[:,tmin:tmax,:].mean(2).mean(1),color='r')
ax.scatter(np.arange(10),left[:,tmin:tmax,:].mean(2).mean(1),color='green')
ax.plot(np.arange(10),right[:,tmin:tmax,:].mean(2).mean(1),color='r')
ax.plot(np.arange(10),left[:,tmin:tmax,:].mean(2).mean(1),color='green')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# # interpolated
# ls = scipy.interpolate.interp1d(np.arange(10),left[:,tmin:tmax,:].mean(2).mean(1),kind='linear')
# ls_sg = savgol_filter(ls(np.linspace(0,9,50)),9,1)

# rs = scipy.interpolate.interp1d(np.arange(10),right[:,tmin:tmax,:].mean(2).mean(1),kind='linear')
# rs_sg = savgol_filter(rs(np.linspace(0,9,50)),9,1)

ax.plot(np.linspace(0,9,50),ls_sg,'green',linewidth=4)
ax.plot(np.linspace(0,9,50),rs_sg,'red',linewidth=4)
ax.vlines(4,.08,.14,linestyle='dashed')
ax.set_xticks([-1,4,9])
ax.set_xticklabels(['-90','0','+90'])
# ax.hlines(.1,0,9,linestyle='dashed')
 
ax.set_title('prev probe 25-50 deg cw or ccw \n from current stim')

ax = plt.subplot(4,2,2)
tsplot(ax=ax, data=shift.T, time=time_, color = 'black', linestyle = 'solid',legend='left-right',chance=0)
ax.set_ylabel('Bias')
t,p = stats.ttest_1samp(shift[tmin:tmax,:].mean(0),0)
ax.set_title('t = %s, p = %s' %(np.round(t,3),np.round(p,4) ))
ax = plt.subplot(4,2,4)

cosfit_bias = phase_cosfit[1,:,:,0]-phase_cosfit[0,:,:,0]
tsplot(ax=ax, data=cosfit_bias.T, time=time_, color = 'blue', linestyle = 'solid',legend='cosfit',chance=0)
ax.set_xlabel('')
ax = plt.subplot(2,1,2)
tsplot(ax=ax, data=classifier_output[:,:,1].T, time=time_, color = 'blue', linestyle = 'solid',legend='prev sensory',chance=0)
tsplot(ax=ax, data=classifier_output[:,:,0].T, time=time_, color = 'grey', linestyle = 'solid',legend='stim 2',chance=0)
tsplot(ax=ax, data=classifier_output[:,:,2].T, time=time_, color = 'green', linestyle = 'solid',legend='stim 1',chance=0)
tsplot(ax=ax, data=classifier_output[:,:,3].T, time=time_, color = 'black', linestyle = 'solid',legend='prev sensory select',chance=0)

t,p = stats.ttest_1samp(classifier_output[tmin:tmax,:,3].mean(0),0)
ax.set_title('cross dec t = %s, p = %s' %(np.round(t,3),np.round(p,4) ))
plt.suptitle('Both stimuli all channels')


plt.tight_layout()
fig.savefig('overview_fig.png',dpi=150)


with open('/Users/jasperhajonides/Documents/EXP8_UpdProtec/Figures/Figure_stim2_bias/data_stim2_bias', 'wb') as fp:
    pickle.dump([stim2_bias,time_], fp)


 # %% stimulus 1

fig = plt.figure(figsize=(11,4))
# ax = plt.subplot(2,3,1)
# tsplot(ax=ax, data=classifier_output[:,:,0].T, time=time_, color = 'red', linestyle = 'solid',legend='stimulus 1',chance=0)

ax = plt.subplot(1,2,1)
ax.axvspan(time_[tmin], time_[tmax], ymin=-1, ymax=2, alpha=0.25, color='gray')

L  = (left[0:4,:,:].mean(0) - left[5:9,:,:].mean(0))
R = (right[0:4,:,:].mean(0) - right[5:9,:,:].mean(0))

tsplot(ax=ax, data=L.T, time=time_, color = 'mediumseagreen', linestyle = 'solid',legend='CCW',chance=0)
tsplot(ax=ax, data=R.T, time=time_, color = 'mediumpurple', linestyle = 'solid',legend='CW',chance=0)
ax.plot(time_,gaussian_filter1d(R.mean(1),5),'darkviolet',linewidth=3)
ax.plot(time_,gaussian_filter1d(L.mean(1),5),'darkgreen',linewidth=3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('Stimulus 2 \n Tuning Curve Shift on Update Trials')
cluster_times,condition_labels,cluster_pval = cluster_based_permutation_test(shift.T,time,threshold)

[ax.hlines(-.03, cluster_times[i,0],cluster_times[i,1],color='black') for i in range(0,int(sum(condition_labels)))]





# ------------
ax = plt.subplot(1,4,3)
ax.scatter(np.arange(10),right[:,tmin:tmax,:].mean(2).mean(1),color='darkviolet')
ax.scatter(np.arange(10),left[:,tmin:tmax,:].mean(2).mean(1),color='darkgreen')
ax.plot(np.arange(10),right[:,tmin:tmax,:].mean(2).mean(1),color='darkviolet',linewidth=0.75)
ax.plot(np.arange(10),left[:,tmin:tmax,:].mean(2).mean(1),color='darkgreen',linewidth=0.75)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# interpolated
ls = scipy.interpolate.interp1d(np.arange(10),left[:,tmin:tmax,:].mean(2).mean(1),kind='linear')
ls_sg = savgol_filter(ls(np.linspace(0,9,50)),9,1)

rs = scipy.interpolate.interp1d(np.arange(10),right[:,tmin:tmax,:].mean(2).mean(1),kind='linear')
rs_sg = savgol_filter(rs(np.linspace(0,9,50)),9,1)

ax.plot(np.linspace(0,9,50),ls_sg,'darkgreen',linewidth=3)
ax.plot(np.linspace(0,9,50),rs_sg,'darkviolet',linewidth=3)
ax.vlines(4,.08,.14,linestyle='dashed',color='grey')
ax.set_xticks([-1,4,9])
ax.set_xticklabels(['-90' + u'\N{DEGREE SIGN}','0','+90'+ u'\N{DEGREE SIGN}'])
ax.set_xlabel('Orientation Bins')

#get gradient map
ax = plt.subplot(1,4,4)
gradient=np.linspace(0,1,256)
gradient = np.fliplr(np.tile(np.vstack((gradient,gradient)),(8,1)))
ax.imshow(gradient,cmap=plt.get_cmap('PRGn'))
ax.axis('off')

box = ax.get_position()
box.x0 = box.x0 - 0.202
box.x1 = box.x1 - 0.202
box.y0 = box.y0 - 0.37
box.y1 = box.y1 - 0.37
ax.set_position(box)



fig.savefig('Stimulus_2_update_cue.png',dpi=100)
#%% cross decoding stimulus 2
fig = plt.figure(figsize=(7,4))

protect = classifier_output[:,:,2]

ax = plt.subplot(1,1,1)
tsplot(ax=ax, data=protect.T, time=time_, color = 'darkgreen', linestyle = 'solid',legend='Protect',chance=0)
tsplot(ax=ax, data=update.T, time=time_, color = 'darkred', linestyle = 'solid',legend='Update',chance=0)

ax.axvspan(time_[tmin], time_[tmax], ymin=-1, ymax=2, alpha=0.25, color='gray')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('Stimulus 2 \n Cross decoding of stimulus 1 \n')


plt.tight_layout()
fig.savefig('Stimulus_2_cross_decoding.png',dpi=100)


#%% previous trial

colors = ['darkblue','teal']
customPalette = sns.set_palette(sns.color_palette(colors))
sns.set_palette(customPalette)


fig = plt.figure(figsize=(14,7))
ax = plt.subplot(2,3,1)

tsplot(ax=ax, data=shift[:,:].T, time=time_, color = 'grey', linestyle = 'solid',legend='Previous probe angle',chance=0)
ax.plot(time_,gaussian_filter1d(classifier_output[:,:,3].mean(1).T,5),'black',linewidth=3)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('Stimulus 1 \n Cross decoding of previous probe angle \n')
ax.set_xlabel(' ')
ax.set_ylabel('Cos Conv Evidence')
ax.axvspan(time_[tmin], time_[tmax], ymin=-1, ymax=2, alpha=0.25, color='gray')
ax.set_xlim([-.2,.9])


# ------------------------------------
ax = plt.subplot(2,3,4)
L  = (left[0:4,:,:].mean(0) - left[5:9,:,:].mean(0))
R = (right[0:4,:,:].mean(0) - right[5:9,:,:].mean(0))

df = pd.DataFrame()
df['Evidence'] = np.concatenate((L[tmin:tmax,:].mean(0),R[tmin:tmax,:].mean(0)),axis=0)
df['Direction'] = np.sort(np.tile([0,1],20))


tsplot(ax=ax, data=L[:,:].T, time=time_, color = 'royalblue', linestyle = 'solid',legend='CCW',chance=0)
tsplot(ax=ax, data=R[:,:].T, time=time_, color = 'c', linestyle = 'solid',legend='CW',chance=0)
ax.plot(time_,gaussian_filter1d(R.mean(1),5),'teal',linewidth=3)
ax.plot(time_,gaussian_filter1d(L.mean(1),5),'darkblue',linewidth=3)


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Time(s)')
ax.axvspan(time_[tmin], time_[tmax], ymin=-1, ymax=2, alpha=0.25, color='gray')
ax.set_ylim([-.04,0.04])
ax.set_xlim([-.2,.9])
# ------------------------------------
ax = plt.subplot(2,11,17)

ax = sns.violinplot(x=np.zeros(40),y="Evidence", hue="Direction",
                    data=df, palette=customPalette, split=True,
                    inner="stick",
                    scale_hue=False, bw=.4,Legend=False)
ax.get_legend().remove()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_ylim([-.04,0.04])
ax.set_yticks([])
ax.set_xticks([])

ax.set_ylabel('')


box = ax.get_position()
box.x0 = box.x0 - 0.122
box.x1 = box.x1 - 0.132
# box.y0 = box.y0 - 0.37
# box.y1 = box.y1 - 0.37
ax.set_position(box)


# ax.y_
# ------------------------------------


# ax = plt.subplot(2,3,3)
# sns.regplot(classifier_output[120:180,:,1].mean(0),shift[120:180,:].mean(0),ax=ax)



plt.tight_layout()

fig.savefig('Stimulus_1_prevPang.png',dpi=100)
