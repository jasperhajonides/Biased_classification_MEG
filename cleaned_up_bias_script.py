#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:39:45 2022

@author: jasperhajonides
"""

#imports
fitting_params = scipy.io.loadmat(projectloc + '/data/Fitting_params_Biases.mat')
post_s = scipy.io.loadmat(projectloc + '/MEG/analyses/Channel_selection/posterior_electrodes.mat')
neighb = scipy.io.loadmat(projectloc + '/data/channels/neighbours.mat')


# %%

    
    

def reject_outliers(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)


def get_config(sb:int(),
               bias_type: str(),
               pca_variance: float() = .90,
               size_window: int() = 30,
               epoch: str() = 'med') -> dict:
    
    config = dict()
    # subject subject ID
    config['subject'] = 'S%02d' %(sb+1)
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
        print('running sb %s' %str(sb_count))

        # # read MEG data
        # with open (config['projectloc'] + '/MEG/data/preprocessed' + '/neurophys/MEG_data_200Hz_' + config['subject'] , 'rb') as fp:
        #     [X] = pickle.load(fp)
        
        # read neural data
        data = scipy.io.loadmat(config['projectloc']  + '/data/decoding_data/306ch_data_' + config['subject']  + suffix)
        X,thetas,stimulus_nr,presented_angle,time= concatenate_data(data,sb_count,toi=[-0.4,3.5])
        
        X = prep_sensor_data(X,data,baseline=False)
           
        with open( config['projectloc']  + '/data/decoding_data/%s_preprocessed_MEG_data' % config['subject'], 'wb') as fp:
            pickle.dump([X,thetas,stimulus_nr,presented_angle,time], fp)
        
        # # read behaviour
        # df_read = pd.read_csv(projectloc + '/data/behavioural/all_behavioural_May2021.csv')
        # df_read = df_read.loc[(df_read['subject'] == sb_count) , :]
        
        
        # # Get neural data and behavioural data for selected trials. 
        # inx = df_read['stimulus_nr'].isin(config['stimulus_nr'])
        # X_all = X[inx,:,:]
        # y,yb = bin_array(df_read['presented'][inx],nr_bins)
        
        # #select time
        # X_all = X_all[:,:,(time >= time_lim[0]) & (time <= time_lim[1]) ] 
        # time_ = time[(time >= time_lim[0]) & (time <= time_lim[1]) ] #adjust the time vector
    
        # # run decoding
        # evidence, nr_pca_components = temporal_decoding(X_all, yb, size_window=config['size_window'], n_folds=10,
        #        classifier='LDA', pca_components=config['pca_var'], demean=False) 
        # with open(config['save_file_name'], 'wb') as fp:
        #     pickle.dump([evidence,time_], fp)
    else:
        print('Load data for %s' %config['subject'])
        with open (config['save_file_name'], 'rb') as fp:
            [evidence,time_] = pickle.load(fp)
    return X #evidence
   
        
# %%
classifier_output = np.zeros((nr_tps,nsubs,3))
shift = np.zeros((nr_tps,nsubs))

for sb_count in range(0,20):
  
    config = get_config(sb_count, 'within_trial_update', size_window = 30)
    # obtain decoding evidence for selected condition
    evidence = decoding_function(config, overwrite = True)

    
    ####
    df_read = pd.read_csv(projectloc + '/data/behavioural/all_behavioural_May2021.csv')
    df_read = df_read.loc[(df_read['subject'] == sb_count), :]
    inx = df_read['stimulus_nr'].isin(config['stimulus_nr'])
    inx2 = df_read[list(config['trial_subselection'].keys())[0]].isin(
        list(config['trial_subselection'].values())[0])[inx]
    ###
    
    
    evidence['single_trial_evidence_store'] = evidence['single_trial_evidence'][inx2]



    # convolve with cosine and obtain evidence
    for i, name in enumerate(['prev_probe_ang','stim1', 'presented']):
        evidence['single_trial_evidence'] = evidence['single_trial_evidence_store']
        y, evidence['y'] = bin_array(np.array(df_read[name])[inx][inx2],nr_bins)
        evidence = cos_convolve(evidence)
        classifier_output[:,sb_count,i] = evidence['cos_convolved']
    

    
    # get the angular difference between the presented grating and the angle 
    # that is expected to generate the bias (eiter stimulus 1 (within trial 
    # bias) or the probed orientation on the previous trial (between trial 
    # bias))
    df_read['angular_difference'] = pycircstat.cdiff(df_read[config['bias_source']],
                                                      df_read['presented'])
  
    # find the trials that have the appropriate angular difference within range
    # now from the decoding evidence we select trials with an angular 
    # difference that biases orientations to the right/negative, or to the 
    # left/positive. 
    right_diffs = ((df_read['angular_difference'][inx][inx2] >= config['min_deg']/90*np.pi) & 
                   (df_read['angular_difference'][inx][inx2] < config['max_dex']/90*np.pi))
    right[:,:,sb_count] = evidence['single_trial_ev_centered'][right_diffs,:,:].mean(0)
    
    left_diffs = ((df_read['angular_difference'][inx][inx2] <= -config['min_deg']/90*np.pi) &
                  (df_read['angular_difference'][inx][inx2] > -config['max_dex']/90*np.pi))
    left[:,:,sb_count] = evidence['single_trial_ev_centered'][left_diffs,:,:].mean(0)
    
    # the decoding evidence provides us with tuning curves. We like to know 
    # if the tuning curves are systematically shifted to either direction. 
    # to this end we subtract the evidence right from the centre from the 
    # decoding evidence left from the centre. 
    # This will give us a metric of asymmetry or shift in the tuning curve.
    shift[:,sb_count] = ((left[0:4,:,sb_count].mean(0) - 
                          left[5:9,:,sb_count].mean(0)) - 
                         (right[0:4,:,sb_count].mean(0) - 
                          right[5:9,:,sb_count].mean(0))) 








