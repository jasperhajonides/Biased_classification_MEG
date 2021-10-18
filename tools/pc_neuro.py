#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:16:45 2020

@author: jasperhvdm
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from scipy import stats
import scipy.stats
import copy

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
    ax.legend()
    ax.axvline(.0, color='k', linestyle='-')
    
def tsplot_neutral(ax, data, time, color = 'k', linestyle = 'solid',legend='target',chance=0.0):
    x = time
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    se = (sd/np.sqrt(len(data)))
    cis = (est - se, est + se)
    
    ax.fill_between(x,cis[0],cis[1],alpha = 0.2, facecolor = color)
    ax.plot(x,est,color = color, linestyle = linestyle,label=legend)
    ax.margins(x=0)

    ax.set_xlabel('Times',fontsize=14)
    ax.legend()
    
    
    
    
def circ_bini(x, nbin, pbin):
    
    quantbeg = np.mod(np.linspace(0-pbin/2, 1-1/nbin - pbin/2, nbin),1)
    quantend = np.mod(quantbeg+pbin,1)
    xbinbeg = np.quantile(x, quantbeg)
    xbinend = np.quantile(x, quantend)
    
    ibin = np.full(shape = (nbin, np.size(x)), fill_value = False)
    
    for i in range(nbin):
        if quantbeg[i] < quantend[i]: #no wrapping needed
            ibin[i,:] = np.logical_and(np.greater_equal(x,xbinbeg[i]), np.less_equal(x, xbinend[i]))
        else: #wrapping
            ibin[i,:] = np.logical_or(np.greater_equal(x, xbinbeg[i]), np.less_equal(x, xbinend[i]))
    
    return ibin


def randomly_avg(X,y,div=2):
    """By randomly averaging trials within the same stimulus category 
    we can decrease noise. """
    X_merged = np.zeros((X.shape))
    y_merged = np.zeros((y.shape))*np.nan
    cc=0
    for stim_inx in range(y.min(),y.max()+1):
        #select stimuli
        X_stim = X[y==stim_inx,:,:]

        # randomly avg
        N = np.sum(y==stim_inx)
        items = np.arange(N/div)
        trl_alloc = random.choices(items, k=N)
        for ii in np.unique(trl_alloc):
            X_merged[cc,:,:] = X_stim[trl_alloc==ii,:,:].mean(0)
            y_merged[cc] = stim_inx
            cc+=1
    
    return X_merged[~np.isnan(y_merged),:,:], y_merged[~np.isnan(y_merged)]



def gen_RSA_matrix(y,nr_bins,mode='continuous'):
    """ Generates dissimilarity matrix of n-by-n where n is the length of the 
        vector provided."""
    
    # first we binarise the 
    bins = np.arange(0.000001,math.pi,math.pi/nr_bins)
    y = np.digitize(y, bins) 
        
    if mode == 'continuous':
        if (nr_bins%2)==1:
          raise Exception("Please provide an even number of bins.")
        #colour cosine
        stim_mat = np.zeros((nr_bins,nr_bins))
        for i in range(nr_bins):
            stim_mat[i,:] = np.roll(np.cos(np.linspace(0,2*np.pi,nr_bins)),int(nr_bins/2+i))*2.5
    elif mode == 'categorical':
        stim_mat = 1-np.eye(2)
        y = y-1
    stim_mat = stats.zscore(stim_mat,axis=None)
    
    return stim_mat,y
    

from sklearn.decomposition import PCA
from sklearn.neighbors import DistanceMetric
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler


def mahalanobis(X,y,nr_bins,mode='continuous'):
    """ Estimate the mahalanobis distance between the average stimulus 
    pattern defined by classes in y across features in X. For every time point 
    """

    stim_mat,y = gen_RSA_matrix(y,nr_bins,mode=mode)

    evidence_RSA = np.zeros(X.shape[2])*np.nan
    matrix = np.zeros((nr_bins,nr_bins,X.shape[2]))*np.nan
    for tp in range(X.shape[2]):
        
        
        X_in = X[:,:,tp]
        
        #pca
        pca = PCA(n_components=.95)
        X_in = pca.fit(X_in).transform(X_in)
        
        #estimate covariance 
        emp_cov = EmpiricalCovariance().fit(X_in)
        maha = DistanceMetric.get_metric('mahalanobis',VI=emp_cov.covariance_)
    
    
        #scale data
        scaler = StandardScaler().fit(X_in)
        X_s = scaler.transform(X_in)
        
        
        X_stim = np.zeros((nr_bins,X_s.shape[1]))
        for i, stim in enumerate(np.unique(y)):
            X_stim[i,:] = X_s[(y==stim) ,:].mean(0).T
        
    
        matrix[:,:,tp] = maha.pairwise(X_stim)
        evidence_RSA[tp] = np.mean(np.mean(np.multiply(matrix[:,:,tp],stim_mat)))
        
    return evidence_RSA, matrix


def gesd(x, alpha = 0.05, p_out = .1, outlier_side = 0):

    import numpy as np

    '''
    Detect outliers using Generalizes ESD test
    based on the code from Romesh Abeysuriya implementation for OSL
    based on the code from Sage Boettchers python translation
      
    Inputs:
    - x : Data set containing outliers - should be a np.array 
    - alpha : Significance level to detect at (default = .05)
    - p_out : percent of max number of outliers to detect (default = 10% of data set)
    - outlier_side : Specify sidedness of the test
        - outlier_side = -1 -> outliers are all smaller
        - outlier_side = 0 -> outliers could be small/negative or large/positive (default)
        - outlier_side = 1 -> outliers are all larger
        
    Outputs
    - idx : Logicial array with True wherever a sample is an outlier
    - x2 : input array with outliers removed
    
    For details about the method, see
    B. Rosner (1983). Percentage Points for a Generalized ESD Many-outlier Procedure, Technometrics 25(2), pp. 165-172.
    http://www.jstor.org/stable/1268549?seq=1
    '''

    if outlier_side == 0:
        alpha = alpha/2
    
    
    if type(x) != np.ndarray:
        x = np.asarray(x)

    n_out = int(np.ceil(len(x)*p_out))

    if any(~np.isfinite(x)):
        #Need to find outliers only in non-finite x
        y = np.where(np.isfinite(x))[0] # these are the indexes of x that are finite
        idx1, x2 = gesd(x[np.isfinite(x)], alpha, n_out, outlier_side)
        # idx1 has the indexes of y which were marked as outliers
        # the value of y contains the corresponding indexes of x that are outliers
        idx = [False] * len(x)
        idx[y[idx1]] = True

    n      = len(x)
    temp   = x
    R      = np.zeros((1, n_out))[0]
    rm_idx = copy.deepcopy(R)
    lam    = copy.deepcopy(R)

    for j in range(0,int(n_out)):
        i = j+1
        if outlier_side == -1:
            rm_idx[j] = np.nanargmin(temp)
            sample    = np.nanmin(temp)
            R[j]      = np.nanmean(temp) - sample
        elif outlier_side == 0:
            rm_idx[j] = int(np.nanargmax(abs(temp-np.nanmean(temp))))
            R[j]      = np.nanmax(abs(temp-np.nanmean(temp)))
        elif outlier_side == 1: 
            rm_idx[j] = np.nanargmax(temp)
            sample    = np.nanmax(temp)
            R[j]      = sample - np.nanmean(temp)
        
        R[j] = R[j] / np.nanstd(temp)
        temp[int(rm_idx[j])] = np.nan
        
        p = 1-alpha/(n-i+1)
        t = scipy.stats.t.ppf(p,n-i-1)
        lam[j] = ((n-i) * t) / (np.sqrt((n-i-1+t**2)*(n-i+1)))
    
    #And return a logical array of outliers
    if any(R>lam):
        idx = np.zeros((1,n))[0]
        idx[np.asarray(rm_idx[range(0,np.max(np.where(R>lam))+1)],int)] = np.nan
        idx = ~np.isfinite(idx)
        
        x2 = x[~idx]
    else:
        idx = np.array([])
        x2  = x
    
    return idx, x2

#%%

# Nr of trials and nr of subjects
#with open('/Users/jasperhvdm/Documents/DPhil/Projects/EXP8_UpdProtec/scripts/MEG_topo_struct.pkl', 'rb') as f:  
#    [epochs] = pickle.load(f)
#evoked = epochs.average()
#
#PcA = pca_components[0]
#evoked.data[0:306,0:PcA.shape[0]] = PcA.T
#
#times = evoked.times
#evoked.plot_topomap(times[0:5], ch_type='grad', time_unit='s')
#
#corr_matrix = np.zeros((pca_components[0].shape[0],pca_components[1].shape[0]))
#for comp in range(0,pca_components[0].shape[0]):
#    for comp2 in range(0,pca_components[1].shape[0]):
#        corr_matrix[comp,comp2],p = stats.pearsonr(pca_components[0][comp,:],pca_components[1][comp2,:])
#    
