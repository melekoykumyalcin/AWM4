#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:38:24 2020

@author: Cora
"""

# MEG Python script of Cora - Analysis of Context MEG
# Step B: ICA, parallel processed possible
# includes: ICA computation per subject and session

#%% Setup ####################################################################

# Import libraries
import os
import locale
import numpy as np
#import scipy
import mne
from mne.preprocessing import ICA
import pandas as pd
#from autoreject import get_rejection_threshold
from joblib import Parallel, delayed
import multiprocessing

# Set paths
homeDir = '/media/headmodel/Elements/AWM4/'
os.chdir(homeDir) # change current directory

# Load in meta information
metaFile = homeDir + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(metaFile)
NrSessions = 1
Subs       = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
Subs = Subs[28:]
#Subs       = Subs[Subs != 3] #exclude sub 3 for no, need noise file
locale.setlocale(locale.LC_ALL, "en_US.utf8")

#%% compute ICA #########################################################
        
# Choose other parameters
n_components = .99  # if float, select n_components by explained variance of PCA
decim = 4  # we need sufficient statistics, not all time points -> saves time (downsampled to 300 Hz)

# we will also set state of the random number generator - ICA is a
# non-deterministic algorithm, but we want to have the same decomposition
# and the same order of components each time this tutorial is run
random_state = 0

# paths and file names
data_path   = homeDir +'/AWM4_data/raw/'

# Loop over subjects:
#def parallel_ica(actSubj):    
for actSubj in Subs: 
    fname = str(homeDir)+'/AWM4_data/processed/ICAs/VP'+str(actSubj)+'_fastica-ica.fif'
    jname = str(homeDir)+'/AWM4_data/processed/ICAs/Jumps'+str(actSubj)+'.npy'
    #if os.path.isfile(fname): 
        #print('already done.')           
    #else:
    #if os.path.isfile(jname): 
        # load cut epochs
    CombTrials = mne.read_epochs(str(homeDir)+'/AWM4_data/processed/CutEpochs/CutData_VP'+str(actSubj)+'-epo.fif')
    filt_trials = CombTrials.copy()
    # to try with 1-30 Hz bandpass filter filter(l_freq=1., h_freq=30)
    filt_trials.load_data().filter(l_freq=1., h_freq=None)

    # use autoreject to exclude trials with jumps etc
    #reject = get_rejection_threshold(filt_trials, decim=2, ch_types = 'mag')
    
    # load epochs with jumps
    if os.path.isfile(jname):
        jump_inds = np.load(jname)
        # attention! check manually if this works as expected. CF
        filt_trials.drop(jump_inds,reason='jump')
    
    # define ICA
    ica = ICA(method='fastica',n_components=n_components, random_state=random_state, max_iter = 1000)
    picks = mne.pick_types(filt_trials.info, meg=True, eeg=False, stim=False, 
                       eog=False, ref_meg=False, include=[], exclude='bads')
    
    # fit ICA
    ica.fit(filt_trials, picks=picks, decim=decim)
    ica.save(fname)
    #else:
        #print('no jumps')
    
           

# Run loop in parallel (subjectwise)
#num_cores = multiprocessing.cpu_count()
#Parallel(n_jobs=int(num_cores/4))(delayed(parallel_ica)(actSubj) for actSubj in Subs)
