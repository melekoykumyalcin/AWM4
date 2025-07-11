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
import matplotlib.pyplot as plt
import matplotlib
# mne.viz.set_browser_backend('matplotlib')
# matplotlib.use('TkAgg')


# Set paths
homeDir = '/media/headmodel/Elements/AWM4/'
os.chdir(homeDir) # change current directory

# Load in meta information
metaFile = homeDir + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(metaFile)
NrSessions = 1
Subs       = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
#Subs       = Subs[Subs != 3] #exclude sub 3 for no, need noise file
locale.setlocale(locale.LC_ALL, "en_US.utf8")

#%% Load subject data and show suspicious sensors##############################
        
# paths and file names
data_path   = homeDir +'/AWM4_data/raw/'

actSubj = Subs[int(input("Subject no [1-30]: "))-1] 
CombTrials = mne.read_epochs(str(homeDir)+'/AWM4_data/processed/CutEpochs/CutData_VP'+str(actSubj)+'-epo.fif')
#CombTrials.plot_psd_topomap()
#CombTrials.plot_sensors(show_names=True)
          
#%% Choose suspicious sensors ################################################# 

picks = mne.pick_types(CombTrials.info, meg=True, eeg=False, stim=False, 
                             eog=False, ref_meg=False, include=[], exclude=[])

# Check MLF 46/single sensor
#CombTrials.plot_image(picks='MLP42-3609', vmin=-2000, vmax=2000) # MZC01

# # left sensors
#actPicks  = np.arange(0,11)+picks[0]
#actPicks  = np.arange(11,23)+picks[0]
#actPicks  = np.arange(23,35)+picks[0]
#actPicks  = np.arange(35,47)+picks[0]
#actPicks  = np.arange(47,59)+picks[0]
#actPicks  = np.arange(59,71)+picks[0]
#actPicks  = np.arange(71,83)+picks[0]
#actPicks  = np.arange(83,95)+picks[0]
#actPicks  = np.arange(95,107)+picks[0]
#actPicks  = np.arange(107,119)+picks[0]
#actPicks  = np.arange(119,131)+picks[0]
# # right sensors
#actPicks = np.arange(131,143)+picks[0]
#actPicks = np.arange(143,155)+picks[0]
#actPicks = np.arange(155,167)+picks[0]
#actPicks = np.arange(167,179)+picks[0]
#actPicks = np.arange(179,191)+picks[0]
#actPicks = np.arange(191,203)+picks[0] #
#actPicks = np.arange(203,215)+picks[0] 
#actPicks = np.arange(215,227)+picks[0]
#actPicks = np.arange(227,239)+picks[0]
#actPicks = np.arange(239,251)+picks[0]
#actPicks = np.arange(251,262)+picks[0]
# # central sensors
#actPicks = np.arange(262,271)+picks[0]

#plot to your liking
#CombTrials.plot_image(actPicks) #plot your current selection of sensors

#%% Plot suspicous sensors to find time point of jump ########################
# sqno= int(np.ceil(np.sqrt(actPicks.shape[0])))
# fig,ax = plt.subplots(sqno,sqno)
# for pi in range(actPicks.shape[0]):
#    subplot_idx = np.unravel_index(pi, (sqno, sqno))
#    CombTrials.plot_image(picks=actPicks[pi], vmin=-2000, vmax=2000, axes=ax[subplot_idx], evoked=False,colorbar=False)
# fig    
#%%
CombTrials.plot(n_epochs=4, block=True)

#%%
jump_inds = list(map(int, input("Please enter the epochs that contained jumps (as: x y z): ").split())) 
fname = str(homeDir)+'/AWM4_data/processed/ICAs/Jumps'+str(actSubj)+'.npy'
np.save(fname,jump_inds) 
plt.close('all')           

# read the jumps
#jumps = np.load(fname)



# %%
