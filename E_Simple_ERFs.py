#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:37:15 2020

@author: Cora
"""

# MEG Python script of Cora - Analysis of Context MEG
# Step D: Compute first ERFs

#%% Setup ####################################################################

# Import libraries
import os
import locale
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 8]
import numpy as np#, h5py
#import scipy
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
import pandas as pd
import autoreject
from autoreject import get_rejection_threshold
from pynput.keyboard import Key, Controller
keyboard = Controller()


#%% Analysis per Subject

# Set paths
homeDir = '/media/headmodel/Elements/AWM4/'
os.chdir(homeDir) # change current directory

# Load in meta information
metaFile = homeDir + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(metaFile)
NrSessions = 1
Subs       = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
Subs = Subs[21:]
locale.setlocale(locale.LC_ALL, "en_US.utf8")

# paths and file names
data_path   = homeDir +'/AWM4_data/raw/' 
#decim = 4
#fig,ax = plt.subplots(Subs.shape[0],2)

#%% Ask for subjects:
cc = 0
for actSubj in Subs:
    # load cut epochs
    CombTrials = mne.read_epochs(str(homeDir)+'/AWM4_data/processed/CutEpochs/CutData_VP'+str(actSubj)+'-epo.fif')
    CombTrials.pick_types(meg= True)
    CombTrials.apply_baseline(baseline = (-.5, -.25))

    # load ICA
    ica   = mne.preprocessing.ica.read_ica(str(homeDir)+'/AWM4_data/processed/ICAs/CleanVP'+str(actSubj)+'-ica.fif')
    picks = mne.pick_types(CombTrials.info, meg=True, eeg=True, stim=False, 
                        eog=False, ref_meg=False, include=[], exclude='bads')
    
    # drop epochs with jump      
    jname = str(homeDir)+'/AWM4_data/processed/ICAs/Jumps'+str(actSubj)+'.npy'
    if os.path.isfile(jname): 
        jump_inds = np.load(jname) 
        CombTrials.drop(jump_inds,reason='jump')
    
    # apply ICA to cut epochs
    ica.apply(CombTrials)
    
    # average all trials
    evoked = CombTrials.average()
    #actDict =  dict(axes=ax[0,0],spatial_colors=True, zorder='std')
    #actDict2 = dict(show=False)
    #ev = evoked.plot_joint(ts_args=actDict,topomap_args = actDict2)
    evoked.filter(l_freq=None, h_freq=30)

    fig,ax = plt.subplots(1)

    evoked.plot(axes=ax,spatial_colors=True, show = False)
    #evoked.plot_joint(times = [.15,1.15,2.15,3.65,4.65,5.15])
    ymin,ymax = ax.get_ylim()

    # S1
    time_event = 0
    text_event = 'S1'
    ax.plot([time_event, time_event], [ymin, ymax],'k')
    ax.text(time_event+.05,ymax, text_event)

    # S2
    time_event = 1
    text_event = 'S2'
    ax.plot([time_event, time_event], [ymin, ymax],'k')
    ax.text(time_event+.05,ymax, text_event)

    # Cue
    time_event = 2
    text_event = 'Cue'
    ax.plot([time_event, time_event], [ymin, ymax],'k')
    ax.text(time_event+.05,ymax, text_event)

    # Ping
    time_event = 3.5
    text_event = 'Ping'
    ax.plot([time_event, time_event], [ymin, ymax],'k')
    ax.text(time_event+.05,ymax, text_event)

    # Probe
    time_event = 4.5
    text_event = 'Probe'
    ax.plot([time_event, time_event], [ymin, ymax],'k')
    ax.text(time_event+.05,ymax, text_event)

    # Response Start
    time_event = 5
    text_event = 'Response'
    ax.plot([time_event, time_event], [ymin, ymax],'k')
    ax.text(time_event+.05,ymax, text_event)

    ax.plot([0, 0], [ymin, ymax],'k')
    ax.text(0.05,ymax,'S1')
    ax.plot([1.0, 1.0], [ymin, ymax],'k')
    ax.text(1.05,ymax,'S2')
    ax.plot([2.0, 2.0], [ymin, ymax],'k')
    ax.text(2.05,ymax,'Cue')
    
    fig.show()
    plt.savefig(str(homeDir)+'/AWM4_data/processed/ERFs/VP'+str(actSubj)+'_simpleERFs.pdf')
    plt.close('all')
        
    cc = cc+1
    print(f"Processed subject {actSubj} ({cc}/{len(Subs)})")
