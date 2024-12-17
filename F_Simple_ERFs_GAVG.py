#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:31:13 2020

@author: Cora
"""
# MEG Python script of Cora - Analysis of Context MEG
# Step F: Compute grandaverage of simple ERFs

#%% Setup ####################################################################

# Import libraries
import os
import locale
import matplotlib
import matplotlib.pyplot as plt
plt.ioff()
# use Agg backend for saving figures
plt.switch_backend('agg')
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
#Subs       = Subs[Subs != 3] #exclude sub 3 for no, need noise file
locale.setlocale(locale.LC_ALL, "en_US.utf8")

# paths and file names
data_path   = homeDir +'/AWM4_data/raw/'
#decim = 4
#fig,ax = plt.subplots(Subs.shape[0],2)

evoked = list()

for actSubj_idx,actSubj in enumerate(Subs):
    print(f'Processing subject {actSubj}')
    
    # # load ICA
    # ica   = mne.preprocessing.ica.read_ica(str(homeDir)+'/AWM4_data/processed/ICAs/CleanVP'+str(actSubj)+'-ica.fif')
    # picks = mne.pick_types(CombTrials.info, meg=True, eeg=True, stim=False, 
    #                     eog=False, ref_meg=False, include=[], exclude='bads')
    
    # # apply ICA to cut epochs
    # ica.apply(CombTrials)
    # CombTrials.pick_types(meg=True,ref_meg=False) # pick only meg channels (default)
    # CombTrials.apply_baseline(baseline = (-.5,-.25))
    
    tmp_evoked_fname = str(homeDir)+'/AWM4_data/processed/ERFs/VP'+str(actSubj)+'_alltrial_ERFs-ave.fif'
    if os.path.isfile(tmp_evoked_fname):
        print('File already exists. Skipping')
        tmp_evoked = mne.read_evokeds(tmp_evoked_fname)
        tmp_evoked = tmp_evoked[0]
    else:
        # load cut epochs
        CleanTrials = mne.read_epochs(str(homeDir)+'/AWM4_data/processed/CutEpochs/CutData_VP'+str(actSubj)+'-cleanedICA-epo.fif')
        # drop epochs with jump      
        jname = str(homeDir)+'/AWM4_data/processed/ICAs/Jumps'+str(actSubj)+'.npy'
        if os.path.isfile(jname): 
            jump_inds = np.load(jname) 
            CleanTrials.drop(jump_inds,reason='jump')
    
        # average all trials
        tmp_evoked = CleanTrials.average(picks='meg')

        #save evoked
        tmp_evoked.save(tmp_evoked_fname)

    evoked.append(tmp_evoked)

    plot_path = str(homeDir)+'/AWM4_data/processed/ERFs/plots'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    fig,ax = plt.subplots(1)
    tmp_evoked.plot(axes=ax,spatial_colors=True)
    ymin,ymax = ax.get_ylim()
    ax.plot([0, 0], [ymin, ymax],'k')
    ax.text(0.05,ymax,'S1')
    ax.plot([1, 1], [ymin, ymax],'k')
    ax.text(1.05,ymax,'S2')
    ax.plot([2, 2], [ymin, ymax],'k')
    ax.text(2.05,ymax,'Cue')
    ax.plot([3.5, 3.5], [ymin, ymax],'k')
    ax.text(3.55,ymax,'Ping')
    ax.plot([4.5, 4.5], [ymin, ymax],'k')
    ax.text(4.55,ymax,'Probe')
    ax.set_title('Subject '+str(actSubj))
    fig.savefig(f'{plot_path}/VP{actSubj}_clean_simpleERFs.pdf')
    plt.close(fig)


fig,ax = plt.subplots(1)
GAVG = mne.grand_average(evoked)
GAVG.plot(axes=ax,spatial_colors=True)
ymin,ymax = ax.get_ylim()
ax.plot([0, 0], [ymin, ymax],'k')
ax.text(0.05,ymax,'S1')
ax.plot([1, 1], [ymin, ymax],'k')
ax.text(1.05,ymax,'S2')
ax.plot([2, 2], [ymin, ymax],'k')
ax.text(2.05,ymax,'Cue')
ax.plot([3.5, 3.5], [ymin, ymax],'k')
ax.text(3.55,ymax,'Ping')
ax.plot([4.5, 4.5], [ymin, ymax],'k')
ax.text(4.55,ymax,'Probe')
ax.set_title('Grand Average all subjects')
fig.show()
plt.savefig(str(homeDir)+'/AWM4_data/processed/ERFs/AllVP_simpleERFs_AVG.pdf')
plt.close('all')
