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
homeDir = '/home/headmodel/Schreibtisch/AWM4/'
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
cc=0

evoked = list()
for actSubj in Subs:
        
    # load cut epochs
    CombTrials = mne.read_epochs(str(homeDir)+'/processed/CutEpochs/CutData_VP'+str(actSubj)+'-epo.fif')
    
    # load ICA
    ica   = mne.preprocessing.ica.read_ica(str(homeDir)+'/processed/ICAs/CleanVP'+str(actSubj)+'-ica.fif')
    picks = mne.pick_types(CombTrials.info, meg=True, eeg=True, stim=False, 
                        eog=False, ref_meg=False, include=[], exclude='bads')
    
    # drop epochs with jump      
    jname = str(homeDir)+'/processed/ICAs/Jumps'+str(actSubj)+'.npy'
    if os.path.isfile(jname): 
        jump_inds = np.load(jname) 
        CombTrials.drop(jump_inds,reason='jump')
    
    # apply ICA to cut epochs
    ica.apply(CombTrials)
    CombTrials.pick_types(meg=True,ref_meg=False) # pick only meg channels (default)
    CombTrials.apply_baseline(baseline = (-.5,-.25))
    
    # average all trials
    evoked.append(CombTrials.average())

    fig,ax = plt.subplots(1)
    evoked[actSubj].plot(axes=ax,spatial_colors=True)
    ymin,ymax = ax.get_ylim()
    ax.plot([0, 0], [ymin, ymax],'k')
    ax.text(0.05,ymax,'S1')
    ax.plot([1, 1], [ymin, ymax],'k')
    ax.text(1.05,ymax,'S2')
    ax.plot([2, 2], [ymin, ymax],'k')
    ax.text(2.05,ymax,'Cue')
    ax.plot([4, 4], [ymin, ymax],'k')
    ax.text(4.05,ymax,'Ping')
    ax.plot([5.5, 5.5], [ymin, ymax],'k')
    ax.text(5.55,ymax,'Probe')
    ax.set_title('Subject '+str(actSubj))
    fig.show()
    plt.savefig(str(homeDir)+'/processed/ERFs/VP'+str(actSubj)+'_simpleERFs_AVG.pdf')
    plt.close('all')
    cc=cc+1
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
ax.plot([4, 4], [ymin, ymax],'k')
ax.text(4.05,ymax,'Ping')
ax.plot([5.5, 5.5], [ymin, ymax],'k')
ax.text(5.55,ymax,'Probe')
ax.set_title('Grand Average all subjects')
fig.show()
plt.savefig(str(homeDir)+'/processed/ERFs/AllVP_simpleERFs_AVG.pdf')
plt.close('all')
