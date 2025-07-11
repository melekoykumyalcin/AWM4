# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:55:01 2019

@author: Cora&Melek
"""
# MEG Python script of Cora - Analysis of Context MEG
# Step A: Preprocessing 
# includes: loading MEG data, notch filter (50 Hz), cutting trials relative to 
# S1 onset

#%% Setup ####################################################################

# Import libraries
import os
import locale
locale.setlocale(locale.LC_ALL, "en_US.utf8") # set date settings to avoid error when reading in raw data
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 8]
import numpy as np#, h5py
#import scipy
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
import pandas as pd
#from autoreject import get_rejection_threshold
from joblib import Parallel, delayed
import multiprocessing

# Set paths
#homeDir = '/home/imp/MEG_Context/' # adapt
#homeDir = '/home/headmodel/Schreibtisch/AWM4/'
homeDir = '/media/headmodel/J/AWM4/'
os.chdir(homeDir) # change current directory

# Load in meta information
#metaFile = homeDir + 'Protocols/MEG_Protocols.xlsx'
metaFile = homeDir + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(metaFile)
NrSessions = 1
Subs       = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
#Subs       = Subs[Subs != 3] #exclude sub 3 for now, need noise file

#%% Load and cut data per subject and session ################################

# paths and file names
#data_path   = homeDir +'/MEGData/'
data_path   = homeDir +'/AWM4_data/raw/'
allFiles    = metaInfo['MEG_Name']
noiseFiles  = metaInfo['Noise_Measurement']

# Loop over subjects:
for actSubj in Subs:
    actInd        = (metaInfo.Subject==actSubj) & (metaInfo.Valid==1)
    actFiles      = allFiles[actInd]
    actNoiseFiles = noiseFiles[actInd] 
    for ff in range(actFiles.count()):
        print(f"processing run {ff+1}...")
        print('')
        print('')

        fname = data_path + actFiles.iloc[ff]
        raw   = mne.io.read_raw_ctf(fname,'truncate',True) 

        if ff==0:
            reference_dev_head_t_ref = raw.info["dev_head_t"]
        else:
            raw.info['dev_head_t']=reference_dev_head_t_ref

        
        # examples for how to fix triggers
        # if actSubj == 7 and se == 2 and ff == 4:
        #     events    = mne.find_events(raw, 'UPPT001',shortest_event=1)
        #     events= events[events[:,1]==0,:]
        # elif actSubj == 7 and se == 1 and ff == 1: #block was restarted; exclude first trial
        #     events    = mne.find_events(raw, 'UPPT001')
        #     events = events[18:,:]
        # else:
        events    = mne.find_events(raw, 'UPPT001', shortest_event=1)
        if np.any(events[:,1]<0):
            raise ValueError('faulty trigger found, please inspect manually.')    # check fo corrupted triggers 

        # replace with you triggers of interest !!!!!
        #event_id  = {'S1/red': 111, 'S1/green': 112,
        #            'S2/red': 121, 'S2/green': 122,
        #            'Cue_S1_R': 211, 'Cue_S1_G': 212, 
        #            'Cue_S2_R': 221, 'Cue_S2_G': 222}
        #trial_id  = {'S1/red': 111, 'S1/green': 112} 
        event_id  = {'S1/Sp1/L1': 111, 'S1/Sp1/L2': 112, 'S1/Sp1/L3': 113, 'S1/Sp1/L4': 114,
                     'S1/Sp2/L1': 121, 'S1/Sp2/L2': 122, 'S1/Sp2/L3': 123, 'S1/Sp2/L4': 124,
                     'S1/Sp3/L1': 131, 'S1/Sp3/L2': 132, 'S1/Sp3/L3': 133, 'S1/Sp3/L4': 134,
                     'S1/Sp4/L1': 141, 'S1/Sp4/L2': 142, 'S1/Sp4/L3': 143, 'S1/Sp4/L4': 144,
                     'S2/Sp1/L1': 211, 'S2/Sp1/L2': 212, 'S2/Sp1/L3': 213, 'S2/Sp1/L4': 214,
                     'S2/Sp2/L1': 221, 'S2/Sp2/L2': 222, 'S2/Sp2/L3': 223, 'S2/Sp2/L4': 224,
                     'S2/Sp3/L1': 231, 'S2/Sp3/L2': 232, 'S2/Sp3/L3': 233, 'S2/Sp3/L4': 234,
                     'S2/Sp4/L1': 241, 'S2/Sp4/L2': 242, 'S2/Sp4/L3': 243, 'S2/Sp4/L4': 244,
                     'Cue_S1': 101, 'Cue_S2': 201, 'Ping': 254, 
                     'Delay/S1/Cue': 100, 'Delay/S1/Ping': 200, 'Delay/S2/Cue': 250, 'Delay/S2/Ping': 251,
                     'Probe/SpNM/LoNM': 150, 'Probe/SpNM/LoM': 151, 'Probe/SpM/LoNM': 160, 'Probe/SpM/LoM': 161,
                     'Probe/L1/Di0': 154, 'Probe/L1/Di-2': 155, 'Probe/L1/Di-1': 156, 'Probe/L1/Di+1': 157, 'Probe/L1/Di+2': 158, # Di0 : perfect match, -/+2: Easy -/+1: Hard 
                     'Probe/L2/Di0': 164, 'Probe/L2/Di-2': 165, 'Probe/L2/Di-1': 166, 'Probe/L2/Di+1': 167, 'Probe/L2/Di+2': 168,
                     'Probe/L3/Di0': 174, 'Probe/L3/Di-2': 175, 'Probe/L3/Di-1': 176, 'Probe/L3/Di+1': 177, 'Probe/L3/Di+2': 178,
                     'Probe/L4/Di0': 184, 'Probe/L4/Di-2': 185, 'Probe/L4/Di-1': 186, 'Probe/L4/Di+1': 187, 'Probe/L4/Di+2': 188,
                     'ResponseStart': 199, 'NM': 190, 'M': 191,  'TooSlow': 192, 'Incorrect': 195, 'Correct': 196, 'BStart': 98, 
                     'B1': 71, 'B2': 72, 'B3': 73, 'B4': 74, 'B5': 75, 'B6': 76, 'B7': 77, 'B8': 78, 'PracticeB': 70, 'EndB': 99}
        trial_id  = {'S1/Sp1/L1': 111, 'S1/Sp1/L2': 112, 'S1/Sp1/L3': 113, 'S1/Sp1/L4': 114,
                     'S1/Sp2/L1': 121, 'S1/Sp2/L2': 122, 'S1/Sp2/L3': 123, 'S1/Sp2/L4': 124,
                     'S1/Sp3/L1': 131, 'S1/Sp3/L2': 132, 'S1/Sp3/L3': 133, 'S1/Sp3/L4': 134,
                     'S1/Sp4/L1': 141, 'S1/Sp4/L2': 142, 'S1/Sp4/L3': 143, 'S1/Sp4/L4': 144} 
        
        
        
        # replace system's noise reduction with individual empty room measurememts
        
        # if (actSubj == 3)&(se==1): # use this if you have a subject without noise file
        #     system_projs = raw.info['projs']
        # else:
        system_projs = raw.info['projs']
        raw.del_proj()
        empty_room_file = data_path + actNoiseFiles.iloc[ff]
        empty_room_raw = mne.io.read_raw_ctf(empty_room_file)
        empty_room_raw.del_proj()
        empty_room_projs = mne.compute_proj_raw(empty_room_raw, n_mag=5)
        raw.add_proj(empty_room_projs, remove_existing=True)
                   
                  
        # apply notch filter to remove line noise before jump detection
        notch_picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, 
                                     eog=True, ref_meg=False, include=[], exclude=[])
        raw.notch_filter(np.arange(50,251,50),notch_picks)
        
        
        # plot head motion
        # head_pos = mne.chpi._calculate_head_pos_ctf(raw)
        # mne.viz.plot_head_positions(head_pos, mode = 'field', show = False,
        # destination = None, info = raw.info);

        # start 1s before stim onset and end .5 s after response onset
        #MEGTrials = mne.Epochs(raw,events,trial_id,tmin = -1,tmax = 4.6,
        #            baseline=(None, 0), reject=None, verbose=False, detrend=0, preload=True) 
        # start 0.5s before stim onset and end 1.5 after response onset, its relative to S1
        MEGTrials = mne.Epochs(raw,events,trial_id,tmin = -0.5,tmax = 7.5,
                    baseline=(None, 0), reject=None, verbose=False, detrend=0, preload=True, on_missing='ignore') 
      
        # combine epochs of one session !FUNCTION EPOCHS.PY MANUALLY CHANGED BY CF
        if ff==0:
            CombTrials = MEGTrials
            CombRaw = raw
        else:
            CombTrials = mne.concatenate_epochs([CombTrials, MEGTrials]) # NOTE: function manually changed 
            CombRaw    = mne.concatenate_raws([CombRaw,raw])             # by CF to concatenate runs
                                                                         
    #CombTrials.save(str(homeDir)+'/Results/CutEpochs/CutData_VP'+str(actSubj)+'-epo.fif',split_size='2GB', overwrite=True)
    #CombRaw.save(str(homeDir)+'/Results/CutEpochs/CutData_VP'+str(actSubj)+'-raw.fif',split_size='2GB', overwrite=True)
    CombTrials.save(str(homeDir)+'/AWM4_data/processed/CutEpochs/CutData_VP'+str(actSubj)+'-epo.fif',split_size='2GB', overwrite=True)
    CombRaw.save(str(homeDir)+'/AWM4_data/processed/CutEpochs/CutData_VP'+str(actSubj)+'-raw.fif',split_size='2GB', overwrite=True)
    
    

#%% End of script ############################################################      