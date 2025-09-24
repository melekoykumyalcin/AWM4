#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created for probe-locked epoch analysis
Analysis of Probe Processing
"""

#%% Setup ####################################################################

# Import libraries
import os
import locale
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt

# Set paths
HOME_DIR = '/media/headmodel/Elements/AWM4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
DATA_PATH = HOME_DIR + '/AWM4_data/raw/'
CORRECTED_DATA = HOME_DIR + '/AWM4_data/raw/correctTriggers/'
OUTPUT_PATH = HOME_DIR + '/AWM4_data/processed/ProbeEpochs/'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(HOME_DIR)

# Load meta information
metaInfo = pd.read_excel(META_FILE)
Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
#Subs = Subs[[9, 16, 23]]

locale.setlocale(locale.LC_ALL, "en_US.utf8")

# Define probe events of interest
probe_event_ids = {
    'Probe/SpNM/LoNM': 150,  # Voice Non-match, Location Non-match
    'Probe/SpNM/LoM': 151,   # Voice Non-match, Location Match  
    'Probe/SpM/LoNM': 160,   # Voice Match, Location Non-match
    'Probe/SpM/LoM': 161     # Voice Match, Location Match
}

print(f"Processing {len(Subs)} subjects for probe-locked analysis...")
print(f"Target events: {list(probe_event_ids.keys())}")

#%% Process each subject

cc = 0
for subject in Subs:
    print(f"\nProcessing subject {subject}...")
    
    try:
        # Get file information (using your existing logic)
        actInd = (metaInfo.Subject==subject) & (metaInfo.Valid==1)
        
        # Determine if subject is in early subjects
        early_subject = subject in np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])[:7]
        
        if early_subject:
            actFiles = pd.Series([f.split('.')[0] + '_correct_triggers.fif' for f in metaInfo['MEG_Name']])[actInd]
        else:
            actFiles = metaInfo['MEG_Name'][actInd]
        
        # Load and concatenate raw data
        all_events = None
        reference_dev_head_t_ref = None
        raw_files = []
        
        for ff in range(actFiles.count()):
            if early_subject:
                fname = CORRECTED_DATA + actFiles.iloc[ff]
                raw = mne.io.read_raw_fif(fname, preload=True)
            else:
                fname = DATA_PATH + actFiles.iloc[ff]
                raw = mne.io.read_raw_ctf(fname, 'truncate', True)
                
            if ff == 0:
                reference_dev_head_t_ref = raw.info["dev_head_t"]
            else:
                raw.info['dev_head_t'] = reference_dev_head_t_ref
                
            events = mne.find_events(raw, 'UPPT001', shortest_event=1)
            if ff != 0:
                events = events[events[:, 1] == 0, :]
                
            if ff == 0:
                all_events = events
                raw_concat = raw.copy()
            else:
                all_events = np.concatenate((all_events, events), axis=0)
                raw_concat.append(raw)
            
            del raw
        
        # Create probe-locked epochs
        probe_epochs = mne.Epochs(
            raw_concat, 
            all_events, 
            probe_event_ids,
            tmin=-0.5,      # 500ms before probe
            tmax=1.0,       # 1000ms after probe
            baseline=(-0.1, 0),  # 100ms pre-probe baseline
            reject=None, 
            verbose=False, 
            detrend=0, 
            preload=True, 
            on_missing='ignore',
            event_repeated='drop' # automatically drop duplicate events that occur at the same time sample
        )
        
        # Clean up raw data
        del raw_concat
        
        # Pick MEG channels only
        probe_epochs.pick_types(meg=True)
        
        # Load and apply ICA (using your existing logic)
        ica_file = HOME_DIR + '/AWM4_data/processed/ICAs/CleanVP' + str(subject) + '-ica.fif'
        if os.path.exists(ica_file):
            ica = mne.preprocessing.read_ica(ica_file)
            ica.apply(probe_epochs)
            print(f"  Applied ICA")
        else:
            print(f"  Warning: No ICA file found for subject {subject}")
        
        # # Drop epochs with jumps (using your existing logic)
        # jump_file = HOME_DIR + '/AWM4_data/processed/ICAs/Jumps' + str(subject) + '.npy'
        # if os.path.exists(jump_file):
        #     jump_inds = np.load(jump_file)
        #     # Note: jump indices from S1-locked epochs may not directly apply
        #     # You might need to adjust this based on your jump detection method
        #     print(f"  Warning: Jump indices from S1-locked epochs may need adjustment")
        
        # Apply lowpass filter
        probe_epochs.filter(l_freq=None, h_freq=30)
        
        # Print trial counts for quality check
        print(f"  Trial counts:")
        for event_name in probe_event_ids.keys():
            if event_name in probe_epochs.event_id:
                count = len(probe_epochs[event_name])
                print(f"    {event_name}: {count} trials")
            else:
                print(f"    {event_name}: 0 trials")
        
        total_trials = len(probe_epochs)
        print(f"  Total probe trials: {total_trials}")
        
        # Save probe-locked epochs
        output_file = OUTPUT_PATH + f'ProbeEpochs_VP{subject}-epo.fif'
        probe_epochs.save(output_file, overwrite=True)
        print(f"  Saved: {output_file}")
        

        cc += 1
        print(f"Completed subject {subject} ({cc}/{len(Subs)})")
        
    except Exception as e:
        print(f"Error processing subject {subject}: {str(e)}")
        continue

print(f"\nProcessing complete! {cc}/{len(Subs)} subjects processed successfully.")
print(f"Probe-locked epochs saved to: {OUTPUT_PATH}")

# Look at trial-level data to spot artifacts
#read probe epochs
# for subject in Subs:
#     print(f"\nLoading probe epochs for subject {subject}...")
#     probe_epochs = mne.read_epochs(OUTPUT_PATH + f'ProbeEpochs_VP{subject}-epo.fif')
#     probe_epochs.plot(n_epochs=10, scalings={'meg': 1e-12})  
#     #save the dropped version
#     probe_epochs.save(OUTPUT_PATH + f'ProbeEpochs_VP{subject}-epo-dropped.fif', overwrite=True)


for subject in Subs:
    print(f"\nLoading probe epochs for subject {subject}...")
    probecorrected_epochs = mne.read_epochs(OUTPUT_PATH + f'ProbeEpochs_VP{subject}-epo-dropped.fif')
    # check how many epochs are there 
    print(f"  Number of epochs before dropping: {len(probecorrected_epochs)}")
    probecorrected_epochs.plot(n_epochs=10, scalings={'meg': 1e-12})  
    #save the dropped version
    probecorrected_epochs.save(OUTPUT_PATH + f'ProbeEpochs_VP{subject}-epo-control.fif', overwrite=True)
    # check how many epochs are there after dropping
    print(f"  Number of epochs after dropping: {len(probecorrected_epochs)}")
    # read the epochs again to ensure they are saved correctly
    probecontrol_epochs = mne.read_epochs(OUTPUT_PATH + f'ProbeEpochs_VP{subject}-epo-control.fif')
    # how many epochs are there after saving
    print(f"  Number of epochs after saving: {len(probecontrol_epochs)}")

# Convert Epochs to Evoked objects first
probe_evoked_list = []
for subject in Subs:
    print(f"\nLoading probe epochs for subject {subject}...")
    probe_epochs = mne.read_epochs(OUTPUT_PATH + f'ProbeEpochs_VP{subject}-epo-dropped.fif')
    # Average epochs to get Evoked object
    probe_evoked = probe_epochs.average()
    probe_evoked_list.append(probe_evoked)

# Now create grand average from Evoked objects
grand_average = mne.grand_average(probe_evoked_list)
# Plot the grand average
grand_average.plot(spatial_colors=True)

