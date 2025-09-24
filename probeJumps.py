#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe Jump Detection Script
Adapted from B_JumpDetection.py for probe-locked epochs
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
OUTPUT_PATH = HOME_DIR + '/AWM4_data/processed/ProbeEpochs/'
JUMP_OUTPUT_PATH = OUTPUT_PATH + 'ProbeJumps'

os.chdir(HOME_DIR)

# Load meta information
metaInfo = pd.read_excel(META_FILE)
Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
locale.setlocale(locale.LC_ALL, "en_US.utf8")

#%% Interactive subject selection and jump detection ########################

print("Available subjects:", Subs)
actSubj = Subs[int(input("Subject no [1-" + str(len(Subs)) + "]: "))-1] 
print(f"Selected subject: {actSubj}")

# Load probe epochs
probe_epochs_file = OUTPUT_PATH + f'ProbeEpochs_VP{actSubj}-epo.fif'

if not os.path.exists(probe_epochs_file):
    print(f"Error: Probe epochs file not found: {probe_epochs_file}")
    print("Please run probe epoching first.")
    exit()

print(f"Loading probe epochs for subject {actSubj}...")
probe_epochs = mne.read_epochs(probe_epochs_file)

print(f"Loaded {len(probe_epochs)} probe epochs")
print(f"Event types: {list(probe_epochs.event_id.keys())}")

# Print trial counts per condition
for event_name, event_id in probe_epochs.event_id.items():
    count = len(probe_epochs[event_name])
    print(f"  {event_name}: {count} trials")

#%% Plot epochs for visual inspection ########################################

print("\nPlotting probe epochs for visual inspection...")
print("Look for jumps, artifacts, or noisy trials.")
print("Use the interface to mark bad epochs.")

# Plot epochs - you can adjust n_epochs and scalings as needed
probe_epochs.plot(n_epochs=10, scalings={'meg': 1e-12}, block=True)

#%% Get jump indices from user ###############################################

print("\nAfter reviewing the epochs:")
jump_inds = list(map(int, input("Please enter the epochs that contained jumps/artifacts (as: x y z): ").split())) 

# Save jump indices specifically for probe epochs
jump_fname = JUMP_OUTPUT_PATH + f'ProbeJumps{actSubj}.npy'
np.save(jump_fname, jump_inds)
print(f"Saved jump indices to: {jump_fname}")
print(f"Jump epochs: {jump_inds}")

#%% Apply jump removal automatically ##########################################

def apply_jump_removal(epochs, subject):
    """
    Apply jump removal using the same logic as analysis scripts
    """
    
    probe_jump_file = JUMP_OUTPUT_PATH + f'ProbeJumps{subject}.npy'
        
    # Apply probe-specific jumps if they exist
    if os.path.isfile(probe_jump_file):
        probe_jump_inds = np.load(probe_jump_file)
        if len(probe_jump_inds) > 0:
            probe_jump_inds = np.array(probe_jump_inds, dtype=int)
            valid_probe_jump_inds = probe_jump_inds[probe_jump_inds < len(epochs)]
            
            if len(valid_probe_jump_inds) > 0:
                epochs.drop(valid_probe_jump_inds, reason='jump_probe')
                print(f"  Removed {len(valid_probe_jump_inds)} epochs with probe-specific jumps")
        
    return epochs

# Update trial counts
print("Trial counts after jump removal:")
for event_name, event_id in probe_epochs.event_id.items():
    count = len(probe_epochs[event_name])
    print(f"  {event_name}: {count} trials")

# Save final cleaned epochs
clean_fname = OUTPUT_PATH + f'ProbeEpochs_VP{actSubj}-epo-dropped.fif'
probe_epochs.save(clean_fname, overwrite=True)
print(f"Saved final cleaned probe epochs to: {clean_fname}")

plt.close('all')
print("Jump detection and removal complete!")

#%% Batch processing function for automatic jump removal ##################

# def process_all_subjects_with_jumps(subjects_to_process=None):
#     """
#     Process all subjects and automatically apply jump removal
#     """
#     if subjects_to_process is None:
#         subjects_to_process = Subs
    
#     for subject in subjects_to_process:
#         print(f"\nProcessing subject {subject}...")
        
#         # Check if probe epochs exist
#         probe_file = OUTPUT_PATH + f'ProbeEpochs_VP{subject}-epo.fif'
#         if not os.path.exists(probe_file):
#             print(f"  Skipping: No probe epochs found for subject {subject}")
#             continue
        
#         # Load probe epochs
#         probe_epochs = mne.read_epochs(probe_file)
#         print(f"  Loaded {len(probe_epochs)} probe epochs")
        
#         # Apply automatic jump removal
#         probe_epochs = apply_jump_removal(probe_epochs, subject)
        
#         # Save cleaned epochs
#         clean_fname = OUTPUT_PATH + f'ProbeEpochs_VP{subject}-epo-clean.fif'
#         probe_epochs.save(clean_fname, overwrite=True)
#         print(f"  Saved cleaned epochs: {clean_fname}")

# # Uncomment to run batch processing for all subjects:
# # process_all_subjects_with_jumps()