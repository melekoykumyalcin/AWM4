#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Probe Artifact Rejection Script
Visual inspection approach following MNE tutorial
Interactive identification of ECG and EOG components
"""

#%% Setup ####################################################################

import os
import locale
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
plt.rcParams['figure.figsize'] = [12, 8]
import numpy as np
import mne
from mne.preprocessing import ICA
import pandas as pd

# Set paths
homeDir = '/media/headmodel/Elements/AWM4/'
PROBE_EPOCHS_PATH = homeDir + '/AWM4_data/processed/ProbeEpochs/'
ICA_PATH = homeDir + '/AWM4_data/processed/ICAs/'

os.chdir(homeDir)

# Load meta information
metaFile = homeDir + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(metaFile)
Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
locale.setlocale(locale.LC_ALL, "en_US.utf8")

#%% Interactive manual artifact rejection ###################################

# Subject selection
print("Available subjects:", Subs)
actSubj = Subs[int(input("Subject no [1-" + str(len(Subs)) + "]: "))-1] 
print(f"Selected subject: {actSubj}")

# Load probe epochs (cleaned from jumps)
probe_epochs_file = PROBE_EPOCHS_PATH + f'ProbeEpochs_VP{actSubj}-epo-dropped.fif'
if not os.path.exists(probe_epochs_file):
    print(f"Error: Cleaned probe epochs not found: {probe_epochs_file}")
    exit()

print(f"Loading cleaned probe epochs...")
ProbeTrials = mne.read_epochs(probe_epochs_file)
print(f"Loaded {len(ProbeTrials)} probe epochs")

# Load raw data
raw_file = homeDir + f'/AWM4_data/processed/CutEpochs/CutData_VP{actSubj}-raw.fif'
if os.path.exists(raw_file):
    raw = mne.io.read_raw_fif(raw_file, preload=True)
    print("Loaded raw data for overlay plots")
else:
    raw = None
    print("Raw data not found - will use epochs for component inspection")

# Load probe-specific ICA
ica_file = ICA_PATH + f'probe_VP{actSubj}_fastica-ica.fif'
if not os.path.exists(ica_file):
    print(f"Error: Probe ICA file not found: {ica_file}")
    exit()

print(f"Loading probe ICA...")
ica = mne.preprocessing.read_ica(ica_file)
print(f"ICA loaded with {ica.n_components_} components")

#%% Step 1: Plot all ICA sources for overview ##############################

print(f"\n=== STEP 1: VISUAL INSPECTION OF ALL COMPONENTS ===")
print("Look for:")
print("- Eye blinks: large frontal components with regular timing")
print("- Heartbeats: regular rhythm, often posterior components") 
print("- Muscle artifacts: high frequency, temporally localized")
print("- Line noise: 50/60 Hz patterns")

# Plot component topographies (first 20-25 components)
n_components_to_show = min(25, ica.n_components_)
fig_topo = ica.plot_components(range(n_components_to_show), 
                              title=f'ICA Components 0-{n_components_to_show-1}', 
                              colorbar=True)

# Plot time series of components
if raw is not None:
    raw.load_data()
    print("Plotting component time series (close when done inspecting)...")
    fig_sources = ica.plot_sources(raw, show_scrollbars=False, title='ICA Component Time Series')
else:
    print("Plotting component time series from epochs...")
    fig_sources = ica.plot_sources(ProbeTrials, show_scrollbars=False, title='ICA Component Time Series')

input("Press Enter after you've identified suspicious components...")

#%% Step 2: Test individual components ######################################

print(f"\n=== STEP 2: TEST INDIVIDUAL COMPONENTS ===")
print("Test components by excluding them and seeing the effect")

while True:
    test_comp = input("Enter component number to test (or 'done' to finish): ").strip()
    
    if test_comp.lower() == 'done':
        break
        
    try:
        comp_idx = int(test_comp)
        if comp_idx >= ica.n_components_:
            print(f"Component {comp_idx} doesn't exist. Max component: {ica.n_components_-1}")
            continue
            
        print(f"\nTesting component {comp_idx}...")
        
        # Show component properties
        if raw is not None:
            fig_props = ica.plot_properties(raw, picks=[comp_idx])
        else:
            fig_props = ica.plot_properties(ProbeTrials, picks=[comp_idx])
        
        # Show overlay comparison
        if raw is not None:
            # Test on MEG magnetometers
            fig_overlay_mag = ica.plot_overlay(raw, exclude=[comp_idx], picks='mag', 
                                              title=f'Effect of excluding component {comp_idx} (MAG)')
            # Test on MEG gradiometers  
            fig_overlay_grad = ica.plot_overlay(raw, exclude=[comp_idx], picks='grad',
                                               title=f'Effect of excluding component {comp_idx} (GRAD)')
        else:
            # Use epochs average if no raw data
            evoked = ProbeTrials.average()
            fig_overlay = ica.plot_overlay(evoked, exclude=[comp_idx], picks='mag')
            
        should_exclude = input(f"Should component {comp_idx} be excluded? [y/n]: ").strip().lower()
        
        if should_exclude == 'y':
            if comp_idx not in ica.exclude:
                ica.exclude.append(comp_idx)
                print(f"Added component {comp_idx} to exclusion list")
            else:
                print(f"Component {comp_idx} already in exclusion list")
        
        plt.close('all')
        
    except ValueError:
        print("Please enter a valid component number or 'done'")
        continue

#%% Step 3: Review final exclusions #########################################

print(f"\n=== STEP 3: FINAL REVIEW ===")
print(f"Components to exclude: {sorted(ica.exclude)}")
print(f"Total components to exclude: {len(ica.exclude)}")

if len(ica.exclude) > 0:
    # Show final overview of excluded components
    fig_excluded = ica.plot_components(ica.exclude, 
                                     title='Components to be Excluded', 
                                     colorbar=True)
    
    # Show final overlay with all exclusions
    if raw is not None:
        fig_final_overlay = ica.plot_overlay(raw, exclude=ica.exclude, picks='mag',
                                           title='Final result with all exclusions (MAG)')
    
    final_confirm = input("Confirm these exclusions? [y/n]: ").strip().lower()
    
    if final_confirm != 'y':
        print("Exclusions cancelled. You can re-run the script to try again.")
        exit()

else:
    print("No components selected for exclusion.")
    no_exclusion_confirm = input("Continue with no exclusions? [y/n]: ").strip().lower()
    if no_exclusion_confirm != 'y':
        print("Process cancelled.")
        exit()

#%% Step 4: Save clean ICA and apply to data ################################

print(f"\n=== STEP 4: SAVING AND APPLYING CLEAN ICA ===")

# Save clean ICA
clean_ica_file = ICA_PATH + f'CleanProbeVP{actSubj}-ica.fif'
ica.save(clean_ica_file, overwrite=True)
print(f"Saved clean probe ICA: {clean_ica_file}")

# Apply ICA to probe epochs
print("Applying clean ICA to probe epochs...")
ica_probe_epochs = ProbeTrials.copy()
ica.apply(ica_probe_epochs)

# Save cleaned probe epochs
clean_epochs_file = PROBE_EPOCHS_PATH + f'ProbeEpochs_VP{actSubj}-cleanedICA-epo.fif'
ica_probe_epochs.save(clean_epochs_file, overwrite=True)
print(f"Saved ICA-cleaned probe epochs: {clean_epochs_file}")

# Print final trial counts
print(f"\nFinal cleaned probe epoch counts:")
for event_name, event_id in ica_probe_epochs.event_id.items():
    count = len(ica_probe_epochs[event_name])
    print(f"  {event_name}: {count} trials")

total_trials = len(ica_probe_epochs)
print(f"Total cleaned probe trials: {total_trials}")

# Optional: Compare before and after
compare = input("\nShow before/after comparison plots? [y/n]: ").strip().lower()
if compare == 'y':
    # Pick some channels for comparison
    picks_for_comparison = ['MEG0113', 'MEG0112', 'MEG0111', 'MEG0122', 'MEG0123']  # front channels
    available_picks = [ch for ch in picks_for_comparison if ch in ProbeTrials.ch_names]
    
    if available_picks:
        print("Plotting before/after comparison...")
        
        # Original data
        fig_before = ProbeTrials.average().plot(picks=available_picks, 
                                               titles='Before ICA cleaning',
                                               show=False)
        
        # Cleaned data  
        fig_after = ica_probe_epochs.average().plot(picks=available_picks,
                                                   titles='After ICA cleaning',
                                                   show=False)
        
        plt.show()

plt.close('all')
print(f"\n=== MANUAL PROBE ARTIFACT REJECTION COMPLETE FOR SUBJECT {actSubj} ===")
print(f"Excluded components: {sorted(ica.exclude)}")