#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Most Basic Match vs Non-Match ERP Analysis
Simple visual inspection of voice matching effects
"""

import os
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt

# Set paths
HOME_DIR = '/media/headmodel/Elements/AWM4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
PROBE_EPOCHS_PATH = HOME_DIR + '/AWM4_data/processed/ProbeEpochs/'
OUTPUT_PATH = HOME_DIR + '/AWM4_data/processed/ProbeMatchAnalysis/'

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(HOME_DIR)

# Load meta information
metaInfo = pd.read_excel(META_FILE)
Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
# Subs = Subs[21:]  # Use your subject selection

print("="*60)
print("BASIC MATCH vs NON-MATCH ERP ANALYSIS")
print("="*60)
print(f"Processing {len(Subs)} subjects")

# Simple conditions 
conditions_of_interest = {
    'Voice_Match': ['Probe/SpM/LoNM', 'Probe/SpM/LoM'],      # Voice matches target
    'Voice_NonMatch': ['Probe/SpNM/LoNM', 'Probe/SpNM/LoM']  # Voice doesn't match target
}

#%% Step 1: Load data and create simple contrasts

match_evokeds = []
nonmatch_evokeds = []
all_subjects = []

for subject in Subs:
    try:
        print(f"Processing subject {subject}...")
        
        # Load epochs
        epoch_files = [
            PROBE_EPOCHS_PATH + f'ProbeEpochs_VP{subject}-epo-dropped.fif'
                            ]
        
        probe_epochs = None
        for epoch_file in epoch_files:
            if os.path.exists(epoch_file):
                probe_epochs = mne.read_epochs(epoch_file, verbose=False)
                break
        
        if probe_epochs is None:
            print(f"  No epochs found for subject {subject}")
            continue
                
        # Crop to focus on probe period
        probe_epochs = probe_epochs.copy().crop(tmin=-0.1, tmax=1.0)
        
        # Create Match condition (combine voice match conditions)
        match_epochs = mne.concatenate_epochs([
            probe_epochs['Probe/SpM/LoNM'], 
            probe_epochs['Probe/SpM/LoM']
        ])
        match_evoked = match_epochs.average()
        
        # Create Non-Match condition (combine voice non-match conditions)  
        nonmatch_epochs = mne.concatenate_epochs([
            probe_epochs['Probe/SpNM/LoNM'],
            probe_epochs['Probe/SpNM/LoM']
        ])
        nonmatch_evoked = nonmatch_epochs.average()
        
        # Store
        match_evokeds.append(match_evoked)
        nonmatch_evokeds.append(nonmatch_evoked)
        all_subjects.append(subject)
        
        print(f"  ✓ Match trials: {len(match_epochs)}, Non-match trials: {len(nonmatch_epochs)}")
        
    except Exception as e:
        print(f"  Error: {e}")
        continue

print(f"\nSuccessfully processed {len(all_subjects)} subjects")

#%% Step 2: Create Grand Averages

# Compute grand averages
match_grand_avg = mne.grand_average(match_evokeds)
nonmatch_grand_avg = mne.grand_average(nonmatch_evokeds)

print(f"Grand averages computed from {len(match_evokeds)} subjects")

#%% Step 3: Basic Visual Inspection

# 'MRT53-3609', 'MLT42-3609', 'MLT54-3609'
temporal_channels = [
    'MLT11-3609', 'MLT12-3609', 'MLT13-3609', 'MLT14-3609', 'MLT15-3609', 'MLT16-3609',
    'MLT21-3609', 'MLT22-3609', 'MLT23-3609', 'MLT24-3609', 'MLT25-3609', 'MLT26-3609', 'MLT27-3609',
    'MLT31-3609', 'MLT32-3609', 'MLT33-3609', 'MLT34-3609', 'MLT35-3609', 'MLT36-3609', 'MLT37-3609',
    'MLT41-3609', 'MLT43-3609', 'MLT44-3609', 'MLT45-3609', 'MLT46-3609', 'MLT47-3609',
    'MLT51-3609', 'MLT52-3609', 'MLT53-3609', 'MLT55-3609', 'MLT56-3609', 'MLT57-3609',
    'MRT11-3609', 'MRT12-3609', 'MRT13-3609', 'MRT14-3609', 'MRT15-3609', 'MRT16-3609',
    'MRT21-3609', 'MRT22-3609', 'MRT23-3609', 'MRT24-3609', 'MRT25-3609', 'MRT26-3609', 'MRT27-3609',
    'MRT31-3609', 'MRT32-3609', 'MRT33-3609', 'MRT34-3609', 'MRT35-3609', 'MRT36-3609', 'MRT37-3609',
    'MRT41-3609', 'MRT42-3609', 'MRT43-3609', 'MRT44-3609', 'MRT45-3609', 'MRT46-3609', 'MRT47-3609',
    'MRT51-3609', 'MRT52-3609', 'MRT54-3609', 'MRT55-3609', 'MRT56-3609', 'MRT57-3609'
]

# Check which temporal channels are available
available_temporal = [ch for ch in temporal_channels if ch in match_grand_avg.ch_names]
print(f"Using {len(available_temporal)} temporal channels for ROI analysis")

#%% Plot 1: All Magnetometers - Butterfly Plot

# Define problematic sensors to exclude
problematic_sensors = ['MRT53-3609', 'MLT42-3609', 'MLT54-3609', 'MRO44-3609', 'MLF41-3609']

# Create copies of the grand averages without problematic sensors
match_clean = match_grand_avg.copy()
nonmatch_clean = nonmatch_grand_avg.copy()

# Drop problematic channels
match_clean.drop_channels(problematic_sensors)
nonmatch_clean.drop_channels(problematic_sensors)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Match condition - all magnetometers (excluding problematic ones)
match_clean.plot(axes=axes[0,0], spatial_colors=True, show=False)
axes[0,0].set_title(f'MATCH Trials (N={len(match_evokeds)} subjects)\nAll Magnetometers')
axes[0,0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Probe Onset')
axes[0,0].legend()

# Non-match condition - all magnetometers (excluding problematic ones)
nonmatch_clean.plot(axes=axes[0,1], spatial_colors=True, show=False)
axes[0,1].set_title(f'NON-MATCH Trials (N={len(nonmatch_evokeds)} subjects)\nAll Magnetometers')
axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.7, label='Probe Onset')
axes[0,1].legend()

# Convert times to milliseconds for better readability
times_ms = match_clean.times * 1000

# Direct comparison - averaged across all magnetometers
match_avg = np.mean(match_clean.data, axis=0)
nonmatch_avg = np.mean(nonmatch_clean.data, axis=0)

axes[1,0].plot(times_ms, match_avg, label='Match', linewidth=3, color='blue')
axes[1,0].plot(times_ms, nonmatch_avg, label='Non-Match', linewidth=3, color='red')
axes[1,0].axvline(0, color='black', linestyle='--', alpha=0.7, label='Probe Onset')
axes[1,0].axhline(0, color='gray', linestyle='-', alpha=0.3)
axes[1,0].set_xlabel('Time (ms)')
axes[1,0].set_ylabel('Amplitude (T)')
axes[1,0].set_title('Direct Comparison - All Magnetometers')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Difference wave (Match - Non-match)
difference = match_avg - nonmatch_avg
axes[1,1].plot(times_ms, difference, linewidth=3, color='green', label='Match - Non-Match')
axes[1,1].axvline(0, color='black', linestyle='--', alpha=0.7, label='Probe Onset')
axes[1,1].axhline(0, color='gray', linestyle='-', alpha=0.3)
axes[1,1].set_xlabel('Time (ms)')
axes[1,1].set_ylabel('Difference (T)')
axes[1,1].set_title('Difference Wave (Match - Non-Match)')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'basic_match_nonmatch_all_channels.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Plot 2: Temporal ROI Focus

if len(available_temporal) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get temporal channel indices
    temporal_picks = mne.pick_channels(match_grand_avg.ch_names, available_temporal)
    
    # Match condition - temporal channels only
    match_grand_avg.plot(picks=temporal_picks, axes=axes[0,0], spatial_colors=True, show=False)
    axes[0,0].set_title(f'MATCH Trials - Temporal Channels\n({len(available_temporal)} channels)')
    axes[0,0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Probe Onset')
    axes[0,0].legend()
    
    # Non-match condition - temporal channels only
    nonmatch_grand_avg.plot(picks=temporal_picks, axes=axes[0,1], spatial_colors=True, show=False)
    axes[0,1].set_title(f'NON-MATCH Trials - Temporal Channels\n({len(available_temporal)} channels)')
    axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.7, label='Probe Onset')
    axes[0,1].legend()
    
    # Direct comparison - averaged across temporal channels
    match_temporal = np.mean(match_grand_avg.data[temporal_picks], axis=0)
    nonmatch_temporal = np.mean(nonmatch_grand_avg.data[temporal_picks], axis=0)
    
    axes[1,0].plot(times_ms, match_temporal, label='Match', linewidth=3, color='blue')
    axes[1,0].plot(times_ms, nonmatch_temporal, label='Non-Match', linewidth=3, color='red')
    axes[1,0].axvline(0, color='black', linestyle='--', alpha=0.7, label='Probe Onset')
    axes[1,0].axhline(0, color='gray', linestyle='-', alpha=0.3)
    axes[1,0].set_xlabel('Time (ms)')
    axes[1,0].set_ylabel('Amplitude (T)')
    axes[1,0].set_title('Direct Comparison - Temporal ROI')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Difference wave - temporal ROI
    difference_temporal = match_temporal - nonmatch_temporal
    axes[1,1].plot(times_ms, difference_temporal, linewidth=3, color='green', label='Match - Non-Match')
    axes[1,1].axvline(0, color='black', linestyle='--', alpha=0.7, label='Probe Onset')
    axes[1,1].axhline(0, color='gray', linestyle='-', alpha=0.3)
    axes[1,1].set_xlabel('Time (ms)')
    axes[1,1].set_ylabel('Difference (T)')
    axes[1,1].set_title('Difference Wave - Temporal ROI')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + 'basic_match_nonmatch_temporal_roi.png', dpi=300, bbox_inches='tight')
    plt.show()

#%% Plot 3: Time Window Analysis

# Define time windows of interest
time_windows = {
    'Early (100-200ms)': (0.1, 0.2),
    'Mid (200-400ms)': (0.2, 0.4),  
    'Late (400-600ms)': (0.4, 0.6),
    'Very Late (600-1000ms)': (0.6, 1.0)
}

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for i, (window_name, (tmin, tmax)) in enumerate(time_windows.items()):
    ax = axes[i//2, i%2]
    
    # Find time indices
    time_mask = (match_grand_avg.times >= tmin) & (match_grand_avg.times <= tmax)
    
    # Extract mean amplitudes in this window
    if len(available_temporal) > 0:
        # Use temporal ROI
        match_window = np.mean(match_grand_avg.data[temporal_picks][:, time_mask])
        nonmatch_window = np.mean(nonmatch_grand_avg.data[temporal_picks][:, time_mask])
    else:
        # Use all channels
        match_window = np.mean(match_grand_avg.data[:, time_mask])
        nonmatch_window = np.mean(nonmatch_grand_avg.data[:, time_mask])
    
    # Simple bar plot
    conditions = ['Match', 'Non-Match']
    amplitudes = [match_window, nonmatch_window]
    colors = ['blue', 'red']
    
    bars = ax.bar(conditions, amplitudes, color=colors, alpha=0.7)
    ax.set_ylabel('Mean Amplitude (T)')
    ax.set_title(f'{window_name}\n({tmin*1000:.0f}-{tmax*1000:.0f}ms)')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, amp in zip(bars, amplitudes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{amp:.2e}', ha='center', va='bottom' if height >= 0 else 'top')

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'basic_match_nonmatch_time_windows.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Print Summary

print("\n" + "="*60)
print("BASIC ANALYSIS SUMMARY")
print("="*60)

print(f"Subjects analyzed: {len(all_subjects)}")
print(f"Temporal channels used: {len(available_temporal)}")

# Simple peak analysis
if len(available_temporal) > 0:
    match_data = np.mean(match_grand_avg.data[temporal_picks], axis=0)
    nonmatch_data = np.mean(nonmatch_grand_avg.data[temporal_picks], axis=0)
else:
    match_data = np.mean(match_grand_avg.data, axis=0) 
    nonmatch_data = np.mean(nonmatch_grand_avg.data, axis=0)

# Find peaks
match_peak_idx = np.argmax(np.abs(match_data))
nonmatch_peak_idx = np.argmax(np.abs(nonmatch_data))

match_peak_time = match_grand_avg.times[match_peak_idx]
nonmatch_peak_time = nonmatch_grand_avg.times[nonmatch_peak_idx]

match_peak_amp = match_data[match_peak_idx]
nonmatch_peak_amp = nonmatch_data[nonmatch_peak_idx]

print(f"\nPEAK ANALYSIS:")
print(f"Match peak: {match_peak_amp:.2e} T at {match_peak_time*1000:.1f}ms")
print(f"Non-match peak: {nonmatch_peak_amp:.2e} T at {nonmatch_peak_time*1000:.1f}ms")

# Simple difference assessment
difference_data = match_data - nonmatch_data
max_diff_idx = np.argmax(np.abs(difference_data))
max_diff_time = match_grand_avg.times[max_diff_idx]
max_diff_amp = difference_data[max_diff_idx]

print(f"\nLARGEST DIFFERENCE:")
print(f"Max difference: {max_diff_amp:.2e} T at {max_diff_time*1000:.1f}ms")
print(f"Direction: {'Match > Non-Match' if max_diff_amp > 0 else 'Non-Match > Match'}")

# Save grand averages
match_grand_avg.save(OUTPUT_PATH + 'match_grand_average-ave.fif', overwrite=True)
nonmatch_grand_avg.save(OUTPUT_PATH + 'nonmatch_grand_average-ave.fif', overwrite=True)

print(f"\nFiles saved to: {OUTPUT_PATH}")
print("✓ Basic analysis complete!")