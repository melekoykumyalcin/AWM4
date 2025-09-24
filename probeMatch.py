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
OUTPUT_PATH = HOME_DIR + '/AWM4_data/processed/BasicMatchAnalysis/'

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

# Simple conditions - both voice and location matching
conditions_of_interest = {
    'Voice_Match': ['Probe/SpM/LoNM', 'Probe/SpM/LoM'],      # Voice matches target (relevant)
    'Voice_NonMatch': ['Probe/SpNM/LoNM', 'Probe/SpNM/LoM'], # Voice doesn't match target (relevant)
    'Location_Match': ['Probe/SpNM/LoM', 'Probe/SpM/LoM'],    # Location matches target (irrelevant)
    'Location_NonMatch': ['Probe/SpNM/LoNM', 'Probe/SpM/LoNM'] # Location doesn't match target (irrelevant)
}

#%% Step 1: Load data and create simple contrasts

# Storage for all conditions
voice_match_evokeds = []
voice_nonmatch_evokeds = []
location_match_evokeds = []
location_nonmatch_evokeds = []
all_subjects = []

for subject in Subs:
    try:
        print(f"Processing subject {subject}...")
        
        # Load epochs
        epoch_files = [
            PROBE_EPOCHS_PATH + f'ProbeEpochs_VP{subject}-epo-dropped.fif',
            PROBE_EPOCHS_PATH + f'ProbeEpochs_VP{subject}-epo.fif'
        ]
        
        probe_epochs = None
        for epoch_file in epoch_files:
            if os.path.exists(epoch_file):
                probe_epochs = mne.read_epochs(epoch_file, verbose=False)
                break
        
        if probe_epochs is None:
            print(f"  No epochs found for subject {subject}")
            continue
        
        # Check if we have the conditions we need
        required_conditions = ['Probe/SpM/LoNM', 'Probe/SpM/LoM', 'Probe/SpNM/LoNM', 'Probe/SpNM/LoM']
        missing = [cond for cond in required_conditions if cond not in probe_epochs.event_id]
        if missing:
            print(f"  Missing conditions: {missing}")
            continue
        
        # Crop to focus on probe period
        probe_epochs = probe_epochs.copy().crop(tmin=-0.1, tmax=1.0)
        
        # Create VOICE conditions
        voice_match_epochs = mne.concatenate_epochs([
            probe_epochs['Probe/SpM/LoNM'], 
            probe_epochs['Probe/SpM/LoM']
        ])
        voice_match_evoked = voice_match_epochs.average()
        
        voice_nonmatch_epochs = mne.concatenate_epochs([
            probe_epochs['Probe/SpNM/LoNM'],
            probe_epochs['Probe/SpNM/LoM']
        ])
        voice_nonmatch_evoked = voice_nonmatch_epochs.average()
        
        # Create LOCATION conditions
        location_match_epochs = mne.concatenate_epochs([
            probe_epochs['Probe/SpNM/LoM'],
            probe_epochs['Probe/SpM/LoM']
        ])
        location_match_evoked = location_match_epochs.average()
        
        location_nonmatch_epochs = mne.concatenate_epochs([
            probe_epochs['Probe/SpNM/LoNM'],
            probe_epochs['Probe/SpM/LoNM']
        ])
        location_nonmatch_evoked = location_nonmatch_epochs.average()
        
        # Store all conditions
        voice_match_evokeds.append(voice_match_evoked)
        voice_nonmatch_evokeds.append(voice_nonmatch_evoked)
        location_match_evokeds.append(location_match_evoked)
        location_nonmatch_evokeds.append(location_nonmatch_evoked)
        all_subjects.append(subject)
        
        print(f"  ✓ Voice Match: {len(voice_match_epochs)}, Voice Non-match: {len(voice_nonmatch_epochs)}")
        print(f"    Location Match: {len(location_match_epochs)}, Location Non-match: {len(location_nonmatch_epochs)}")
        
    except Exception as e:
        print(f"  Error: {e}")
        continue

print(f"\nSuccessfully processed {len(all_subjects)} subjects")

#%% Step 2: Create Grand Averages

# Compute grand averages for all conditions
voice_match_grand_avg = mne.grand_average(voice_match_evokeds)
voice_nonmatch_grand_avg = mne.grand_average(voice_nonmatch_evokeds)
location_match_grand_avg = mne.grand_average(location_match_evokeds)
location_nonmatch_grand_avg = mne.grand_average(location_nonmatch_evokeds)

print(f"Grand averages computed from {len(voice_match_evokeds)} subjects")
print("Conditions created:")
print("✓ Voice Match (relevant feature)")
print("✓ Voice Non-Match (relevant feature)")
print("✓ Location Match (irrelevant feature)")  
print("✓ Location Non-Match (irrelevant feature)")

#%% Step 3: Basic Visual Inspection

# Define temporal channels for ROI analysis (adjust channel names to your system)
temporal_channels = [
    'MLT11-3609', 'MLT12-3609', 'MLT13-3609', 'MLT14-3609', 'MLT15-3609',
    'MLT21-3609', 'MLT22-3609', 'MLT23-3609', 'MLT24-3609', 'MLT25-3609',
    'MRT11-3609', 'MRT12-3609', 'MRT13-3609', 'MRT14-3609', 'MRT15-3609',
    'MRT21-3609', 'MRT22-3609', 'MRT23-3609', 'MRT24-3609', 'MRT25-3609'
]

# Check which temporal channels are available
available_temporal = [ch for ch in temporal_channels if ch in voice_match_grand_avg.ch_names]
print(f"Using {len(available_temporal)} temporal channels for ROI analysis")

#%% Plot 1: VOICE MATCHING (Relevant Feature) - All Magnetometers

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Voice Match condition - all magnetometers
voice_match_grand_avg.plot(axes=axes[0,0], spatial_colors=True, show=False)
axes[0,0].set_title(f'VOICE MATCH (Relevant Feature)\nAll Magnetometers (N={len(voice_match_evokeds)})')
axes[0,0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Probe Onset')
axes[0,0].legend()

# Voice Non-match condition - all magnetometers  
voice_nonmatch_grand_avg.plot(axes=axes[0,1], spatial_colors=True, show=False)
axes[0,1].set_title(f'VOICE NON-MATCH (Relevant Feature)\nAll Magnetometers (N={len(voice_nonmatch_evokeds)})')
axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.7, label='Probe Onset')
axes[0,1].legend()

# Convert times to milliseconds for better readability
times_ms = voice_match_grand_avg.times * 1000

# Direct comparison - averaged across all magnetometers
voice_match_avg = np.mean(voice_match_grand_avg.data, axis=0)
voice_nonmatch_avg = np.mean(voice_nonmatch_grand_avg.data, axis=0)

axes[1,0].plot(times_ms, voice_match_avg, label='Voice Match', linewidth=3, color='blue')
axes[1,0].plot(times_ms, voice_nonmatch_avg, label='Voice Non-Match', linewidth=3, color='red')
axes[1,0].axvline(0, color='black', linestyle='--', alpha=0.7, label='Probe Onset')
axes[1,0].axhline(0, color='gray', linestyle='-', alpha=0.3)
axes[1,0].set_xlabel('Time (ms)')
axes[1,0].set_ylabel('Amplitude (T)')
axes[1,0].set_title('Voice Comparison - All Magnetometers')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Voice difference wave (Match - Non-match)
voice_difference = voice_match_avg - voice_nonmatch_avg
axes[1,1].plot(times_ms, voice_difference, linewidth=3, color='darkblue', label='Voice Match - Non-Match')
axes[1,1].axvline(0, color='black', linestyle='--', alpha=0.7, label='Probe Onset')
axes[1,1].axhline(0, color='gray', linestyle='-', alpha=0.3)
axes[1,1].set_xlabel('Time (ms)')
axes[1,1].set_ylabel('Difference (T)')
axes[1,1].set_title('Voice Difference Wave (Relevant Feature)')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'voice_match_nonmatch_all_channels.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Plot 2: LOCATION MATCHING (Irrelevant Feature) - All Magnetometers

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Location Match condition - all magnetometers
location_match_grand_avg.plot(axes=axes[0,0], spatial_colors=True, show=False)
axes[0,0].set_title(f'LOCATION MATCH (Irrelevant Feature)\nAll Magnetometers (N={len(location_match_evokeds)})')
axes[0,0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Probe Onset')
axes[0,0].legend()

# Location Non-match condition - all magnetometers  
location_nonmatch_grand_avg.plot(axes=axes[0,1], spatial_colors=True, show=False)
axes[0,1].set_title(f'LOCATION NON-MATCH (Irrelevant Feature)\nAll Magnetometers (N={len(location_nonmatch_evokeds)})')
axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.7, label='Probe Onset')
axes[0,1].legend()

# Direct comparison - averaged across all magnetometers
location_match_avg = np.mean(location_match_grand_avg.data, axis=0)
location_nonmatch_avg = np.mean(location_nonmatch_grand_avg.data, axis=0)

axes[1,0].plot(times_ms, location_match_avg, label='Location Match', linewidth=3, color='orange')
axes[1,0].plot(times_ms, location_nonmatch_avg, label='Location Non-Match', linewidth=3, color='purple')
axes[1,0].axvline(0, color='black', linestyle='--', alpha=0.7, label='Probe Onset')
axes[1,0].axhline(0, color='gray', linestyle='-', alpha=0.3)
axes[1,0].set_xlabel('Time (ms)')
axes[1,0].set_ylabel('Amplitude (T)')
axes[1,0].set_title('Location Comparison - All Magnetometers')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Location difference wave (Match - Non-match)
location_difference = location_match_avg - location_nonmatch_avg
axes[1,1].plot(times_ms, location_difference, linewidth=3, color='darkorange', label='Location Match - Non-Match')
axes[1,1].axvline(0, color='black', linestyle='--', alpha=0.7, label='Probe Onset')
axes[1,1].axhline(0, color='gray', linestyle='-', alpha=0.3)
axes[1,1].set_xlabel('Time (ms)')
axes[1,1].set_ylabel('Difference (T)')
axes[1,1].set_title('Location Difference Wave (Irrelevant Feature)')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'location_match_nonmatch_all_channels.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Plot 3: DIRECT COMPARISON - Relevant vs Irrelevant Features

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# All four conditions together
axes[0,0].plot(times_ms, voice_match_avg, label='Voice Match (Relevant)', linewidth=3, color='blue')
axes[0,0].plot(times_ms, voice_nonmatch_avg, label='Voice Non-Match (Relevant)', linewidth=3, color='red')
axes[0,0].plot(times_ms, location_match_avg, label='Location Match (Irrelevant)', linewidth=2, color='orange', linestyle='--')
axes[0,0].plot(times_ms, location_nonmatch_avg, label='Location Non-Match (Irrelevant)', linewidth=2, color='purple', linestyle='--')
axes[0,0].axvline(0, color='black', linestyle='--', alpha=0.7, label='Probe Onset')
axes[0,0].axhline(0, color='gray', linestyle='-', alpha=0.3)
axes[0,0].set_xlabel('Time (ms)')
axes[0,0].set_ylabel('Amplitude (T)')
axes[0,0].set_title('All Conditions - Direct Comparison')
axes[0,0].legend(fontsize=9)
axes[0,0].grid(True, alpha=0.3)

# Difference waves comparison
axes[0,1].plot(times_ms, voice_difference, label='Voice Effect (Relevant)', linewidth=3, color='darkblue')
axes[0,1].plot(times_ms, location_difference, label='Location Effect (Irrelevant)', linewidth=3, color='darkorange')
axes[0,1].axvline(0, color='black', linestyle='--', alpha=0.7, label='Probe Onset')
axes[0,1].axhline(0, color='gray', linestyle='-', alpha=0.3)
axes[0,1].set_xlabel('Time (ms)')
axes[0,1].set_ylabel('Difference (T)')
axes[0,1].set_title('Difference Waves: Relevant vs Irrelevant')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Effect magnitude comparison over time
voice_abs_diff = np.abs(voice_difference)
location_abs_diff = np.abs(location_difference)

axes[1,0].plot(times_ms, voice_abs_diff, label='Voice Effect Magnitude', linewidth=3, color='darkblue')
axes[1,0].plot(times_ms, location_abs_diff, label='Location Effect Magnitude', linewidth=3, color='darkorange')
axes[1,0].axvline(0, color='black', linestyle='--', alpha=0.7, label='Probe Onset')
axes[1,0].set_xlabel('Time (ms)')
axes[1,0].set_ylabel('|Difference| (T)')
axes[1,0].set_title('Effect Magnitude Over Time')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Simple effect size comparison
voice_max_effect = np.max(voice_abs_diff)
location_max_effect = np.max(location_abs_diff)
voice_max_time = times_ms[np.argmax(voice_abs_diff)]
location_max_time = times_ms[np.argmax(location_abs_diff)]

features = ['Voice\n(Relevant)', 'Location\n(Irrelevant)']
max_effects = [voice_max_effect, location_max_effect]
colors = ['darkblue', 'darkorange']

bars = axes[1,1].bar(features, max_effects, color=colors, alpha=0.7)
axes[1,1].set_ylabel('Max Effect Size (T)')
axes[1,1].set_title('Peak Effect Comparison')
axes[1,1].grid(True, alpha=0.3)

# Add value labels
for bar, effect, time in zip(bars, max_effects, [voice_max_time, location_max_time]):
    height = bar.get_height()
    axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{effect:.2e}\n@{time:.0f}ms', 
                   ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'relevant_vs_irrelevant_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Plot 4: Temporal ROI Focus - All Conditions

if len(available_temporal) > 0:
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # Get temporal channel indices
    temporal_picks = mne.pick_channels(voice_match_grand_avg.ch_names, available_temporal)
    
    # Row 1: Voice conditions - temporal channels
    voice_match_grand_avg.plot(picks=temporal_picks, axes=axes[0,0], spatial_colors=True, show=False)
    axes[0,0].set_title(f'VOICE MATCH - Temporal Channels\n({len(available_temporal)} channels)')
    axes[0,0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Probe Onset')
    axes[0,0].legend()
    
    voice_nonmatch_grand_avg.plot(picks=temporal_picks, axes=axes[0,1], spatial_colors=True, show=False)
    axes[0,1].set_title(f'VOICE NON-MATCH - Temporal Channels\n({len(available_temporal)} channels)')
    axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.7, label='Probe Onset')
    axes[0,1].legend()
    
    # Row 2: Location conditions - temporal channels
    location_match_grand_avg.plot(picks=temporal_picks, axes=axes[1,0], spatial_colors=True, show=False)
    axes[1,0].set_title(f'LOCATION MATCH - Temporal Channels\n({len(available_temporal)} channels)')
    axes[1,0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Probe Onset')
    axes[1,0].legend()
    
    location_nonmatch_grand_avg.plot(picks=temporal_picks, axes=axes[1,1], spatial_colors=True, show=False)
    axes[1,1].set_title(f'LOCATION NON-MATCH - Temporal Channels\n({len(available_temporal)} channels)')
    axes[1,1].axvline(0, color='red', linestyle='--', alpha=0.7, label='Probe Onset')
    axes[1,1].legend()
    
    # Row 3: Direct comparisons - averaged across temporal channels
    voice_match_temporal = np.mean(voice_match_grand_avg.data[temporal_picks], axis=0)
    voice_nonmatch_temporal = np.mean(voice_nonmatch_grand_avg.data[temporal_picks], axis=0)
    location_match_temporal = np.mean(location_match_grand_avg.data[temporal_picks], axis=0)
    location_nonmatch_temporal = np.mean(location_nonmatch_grand_avg.data[temporal_picks], axis=0)
    
    # Voice comparison
    axes[2,0].plot(times_ms, voice_match_temporal, label='Voice Match', linewidth=3, color='blue')
    axes[2,0].plot(times_ms, voice_nonmatch_temporal, label='Voice Non-Match', linewidth=3, color='red')
    axes[2,0].axvline(0, color='black', linestyle='--', alpha=0.7, label='Probe Onset')
    axes[2,0].axhline(0, color='gray', linestyle='-', alpha=0.3)
    axes[2,0].set_xlabel('Time (ms)')
    axes[2,0].set_ylabel('Amplitude (T)')
    axes[2,0].set_title('Voice Comparison - Temporal ROI')
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)
    
    # Location comparison
    axes[2,1].plot(times_ms, location_match_temporal, label='Location Match', linewidth=3, color='orange')
    axes[2,1].plot(times_ms, location_nonmatch_temporal, label='Location Non-Match', linewidth=3, color='purple')
    axes[2,1].axvline(0, color='black', linestyle='--', alpha=0.7, label='Probe Onset')
    axes[2,1].axhline(0, color='gray', linestyle='-', alpha=0.3)
    axes[2,1].set_xlabel('Time (ms)')
    axes[2,1].set_ylabel('Amplitude (T)')
    axes[2,1].set_title('Location Comparison - Temporal ROI')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + 'all_conditions_temporal_roi.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create difference waves for temporal ROI
    voice_difference_temporal = voice_match_temporal - voice_nonmatch_temporal
    location_difference_temporal = location_match_temporal - location_nonmatch_temporal
    
    # Plot difference waves comparison - temporal ROI
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(times_ms, voice_difference_temporal, linewidth=3, color='darkblue', label='Voice Effect (Relevant)')
    ax.plot(times_ms, location_difference_temporal, linewidth=3, color='darkorange', label='Location Effect (Irrelevant)')
    ax.axvline(0, color='black', linestyle='--', alpha=0.7, label='Probe Onset')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Difference Wave Amplitude (T)')
    ax.set_title('Difference Waves Comparison - Temporal ROI\n(Relevant vs Irrelevant Features)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + 'difference_waves_temporal_roi.png', dpi=300, bbox_inches='tight')
    plt.show()
    
else:
    print("No temporal channels available for ROI analysis")

#%% Plot 5: Time Window Analysis - All Conditions

# Define time windows of interest
time_windows = {
    'Early (100-200ms)': (0.1, 0.2),
    'Mid (200-400ms)': (0.2, 0.4),  
    'Late (400-600ms)': (0.4, 0.6),
    'Very Late (600-1000ms)': (0.6, 1.0)
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for i, (window_name, (tmin, tmax)) in enumerate(time_windows.items()):
    ax = axes[i//2, i%2]
    
    # Find time indices
    time_mask = (voice_match_grand_avg.times >= tmin) & (voice_match_grand_avg.times <= tmax)
    
    # Extract mean amplitudes in this window
    if len(available_temporal) > 0:
        # Use temporal ROI
        voice_match_window = np.mean(voice_match_grand_avg.data[temporal_picks][:, time_mask])
        voice_nonmatch_window = np.mean(voice_nonmatch_grand_avg.data[temporal_picks][:, time_mask])
        location_match_window = np.mean(location_match_grand_avg.data[temporal_picks][:, time_mask])
        location_nonmatch_window = np.mean(location_nonmatch_grand_avg.data[temporal_picks][:, time_mask])
    else:
        # Use all channels
        voice_match_window = np.mean(voice_match_grand_avg.data[:, time_mask])
        voice_nonmatch_window = np.mean(voice_nonmatch_grand_avg.data[:, time_mask])
        location_match_window = np.mean(location_match_grand_avg.data[:, time_mask])
        location_nonmatch_window = np.mean(location_nonmatch_grand_avg.data[:, time_mask])
    
    # Create grouped bar plot
    x_pos = np.arange(2)
    width = 0.35
    
    voice_amplitudes = [voice_match_window, voice_nonmatch_window]
    location_amplitudes = [location_match_window, location_nonmatch_window]
    
    bars1 = ax.bar(x_pos - width/2, voice_amplitudes, width, label='Voice (Relevant)', 
                   color=['blue', 'red'], alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, location_amplitudes, width, label='Location (Irrelevant)', 
                   color=['orange', 'purple'], alpha=0.7)
    
    ax.set_ylabel('Mean Amplitude (T)')
    ax.set_title(f'{window_name}\n({tmin*1000:.0f}-{tmax*1000:.0f}ms)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Match', 'Non-Match'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1e}', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8, rotation=90)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'all_conditions_time_windows.png', dpi=300, bbox_inches='tight')
plt.show()

# Create effect size comparison plot
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

window_names = list(time_windows.keys())
voice_effects = []
location_effects = []

for window_name, (tmin, tmax) in time_windows.items():
    time_mask = (voice_match_grand_avg.times >= tmin) & (voice_match_grand_avg.times <= tmax)
    
    if len(available_temporal) > 0:
        voice_match_amp = np.mean(voice_match_grand_avg.data[temporal_picks][:, time_mask])
        voice_nonmatch_amp = np.mean(voice_nonmatch_grand_avg.data[temporal_picks][:, time_mask])
        location_match_amp = np.mean(location_match_grand_avg.data[temporal_picks][:, time_mask])
        location_nonmatch_amp = np.mean(location_nonmatch_grand_avg.data[temporal_picks][:, time_mask])
    else:
        voice_match_amp = np.mean(voice_match_grand_avg.data[:, time_mask])
        voice_nonmatch_amp = np.mean(voice_nonmatch_grand_avg.data[:, time_mask])
        location_match_amp = np.mean(location_match_grand_avg.data[:, time_mask])
        location_nonmatch_amp = np.mean(location_nonmatch_grand_avg.data[:, time_mask])
    
    voice_effect = abs(voice_match_amp - voice_nonmatch_amp)
    location_effect = abs(location_match_amp - location_nonmatch_amp)
    
    voice_effects.append(voice_effect)
    location_effects.append(location_effect)

x_pos = np.arange(len(window_names))
width = 0.35

bars1 = ax.bar(x_pos - width/2, voice_effects, width, label='Voice Effect (Relevant)', 
               color='darkblue', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, location_effects, width, label='Location Effect (Irrelevant)', 
               color='darkorange', alpha=0.7)

ax.set_ylabel('Effect Size |Match - Non-Match| (T)')
ax.set_title('Effect Size Comparison Across Time Windows')
ax.set_xticks(x_pos)
ax.set_xticklabels(window_names, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1e}', ha='center', va='bottom',
                fontsize=9, rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'effect_size_comparison_time_windows.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Print Summary

print("\n" + "="*60)
print("BASIC ANALYSIS SUMMARY")
print("="*60)

print(f"Subjects analyzed: {len(all_subjects)}")
print(f"Temporal channels used: {len(available_temporal)}")

# Simple peak analysis for both features
if len(available_temporal) > 0:
    voice_match_data = np.mean(voice_match_grand_avg.data[temporal_picks], axis=0)
    voice_nonmatch_data = np.mean(voice_nonmatch_grand_avg.data[temporal_picks], axis=0)
    location_match_data = np.mean(location_match_grand_avg.data[temporal_picks], axis=0)
    location_nonmatch_data = np.mean(location_nonmatch_grand_avg.data[temporal_picks], axis=0)
else:
    voice_match_data = np.mean(voice_match_grand_avg.data, axis=0) 
    voice_nonmatch_data = np.mean(voice_nonmatch_grand_avg.data, axis=0)
    location_match_data = np.mean(location_match_grand_avg.data, axis=0)
    location_nonmatch_data = np.mean(location_nonmatch_grand_avg.data, axis=0)

# VOICE (Relevant Feature) Analysis
print(f"\n" + "="*40)
print("VOICE MATCHING (RELEVANT FEATURE)")
print("="*40)

voice_match_peak_idx = np.argmax(np.abs(voice_match_data))
voice_nonmatch_peak_idx = np.argmax(np.abs(voice_nonmatch_data))

voice_match_peak_time = voice_match_grand_avg.times[voice_match_peak_idx]
voice_nonmatch_peak_time = voice_match_grand_avg.times[voice_nonmatch_peak_idx]

voice_match_peak_amp = voice_match_data[voice_match_peak_idx]
voice_nonmatch_peak_amp = voice_nonmatch_data[voice_nonmatch_peak_idx]

print(f"Voice Match peak: {voice_match_peak_amp:.2e} T at {voice_match_peak_time*1000:.1f}ms")
print(f"Voice Non-match peak: {voice_nonmatch_peak_amp:.2e} T at {voice_nonmatch_peak_time*1000:.1f}ms")

# Voice difference analysis
voice_diff_data = voice_match_data - voice_nonmatch_data
voice_max_diff_idx = np.argmax(np.abs(voice_diff_data))
voice_max_diff_time = voice_match_grand_avg.times[voice_max_diff_idx]
voice_max_diff_amp = voice_diff_data[voice_max_diff_idx]

print(f"Voice max difference: {voice_max_diff_amp:.2e} T at {voice_max_diff_time*1000:.1f}ms")
print(f"Voice effect direction: {'Match > Non-Match' if voice_max_diff_amp > 0 else 'Non-Match > Match'}")

# LOCATION (Irrelevant Feature) Analysis
print(f"\n" + "="*40)
print("LOCATION MATCHING (IRRELEVANT FEATURE)")
print("="*40)

location_match_peak_idx = np.argmax(np.abs(location_match_data))
location_nonmatch_peak_idx = np.argmax(np.abs(location_nonmatch_data))

location_match_peak_time = location_match_grand_avg.times[location_match_peak_idx]
location_nonmatch_peak_time = location_match_grand_avg.times[location_nonmatch_peak_idx]

location_match_peak_amp = location_match_data[location_match_peak_idx]
location_nonmatch_peak_amp = location_nonmatch_data[location_nonmatch_peak_idx]

print(f"Location Match peak: {location_match_peak_amp:.2e} T at {location_match_peak_time*1000:.1f}ms")
print(f"Location Non-match peak: {location_nonmatch_peak_amp:.2e} T at {location_nonmatch_peak_time*1000:.1f}ms")

# Location difference analysis
location_diff_data = location_match_data - location_nonmatch_data
location_max_diff_idx = np.argmax(np.abs(location_diff_data))
location_max_diff_time = location_match_grand_avg.times[location_max_diff_idx]
location_max_diff_amp = location_diff_data[location_max_diff_idx]

print(f"Location max difference: {location_max_diff_amp:.2e} T at {location_max_diff_time*1000:.1f}ms")
print(f"Location effect direction: {'Match > Non-Match' if location_max_diff_amp > 0 else 'Non-Match > Match'}")

# COMPARISON
print(f"\n" + "="*40)
print("RELEVANT vs IRRELEVANT COMPARISON")
print("="*40)

voice_effect_magnitude = abs(voice_max_diff_amp)
location_effect_magnitude = abs(location_max_diff_amp)
effect_ratio = voice_effect_magnitude / location_effect_magnitude if location_effect_magnitude > 0 else float('inf')

print(f"Voice effect magnitude: {voice_effect_magnitude:.2e} T")
print(f"Location effect magnitude: {location_effect_magnitude:.2e} T") 
print(f"Voice/Location effect ratio: {effect_ratio:.2f}")

if effect_ratio > 2:
    print("✓ Voice effect is much stronger than location effect")
elif effect_ratio > 1.5:
    print("✓ Voice effect is stronger than location effect")  
elif effect_ratio > 0.67:
    print("~ Voice and location effects are similar")
else:
    print("⚠ Location effect is stronger than voice effect")

# Save all grand averages
voice_match_grand_avg.save(OUTPUT_PATH + 'voice_match_grand_average-ave.fif', overwrite=True)
voice_nonmatch_grand_avg.save(OUTPUT_PATH + 'voice_nonmatch_grand_average-ave.fif', overwrite=True)
location_match_grand_avg.save(OUTPUT_PATH + 'location_match_grand_average-ave.fif', overwrite=True)
location_nonmatch_grand_avg.save(OUTPUT_PATH + 'location_nonmatch_grand_average-ave.fif', overwrite=True)

print(f"\nFiles saved to: {OUTPUT_PATH}")
print("✓ All condition grand averages saved")
print("✓ Basic analysis complete!")

print(f"\n" + "="*60)
print("INTERPRETATION SUMMARY")
print("="*60)
print("1. Look at the difference wave plots:")
print("   - Is the voice difference wave larger than location?")  
print("   - Are there clear peaks in voice but not location?")
print("2. Check timing:")
print(f"   - Voice effect peaks at {voice_max_diff_time*1000:.0f}ms")
print(f"   - Location effect peaks at {location_max_diff_time*1000:.0f}ms")
print("3. Effect magnitude comparison shows voice vs location processing")
print("4. If voice >> location: participants focus on relevant feature ✓")
print("5. If voice ≈ location: both features processed equally")  
print("6. If location >> voice: unexpected pattern!")