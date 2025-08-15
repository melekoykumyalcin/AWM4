#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Probe Feature Relevance Analysis
From raw epochs to cluster-based permutation testing

Analysis of relevant (voice) vs irrelevant (location) feature processing
during probe presentation in auditory working memory task.
"""

#%% Setup ####################################################################

import os
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_test, permutation_cluster_1samp_test
from itertools import combinations
import seaborn as sns
import json
from scipy import stats

# Set paths
HOME_DIR = '/media/headmodel/Elements/AWM4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
PROBE_EPOCHS_PATH = HOME_DIR + '/AWM4_data/processed/ProbeEpochs/'
OUTPUT_PATH = HOME_DIR + '/AWM4_data/processed/CompleteProbeAnalysis/'

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(HOME_DIR)

# Load meta information
metaInfo = pd.read_excel(META_FILE)
Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])

# Define the four probe conditions
conditions = {
    'Probe/SpNM/LoNM': 150,  # Voice Non-match, Location Non-match
    'Probe/SpNM/LoM': 151,   # Voice Non-match, Location Match  
    'Probe/SpM/LoNM': 160,   # Voice Match, Location Non-match
    'Probe/SpM/LoM': 161     # Voice Match, Location Match
}

condition_labels = {
    'Probe/SpNM/LoNM': 'Voice NoMatch + Location NoMatch',
    'Probe/SpNM/LoM': 'Voice NoMatch + Location Match',
    'Probe/SpM/LoNM': 'Voice Match + Location NoMatch',
    'Probe/SpM/LoM': 'Voice Match + Location Match'
}

# Define channel regions for ROI analysis
regions = {
    'frontal_left': [
        'MLF11-3609', 'MLF12-3609', 'MLF13-3609', 'MLF14-3609',
        'MLF21-3609', 'MLF22-3609', 'MLF23-3609', 'MLF24-3609', 'MLF25-3609',
        'MLF31-3609', 'MLF32-3609', 'MLF33-3609', 'MLF34-3609', 'MLF35-3609',
        'MLF41-3609', 'MLF42-3609', 'MLF43-3609', 'MLF44-3609', 'MLF45-3609', 'MLF46-3609',
        'MLF51-3609', 'MLF52-3609', 'MLF53-3609', 'MLF54-3609', 'MLF55-3609', 'MLF56-3609',
        'MLF61-3609', 'MLF62-3609', 'MLF63-3609', 'MLF64-3609', 'MLF65-3609', 'MLF66-3609', 'MLF67-3609'
    ],
    'frontal_right': [
        'MRF11-3609', 'MRF12-3609', 'MRF13-3609', 'MRF14-3609',
        'MRF21-3609', 'MRF22-3609', 'MRF23-3609', 'MRF24-3609', 'MRF25-3609',
        'MRF31-3609', 'MRF32-3609', 'MRF33-3609', 'MRF34-3609', 'MRF35-3609',
        'MRF41-3609', 'MRF42-3609', 'MRF43-3609', 'MRF44-3609', 'MRF45-3609', 'MRF46-3609',
        'MRF51-3609', 'MRF52-3609', 'MRF53-3609', 'MRF54-3609', 'MRF55-3609', 'MRF56-3609',
        'MRF61-3609', 'MRF62-3609', 'MRF63-3609', 'MRF64-3609', 'MRF65-3609', 'MRF66-3609', 'MRF67-3609'
    ],
    'temporal_left': [
        'MLT11-3609', 'MLT12-3609', 'MLT13-3609', 'MLT14-3609', 'MLT15-3609', 'MLT16-3609',
        'MLT21-3609', 'MLT22-3609', 'MLT23-3609', 'MLT24-3609', 'MLT25-3609', 'MLT26-3609', 'MLT27-3609',
        'MLT31-3609', 'MLT32-3609', 'MLT33-3609', 'MLT34-3609', 'MLT35-3609', 'MLT36-3609', 'MLT37-3609',
        'MLT41-3609', 'MLT42-3609', 'MLT43-3609', 'MLT44-3609', 'MLT45-3609', 'MLT46-3609', 'MLT47-3609',
        'MLT51-3609', 'MLT52-3609', 'MLT53-3609', 'MLT54-3609', 'MLT55-3609', 'MLT56-3609', 'MLT57-3609'
    ],
    'temporal_right': [
        'MRT11-3609', 'MRT12-3609', 'MRT13-3609', 'MRT14-3609', 'MRT15-3609', 'MRT16-3609',
        'MRT21-3609', 'MRT22-3609', 'MRT23-3609', 'MRT24-3609', 'MRT25-3609', 'MRT26-3609', 'MRT27-3609',
        'MRT31-3609', 'MRT32-3609', 'MRT33-3609', 'MRT34-3609', 'MRT35-3609', 'MRT36-3609', 'MRT37-3609',
        'MRT41-3609', 'MRT42-3609', 'MRT43-3609', 'MRT44-3609', 'MRT45-3609', 'MRT46-3609', 'MRT47-3609',
        'MRT51-3609', 'MRT52-3609', 'MRT53-3609', 'MRT54-3609', 'MRT55-3609', 'MRT56-3609', 'MRT57-3609'
    ],
    'central_left': [
        'MLC11-3609', 'MLC12-3609', 'MLC13-3609', 'MLC14-3609', 'MLC15-3609', 'MLC16-3609', 'MLC17-3609',
        'MLC21-3609', 'MLC22-3609', 'MLC23-3609', 'MLC24-3609', 'MLC25-3609',
        'MLC31-3609', 'MLC32-3609', 'MLC41-3609', 'MLC42-3609',
        'MLC51-3609', 'MLC52-3609', 'MLC53-3609', 'MLC54-3609', 'MLC55-3609',
        'MLC61-3609', 'MLC62-3609', 'MLC63-3609'
    ],
    'central_right': [
        'MRC11-3609', 'MRC12-3609', 'MRC14-3609', 'MRC15-3609', 'MRC16-3609', 'MRC17-3609',
        'MRC21-3609', 'MRC22-3609', 'MRC23-3609', 'MRC24-3609', 'MRC25-3609',
        'MRC31-3609', 'MRC32-3609', 'MRC41-3609', 'MRC42-3609',
        'MRC51-3609', 'MRC52-3609', 'MRC53-3609', 'MRC54-3609', 'MRC55-3609',
        'MRC61-3609', 'MRC62-3609', 'MRC63-3609'
    ],
    'parietal_left': [
        'MLP11-3609', 'MLP12-3609', 'MLP21-3609', 'MLP22-3609', 'MLP23-3609',
        'MLP31-3609', 'MLP32-3609', 'MLP33-3609', 'MLP34-3609', 'MLP35-3609',
        'MLP41-3609', 'MLP42-3609', 'MLP43-3609', 'MLP44-3609', 'MLP45-3609',
        'MLP51-3609', 'MLP52-3609', 'MLP53-3609', 'MLP54-3609', 'MLP55-3609', 'MLP56-3609', 'MLP57-3609'
    ],
    'parietal_right': [
        'MRP11-3609', 'MRP12-3609', 'MRP21-3609', 'MRP22-3609', 'MRP23-3609',
        'MRP31-3609', 'MRP32-3609', 'MRP33-3609', 'MRP34-3609', 'MRP35-3609',
        'MRP41-3609', 'MRP42-3609', 'MRP43-3609', 'MRP44-3609', 'MRP45-3609',
        'MRP51-3609', 'MRP52-3609', 'MRP53-3609', 'MRP54-3609', 'MRP55-3609', 'MRP56-3609', 'MRP57-3609'
    ]
}

# Combine temporal channels for auditory analysis (main ROI)
temporal_channels = regions['temporal_left'] + regions['temporal_right']
available_temporal = []  # Will be populated after finding common channels

print("="*80)
print("COMPLETE PROBE FEATURE RELEVANCE ANALYSIS")
print("="*80)
print(f"Subjects to process: {len(Subs)}")
print(f"Conditions: {len(conditions)}")
print(f"Main ROI: {len(temporal_channels)} temporal channels (will check availability)")
print(f"Output directory: {OUTPUT_PATH}")

#%% Step 1: Load Probe Epochs and Compute Evoked Responses

print("\n" + "="*60)
print("STEP 1: LOADING EPOCHS AND COMPUTING EVOKED RESPONSES")
print("="*60)

# Initialize storage for evoked responses
all_condition_evokeds = {cond: [] for cond in conditions.keys()}
all_subjects = []

for subject in Subs:
    try:
        print(f"\nProcessing subject {subject}...")
        
        # Try both with and without '-dropped' suffix
        epoch_files = [
            PROBE_EPOCHS_PATH + f'ProbeEpochs_VP{subject}-epo-dropped.fif',
            PROBE_EPOCHS_PATH + f'ProbeEpochs_VP{subject}-epo.fif'
        ]
        
        probe_epochs = None
        for epoch_file in epoch_files:
            if os.path.exists(epoch_file):
                probe_epochs = mne.read_epochs(epoch_file, verbose=False)
                print(f"  Loaded: {os.path.basename(epoch_file)}")
                break
        
        if probe_epochs is None:
            print(f"  No epoch file found for subject {subject}")
            continue
        
        # Check if all conditions are present
        missing_conditions = [cond for cond in conditions.keys() if cond not in probe_epochs.event_id]
        if missing_conditions:
            print(f"  Missing conditions: {missing_conditions}, skipping...")
            continue
        
        # Print trial counts
        print(f"  Trial counts:")
        trial_counts = []
        for cond in conditions.keys():
            count = len(probe_epochs[cond])
            trial_counts.append(count)
            print(f"    {cond}: {count}")
        
        # Check minimum trial count
        min_trials = min(trial_counts)
        if min_trials < 15:  # Minimum threshold for reliable averaging
            print(f"  Too few trials (min: {min_trials}) for subject {subject}, skipping...")
            continue
        
        # Crop to start at -100ms (as in your code)
        probe_epochs = probe_epochs.copy().crop(tmin=-0.1)
        
        # Compute evoked responses for each condition
        for cond in conditions.keys():
            evoked = probe_epochs[cond].average()
            all_condition_evokeds[cond].append(evoked.copy())
        
        all_subjects.append(subject)
        print(f"  ✓ Processed subject {subject} - Total: {sum(trial_counts)} trials")
        
    except Exception as e:
        print(f"  Error processing subject {subject}: {str(e)}")
        continue

print(f"\nSuccessfully processed {len(all_subjects)} subjects")
print(f"Subjects included: {all_subjects}")

if len(all_subjects) == 0:
    print("ERROR: No subjects processed successfully!")
    exit()

#%% Step 2: Basic Visualization of Individual Conditions

print("\n" + "="*60)
print("STEP 2: BASIC VISUALIZATION OF CONDITIONS")
print("="*60)

# Compute grand averages for visualization
grand_averages = {}
for cond in conditions.keys():
    grand_averages[cond] = mne.grand_average(all_condition_evokeds[cond])
    print(f"Grand average for {cond}: {len(all_condition_evokeds[cond])} subjects")

# Create comparison plots using MNE's plot_compare_evokeds
print("Creating condition comparison plots...")

# Plot all conditions using your approach
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Use MNE's compare_evokeds for nice formatting
evokeds_dict = {
    'SpNM_LoNM': grand_averages['Probe/SpNM/LoNM'],
    'SpNM_LoM': grand_averages['Probe/SpNM/LoM'],
    'SpM_LoNM': grand_averages['Probe/SpM/LoNM'],
    'SpM_LoM': grand_averages['Probe/SpM/LoM']
}

# Plot with temporal channels focus
try:
    # This uses the MNE plot_compare_evokeds function
    mne.viz.plot_compare_evokeds(
        evokeds_dict,
        picks=available_temporal,  # Use available temporal channels
        colors={'SpNM_LoNM': 'blue', 'SpNM_LoM': 'orange', 'SpM_LoNM': 'green', 'SpM_LoM': 'red'},
        time_unit="ms",
        show=False
    )
    plt.suptitle(f'All Conditions - Temporal Channels (N={len(all_subjects)})')
    plt.savefig(OUTPUT_PATH + 'individual_conditions_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Note: MNE comparison plot failed ({e}), will create custom plots")

#%% Step 3: Prepare Data for Statistical Testing

print("\n" + "="*60)
print("STEP 3: PREPARING DATA FOR STATISTICAL TESTING")
print("="*60)

# First, identify common channels across all subjects and conditions
print("Identifying common channels across all subjects...")

all_ch_names_sets = []
for cond in conditions.keys():
    for evoked in all_condition_evokeds[cond]:
        all_ch_names_sets.append(set(evoked.ch_names))

# Find intersection of all channel sets (channels present in ALL subjects)
common_channels = list(set.intersection(*all_ch_names_sets))
common_channels.sort()  # Sort for consistent ordering

print(f"Common channels across all subjects: {len(common_channels)}")

# Check which channels were dropped
first_evoked = all_condition_evokeds['Probe/SpNM/LoNM'][0]
missing_channels = set(first_evoked.ch_names) - set(common_channels)
if missing_channels:
    print(f"Channels not present in all subjects: {sorted(missing_channels)}")

# Update temporal channels to only include those in common_channels
available_temporal = [ch for ch in temporal_channels if ch in common_channels]
print(f"Temporal channels available: {len(available_temporal)} out of {len(temporal_channels)}")

if len(available_temporal) < len(temporal_channels):
    missing_temporal = set(temporal_channels) - set(available_temporal)
    print(f"Missing temporal channels: {sorted(missing_temporal)}")

# Convert evoked objects to data arrays, picking only common channels
condition_data = {}
times = all_condition_evokeds['Probe/SpNM/LoNM'][0].times  # Get times from first subject

print("\nConverting evoked objects to arrays...")

for cond in conditions.keys():
    # Pick common channels and convert to array
    evoked_data_list = []
    for evoked in all_condition_evokeds[cond]:
        # Pick common channels
        evoked_common = evoked.copy().pick_channels(common_channels)
        evoked_data_list.append(evoked_common.data)
    
    # Now stack into array (n_subjects, n_channels, n_times)
    condition_data[cond] = np.stack(evoked_data_list, axis=0)
    print(f"{cond}: {condition_data[cond].shape}")

print(f"Time range: {times[0]:.3f} to {times[-1]:.3f} seconds")
print(f"Number of subjects: {len(all_subjects)}")
print(f"Number of magnetometers: {condition_data[list(conditions.keys())[0]].shape[1]}")

# Get temporal channel indices for ROI analysis (from common channels)
temporal_picks = mne.pick_channels(common_channels, available_temporal, ordered=True)
print(f"Temporal ROI: {len(temporal_picks)} channels")

#%% Step 4: Location Effect Analysis (Main Question - Irrelevant Feature)

print("\n" + "="*60)
print("STEP 4: LOCATION EFFECT ANALYSIS (IRRELEVANT FEATURE)")
print("="*60)

print("Creating location contrast (averaging across voice conditions)...")

# Create location match vs non-match conditions (averaging across voice)
location_match_data = []
location_nonmatch_data = []

for subj_idx in range(len(all_subjects)):
    # Location Match: average of SpNM/LoM and SpM/LoM
    loc_match = (condition_data['Probe/SpNM/LoM'][subj_idx] + 
                 condition_data['Probe/SpM/LoM'][subj_idx]) / 2
    
    # Location Non-match: average of SpNM/LoNM and SpM/LoNM  
    loc_nonmatch = (condition_data['Probe/SpNM/LoNM'][subj_idx] + 
                    condition_data['Probe/SpM/LoNM'][subj_idx]) / 2
    
    location_match_data.append(loc_match)
    location_nonmatch_data.append(loc_nonmatch)

location_match_data = np.array(location_match_data)
location_nonmatch_data = np.array(location_nonmatch_data)

print(f"Location Match data shape: {location_match_data.shape}")
print(f"Location Non-match data shape: {location_nonmatch_data.shape}")

# Statistical parameters
threshold = None  # Let MNE choose threshold
n_permutations = 1000
alpha = 0.05

# Test location effect with cluster-based permutation
print(f"Running cluster-based permutation test...")
print(f"Parameters: {n_permutations} permutations, alpha = {alpha}")

location_test_results = permutation_cluster_test(
    [location_match_data, location_nonmatch_data],
    threshold=threshold,
    n_permutations=n_permutations,
    tail=0,  # Two-tailed test
    out_type='mask',
    verbose=True
)

# Handle different return formats (MNE versions may vary)
print(f"Permutation test returned {len(location_test_results)} values")
print(f"Types of returned values: {[type(x) for x in location_test_results]}")

# The correct order should be: T_obs, clusters, cluster_pv, H0
T_obs, location_clusters, location_cluster_pv, location_H0 = location_test_results

print(f"T_obs type: {type(T_obs)}, shape: {T_obs.shape}")
print(f"Clusters type: {type(location_clusters)}, length: {len(location_clusters)}")
print(f"Cluster_pv type: {type(location_cluster_pv)}")

if hasattr(location_cluster_pv, 'shape'):
    print(f"Cluster_pv shape: {location_cluster_pv.shape}")
else:
    print(f"Cluster_pv length: {len(location_cluster_pv)}")

print(f"H0 type: {type(location_H0)}")

# Check the actual cluster p-values
if hasattr(location_cluster_pv, '__len__') and len(location_cluster_pv) > 0:
    print(f"First few cluster p-values: {location_cluster_pv[:5]}")
    print(f"P-value range: {np.min(location_cluster_pv):.6f} to {np.max(location_cluster_pv):.6f}")
else:
    print("No cluster p-values found")

print(f"\nLOCATION EFFECT RESULTS:")
print(f"Total clusters found: {len(location_clusters)}")
print(f"Cluster p-values type: {type(location_cluster_pv)}")
print(f"Number of p-values: {len(location_cluster_pv)}")

# Now check if we have actual p-values
if len(location_cluster_pv) > 0:
    first_pval = location_cluster_pv[0]
    print(f"First p-value type: {type(first_pval)}")
    if isinstance(first_pval, (int, float, np.number)):
        print(f"✓ P-values look correct: {location_cluster_pv[:3]}")
        location_pvals = np.array(location_cluster_pv)
    else:
        print(f"✗ P-values still wrong type: {type(first_pval)}")
        print("Cannot proceed with statistical analysis - p-values not found")
        location_sig_clusters = []
        location_pvals = np.array([])

if len(location_pvals) > 0:
    print(f"P-values range: {np.min(location_pvals):.6f} to {np.max(location_pvals):.6f}")
    
    # Find significant clusters
    location_sig_clusters = []
    for i, p in enumerate(location_pvals):
        if i < len(location_clusters):
            p_val = float(p)
            if p_val < alpha:
                location_sig_clusters.append((i, p_val))

print(f"Significant clusters (p < {alpha}): {len(location_sig_clusters)}")

for cluster_idx, p_val in location_sig_clusters:
    if cluster_idx < len(location_clusters):
        cluster = location_clusters[cluster_idx]
        print(f"  Cluster {cluster_idx + 1}: p = {p_val:.4f}")
        print(f"    Cluster shape: {cluster.shape}")
        
        # Handle different cluster formats
        try:
            if cluster.ndim == 2:
                # Standard format: (n_channels, n_times)
                cluster_times = times[np.any(cluster, axis=0)]
                n_channels = np.sum(np.any(cluster, axis=1))
            elif cluster.ndim == 1:
                # 1D format: might be flattened or time-only
                print(f"    Warning: 1D cluster format")
                if len(cluster) == len(times):
                    # Time-only cluster
                    cluster_times = times[cluster]
                    n_channels = "N/A (time-only cluster)"
                else:
                    print(f"    Unknown 1D cluster format with length {len(cluster)}")
                    continue
            else:
                print(f"    Unknown cluster format with {cluster.ndim} dimensions")
                continue
                
            if len(cluster_times) > 0:
                print(f"    Time range: {cluster_times[0]:.3f} - {cluster_times[-1]:.3f}s")
                print(f"    Channels involved: {n_channels}")
            else:
                print(f"    No significant time points found")
        except Exception as e:
            print(f"    Error processing cluster: {e}")

#%% Step 5: Voice Effect Analysis (Relevant Feature)

print("\n" + "="*60)
print("STEP 5: VOICE EFFECT ANALYSIS (RELEVANT FEATURE)")
print("="*60)

print("Creating voice contrast (averaging across location conditions)...")

voice_match_data = []
voice_nonmatch_data = []

for subj_idx in range(len(all_subjects)):
    # Voice Match: average of SpM/LoNM and SpM/LoM
    voice_match = (condition_data['Probe/SpM/LoNM'][subj_idx] + 
                   condition_data['Probe/SpM/LoM'][subj_idx]) / 2
    
    # Voice Non-match: average of SpNM/LoNM and SpNM/LoM
    voice_nonmatch = (condition_data['Probe/SpNM/LoNM'][subj_idx] + 
                      condition_data['Probe/SpNM/LoM'][subj_idx]) / 2
    
    voice_match_data.append(voice_match)
    voice_nonmatch_data.append(voice_nonmatch)

voice_match_data = np.array(voice_match_data)
voice_nonmatch_data = np.array(voice_nonmatch_data)

print(f"Voice Match data shape: {voice_match_data.shape}")
print(f"Voice Non-match data shape: {voice_nonmatch_data.shape}")

# Test voice effect
print("Running cluster-based permutation test for voice effect...")

voice_test_results = permutation_cluster_test(
    [voice_match_data, voice_nonmatch_data],
    threshold=threshold,
    n_permutations=n_permutations,
    tail=0,
    out_type='mask',
    verbose=True
)

# Handle different return formats
print(f"Voice permutation test returned {len(voice_test_results)} values")
print(f"Types of returned values: {[type(x) for x in voice_test_results]}")

# The correct order should be: T_obs, clusters, cluster_pv, H0
T_obs_voice, voice_clusters, voice_cluster_pv, voice_H0 = voice_test_results

print(f"Voice T_obs type: {type(T_obs_voice)}, shape: {T_obs_voice.shape}")
print(f"Voice clusters type: {type(voice_clusters)}, length: {len(voice_clusters)}")
print(f"Voice cluster_pv type: {type(voice_cluster_pv)}")

if hasattr(voice_cluster_pv, 'shape'):
    print(f"Voice cluster_pv shape: {voice_cluster_pv.shape}")
else:
    print(f"Voice cluster_pv length: {len(voice_cluster_pv)}")

# Check the actual cluster p-values
if hasattr(voice_cluster_pv, '__len__') and len(voice_cluster_pv) > 0:
    print(f"First few voice cluster p-values: {voice_cluster_pv[:5]}")
    print(f"Voice p-value range: {np.min(voice_cluster_pv):.6f} to {np.max(voice_cluster_pv):.6f}")
else:
    print("No voice cluster p-values found")

voice_sig_clusters = [(i, float(p)) for i, p in enumerate(voice_cluster_pv) if float(p) < alpha]

print(f"\nVOICE EFFECT RESULTS:")
print(f"Total clusters found: {len(voice_clusters)}")
print(f"Significant clusters (p < {alpha}): {len(voice_sig_clusters)}")

for cluster_idx, p_val in voice_sig_clusters:
    if cluster_idx < len(voice_clusters):
        cluster = voice_clusters[cluster_idx]
        print(f"  Cluster {cluster_idx + 1}: p = {p_val:.4f}")
        
        # Handle different cluster formats
        try:
            if cluster.ndim == 2:
                cluster_times = times[np.any(cluster, axis=0)]
                n_channels = np.sum(np.any(cluster, axis=1))
            elif cluster.ndim == 1:
                if len(cluster) == len(times):
                    cluster_times = times[cluster]
                    n_channels = "N/A (time-only cluster)"
                else:
                    print(f"    Unknown 1D cluster format")
                    continue
            else:
                print(f"    Unknown cluster format")
                continue
                
            if len(cluster_times) > 0:
                print(f"    Time range: {cluster_times[0]:.3f} - {cluster_times[-1]:.3f}s")
                print(f"    Channels involved: {n_channels}")
        except Exception as e:
            print(f"    Error processing cluster: {e}")

#%% Step 6: Direct Comparison - Voice vs Location Effect Magnitude

print("\n" + "="*60)
print("STEP 6: COMPARING VOICE VS LOCATION EFFECT MAGNITUDES")
print("="*60)

# Create difference waves for each subject
voice_effect_data = voice_match_data - voice_nonmatch_data
location_effect_data = location_match_data - location_nonmatch_data

print("Testing if Voice effect > Location effect...")

# Test if voice effect is larger than location effect (one-tailed)
effect_difference = voice_effect_data - location_effect_data

diff_clusters, diff_cluster_pv, diff_H0 = permutation_cluster_1samp_test(
    effect_difference,
    threshold=threshold,
    n_permutations=n_permutations,
    tail=1,  # One-tailed: voice > location
    out_type='mask',
    verbose=True
)

diff_sig_clusters = [(i, float(p)) for i, p in enumerate(diff_cluster_pv) if float(p) < alpha]

print(f"\nVOICE > LOCATION COMPARISON:")
print(f"Total clusters found: {len(diff_clusters)}")
print(f"Significant clusters where Voice > Location: {len(diff_sig_clusters)}")

for cluster_idx, p_val in diff_sig_clusters:
    if cluster_idx < len(diff_clusters):
        cluster = diff_clusters[cluster_idx]
        print(f"  Cluster {cluster_idx + 1}: p = {p_val:.4f}")
        
        # Handle different cluster formats
        try:
            if cluster.ndim == 2:
                cluster_times = times[np.any(cluster, axis=0)]
                n_channels = np.sum(np.any(cluster, axis=1))
            elif cluster.ndim == 1:
                if len(cluster) == len(times):
                    cluster_times = times[cluster]
                    n_channels = "N/A (time-only cluster)"
                else:
                    print(f"    Unknown 1D cluster format")
                    continue
            else:
                print(f"    Unknown cluster format")
                continue
                
            if len(cluster_times) > 0:
                print(f"    Time range: {cluster_times[0]:.3f} - {cluster_times[-1]:.3f}s")
                print(f"    Channels involved: {n_channels}")
        except Exception as e:
            print(f"    Error processing cluster: {e}")

#%% Step 7: Pairwise Comparisons Between All Conditions

print("\n" + "="*60)
print("STEP 7: PAIRWISE COMPARISONS BETWEEN ALL CONDITIONS")
print("="*60)

# Define condition pairs and their meaningful names
condition_pairs = [
    ('Probe/SpNM/LoNM', 'Probe/SpNM/LoM'),   # Location effect within Voice Non-match
    ('Probe/SpM/LoNM', 'Probe/SpM/LoM'),     # Location effect within Voice Match
    ('Probe/SpNM/LoNM', 'Probe/SpM/LoNM'),   # Voice effect within Location Non-match
    ('Probe/SpNM/LoM', 'Probe/SpM/LoM'),     # Voice effect within Location Match
    ('Probe/SpNM/LoNM', 'Probe/SpM/LoM'),    # Main diagonal: Both different
    ('Probe/SpNM/LoM', 'Probe/SpM/LoNM')     # Anti-diagonal: Cross comparison
]

comparison_names = [
    'Location effect (Voice Non-match)',
    'Location effect (Voice Match)', 
    'Voice effect (Location Non-match)',
    'Voice effect (Location Match)',
    'SpNM/LoNM vs SpM/LoM',
    'SpNM/LoM vs SpM/LoNM'
]

# Storage for results
pairwise_results = {}

print(f"Running {len(condition_pairs)} pairwise comparisons...")

for i, ((cond1, cond2), comp_name) in enumerate(zip(condition_pairs, comparison_names)):
    print(f"\n{i+1}. {comp_name}")
    print(f"   Comparing: {cond1} vs {cond2}")
    
    # Run cluster-based permutation test
    test_results = permutation_cluster_test(
        [condition_data[cond1], condition_data[cond2]],
        threshold=threshold,
        n_permutations=n_permutations,
        tail=0,  # Two-tailed test
        out_type='mask',
        verbose=False
    )
    
    # Use correct unpacking order: T_obs, clusters, cluster_pv, H0
    T_obs_pair, clusters, cluster_pv, H0_pair = test_results
    
    # Find significant clusters
    sig_clusters = []
    if hasattr(cluster_pv, '__len__') and len(cluster_pv) > 0:
        for i, p in enumerate(cluster_pv):
            if i < len(clusters):
                p_val = float(p)
                if p_val < alpha:
                    sig_clusters.append((i, p_val))
    
    pairwise_results[comp_name] = {
        'clusters': clusters,
        'cluster_pv': cluster_pv,
        'sig_clusters': sig_clusters,
        'n_sig': len(sig_clusters)
    }
    
    print(f"   Found {len(sig_clusters)} significant clusters")
    for cluster_idx, p_val in sig_clusters:
        print(f"     Cluster {cluster_idx + 1}: p = {p_val:.4f}")

#%% Step 8: Comprehensive Visualization

print("\n" + "="*60)
print("STEP 8: CREATING COMPREHENSIVE VISUALIZATION")
print("="*60)

fig, axes = plt.subplots(3, 2, figsize=(18, 20))

# Convert times to milliseconds for plotting
times_ms = times * 1000

# Get temporal channel data for cleaner plotting
def get_temporal_data(data_array):
    """Extract temporal channel data and average across channels"""
    return np.mean(data_array[:, temporal_picks, :], axis=1)  # Average across temporal channels

# 1. All individual conditions
colors = ['blue', 'orange', 'green', 'red']
labels = ['Voice NM + Location NM', 'Voice NM + Location M', 'Voice M + Location NM', 'Voice M + Location M']

for i, (cond, label, color) in enumerate(zip(conditions.keys(), labels, colors)):
    # Get grand average for temporal channels
    temp_data = np.mean(grand_averages[cond].data[temporal_picks], axis=0)
    axes[0,0].plot(times_ms, temp_data, label=label, linewidth=2, color=color)

axes[0,0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Probe Onset')
axes[0,0].axhline(0, color='black', linestyle='-', alpha=0.3)
axes[0,0].set_xlabel('Time (ms)')
axes[0,0].set_ylabel('Amplitude (T)')
axes[0,0].set_title(f'All Conditions - Temporal Channels (N={len(all_subjects)})')
axes[0,0].legend(fontsize=9)
axes[0,0].grid(True, alpha=0.3)

# 2. Location effect with significant clusters
location_diff_temporal = np.mean(get_temporal_data(location_match_data - location_nonmatch_data), axis=0)

axes[0,1].plot(times_ms, location_diff_temporal, linewidth=3, color='orange', label='Location Effect')
axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.7)
axes[0,1].axhline(0, color='black', linestyle='-', alpha=0.3)

# Highlight significant clusters (if any)
if len(location_sig_clusters) > 0:
    for cluster_idx, p_val in location_sig_clusters[:5]:  # Limit to first 5 for readability
        if cluster_idx < len(location_clusters):
            cluster = location_clusters[cluster_idx]
            try:
                # Handle different cluster formats
                if cluster.ndim == 2:
                    # Check if temporal channels are involved
                    temporal_cluster = cluster[temporal_picks, :]
                    cluster_times_ms = times_ms[np.any(temporal_cluster, axis=0)]
                elif cluster.ndim == 1 and len(cluster) == len(times):
                    # Time-only cluster
                    cluster_times_ms = times_ms[cluster]
                else:
                    continue  # Skip unknown formats
                    
                if len(cluster_times_ms) > 0:
                    axes[0,1].axvspan(cluster_times_ms[0], cluster_times_ms[-1], alpha=0.3, color='orange',
                                     label=f'p = {p_val:.3f}')
            except Exception as e:
                print(f"Warning: Could not plot cluster {cluster_idx}: {e}")
                continue

axes[0,1].set_xlabel('Time (ms)')
axes[0,1].set_ylabel('Amplitude (T)')
axes[0,1].set_title('Location Effect (Irrelevant Feature)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. Voice effect with significant clusters
voice_diff_temporal = np.mean(get_temporal_data(voice_match_data - voice_nonmatch_data), axis=0)

axes[1,0].plot(times_ms, voice_diff_temporal, linewidth=3, color='blue', label='Voice Effect')
axes[1,0].axvline(0, color='red', linestyle='--', alpha=0.7)
axes[1,0].axhline(0, color='black', linestyle='-', alpha=0.3)

# Highlight significant clusters (if any)
if len(voice_sig_clusters) > 0:
    for cluster_idx, p_val in voice_sig_clusters[:5]:  # Limit to first 5 for readability
        if cluster_idx < len(voice_clusters):
            cluster = voice_clusters[cluster_idx]
            try:
                # Handle different cluster formats
                if cluster.ndim == 2:
                    temporal_cluster = cluster[temporal_picks, :]
                    cluster_times_ms = times_ms[np.any(temporal_cluster, axis=0)]
                elif cluster.ndim == 1 and len(cluster) == len(times):
                    cluster_times_ms = times_ms[cluster]
                else:
                    continue
                    
                if len(cluster_times_ms) > 0:
                    axes[1,0].axvspan(cluster_times_ms[0], cluster_times_ms[-1], alpha=0.3, color='blue',
                                     label=f'p = {p_val:.3f}')
            except Exception as e:
                print(f"Warning: Could not plot voice cluster {cluster_idx}: {e}")
                continue

axes[1,0].set_xlabel('Time (ms)')
axes[1,0].set_ylabel('Amplitude (T)')
axes[1,0].set_title('Voice Effect (Relevant Feature)')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 4. Direct comparison of effects
axes[1,1].plot(times_ms, voice_diff_temporal, linewidth=3, color='blue', label='Voice Effect')
axes[1,1].plot(times_ms, location_diff_temporal, linewidth=3, color='orange', label='Location Effect')
axes[1,1].axvline(0, color='red', linestyle='--', alpha=0.7)
axes[1,1].axhline(0, color='black', linestyle='-', alpha=0.3)

# Highlight where voice > location (if any)
if len(diff_sig_clusters) > 0:
    for cluster_idx, p_val in diff_sig_clusters[:5]:  # Limit to first 5 for readability
        if cluster_idx < len(diff_clusters):
            cluster = diff_clusters[cluster_idx]
            try:
                # Handle different cluster formats
                if cluster.ndim == 2:
                    temporal_cluster = cluster[temporal_picks, :]
                    cluster_times_ms = times_ms[np.any(temporal_cluster, axis=0)]
                elif cluster.ndim == 1 and len(cluster) == len(times):
                    cluster_times_ms = times_ms[cluster]
                else:
                    continue
                    
                if len(cluster_times_ms) > 0:
                    axes[1,1].axvspan(cluster_times_ms[0], cluster_times_ms[-1], alpha=0.2, color='green',
                                     label=f'Voice > Location (p = {p_val:.3f})')
            except Exception as e:
                print(f"Warning: Could not plot diff cluster {cluster_idx}: {e}")
                continue

axes[1,1].set_xlabel('Time (ms)')
axes[1,1].set_ylabel('Amplitude (T)')
axes[1,1].set_title('Direct Comparison: Voice vs Location Effects')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

# 5. Summary statistics
axes[2,0].axis('off')
summary_text = f"""
CLUSTER-BASED PERMUTATION RESULTS

PARTICIPANTS: {len(all_subjects)} subjects
TEMPORAL CHANNELS: {len(available_temporal)} channels  
PERMUTATIONS: {n_permutations}, α = {alpha}

LOCATION EFFECT (Irrelevant):
• Total clusters: {len(location_clusters) if 'location_clusters' in locals() else 'N/A'}
• Significant clusters: {len(location_sig_clusters) if 'location_sig_clusters' in locals() else 'N/A'}
• Interpretation: {'Location still processed' if ('location_sig_clusters' in locals() and len(location_sig_clusters) > 0) else 'Location successfully ignored'}

VOICE EFFECT (Relevant):
• Total clusters: {len(voice_clusters) if 'voice_clusters' in locals() else 'N/A'}
• Significant clusters: {len(voice_sig_clusters) if 'voice_sig_clusters' in locals() else 'N/A'}
• Interpretation: {'Voice processing confirmed' if ('voice_sig_clusters' in locals() and len(voice_sig_clusters) > 0) else 'No voice effect detected'}

COMPARISON (Voice > Location):
• Significant clusters: {len(diff_sig_clusters) if 'diff_sig_clusters' in locals() else 'N/A'}
• Interpretation: {'Voice processing stronger' if ('diff_sig_clusters' in locals() and len(diff_sig_clusters) > 0) else 'Effects not significantly different'}

PAIRWISE COMPARISONS:
"""

for name, results in pairwise_results.items():
    summary_text += f"• {name}: {results['n_sig']} clusters\n"

axes[2,0].text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
               transform=axes[2,0].transAxes, fontfamily='monospace')

# 6. Effect sizes over time windows
time_windows = [(0, 100), (100, 200), (200, 400), (400, 600), (600, 1000)]
voice_windows = []
location_windows = []

for tmin, tmax in time_windows:
    time_mask = (times_ms >= tmin) & (times_ms <= tmax)
    if np.any(time_mask):
        voice_windows.append(np.mean(np.abs(voice_diff_temporal[time_mask])))
        location_windows.append(np.mean(np.abs(location_diff_temporal[time_mask])))
    else:
        voice_windows.append(0)
        location_windows.append(0)

x_pos = np.arange(len(time_windows))
width = 0.35

bars1 = axes[2,1].bar(x_pos - width/2, voice_windows, width, label='Voice Effect', color='blue', alpha=0.7)
bars2 = axes[2,1].bar(x_pos + width/2, location_windows, width, label='Location Effect', color='orange', alpha=0.7)

axes[2,1].set_xlabel('Time Windows (ms)')
axes[2,1].set_ylabel('Mean Absolute Effect Size (T)')
axes[2,1].set_title('Effect Magnitudes by Time Window')
axes[2,1].set_xticks(x_pos)
axes[2,1].set_xticklabels([f'{tmin}-{tmax}' for tmin, tmax in time_windows])
axes[2,1].legend()
axes[2,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'complete_probe_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Step 9: Save Results and Create Summary Report

print("\n" + "="*60)
print("STEP 9: SAVING RESULTS AND CREATING SUMMARY")
print("="*60)

# Prepare comprehensive results summary
results_summary = {
    'analysis_info': {
        'n_subjects': len(all_subjects),
        'subjects': all_subjects,
        'n_permutations': n_permutations,
        'alpha': alpha,
        'temporal_channels': len(available_temporal),
        'time_range': f"{times[0]:.3f} to {times[-1]:.3f} seconds"
    },
    'location_effect': {
        'total_clusters': len(location_clusters) if 'location_clusters' in locals() else 0,
        'significant_clusters': len(location_sig_clusters) if 'location_sig_clusters' in locals() else 0,
        'cluster_details': [{'cluster_id': i+1, 'p_value': float(p) if np.isscalar(p) else float(p.item() if hasattr(p, 'item') else p[0])} for i, p in location_sig_clusters] if 'location_sig_clusters' in locals() else [],
        'interpretation': 'Location still processed' if ('location_sig_clusters' in locals() and len(location_sig_clusters) > 0) else 'Location successfully ignored'
    },
    'voice_effect': {
        'total_clusters': len(voice_clusters) if 'voice_clusters' in locals() else 0,
        'significant_clusters': len(voice_sig_clusters) if 'voice_sig_clusters' in locals() else 0,
        'cluster_details': [{'cluster_id': i+1, 'p_value': float(p) if np.isscalar(p) else float(p.item() if hasattr(p, 'item') else p[0])} for i, p in voice_sig_clusters] if 'voice_sig_clusters' in locals() else [],
        'interpretation': 'Voice processing confirmed' if ('voice_sig_clusters' in locals() and len(voice_sig_clusters) > 0) else 'No voice effect detected'
    },
    'comparison_voice_vs_location': {
        'significant_clusters': len(diff_sig_clusters) if 'diff_sig_clusters' in locals() else 0,
        'cluster_details': [{'cluster_id': i+1, 'p_value': float(p) if np.isscalar(p) else float(p.item() if hasattr(p, 'item') else p[0])} for i, p in diff_sig_clusters] if 'diff_sig_clusters' in locals() else [],
        'interpretation': 'Voice processing stronger' if ('diff_sig_clusters' in locals() and len(diff_sig_clusters) > 0) else 'Effects not significantly different'
    },
    'pairwise_results': {name: {'n_significant': res['n_sig']} for name, res in pairwise_results.items()}
}

# Save results as JSON
with open(OUTPUT_PATH + 'complete_analysis_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

# Save individual grand averages for further analysis
for cond, grand_avg in grand_averages.items():
    filename = cond.replace('/', '_') + '_grand_average-ave.fif'
    grand_avg.save(OUTPUT_PATH + filename, overwrite=True)

# Create effect difference waves and save them
location_effect_avg = mne.combine_evoked([
    mne.grand_average([mne.combine_evoked([all_condition_evokeds['Probe/SpNM/LoM'][i], all_condition_evokeds['Probe/SpM/LoM'][i]], weights=[0.5, 0.5]) for i in range(len(all_subjects))]),
    mne.grand_average([mne.combine_evoked([all_condition_evokeds['Probe/SpNM/LoNM'][i], all_condition_evokeds['Probe/SpM/LoNM'][i]], weights=[0.5, 0.5]) for i in range(len(all_subjects))])
], weights=[1, -1])

voice_effect_avg = mne.combine_evoked([
    mne.grand_average([mne.combine_evoked([all_condition_evokeds['Probe/SpM/LoNM'][i], all_condition_evokeds['Probe/SpM/LoM'][i]], weights=[0.5, 0.5]) for i in range(len(all_subjects))]),
    mne.grand_average([mne.combine_evoked([all_condition_evokeds['Probe/SpNM/LoNM'][i], all_condition_evokeds['Probe/SpNM/LoM'][i]], weights=[0.5, 0.5]) for i in range(len(all_subjects))])
], weights=[1, -1])

location_effect_avg.save(OUTPUT_PATH + 'location_effect_difference_wave-ave.fif', overwrite=True)
voice_effect_avg.save(OUTPUT_PATH + 'voice_effect_difference_wave-ave.fif', overwrite=True)

# Print final summary
print("\nFILES CREATED:")
print("• complete_probe_analysis_results.png - Comprehensive visualization")
print("• complete_analysis_results.json - Detailed statistical summary")
print("• [condition]_grand_average-ave.fif - Individual condition grand averages")
print("• [effect]_difference_wave-ave.fif - Effect difference waves")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

# Print interpretation
print("\nINTERPRETATION:")
location_clusters_count = len(location_sig_clusters) if 'location_sig_clusters' in locals() else 0
voice_clusters_count = len(voice_sig_clusters) if 'voice_sig_clusters' in locals() else 0  
diff_clusters_count = len(diff_sig_clusters) if 'diff_sig_clusters' in locals() else 0

if location_clusters_count == 0:
    print("✓ SUCCESS: Location (irrelevant feature) was successfully ignored")
else:
    print("⚠ ATTENTION: Location (irrelevant feature) still shows significant processing")

if voice_clusters_count > 0:
    print("✓ SUCCESS: Voice (relevant feature) shows significant processing")
else:
    print("⚠ ATTENTION: Voice (relevant feature) shows no significant effects")

if diff_clusters_count > 0:
    print("✓ SUCCESS: Voice processing is significantly stronger than location processing")
else:
    print("⚠ NEUTRAL: Voice and location effects are not significantly different")

print(f"\nResults saved to: {OUTPUT_PATH}")