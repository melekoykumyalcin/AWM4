#!/usr/bin/env python
"""
Temporal-Only Spatio-Temporal Delay Period Analysis Script
- Uses temporal channels only (68 channels)
- No averaging scheme (individual trials)
- Adaptive regularization for different window sizes
- 10ms sliding window steps
- All window sizes [100, 200, 300, 400, 500]ms

Usage: python temporal_spatiotemporal_analysis.py --subject 23
"""

import os
import sys
import argparse
import locale
from tqdm import tqdm
import numpy as np
import mne
mne.set_log_level('WARNING')
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from collections import defaultdict, Counter
from joblib import Parallel, delayed
import json
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run temporal-only spatio-temporal delay period analysis')
parser.add_argument('--subject', type=int, required=True, help='Subject ID')
parser.add_argument('--n_iterations', type=int, default=10, help='Number of iterations')
parser.add_argument('--n_jobs', type=int, default=20, help='Number of parallel jobs')
parser.add_argument('--tmin', type=float, default=2.0, help='Start time of analysis window')
parser.add_argument('--tmax', type=float, default=4.5, help='End time of analysis window')
args = parser.parse_args()

# Set paths for HPC
HOME_DIR = '/mnt/hpc/projects/awm4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
DATA_PATH = HOME_DIR + '/AWM4_data/raw/'
CORRECTED_DATA = HOME_DIR + '/AWM4_data/raw/correctTriggers/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'

# Create output directory
OUTPUT_DIR = PROCESSED_DIR + f'temporalSpatiotemporalAnalysis/subject_{args.subject}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log file path
LOG_FILE_PATH = f'{OUTPUT_DIR}/processing_log.txt'

def write_log(message):
    """Write message to log file"""
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(message + '\n')

# Initialize log
write_log(f"Temporal spatio-temporal analysis started at: {datetime.now()}")
write_log(f"Subject: {args.subject}")
write_log(f"Analysis window: {args.tmin}s to {args.tmax}s")
write_log(f"Using temporal channels only (no averaging scheme)")

# Analysis parameters
TEMPORAL_CONFIG = {
    'sampling_rate': 100,  # Hz (after initial downsampling)
    'within_window_downsampling': 20,  # ms (averaging interval)
    'sliding_step': 10,  # ms
    'window_sizes': [100, 200, 300, 400, 500],  # ms
    'analysis_period': [args.tmin, args.tmax],  # seconds
    'use_averaging_scheme': False,  # Individual trials
}

# Define temporal channels
TEMPORAL_CHANNELS = {
    'left': [
        'MLT11-3609', 'MLT12-3609', 'MLT13-3609', 'MLT14-3609', 'MLT15-3609', 'MLT16-3609',
        'MLT21-3609', 'MLT22-3609', 'MLT23-3609', 'MLT24-3609', 'MLT25-3609', 'MLT26-3609', 'MLT27-3609',
        'MLT31-3609', 'MLT32-3609', 'MLT33-3609', 'MLT34-3609', 'MLT35-3609', 'MLT36-3609', 'MLT37-3609',
        'MLT41-3609', 'MLT42-3609', 'MLT43-3609', 'MLT44-3609', 'MLT45-3609', 'MLT46-3609', 'MLT47-3609',
        'MLT51-3609', 'MLT52-3609', 'MLT53-3609', 'MLT54-3609', 'MLT55-3609', 'MLT56-3609', 'MLT57-3609'
    ],
    'right': [
        'MRT11-3609', 'MRT12-3609', 'MRT13-3609', 'MRT14-3609', 'MRT15-3609', 'MRT16-3609',
        'MRT21-3609', 'MRT22-3609', 'MRT23-3609', 'MRT24-3609', 'MRT25-3609', 'MRT26-3609', 'MRT27-3609',
        'MRT31-3609', 'MRT32-3609', 'MRT33-3609', 'MRT34-3609', 'MRT35-3609', 'MRT36-3609', 'MRT37-3609',
        'MRT41-3609', 'MRT42-3609', 'MRT43-3609', 'MRT44-3609', 'MRT45-3609', 'MRT46-3609', 'MRT47-3609',
        'MRT51-3609', 'MRT52-3609', 'MRT53-3609', 'MRT54-3609', 'MRT55-3609', 'MRT56-3609', 'MRT57-3609'
    ]
}

ALL_TEMPORAL_CHANNELS = TEMPORAL_CHANNELS['left'] + TEMPORAL_CHANNELS['right']

# Important time markers
CUE_TIME = 2.0  # Cue presentation
PING_TIME = 3.5  # Ping occurs at 3.5s
PROBE_TIME = 4.5  # Probe presentation

# Event dictionary
EVENT_DICT = {
    'S1/Sp1/L1': 111, 'S1/Sp1/L2': 112, 'S1/Sp1/L3': 113, 'S1/Sp1/L4': 114,
    'S1/Sp2/L1': 121, 'S1/Sp2/L2': 122, 'S1/Sp2/L3': 123, 'S1/Sp2/L4': 124,
    'S1/Sp3/L1': 131, 'S1/Sp3/L2': 132, 'S1/Sp3/L3': 133, 'S1/Sp3/L4': 134,
    'S1/Sp4/L1': 141, 'S1/Sp4/L2': 142, 'S1/Sp4/L3': 143, 'S1/Sp4/L4': 144,
    'S2/Sp1/L1': 211, 'S2/Sp1/L2': 212, 'S2/Sp1/L3': 213, 'S2/Sp1/L4': 214,
    'S2/Sp2/L1': 221, 'S2/Sp2/L2': 222, 'S2/Sp2/L3': 223, 'S2/Sp2/L4': 224,
    'S2/Sp3/L1': 231, 'S2/Sp3/L2': 232, 'S2/Sp3/L3': 233, 'S2/Sp3/L4': 234,
    'S2/Sp4/L1': 241, 'S2/Sp4/L2': 242, 'S2/Sp4/L3': 243, 'S2/Sp4/L4': 244,
    'Cue_S1': 101, 'Cue_S2': 201
}

def get_adaptive_regularization_strategy(n_features, n_samples):
    """
    Get adaptive regularization strategy based on feature dimensionality
    """
    ratio = n_features / n_samples
    
    strategies = {
        'excellent': {  # ratio < 1.0
            'description': 'Low dimensionality - standard regularization',
            'c_values': [0.1, 1.0, 10.0],
            'cv_folds': 5,
            'n_jobs': args.n_jobs
        },
        'good': {  # 1.0 <= ratio < 2.0
            'description': 'Moderate dimensionality - balanced regularization',
            'c_values': [0.01, 0.1, 1.0],
            'cv_folds': 5,
            'n_jobs': args.n_jobs
        },
        'challenging': {  # 2.0 <= ratio < 4.0
            'description': 'High dimensionality - strong regularization',
            'c_values': [0.001, 0.01, 0.1],
            'cv_folds': 3,  # Fewer folds = more training data per fold
            'n_jobs': min(args.n_jobs, 10)  # Reduce parallel load
        },
        'extreme': {  # ratio >= 4.0
            'description': 'Very high dimensionality - maximum regularization',
            'c_values': [0.0001, 0.001, 0.01],
            'cv_folds': 3,
            'n_jobs': min(args.n_jobs, 5)
        }
    }
    
    if ratio < 1.0:
        return strategies['excellent'], ratio
    elif ratio < 2.0:
        return strategies['good'], ratio
    elif ratio < 4.0:
        return strategies['challenging'], ratio
    else:
        return strategies['extreme'], ratio

def extract_wolff_temporal_features(data, window_length_ms, sfreq=100):
    """
    Extract Wolff-style spatio-temporal features from temporal channels
    100Hz → within-window 20ms averaging → flatten
    
    Parameters:
    data: numpy array of shape (n_trials, n_channels, n_timepoints)
    window_length_ms: length of window in ms
    sfreq: sampling frequency (should be 100Hz)
    """
    
    # Calculate window parameters at specified sampling rate
    window_samples = int(window_length_ms * sfreq / 1000)  # e.g., 200ms = 20 samples at 100Hz
    downsample_interval_samples = int(TEMPORAL_CONFIG['within_window_downsampling'] * sfreq / 1000)  # 20ms = 2 samples at 100Hz
    
    # Calculate number of downsampled timepoints
    n_downsampled_timepoints = window_samples // downsample_interval_samples
    
    # Extract window data
    window_data = data[:, :, :window_samples]  # Shape: (n_trials, n_channels, window_samples)
    
    # Downsample within window by averaging every 20ms
    downsampled_data = []
    
    for timepoint_idx in range(n_downsampled_timepoints):
        # Calculate which samples to average for this 20ms interval
        start_sample = timepoint_idx * downsample_interval_samples
        end_sample = start_sample + downsample_interval_samples
        
        # Average across the interval for all trials and channels
        timepoint_avg = np.mean(window_data[:, :, start_sample:end_sample], axis=2)
        downsampled_data.append(timepoint_avg)
    
    # Stack to get final downsampled data
    processed_data = np.stack(downsampled_data, axis=2)  # (n_trials, n_channels, n_downsampled_timepoints)
    
    # Remove mean per channel per trial within window (Wolff normalization)
    for trial in range(processed_data.shape[0]):
        for channel in range(processed_data.shape[1]):
            channel_mean = np.mean(processed_data[trial, channel, :])
            processed_data[trial, channel, :] -= channel_mean
    
    # Flatten to feature vectors
    n_trials, n_channels, n_timepoints = processed_data.shape
    features = processed_data.reshape(n_trials, n_channels * n_timepoints)
    
    return features

def decode_with_adaptive_svm(features, labels, window_length_ms):
    """
    Decode using SVM with adaptive regularization based on dimensionality
    """
    n_features = features.shape[1]
    n_samples = features.shape[0]
    
    # Get adaptive strategy
    strategy, ratio = get_adaptive_regularization_strategy(n_features, n_samples)
    
    write_log(f"    {window_length_ms}ms window: {n_features} features, {n_samples} samples, ratio={ratio:.2f}")
    write_log(f"    Strategy: {strategy['description']}")
    write_log(f"    C values: {strategy['c_values']}, CV folds: {strategy['cv_folds']}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', probability=True))
    ])
    
    # Parameter grid
    param_grid = {'svm__C': strategy['c_values']}
    
    # Grid search with adaptive CV
    try:
        grid_search = GridSearchCV(
            pipeline, param_grid, 
            cv=strategy['cv_folds'], 
            scoring='accuracy',
            n_jobs=strategy['n_jobs']
        )
        
        grid_search.fit(features, labels)
        
        return grid_search.best_score_, grid_search.best_params_, strategy
        
    except Exception as e:
        write_log(f"    ERROR in SVM fitting: {str(e)}")
        return 0.5, {'svm__C': strategy['c_values'][0]}, strategy

def sliding_window_temporal_svm(epochs_data, events, window_length_ms, feature_name):
    """
    Sliding window SVM analysis with temporal channels and adaptive regularization
    """
    write_log(f"\n  Temporal spatio-temporal decoding: {feature_name}")
    write_log(f"  Window length: {window_length_ms}ms")
    write_log(f"  Sliding step: {TEMPORAL_CONFIG['sliding_step']}ms")
    
    # Calculate sliding window parameters
    sfreq = epochs_data.info['sfreq']
    step_samples = int(TEMPORAL_CONFIG['sliding_step'] * sfreq / 1000)  # 10ms steps
    window_samples = int(window_length_ms * sfreq / 1000)
    
    data = epochs_data.get_data()
    n_times = data.shape[2]
    
    # Calculate number of windows to cover full period
    n_windows = (n_times - window_samples) // step_samples + 1
    
    # Calculate coverage
    first_timepoint = epochs_data.tmin + (window_samples / 2) / sfreq
    last_timepoint = epochs_data.tmin + ((n_windows - 1) * step_samples + window_samples / 2) / sfreq
    coverage = last_timepoint - first_timepoint
    
    write_log(f"  Number of windows: {n_windows}")
    write_log(f"  Coverage: {first_timepoint:.2f}s to {last_timepoint:.2f}s ({coverage:.2f}s)")
    
    # Storage for results across iterations
    iteration_results = []
    
    for iteration in range(args.n_iterations):
        write_log(f"    Iteration {iteration + 1}/{args.n_iterations}")
        
        window_scores = np.zeros(n_windows)
        window_strategies = []
        
        # Process each sliding window
        for window_idx in tqdm(range(n_windows), desc=f"{feature_name} {window_length_ms}ms"):
            win_start = window_idx * step_samples
            win_end = win_start + window_samples
            
            # Extract window data
            window_data = data[:, :, win_start:win_end]
            
            # Extract spatio-temporal features directly from window data
            features = extract_wolff_temporal_features(window_data, window_length_ms, epochs_data.info['sfreq'])
            
            # Create labels based on feature type
            if feature_name == 'maintained_voice':
                # Speaker 1+2 vs 3+4
                labels = np.array([0 if (e//10)%10 in [1,2] else 1 for e in events])
            else:  # maintained_location
                # Location 1+2 vs 3+4
                labels = np.array([0 if e%10 in [1,2] else 1 for e in events])
            
            # Skip if only one class
            if len(np.unique(labels)) < 2:
                window_scores[window_idx] = 0.5
                window_strategies.append({'strategy': 'skipped'})
                continue
            
            # Decode with adaptive SVM
            accuracy, best_params, strategy = decode_with_adaptive_svm(features, labels, window_length_ms)
            
            window_scores[window_idx] = accuracy
            window_strategies.append({
                'strategy': strategy['description'], 
                'best_c': best_params.get('svm__C', 'unknown')
            })
        
        # Store iteration results
        iteration_results.append({
            'scores': window_scores,
            'strategies': window_strategies
        })
    
    # Aggregate results across iterations
    all_scores = np.array([r['scores'] for r in iteration_results])
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)
    
    # Create time points (center of each window)
    timepoints = np.array([epochs_data.tmin + (i * step_samples + window_samples / 2) / sfreq 
                          for i in range(n_windows)])
    
    write_log(f"    Mean accuracy: {np.mean(mean_scores):.3f}")
    write_log(f"    Peak accuracy: {np.max(mean_scores):.3f}")
    write_log(f"    Peak time: {timepoints[np.argmax(mean_scores)]:.3f}s")
    
    # Find accuracy at ping time
    ping_idx = np.argmin(np.abs(timepoints - PING_TIME))
    ping_accuracy = mean_scores[ping_idx]
    write_log(f"    Accuracy at ping ({PING_TIME}s): {ping_accuracy:.3f}")
    
    return mean_scores, std_scores, timepoints, all_scores, iteration_results

def extract_maintained_information(subject, metaInfo):
    """Extract which information is maintained in working memory based on retro-cues"""
    write_log(f"\nExtracting maintained information for subject {subject}...")
    
    try:
        # Get file information
        actInd = (metaInfo.Subject==subject) & (metaInfo.Valid==1)
        
        # Determine if subject is in early subjects
        early_subject = subject in np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])[:7]
        
        if early_subject:
            actFiles = pd.Series([f.split('.')[0] + '_correct_triggers.fif' for f in metaInfo['MEG_Name']])[actInd]
        else:
            actFiles = metaInfo['MEG_Name'][actInd]
        
        # Load and concatenate events
        all_events = None
        reference_dev_head_t_ref = None
        
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
            else:
                all_events = np.concatenate((all_events, events), axis=0)
            del raw
        
        # Find retro-cue events
        S1_cue_indices = np.where(all_events[:,2] == 101)[0]
        S2_cue_indices = np.where(all_events[:,2] == 201)[0]
        
        write_log(f"Found {len(S1_cue_indices)} S1 cues and {len(S2_cue_indices)} S2 cues")
        
        # Create memorized array
        memorized = np.zeros(len(all_events[:,2]))
        
        # For S1 cues, maintained stimulus is 4 positions before
        for i in S1_cue_indices:
            if i >= 4:
                memorized[i - 4] = all_events[i - 4, 2]
        
        # For S2 cues, maintained stimulus is 2 positions before
        for i in S2_cue_indices:
            if i >= 2:
                memorized[i - 2] = all_events[i - 2, 2]
        
        # Keep only non-zero values
        memorized_values = memorized[memorized != 0]
        
        write_log(f"Extracted {len(memorized_values)} maintained trials")
        
        return memorized_values, all_events
        
    except Exception as e:
        write_log(f"Error extracting maintained information: {str(e)}")
        return None, None

def load_temporal_subject_data(subject, meta_info):
    """Load and preprocess data using temporal channels only"""
    write_log(f"\nLoading temporal channel data for subject {subject}...")
    
    try:
        # First extract maintained information
        memorized, all_events = extract_maintained_information(subject, meta_info)
        
        if memorized is None:
            write_log("Could not extract maintained information")
            return None, None, None
        
        # Load cleaned epochs
        clean_trials = mne.read_epochs(
            f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif",
            preload=True,
            verbose='ERROR'
        )
        
        # Special case handling
        if subject == 23:
            drop_idx = 64 * 7
            if drop_idx < len(clean_trials):
                memorized = np.delete(memorized, drop_idx)

        if subject == 28:
            drop_idx = 63
            if drop_idx < len(clean_trials):
                clean_trials.drop(drop_idx)
                        
        # Check for jump artifacts
        jname = f"{PROCESSED_DIR}/ICAs/Jumps{subject}.npy"
        if os.path.isfile(jname):
            jump_inds = np.load(jname)
            if len(jump_inds) > 0:
                jump_inds = np.array(jump_inds, dtype=int)
                valid_jump_inds = jump_inds[jump_inds < len(clean_trials)]
                
                if len(valid_jump_inds) > 0:
                    clean_trials.drop(valid_jump_inds, reason='jump')
                    memorized = np.delete(memorized, valid_jump_inds)
        
        # Update event codes to reflect maintained information
        clean_trials.events[:, 2] = memorized[:len(clean_trials.events)]
        clean_trials.event_id = EVENT_DICT
        
        # Crop to analysis window
        delay_epochs = clean_trials.copy()
        delay_epochs.crop(tmin=TEMPORAL_CONFIG['analysis_period'][0], 
                         tmax=TEMPORAL_CONFIG['analysis_period'][1])
        
        # Select ONLY temporal channels (68 channels)
        available_temporal = [ch for ch in ALL_TEMPORAL_CHANNELS if ch in delay_epochs.ch_names]
        
        if len(available_temporal) == 0:
            write_log("ERROR: No temporal channels found in data!")
            return None, None, None
        
        temporal_epochs = delay_epochs.copy().pick_channels(available_temporal)
        write_log(f"Selected {len(available_temporal)} temporal channels")
        
        # Resample to 100Hz
        temporal_epochs = temporal_epochs.resample(TEMPORAL_CONFIG['sampling_rate'], npad='auto')
        
        # Get data
        epochs_data = temporal_epochs.get_data(copy=False)
        maintained_events = temporal_epochs.events[:, 2]
        
        write_log(f"Data loaded successfully. Shape: {epochs_data.shape}")
        write_log(f"Number of temporal channels: {epochs_data.shape[1]}")
        write_log(f"Maintained events: {len(maintained_events)}")
        
        # Log distribution
        speakers = {
            'Sp1+2': sum(1 for v in maintained_events if (v//10)%10 in [1, 2]),
            'Sp3+4': sum(1 for v in maintained_events if (v//10)%10 in [3, 4])
        }
        locations = {
            'L1+2': sum(1 for v in maintained_events if v%10 in [1, 2]),
            'L3+4': sum(1 for v in maintained_events if v%10 in [3, 4])
        }
        write_log(f"Speaker distribution: {speakers}")
        write_log(f"Location distribution: {locations}")
        
        return temporal_epochs, maintained_events, temporal_epochs.info['sfreq']
        
    except Exception as e:
        write_log(f"Error loading data: {str(e)}")
        import traceback
        write_log(traceback.format_exc())
        return None, None, None

def plot_temporal_results(all_window_results):
    """Create a summary plot of temporal spatio-temporal results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Subject {args.subject} - Temporal Channels Spatio-Temporal Analysis', fontsize=16)
    
    # Colors for different window sizes
    window_colors = {100: '#e41a1c', 200: '#377eb8', 300: '#4daf4a', 400: '#984ea3', 500: '#ff7f00'}
    
    for idx, feature_name in enumerate(['maintained_voice', 'maintained_location']):
        # Time courses (larger subplot)
        ax_main = axes[idx, :2].flatten()
        ax_main = plt.subplot(2, 3, idx*3 + 1)
        plt.subplot(2, 3, idx*3 + 2)
        ax_main = plt.subplot2grid((2, 3), (idx, 0), colspan=2)
        
        # Summary plot
        ax_summary = axes[idx, 2]
        
        peak_accuracies = []
        peak_times = []
        window_labels = []
        
        for window_ms, results in all_window_results.items():
            if feature_name in results:
                data = results[feature_name]
                
                # Plot time course
                ax_main.plot(data['timepoints'], data['mean_scores'], 
                           label=f'{window_ms}ms', 
                           color=window_colors[window_ms], 
                           linewidth=2.5)
                
                # Add standard error
                ax_main.fill_between(data['timepoints'],
                                   data['mean_scores'] - data['std_scores']/np.sqrt(args.n_iterations),
                                   data['mean_scores'] + data['std_scores']/np.sqrt(args.n_iterations),
                                   color=window_colors[window_ms], alpha=0.2)
                
                # Store peak info
                peak_idx = np.argmax(data['mean_scores'])
                peak_accuracies.append(data['mean_scores'][peak_idx])
                peak_times.append(data['timepoints'][peak_idx])
                window_labels.append(f'{window_ms}ms')
        
        # Time course formatting
        ax_main.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7, label='Ping')
        ax_main.axvline(x=CUE_TIME, color='green', linestyle='--', alpha=0.7, label='Cue')
        ax_main.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
        ax_main.set_xlabel('Time (s)')
        ax_main.set_ylabel('Decoding Accuracy')
        ax_main.set_title(f'{feature_name.replace("maintained_", "").title()} - Temporal Channels')
        ax_main.set_ylim(0.45, 0.8)
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        
        # Peak summary
        x_pos = np.arange(len(peak_accuracies))
        bars = ax_summary.bar(x_pos, peak_accuracies, color=[window_colors[int(w.replace('ms', ''))] for w in window_labels])
        ax_summary.set_xlabel('Window Size')
        ax_summary.set_ylabel('Peak Accuracy')
        ax_summary.set_title(f'Peak Performance\n{feature_name.replace("maintained_", "").title()}')
        ax_summary.set_xticks(x_pos)
        ax_summary.set_xticklabels(window_labels, rotation=45)
        ax_summary.set_ylim(0.5, 0.8)
        ax_summary.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
        ax_summary.grid(True, alpha=0.3, axis='y')
        
        # Add peak times as text
        for i, (acc, time) in enumerate(zip(peak_accuracies, peak_times)):
            ax_summary.text(i, acc + 0.01, f'{time:.2f}s', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/temporal_spatiotemporal_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    write_log(f"\nSummary plot saved to {OUTPUT_DIR}/temporal_spatiotemporal_results.png")

def main():
    """Main processing function"""
    # Load metadata
    meta_info = pd.read_excel(META_FILE)
    
    # Load subject data with temporal channels
    temporal_epochs, events, sfreq = load_temporal_subject_data(args.subject, meta_info)
    
    if temporal_epochs is None:
        write_log("Failed to load data. Exiting.")
        sys.exit(1)
    
    write_log(f"\n=== TEMPORAL SPATIO-TEMPORAL ANALYSIS ===")
    write_log(f"=== Channels: {temporal_epochs.get_data().shape[1]} temporal channels ===")
    write_log(f"=== Window lengths: {TEMPORAL_CONFIG['window_sizes']}ms ===")
    write_log(f"=== Sliding step: {TEMPORAL_CONFIG['sliding_step']}ms ===")
    write_log(f"=== Sampling rate: {TEMPORAL_CONFIG['sampling_rate']}Hz ===")
    write_log(f"=== No averaging scheme (individual trials) ===")
    
    # Results storage for all window lengths
    all_window_results = {}
    
    # Iterate through different window lengths
    for window_length_ms in TEMPORAL_CONFIG['window_sizes']:
        write_log(f"\n{'='*60}")
        write_log(f"Processing temporal spatio-temporal window: {window_length_ms}ms")
        write_log(f"{'='*60}")
        
        # Calculate expected features
        timepoints_per_channel = window_length_ms // TEMPORAL_CONFIG['within_window_downsampling']
        n_channels = temporal_epochs.get_data().shape[1]
        expected_features = n_channels * timepoints_per_channel
        n_samples = temporal_epochs.get_data().shape[0]
        
        write_log(f"Expected features: {n_channels} channels × {timepoints_per_channel} timepoints = {expected_features}")
        write_log(f"Available samples: {n_samples}")
        write_log(f"Feature/sample ratio: {expected_features/n_samples:.2f}")
        
        # Results storage for this window length
        window_results = {}
        
        # Process each feature (voice and location)
        for feature_name in ['maintained_voice', 'maintained_location']:
            
            # Run temporal spatio-temporal decoding
            mean_scores, std_scores, timepoints, all_scores, iteration_results = sliding_window_temporal_svm(
                temporal_epochs, events, window_length_ms, feature_name
            )
            
            # Store results
            window_results[feature_name] = {
                'mean_scores': mean_scores,
                'std_scores': std_scores,
                'timepoints': timepoints,
                'all_scores': all_scores,
                'iteration_results': iteration_results,
                'window_length_ms': window_length_ms,
                'n_features': expected_features,
                'n_samples': n_samples,
                'feature_sample_ratio': expected_features/n_samples
            }
            
            write_log(f"  {feature_name}: Mean={np.mean(mean_scores):.3f}, Peak={np.max(mean_scores):.3f}")
        
        # Store results for this window length
        all_window_results[window_length_ms] = window_results
        
        # Save results for this window length
        write_log(f"\nSaving results for {window_length_ms}ms temporal spatio-temporal window...")
        
        for feature_name, results in window_results.items():
            # Save temporal data
            np.save(f'{OUTPUT_DIR}/temporal_spatiotemporal_{feature_name}_{window_length_ms}ms_mean_scores.npy', 
                   results['mean_scores'])
            np.save(f'{OUTPUT_DIR}/temporal_spatiotemporal_{feature_name}_{window_length_ms}ms_std_scores.npy', 
                   results['std_scores'])
            np.save(f'{OUTPUT_DIR}/temporal_spatiotemporal_{feature_name}_{window_length_ms}ms_timepoints.npy', 
                   results['timepoints'])
            np.save(f'{OUTPUT_DIR}/temporal_spatiotemporal_{feature_name}_{window_length_ms}ms_all_scores.npy', 
                   results['all_scores'])
    
    # Create summary plot
    plot_temporal_results(all_window_results)
    
    # Save comprehensive summary
    summary = {
        'subject': args.subject,
        'n_iterations': args.n_iterations,
        'analysis_window': f"{TEMPORAL_CONFIG['analysis_period'][0]}-{TEMPORAL_CONFIG['analysis_period'][1]}s",
        'features': ['maintained_voice', 'maintained_location'],
        'window_sizes_ms': TEMPORAL_CONFIG['window_sizes'],
        'window_step_ms': TEMPORAL_CONFIG['sliding_step'],
        'within_window_downsampling_ms': TEMPORAL_CONFIG['within_window_downsampling'],
        'sampling_rate': TEMPORAL_CONFIG['sampling_rate'],
        'channels': 'temporal_only',
        'n_channels': temporal_epochs.get_data().shape[1],
        'n_trials': temporal_epochs.get_data().shape[0],
        'averaging_scheme': TEMPORAL_CONFIG['use_averaging_scheme'],
        'ping_time': PING_TIME,
        'processing_time': str(datetime.now()),
        'method': 'temporal_spatiotemporal_sliding_window',
        'adaptive_regularization': True,
        'results': {}
    }
    
    # Add detailed results for each window length
    for window_length_ms, window_results in all_window_results.items():
        summary['results'][f'temporal_spatiotemporal_{window_length_ms}ms'] = {}
        
        for feature_name, results in window_results.items():
            # Find ping time accuracy
            ping_idx = np.argmin(np.abs(results['timepoints'] - PING_TIME))
            ping_accuracy = results['mean_scores'][ping_idx]
            
            summary['results'][f'temporal_spatiotemporal_{window_length_ms}ms'][feature_name] = {
                'mean_accuracy': float(np.mean(results['mean_scores'])),
                'peak_accuracy': float(np.max(results['mean_scores'])),
                'peak_time': float(results['timepoints'][np.argmax(results['mean_scores'])]),
                'ping_accuracy': float(ping_accuracy),
                'n_timepoints': len(results['timepoints']),
                'n_features': results['n_features'],
                'n_samples': results['n_samples'],
                'feature_sample_ratio': results['feature_sample_ratio'],
                'time_above_chance': float(np.sum(results['mean_scores'] > 0.5) * TEMPORAL_CONFIG['sliding_step'] / 1000)
            }
    
    # Add comparison across window lengths
    write_log(f"\n=== COMPARISON ACROSS TEMPORAL SPATIO-TEMPORAL WINDOWS ===")
    for feature_name in ['maintained_voice', 'maintained_location']:
        write_log(f"\n{feature_name}:")
        for window_length_ms in sorted(all_window_results.keys()):
            results = all_window_results[window_length_ms][feature_name]
            mean_acc = np.mean(results['mean_scores'])
            peak_acc = np.max(results['mean_scores'])
            peak_time = results['timepoints'][np.argmax(results['mean_scores'])]
            n_features = results['n_features']
            ratio = results['feature_sample_ratio']
            write_log(f"  {window_length_ms}ms: Mean={mean_acc:.3f}, Peak={peak_acc:.3f} @ {peak_time:.2f}s, Features={n_features}, Ratio={ratio:.2f}")
    
    with open(f'{OUTPUT_DIR}/temporal_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    write_log(f"\nProcessing completed at: {datetime.now()}")
    
    print(f"Subject {args.subject} temporal spatio-temporal analysis completed successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()