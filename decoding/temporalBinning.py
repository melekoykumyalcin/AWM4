#!/usr/bin/env python
"""
Temporal Binning Analysis Script
Analyzes delay period using discrete temporal bins with averaging
Implements the methodology: "epoched data were averaged across time bins"

Usage: python temporal_binning_analysis.py --subject 23
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
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from collections import defaultdict, Counter
from joblib import Parallel, delayed
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run temporal binning analysis with multiple bin sizes')
parser.add_argument('--subject', type=int, required=True, help='Subject ID')
parser.add_argument('--n_iterations', type=int, default=10, help='Number of iterations')
parser.add_argument('--n_jobs', type=int, default=20, help='Number of parallel jobs')
args = parser.parse_args()

# Set paths for HPC
HOME_DIR = '/mnt/hpc/projects/awm4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
DATA_PATH = HOME_DIR + '/AWM4_data/raw/'
CORRECTED_DATA = HOME_DIR + '/AWM4_data/raw/correctTriggers/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'

# Create output directory
OUTPUT_DIR = PROCESSED_DIR + f'temporalBinning/subject_{args.subject}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log file path
LOG_FILE_PATH = f'{OUTPUT_DIR}/processing_log.txt'

def write_log(message):
    """Write message to log file"""
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(message + '\n')

# Initialize log
write_log(f"Temporal binning analysis started at: {datetime.now()}")
write_log(f"Subject: {args.subject}")
write_log(f"Using temporal binning with channel-wise z-scoring")

# Analysis parameters
RESAMPLE_FREQ = 100  # Hz
PING_TIME = 3.5  # Ping occurs at 3.5s
CUE_TIME = 2.0  # Cue presentation
PROBE_TIME = 4.7  # Probe presentation

# Define adaptive time windows for each bin size
TIME_WINDOWS = {
    100: (2.0, 4.5),   # 25 bins
    200: (2.0, 4.4),   # 12 bins  
    500: (2.0, 4.5),   # 5 bins
    1000: (2.5, 4.5),  # 2 bins - ping at exact boundary!
    2000: (2.0, 4.5)   # 1 bin
}

# Bin sizes to analyze (in ms)
BIN_SIZES_MS = [100, 200, 500, 1000, 2000]

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

# Define the four corners - includes both S1 and S2 stimuli
CORNERS = {
    'Corner1_Sp12L12': [111, 112, 121, 122, 211, 212, 221, 222],
    'Corner2_Sp12L34': [113, 114, 123, 124, 213, 214, 223, 224],
    'Corner3_Sp34L12': [131, 132, 141, 142, 231, 232, 241, 242],
    'Corner4_Sp34L34': [133, 134, 143, 144, 233, 234, 243, 244]
}

# Averaging scheme
AVERAGING_SCHEME = {'trials_per_condition': 3, 'total_trials': 12}

# SVM parameters
PARAM_GRID = [0.1, 1.0, 10.0]

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

def create_pseudo_trials_from_indices(epochs_data, events, trial_indices, corner_conditions, n_trials_per_condition):
    """Create pseudo-trials from specific trial indices only"""
    # Get only the specified trials
    subset_data = epochs_data[trial_indices]
    subset_events = events[trial_indices]
    
    # Organize trials by condition
    condition_trials = defaultdict(list)
    for idx, event in enumerate(subset_events):
        if event in corner_conditions:
            condition_trials[event].append(idx)
    
    # Check if we have enough trials
    if not condition_trials:
        return None
        
    # Count valid conditions (those with at least n_trials_per_condition)
    valid_conditions = [cond for cond in corner_conditions 
                       if cond in condition_trials and len(condition_trials[cond]) >= n_trials_per_condition]
    
    if len(valid_conditions) < 2:  # Need at least 2 conditions
        return None
    
    # Calculate how many pseudo-trials we can create
    min_trials = min(len(condition_trials[cond]) for cond in valid_conditions)
    n_pseudo_trials = min_trials // n_trials_per_condition
    
    if n_pseudo_trials == 0:
        return None
    
    pseudo_trials = []
    
    # Create pseudo-trials
    for _ in range(n_pseudo_trials):
        # Sample trials from each valid condition
        sampled_data = []
        for condition in valid_conditions:
            # Random sample without replacement
            indices = np.random.choice(
                condition_trials[condition], 
                size=n_trials_per_condition, 
                replace=False
            )
            sampled_data.extend(subset_data[indices])
            
            # Remove used indices to avoid resampling
            condition_trials[condition] = [i for i in condition_trials[condition] if i not in indices]
        
        # Average the sampled trials
        pseudo_trial = np.mean(sampled_data, axis=0)
        pseudo_trials.append(pseudo_trial)
    
    return np.array(pseudo_trials) if pseudo_trials else None

def z_score_channels(X_train, X_test):
    """Apply channel-wise z-scoring based on training data statistics"""
    X_train_z = X_train.copy()
    X_test_z = X_test.copy()
    
    for channel in range(X_train.shape[1]):  # For each channel
        # Compute mean and std on training trials only
        train_mean = np.mean(X_train[:, channel])
        train_std = np.std(X_train[:, channel])
        
        # Avoid division by zero
        if train_std == 0:
            train_std = 1.0
        
        # Apply to both training and test data
        X_train_z[:, channel] = (X_train[:, channel] - train_mean) / train_std
        X_test_z[:, channel] = (X_test[:, channel] - train_mean) / train_std
    
    return X_train_z, X_test_z

def calculate_bin_boundaries(bin_size_ms):
    """Calculate bin boundaries for given bin size"""
    tmin, tmax = TIME_WINDOWS[bin_size_ms]
    duration = tmax - tmin
    bin_size_sec = bin_size_ms / 1000.0
    n_bins = int(duration / bin_size_sec)
    
    bin_boundaries = []
    bin_centers = []
    
    for i in range(n_bins):
        bin_start = tmin + (i * bin_size_sec)
        bin_end = bin_start + bin_size_sec
        bin_center = (bin_start + bin_end) / 2
        
        bin_boundaries.append((bin_start, bin_end))
        bin_centers.append(bin_center)
    
    return bin_boundaries, bin_centers

def extract_bin_data(epochs_data, times, bin_start, bin_end):
    """Extract and average data within a temporal bin"""
    # Find time indices for this bin
    time_mask = (times >= bin_start) & (times < bin_end)
    
    if not np.any(time_mask):
        return None
    
    # Extract data for this time window
    bin_data = epochs_data[:, :, time_mask]
    
    # Average across time points within bin
    averaged_data = np.mean(bin_data, axis=2)  # Shape: (trials, channels)
    
    return averaged_data

def decode_temporal_bin(bin_data, events, feature_name, n_iterations):
    """
    Decode a single temporal bin using pseudo-trials and cross-validation
    """
    # Storage for results across iterations
    iteration_scores = []
    iteration_c_values = []
    
    for iteration in range(n_iterations):
        # Outer cross-validation
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
        outer_scores = []
        best_cs = []
        
        # Create labels based on feature type
        if feature_name == 'maintained_voice':
            # Speaker 1+2 vs 3+4
            original_labels = np.array([0 if (e//10)%10 in [1,2] else 1 for e in events])
        else:  # maintained_location
            # Location 1+2 vs 3+4
            original_labels = np.array([0 if e%10 in [1,2] else 1 for e in events])
        
        # Skip if only one class
        if len(np.unique(original_labels)) < 2:
            iteration_scores.append(0.5)
            iteration_c_values.append(1.0)
            continue
        
        # Outer CV loop
        for train_idx, test_idx in outer_cv.split(bin_data, original_labels):
            
            # Step 1: Split original trials
            train_data = bin_data[train_idx]
            test_data = bin_data[test_idx]
            train_events = events[train_idx]
            test_events = events[test_idx]
            
            # Step 2: Create pseudo-trials separately for train and test
            train_pseudo_data = []
            train_pseudo_labels = []
            test_pseudo_data = []
            test_pseudo_labels = []
            
            for corner_name, corner_conditions in CORNERS.items():
                # Create training pseudo-trials
                train_pseudo = create_pseudo_trials_from_indices(
                    train_data, train_events, 
                    np.arange(len(train_events)),
                    corner_conditions,
                    AVERAGING_SCHEME['trials_per_condition']
                )
                
                if train_pseudo is not None:
                    train_pseudo_data.append(train_pseudo)
                    # Assign labels based on corner
                    if feature_name == 'maintained_voice':
                        labels = [0 if 'Sp12' in corner_name else 1] * len(train_pseudo)
                    else:
                        labels = [0 if 'L12' in corner_name else 1] * len(train_pseudo)
                    train_pseudo_labels.extend(labels)
                
                # Create test pseudo-trials
                test_pseudo = create_pseudo_trials_from_indices(
                    test_data, test_events,
                    np.arange(len(test_events)),
                    corner_conditions,
                    AVERAGING_SCHEME['trials_per_condition']
                )
                
                if test_pseudo is not None:
                    test_pseudo_data.append(test_pseudo)
                    # Assign labels based on corner
                    if feature_name == 'maintained_voice':
                        labels = [0 if 'Sp12' in corner_name else 1] * len(test_pseudo)
                    else:
                        labels = [0 if 'L12' in corner_name else 1] * len(test_pseudo)
                    test_pseudo_labels.extend(labels)
            
            # Skip if not enough data
            if not train_pseudo_data or not test_pseudo_data:
                continue
            
            # Check if we have both classes in train and test
            if len(train_pseudo_data) < 2 or len(test_pseudo_data) < 2:
                continue
            
            # Combine pseudo-trials
            X_train = np.vstack(train_pseudo_data)
            y_train = np.array(train_pseudo_labels)
            X_test = np.vstack(test_pseudo_data)
            y_test = np.array(test_pseudo_labels)
            
            # Verify both classes present
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            
            # Step 3: Apply channel-wise z-scoring
            X_train_z, X_test_z = z_score_channels(X_train, X_test)
            
            # Step 4: Inner CV for hyperparameter selection
            best_score = -1
            best_c = 1.0
            
            # Use simple 3-fold CV for inner loop
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=None)
            
            for c_value in PARAM_GRID:
                # Create SVM classifier
                svm = SVC(kernel='linear', C=c_value, probability=True)
                
                # Evaluate on inner CV
                try:
                    inner_scores = []
                    for inner_train_idx, inner_val_idx in inner_cv.split(X_train_z, y_train):
                        X_inner_train = X_train_z[inner_train_idx]
                        y_inner_train = y_train[inner_train_idx]
                        X_inner_val = X_train_z[inner_val_idx]
                        y_inner_val = y_train[inner_val_idx]
                        
                        svm.fit(X_inner_train, y_inner_train)
                        score = svm.score(X_inner_val, y_inner_val)
                        inner_scores.append(score)
                    
                    mean_inner_score = np.mean(inner_scores)
                    
                    if mean_inner_score > best_score:
                        best_score = mean_inner_score
                        best_c = c_value
                except:
                    continue
            
            # Step 5: Train final model with best C and evaluate on test set
            final_svm = SVC(kernel='linear', C=best_c, probability=True)
            
            try:
                final_svm.fit(X_train_z, y_train)
                test_score = final_svm.score(X_test_z, y_test)
                outer_scores.append(test_score)
                best_cs.append(best_c)
            except:
                continue
        
        # Store results for this iteration
        if outer_scores:
            iteration_scores.append(np.mean(outer_scores))
            
            # Use most common C
            c_counter = Counter(best_cs)
            most_common_c = c_counter.most_common(1)[0][0] if best_cs else 1.0
            iteration_c_values.append(most_common_c)
        else:
            iteration_scores.append(0.5)
            iteration_c_values.append(1.0)
    
    return iteration_scores, iteration_c_values

def decode_temporal_bins(epochs_data, times, events, feature_name, bin_size_ms, n_iterations):
    """
    Decode across temporal bins for a given bin size
    """
    write_log(f"\n  Decoding {feature_name} with {bin_size_ms}ms bins")
    
    # Calculate bin boundaries
    bin_boundaries, bin_centers = calculate_bin_boundaries(bin_size_ms)
    tmin, tmax = TIME_WINDOWS[bin_size_ms]
    
    write_log(f"  Time window: {tmin}s to {tmax}s")
    write_log(f"  Number of bins: {len(bin_boundaries)}")
    write_log(f"  Bin size: {bin_size_ms}ms")
    
    # Storage for results
    bin_scores = []
    bin_stds = []
    bin_c_values = []
    all_iteration_scores = []
    
    # Process each bin
    for bin_idx, (bin_start, bin_end) in enumerate(tqdm(bin_boundaries, desc=f"{feature_name} bins")):
        
        # Extract data for this bin
        bin_data = extract_bin_data(epochs_data, times, bin_start, bin_end)
        
        if bin_data is None:
            write_log(f"    Bin {bin_idx+1} ({bin_start:.2f}-{bin_end:.2f}s): No data")
            bin_scores.append(0.5)
            bin_stds.append(0.0)
            bin_c_values.append(1.0)
            all_iteration_scores.append([0.5] * n_iterations)
            continue
        
        # Decode this bin
        iteration_scores, iteration_c_values = decode_temporal_bin(
            bin_data, events, feature_name, n_iterations
        )
        
        # Store results
        mean_score = np.mean(iteration_scores)
        std_score = np.std(iteration_scores)
        
        bin_scores.append(mean_score)
        bin_stds.append(std_score)
        all_iteration_scores.append(iteration_scores)
        
        # Use most common C value
        c_counter = Counter(iteration_c_values)
        most_common_c = c_counter.most_common(1)[0][0] if iteration_c_values else 1.0
        bin_c_values.append(most_common_c)
        
        write_log(f"    Bin {bin_idx+1} ({bin_start:.2f}-{bin_end:.2f}s): {mean_score:.3f} ± {std_score:.3f}")
    
    # Convert to numpy arrays
    bin_scores = np.array(bin_scores)
    bin_stds = np.array(bin_stds)
    bin_centers = np.array(bin_centers)
    all_iteration_scores = np.array(all_iteration_scores)
    
    # Log summary
    write_log(f"    Mean accuracy: {np.mean(bin_scores):.3f}")
    write_log(f"    Peak accuracy: {np.max(bin_scores):.3f}")
    write_log(f"    Peak bin: {np.argmax(bin_scores)+1} (centered at {bin_centers[np.argmax(bin_scores)]:.2f}s)")
    
    # Find bin closest to ping time
    ping_idx = np.argmin(np.abs(bin_centers - PING_TIME))
    if ping_idx < len(bin_scores):
        write_log(f"    Accuracy at ping time: {bin_scores[ping_idx]:.3f}")
    
    return bin_scores, bin_stds, bin_centers, all_iteration_scores, bin_c_values

def load_subject_data(subject, meta_info):
    """Load and preprocess data for temporal binning analysis"""
    write_log(f"\nLoading data for subject {subject}...")
    
    try:
        # First extract maintained information
        memorized, all_events = extract_maintained_information(subject, meta_info)
        
        if memorized is None:
            write_log("Could not extract maintained information")
            return None, None, None, None
        
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
        
        # Get the maximum time window needed (2.0-4.5s)
        max_tmin = min(TIME_WINDOWS[bs][0] for bs in BIN_SIZES_MS)
        max_tmax = max(TIME_WINDOWS[bs][1] for bs in BIN_SIZES_MS)
        
        # Crop to maximum analysis window
        delay_epochs = clean_trials.copy()
        delay_epochs.crop(tmin=max_tmin, tmax=max_tmax)
        
        # Select magnetometers and resample
        mag_epochs = delay_epochs.copy().pick_types(meg='mag')
        mag_epochs = mag_epochs.resample(RESAMPLE_FREQ, npad='auto')
        
        # Get data and times
        epochs_data = mag_epochs.get_data(copy=False)
        maintained_events = mag_epochs.events[:, 2]
        times = mag_epochs.times
        
        write_log(f"Data loaded successfully. Shape: {epochs_data.shape}")
        write_log(f"Maintained events: {len(maintained_events)}")
        write_log(f"Time range: {times[0]:.2f}s to {times[-1]:.2f}s")
        
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
        
        return epochs_data, maintained_events, times, mag_epochs.info['sfreq']
        
    except Exception as e:
        write_log(f"Error loading data: {str(e)}")
        import traceback
        write_log(traceback.format_exc())
        return None, None, None, None

def plot_results_summary(all_bin_results):
    """Create a summary plot of results across bin sizes"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Subject {args.subject} - Temporal Binning Analysis', fontsize=16)
    
    # Colors for different bin sizes
    bin_colors = {
        100: '#e41a1c',
        200: '#377eb8',
        500: '#4daf4a',
        1000: '#984ea3',
        2000: '#ff7f00'
    }
    
    for idx, feature_name in enumerate(['maintained_voice', 'maintained_location']):
        ax = axes[idx, 0]  # Time courses
        ax2 = axes[idx, 1]  # Peak summary
        
        peak_accuracies = []
        peak_times = []
        
        for bin_size_ms, results in all_bin_results.items():
            if feature_name in results:
                data = results[feature_name]
                
                # Plot time course
                ax.plot(data['bin_centers'], data['bin_scores'], 
                       'o-', label=f'{bin_size_ms}ms', color=bin_colors[bin_size_ms], 
                       linewidth=2, markersize=6)
                
                # Add error bars
                ax.errorbar(data['bin_centers'], data['bin_scores'], 
                          yerr=data['bin_stds']/np.sqrt(args.n_iterations),
                          color=bin_colors[bin_size_ms], alpha=0.7, capsize=3)
                
                # Store peak info
                peak_idx = np.argmax(data['bin_scores'])
                peak_accuracies.append(data['bin_scores'][peak_idx])
                peak_times.append(data['bin_centers'][peak_idx])
        
        # Time course formatting
        ax.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7, label='Ping', linewidth=2)
        ax.axvline(x=CUE_TIME, color='green', linestyle='--', alpha=0.7, label='Cue', linewidth=2)
        ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Decoding Accuracy')
        ax.set_title(f'{feature_name.replace("maintained_", "").title()} Decoding')
        ax.set_ylim(0.45, 0.8)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Peak summary
        x_pos = np.arange(len(peak_accuracies))
        ax2.bar(x_pos, peak_accuracies, color=[bin_colors[bs] for bs in all_bin_results.keys()])
        ax2.set_xlabel('Bin Size')
        ax2.set_ylabel('Peak Accuracy')
        ax2.set_title(f'Peak Performance Summary')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{bs}ms' for bs in all_bin_results.keys()])
        ax2.set_ylim(0.5, 0.8)
        ax2.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add peak times as text
        for i, (acc, time) in enumerate(zip(peak_accuracies, peak_times)):
            ax2.text(i, acc + 0.01, f'{time:.2f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/results_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    write_log(f"\nSummary plot saved to {OUTPUT_DIR}/results_summary.png")

def main():
    """Main processing function"""
    # Load metadata
    meta_info = pd.read_excel(META_FILE)
    
    # Load subject data
    epochs_data, events, times, sfreq = load_subject_data(args.subject, meta_info)
    
    if epochs_data is None:
        write_log("Failed to load data. Exiting.")
        sys.exit(1)
    
    write_log("\n=== Temporal Binning Analysis ===")
    write_log(f"=== Testing bin sizes: {BIN_SIZES_MS}ms ===")
    
    # Results storage for all bin sizes
    all_bin_results = {}
    
    # Iterate through different bin sizes
    for bin_size_ms in BIN_SIZES_MS:
        write_log(f"\n{'='*60}")
        write_log(f"Processing with bin size: {bin_size_ms}ms")
        write_log(f"{'='*60}")
        
        # Results storage for this bin size
        bin_results = {}
        
        # Process each feature (voice and location)
        for feature_name in ['maintained_voice', 'maintained_location']:
            
            # Run decoding with current bin size
            bin_scores, bin_stds, bin_centers, all_iteration_scores, bin_c_values = decode_temporal_bins(
                epochs_data, times, events, feature_name, 
                bin_size_ms, args.n_iterations
            )
            
            # Store results
            bin_results[feature_name] = {
                'bin_scores': bin_scores,
                'bin_stds': bin_stds,
                'bin_centers': bin_centers,
                'all_iteration_scores': all_iteration_scores,
                'bin_c_values': bin_c_values,
                'bin_size_ms': bin_size_ms
            }
            
            write_log(f"  {feature_name}: Mean={np.mean(bin_scores):.3f}, Peak={np.max(bin_scores):.3f}")
        
        # Store results for this bin size
        all_bin_results[bin_size_ms] = bin_results
        
        # Save results for this bin size
        write_log(f"\nSaving results for {bin_size_ms}ms bins...")
        
        # Save results
        for feature_name, results in bin_results.items():
            # Save bin data
            np.save(f'{OUTPUT_DIR}/{feature_name}_{bin_size_ms}ms_bin_scores.npy', 
                   results['bin_scores'])
            np.save(f'{OUTPUT_DIR}/{feature_name}_{bin_size_ms}ms_bin_stds.npy', 
                   results['bin_stds'])
            np.save(f'{OUTPUT_DIR}/{feature_name}_{bin_size_ms}ms_bin_centers.npy', 
                   results['bin_centers'])
            np.save(f'{OUTPUT_DIR}/{feature_name}_{bin_size_ms}ms_all_iteration_scores.npy', 
                   results['all_iteration_scores'])
            np.save(f'{OUTPUT_DIR}/{feature_name}_{bin_size_ms}ms_c_values.npy', 
                   results['bin_c_values'])
    
    # Create summary plot
    plot_results_summary(all_bin_results)
    
    # Save comprehensive summary
    summary = {
        'subject': args.subject,
        'n_iterations': args.n_iterations,
        'bin_sizes_ms': BIN_SIZES_MS,
        'time_windows': TIME_WINDOWS,
        'features': ['maintained_voice', 'maintained_location'],
        'averaging_scheme': AVERAGING_SCHEME,
        'processing_time': str(datetime.now()),
        'method': 'temporal_binning_analysis',
        'normalization': 'channel_wise_z_scoring',
        'ping_time': PING_TIME,
        'results': {}
    }
    
    # Add detailed results for each bin size
    for bin_size_ms, bin_results in all_bin_results.items():
        summary['results'][f'bin_{bin_size_ms}ms'] = {}
        
        for feature_name, results in bin_results.items():
            # Find bin closest to ping time
            ping_idx = np.argmin(np.abs(results['bin_centers'] - PING_TIME))
            ping_accuracy = results['bin_scores'][ping_idx] if ping_idx < len(results['bin_scores']) else np.nan
            
            summary['results'][f'bin_{bin_size_ms}ms'][feature_name] = {
                'mean_accuracy': float(np.mean(results['bin_scores'])),
                'peak_accuracy': float(np.max(results['bin_scores'])),
                'peak_time': float(results['bin_centers'][np.argmax(results['bin_scores'])]),
                'ping_accuracy': float(ping_accuracy),
                'n_bins': len(results['bin_centers']),
                'bins_above_chance': int(np.sum(results['bin_scores'] > 0.5))
            }
    
    # Add comparison across bin sizes
    write_log(f"\n=== COMPARISON ACROSS BIN SIZES ===")
    for feature_name in ['maintained_voice', 'maintained_location']:
        write_log(f"\n{feature_name}:")
        for bin_size_ms in sorted(all_bin_results.keys()):
            mean_acc = np.mean(all_bin_results[bin_size_ms][feature_name]['bin_scores'])
            peak_acc = np.max(all_bin_results[bin_size_ms][feature_name]['bin_scores'])
            peak_time = all_bin_results[bin_size_ms][feature_name]['bin_centers'][
                np.argmax(all_bin_results[bin_size_ms][feature_name]['bin_scores'])]
            n_bins = len(all_bin_results[bin_size_ms][feature_name]['bin_centers'])
            write_log(f"  {bin_size_ms}ms: Mean={mean_acc:.3f}, Peak={peak_acc:.3f} @ {peak_time:.2f}s, Bins={n_bins}")
    
    with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    write_log(f"\nProcessing completed at: {datetime.now()}")
    
    print(f"Subject {args.subject} temporal binning analysis completed successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()