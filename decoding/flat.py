#!/usr/bin/env python
"""
Spatio-Temporal Delay Period Analysis Script
Implements the approach from Wolff et al. and similar papers:
- Extract time window → remove trial means → downsample → flatten space×time dimensions → decode

Usage: python spatiotemporal_ping_analysis.py --subject 23
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from mne.decoding import Vectorizer
from collections import defaultdict, Counter
from joblib import Parallel, delayed
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run spatio-temporal delay period analysis')
parser.add_argument('--subject', type=int, required=True, help='Subject ID')
parser.add_argument('--n_iterations', type=int, default=10, help='Number of iterations')
parser.add_argument('--n_jobs', type=int, default=20, help='Number of parallel jobs')
parser.add_argument('--tmin', type=float, default=2.0, help='Start time of analysis window')
parser.add_argument('--tmax', type=float, default=4.7, help='End time of analysis window')
args = parser.parse_args()

# Set paths for HPC
HOME_DIR = '/mnt/hpc/projects/awm4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
DATA_PATH = HOME_DIR + '/AWM4_data/raw/'
CORRECTED_DATA = HOME_DIR + '/AWM4_data/raw/correctTriggers/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'

# Create output directory
OUTPUT_DIR = PROCESSED_DIR + f'spatiotemporalAnalysis/subject_{args.subject}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log file path
LOG_FILE_PATH = f'{OUTPUT_DIR}/processing_log.txt'

def write_log(message):
    """Write message to log file"""
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(message + '\n')

# Initialize log
write_log(f"Spatio-temporal analysis started at: {datetime.now()}")
write_log(f"Subject: {args.subject}")
write_log(f"Analysis window: {args.tmin}s to {args.tmax}s")
write_log(f"Using spatio-temporal approach following Wolff et al.")

# Spatio-temporal parameters
SPATIOTEMPORAL_WINDOWS = [100, 400]  # ms - at 100ms, 5 timepoints and at 400ms 20 timepoints 
TARGET_SAMPLING_RATE = 50  # Hz - 20ms per sample
SPATIOTEMPORAL_STEP = 0.02  # 20ms step so 1 sample at 50Hz 
BASELINE_REMOVAL = 'per_trial_per_channel'  #normalizes temporal patterns within each trial/channel

# Important time markers
CUE_TIME = 2.0  # Cue presentation
PING_TIME = 3.5  # Ping occurs at 3.5s
PROBE_TIME = 4.7  # Probe presentation

# Full delay period for analysis
ANALYSIS_CONFIG = {
    'tmin': args.tmin,
    'tmax': args.tmax,
    'duration': args.tmax - args.tmin,
    'description': f'Full delay period ({args.tmin}-{args.tmax}s)'
}

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

# Conservative CV parameters
NUM_JOBS = args.n_jobs
PARAM_GRID = [0.1, 1.0, 10.0]

def apply_baseline_removal(window_data, method='per_trial_per_channel'):
    """
    Apply baseline removal to spatio-temporal data following the papers
    
    Parameters:
    window_data: array of shape (n_trials, n_channels, n_timepoints)
    method: 'per_trial_per_channel' (default, following papers)
    
    Returns:
    data_corrected: baseline-corrected data
    """
    if method == 'per_trial_per_channel':
        # Remove mean from each trial/channel combination separately
        data_corrected = window_data.copy()
        for trial in range(window_data.shape[0]):
            for channel in range(window_data.shape[1]):
                trial_channel_mean = np.mean(window_data[trial, channel, :])
                data_corrected[trial, channel, :] -= trial_channel_mean
        return data_corrected
    else:
        raise ValueError(f"Method {method} not implemented. Use 'per_trial_per_channel'")

def create_spatiotemporal_features(window_data, target_hz=None, current_hz=None):
    """
    Create spatio-temporal features - simplified version since we're already at 50Hz
    1. Remove trial means per channel
    2. Flatten space×time dimensions (no downsampling needed!)
    
    Parameters:
    window_data: array of shape (n_trials, n_channels, n_timepoints) already at 50Hz
    target_hz: ignored (kept for compatibility)
    current_hz: ignored (kept for compatibility)
    
    Returns:
    X_spatiotemporal: flattened spatio-temporal feature matrix
    """
    # Step 1: Remove baseline per trial per channel (following papers)
    data_demeaned = apply_baseline_removal(window_data, BASELINE_REMOVAL)
    
    # Step 2: No downsampling needed - we're already at 50Hz!
    # This makes the pipeline much cleaner
    
    # Step 3: Flatten space×time dimensions
    n_trials, n_channels, n_times = data_demeaned.shape
    X_spatiotemporal = data_demeaned.reshape(n_trials, n_channels * n_times)
    
    return X_spatiotemporal

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

def decode_spatiotemporal_sliding_window(epochs_data, events, sfreq, feature_name, n_iterations, window_length_ms):
    """
    Decode using spatio-temporal approach with sliding windows
    Now simplified since we're already at 50Hz!
    """
    write_log(f"\n  Spatio-temporal decoding: {feature_name}")
    write_log(f"  Window length: {window_length_ms}ms")
    write_log(f"  Data already at target sampling rate: {sfreq}Hz")
    
    # Convert window length to samples
    window_length_sec = window_length_ms / 1000.0
    window_length_samples = int(sfreq * window_length_sec)
    window_step_samples = int(sfreq * SPATIOTEMPORAL_STEP)
    
    # Calculate number of windows - THIS WAS MISSING!
    n_times = epochs_data.shape[2]
    n_windows = int((n_times - window_length_samples) / window_step_samples) + 1
    
    # Calculate expected features (much simpler now!)
    expected_timepoints = int(window_length_ms * sfreq / 1000)
    n_channels = epochs_data.shape[1]
    expected_features = n_channels * expected_timepoints
    
    write_log(f"  Data shape: {epochs_data.shape}")
    write_log(f"  Number of sliding windows: {n_windows}")
    write_log(f"  Timepoints per window: {expected_timepoints}")
    write_log(f"  Features per window: {expected_features}")
    
    # Storage for results across iterations
    iteration_results = []
    
    for iteration in range(n_iterations):
        write_log(f"    Iteration {iteration + 1}/{n_iterations}")
        
        # Results for this iteration
        window_scores = np.zeros(n_windows)  # Now n_windows is properly defined
        window_c_values = []
        
        # Process each sliding window
        for window_idx in tqdm(range(n_windows), desc=f"{feature_name} {window_length_ms}ms windows"):
            win_start = window_idx * window_step_samples
            win_end = win_start + window_length_samples
            
            # Extract window data
            window_data = epochs_data[:, :, win_start:win_end]
            
            # Create spatio-temporal features (simplified!)
            X_spatiotemporal = create_spatiotemporal_features(window_data)
                        
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
                window_scores[window_idx] = 0.5
                window_c_values.append(1.0)
                continue
            
            # Outer CV loop
            for train_idx, test_idx in outer_cv.split(X_spatiotemporal, original_labels):
                
                # Step 1: Split original trials
                train_data = X_spatiotemporal[train_idx]
                test_data = X_spatiotemporal[test_idx]
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
                
                # Step 3: Inner CV for hyperparameter selection
                best_score = -1
                best_c = 1.0
                
                # Use simple 3-fold CV for inner loop
                inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=None)
                
                for c_value in PARAM_GRID:
                    # Create pipeline with feature-wise normalization
                    clf = make_pipeline(
                        StandardScaler(),  # Feature-wise normalization
                        SVC(kernel='linear', C=c_value, probability=True)
                    )
                    
                    # Evaluate on inner CV
                    try:
                        inner_scores = cross_val_score(
                            clf, X_train, y_train, 
                            cv=inner_cv, 
                            scoring='accuracy',
                            n_jobs=1
                        )
                        mean_inner_score = np.mean(inner_scores)
                        
                        if mean_inner_score > best_score:
                            best_score = mean_inner_score
                            best_c = c_value
                    except:
                        continue
                
                # Step 4: Train final model with best C and evaluate on test set
                final_clf = make_pipeline(
                    StandardScaler(),  # Feature-wise normalization
                    SVC(kernel='linear', C=best_c, probability=True)
                )
                
                try:
                    final_clf.fit(X_train, y_train)
                    test_score = final_clf.score(X_test, y_test)
                    outer_scores.append(test_score)
                    best_cs.append(best_c)
                except:
                    continue
            
            # Store results for this window
            if outer_scores:
                window_scores[window_idx] = np.mean(outer_scores)
                
                # Use most common C
                c_counter = Counter(best_cs)
                most_common_c = c_counter.most_common(1)[0][0] if best_cs else 1.0
                window_c_values.append(most_common_c)
            else:
                window_scores[window_idx] = 0.5
                window_c_values.append(1.0)
        
        # Store iteration results
        iteration_results.append({
            'scores': window_scores,
            'c_values': window_c_values
        })
    
    # Aggregate results across iterations
    all_scores = np.array([r['scores'] for r in iteration_results])
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)
    
    # Create time points (center of each window)
    timepoints = np.array([ANALYSIS_CONFIG['tmin'] + (i * SPATIOTEMPORAL_STEP) + (window_length_sec / 2) 
                          for i in range(n_windows)])
    
    write_log(f"    Mean accuracy: {np.mean(mean_scores):.3f}")
    write_log(f"    Peak accuracy: {np.max(mean_scores):.3f}")
    write_log(f"    Peak time: {timepoints[np.argmax(mean_scores)]:.3f}s")
    
    # Find accuracy at ping time
    ping_idx = np.argmin(np.abs(timepoints - PING_TIME))
    ping_accuracy = mean_scores[ping_idx]
    write_log(f"    Accuracy at ping ({PING_TIME}s): {ping_accuracy:.3f}")
    
    return mean_scores, std_scores, timepoints, all_scores, iteration_results

def load_subject_data(subject, meta_info):
    """Load and preprocess data for the full delay period analysis"""
    write_log(f"\nLoading data for subject {subject}...")
    
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
        delay_epochs.crop(tmin=ANALYSIS_CONFIG['tmin'], tmax=ANALYSIS_CONFIG['tmax'])
        
        # Select ONLY magnetometers and resample
        mag_epochs = delay_epochs.copy().pick_types(meg='mag')
        mag_epochs = mag_epochs.resample(TARGET_SAMPLING_RATE, npad='auto')
        
        # Get data
        epochs_data = mag_epochs.get_data(copy=False)
        maintained_events = mag_epochs.events[:, 2]
        
        write_log(f"Data loaded successfully. Shape: {epochs_data.shape}")
        write_log(f"Number of magnetometers: {epochs_data.shape[1]}")
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
        
        return epochs_data, maintained_events, mag_epochs.info['sfreq']
        
    except Exception as e:
        write_log(f"Error loading data: {str(e)}")
        import traceback
        write_log(traceback.format_exc())
        return None, None, None

def plot_spatiotemporal_results(all_window_results):
    """Create a summary plot of spatio-temporal results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Subject {args.subject} - Spatio-Temporal Analysis', fontsize=16)
    
    # Colors for different window sizes
    window_colors = {100: '#e41a1c', 400: '#377eb8'}
    
    for idx, feature_name in enumerate(['maintained_voice', 'maintained_location']):
        ax = axes[idx, 0]  # Time courses
        ax2 = axes[idx, 1]  # Peak summary
        
        peak_accuracies = []
        peak_times = []
        
        for window_ms, results in all_window_results.items():
            if feature_name in results:
                data = results[feature_name]
                
                # Plot time course
                ax.plot(data['timepoints'], data['mean_scores'], 
                       label=f'{window_ms}ms spatio-temporal', 
                       color=window_colors[window_ms], 
                       linewidth=2.5)
                
                # Add standard error
                ax.fill_between(data['timepoints'],
                              data['mean_scores'] - data['std_scores']/np.sqrt(args.n_iterations),
                              data['mean_scores'] + data['std_scores']/np.sqrt(args.n_iterations),
                              color=window_colors[window_ms], alpha=0.2)
                
                # Store peak info
                peak_idx = np.argmax(data['mean_scores'])
                peak_accuracies.append(data['mean_scores'][peak_idx])
                peak_times.append(data['timepoints'][peak_idx])
        
        # Time course formatting
        ax.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7, label='Ping')
        ax.axvline(x=CUE_TIME, color='green', linestyle='--', alpha=0.7, label='Cue')
        ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Decoding Accuracy')
        ax.set_title(f'{feature_name.replace("maintained_", "").title()} - Spatio-Temporal')
        ax.set_ylim(0.45, 0.75)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Peak summary
        x_pos = np.arange(len(peak_accuracies))
        ax2.bar(x_pos, peak_accuracies, color=[window_colors[w] for w in all_window_results.keys()])
        ax2.set_xlabel('Spatio-Temporal Window')
        ax2.set_ylabel('Peak Accuracy')
        ax2.set_title(f'Peak Performance Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{w}ms' for w in all_window_results.keys()])
        ax2.set_ylim(0.5, 0.75)
        ax2.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add peak times as text
        for i, (acc, time) in enumerate(zip(peak_accuracies, peak_times)):
            ax2.text(i, acc + 0.01, f'{time:.2f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/spatiotemporal_results_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    write_log(f"\nSummary plot saved to {OUTPUT_DIR}/spatiotemporal_results_summary.png")

def main():
    """Main processing function"""
    # Load metadata
    meta_info = pd.read_excel(META_FILE)
    
    # Load subject data
    epochs_data, events, sfreq = load_subject_data(args.subject, meta_info)
    
    if epochs_data is None:
        write_log("Failed to load data. Exiting.")
        sys.exit(1)
    
    # Verify we got the expected sampling rate
    write_log(f"Loaded data sampling rate: {sfreq}Hz (expected: {TARGET_SAMPLING_RATE}Hz)")
    if abs(sfreq - TARGET_SAMPLING_RATE) > 0.1:
        write_log(f"WARNING: Sampling rate mismatch!")
    
    write_log("\n=== Spatio-Temporal Analysis (Direct 50Hz Resampling) ===")
    write_log(f"=== Window lengths: {SPATIOTEMPORAL_WINDOWS}ms ===")
    write_log(f"=== Already at target rate: {TARGET_SAMPLING_RATE}Hz (no further downsampling needed) ===")
    write_log(f"=== Baseline removal: {BASELINE_REMOVAL} ===")
    
    # Results storage for all window lengths
    all_window_results = {}
    
    # Iterate through different window lengths
    for window_length_ms in SPATIOTEMPORAL_WINDOWS:
        write_log(f"\n{'='*60}")
        write_log(f"Processing spatio-temporal window: {window_length_ms}ms")
        write_log(f"{'='*60}")
        
        # Calculate expected features (updated calculation)
        timepoints_per_channel = int(window_length_ms * TARGET_SAMPLING_RATE / 1000)
        n_channels = epochs_data.shape[1]
        expected_features = n_channels * timepoints_per_channel
        write_log(f"Expected features: {n_channels} channels × {timepoints_per_channel} timepoints = {expected_features}")
        
        # Results storage for this window length
        window_results = {}
        
        # Process each feature (voice and location)
        for feature_name in ['maintained_voice', 'maintained_location']:
            
            # Run spatio-temporal decoding
            mean_scores, std_scores, timepoints, all_scores, iteration_results = decode_spatiotemporal_sliding_window(
                epochs_data, events, sfreq, feature_name, 
                args.n_iterations, window_length_ms
            )
            
            # Store results
            window_results[feature_name] = {
                'mean_scores': mean_scores,
                'std_scores': std_scores,
                'timepoints': timepoints,
                'all_scores': all_scores,
                'iteration_results': iteration_results,
                'window_length_ms': window_length_ms,
                'n_features': expected_features
            }
            
            write_log(f"  {feature_name}: Mean={np.mean(mean_scores):.3f}, Peak={np.max(mean_scores):.3f}")
        
        # Store results for this window length
        all_window_results[window_length_ms] = window_results
        
        # Save results for this window length
        write_log(f"\nSaving results for {window_length_ms}ms spatio-temporal window...")
        
        # Save temporal results with window length in filename
        for feature_name, results in window_results.items():
            # Save temporal data
            np.save(f'{OUTPUT_DIR}/spatiotemporal_{feature_name}_{window_length_ms}ms_mean_scores.npy', 
                   results['mean_scores'])
            np.save(f'{OUTPUT_DIR}/spatiotemporal_{feature_name}_{window_length_ms}ms_std_scores.npy', 
                   results['std_scores'])
            np.save(f'{OUTPUT_DIR}/spatiotemporal_{feature_name}_{window_length_ms}ms_timepoints.npy', 
                   results['timepoints'])
            np.save(f'{OUTPUT_DIR}/spatiotemporal_{feature_name}_{window_length_ms}ms_all_scores.npy', 
                   results['all_scores'])
            np.save(f'{OUTPUT_DIR}/spatiotemporal_{feature_name}_{window_length_ms}ms_c_values.npy', 
                   [r['c_values'] for r in results['iteration_results']])
    
    # Create summary plot
    plot_spatiotemporal_results(all_window_results)
    
    # Save comprehensive summary
    summary = {
        'subject': args.subject,
        'n_iterations': args.n_iterations,
        'analysis_window': f"{ANALYSIS_CONFIG['tmin']}-{ANALYSIS_CONFIG['tmax']}s",
        'features': ['maintained_voice', 'maintained_location'],
        'spatiotemporal_windows_ms': SPATIOTEMPORAL_WINDOWS,
        'window_step_ms': SPATIOTEMPORAL_STEP * 1000,
        'baseline_removal': BASELINE_REMOVAL,
        'averaging_scheme': AVERAGING_SCHEME,
        'processing_time': str(datetime.now()),
        'method': 'spatiotemporal_sliding_window_analysis',
        'normalization': 'StandardScaler (feature-wise)',
        'sensor_type': 'magnetometers_only',
        'n_channels': epochs_data.shape[1],
        'ping_time': PING_TIME,
        'results': {}
    }
    
    # Add detailed results for each window length
    for window_length_ms, window_results in all_window_results.items():
        summary['results'][f'spatiotemporal_{window_length_ms}ms'] = {}
        
        for feature_name, results in window_results.items():
            # Find ping time accuracy
            ping_idx = np.argmin(np.abs(results['timepoints'] - PING_TIME))
            ping_accuracy = results['mean_scores'][ping_idx]
            
            summary['results'][f'spatiotemporal_{window_length_ms}ms'][feature_name] = {
                'mean_accuracy': float(np.mean(results['mean_scores'])),
                'peak_accuracy': float(np.max(results['mean_scores'])),
                'peak_time': float(results['timepoints'][np.argmax(results['mean_scores'])]),
                'ping_accuracy': float(ping_accuracy),
                'n_timepoints': len(results['timepoints']),
                'n_features': results['n_features'],
                'time_above_chance': float(np.sum(results['mean_scores'] > 0.5) * SPATIOTEMPORAL_STEP)
            }
    
    # Add comparison across window lengths
    write_log(f"\n=== COMPARISON ACROSS SPATIO-TEMPORAL WINDOWS ===")
    for feature_name in ['maintained_voice', 'maintained_location']:
        write_log(f"\n{feature_name}:")
        for window_length_ms in sorted(all_window_results.keys()):
            mean_acc = np.mean(all_window_results[window_length_ms][feature_name]['mean_scores'])
            peak_acc = np.max(all_window_results[window_length_ms][feature_name]['mean_scores'])
            peak_time = all_window_results[window_length_ms][feature_name]['timepoints'][
                np.argmax(all_window_results[window_length_ms][feature_name]['mean_scores'])]
            n_features = all_window_results[window_length_ms][feature_name]['n_features']
            write_log(f"  {window_length_ms}ms: Mean={mean_acc:.3f}, Peak={peak_acc:.3f} @ {peak_time:.2f}s, Features={n_features}")
    
    with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    write_log(f"\nProcessing completed at: {datetime.now()}")
    
    print(f"Subject {args.subject} spatio-temporal analysis completed successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()