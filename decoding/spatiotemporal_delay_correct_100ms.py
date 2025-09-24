#!/usr/bin/env python
"""
Delay Period Pseudotrial Sliding Window Analysis Script - FIXED VERSION
Analyzes maintained information during delay period (2.0-4.5s) using:
- 400ms sliding windows with 100ms steps (CHANGED from 10ms)
- Wolff-style feature extraction (10ms averaging, per-trial mean-centering)
- avg12 pseudotrial scheme (3 trials per condition, systematic selection)
- Nested cross-validation with proper data leakage prevention
- PCA dimensionality reduction (90% variance) per window per fold
- Integration with behavioral correctness data
- FIXED: Sampling rate consistency issues

Usage: python spatiotemporal_delay_correct.py --subject 23
"""

import os
import sys
import argparse
from tqdm import tqdm
import numpy as np
import mne
mne.set_log_level('WARNING')
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
parser = argparse.ArgumentParser(description='Run delay period pseudotrial sliding window analysis')
parser.add_argument('--subject', type=int, required=True, help='Subject ID')
parser.add_argument('--n_iterations', type=int, default=10, help='Number of iterations')
parser.add_argument('--n_jobs', type=int, default=20, help='Number of parallel jobs')
parser.add_argument('--correct_only', action='store_true', help='Use only behaviorally correct trials')
args = parser.parse_args()

# Set paths for HPC
HOME_DIR = '/mnt/hpc/projects/awm4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
DATA_PATH = HOME_DIR + '/AWM4_data/raw/'
CORRECTED_DATA = HOME_DIR + '/AWM4_data/raw/correctTriggers/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
BEHAVIORAL_PATH = HOME_DIR + '/AWM4_data/behavioral/'

# Create output directory
OUTPUT_DIR = PROCESSED_DIR + f'spatiotemporal_delay_400/200Hz/subject_{args.subject}/'
if args.correct_only:
    OUTPUT_DIR = PROCESSED_DIR + f'spatiotemporal_delay_400_correct/200ms200Hz/subject_{args.subject}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log file path
LOG_FILE_PATH = f'{OUTPUT_DIR}/processing_log.txt'

def write_log(message):
    """Write message to log file"""
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(message + '\n')

# Initialize log
write_log(f"Delay period pseudotrial analysis started at: {datetime.now()}")
write_log(f"Subject: {args.subject}")
write_log(f"Analysis window: 2.0-4.5s (delay period)")
write_log(f"Sliding windows: 400ms length, 100ms step") # CHANGED from 10ms
write_log(f"Pseudotrial scheme: avg12 (3 trials per condition)")
write_log(f"Correct trials only: {args.correct_only}")
write_log(f"FIXED VERSION: Sampling rate consistency ensured")

# Analysis parameters - FIXED AND UPDATED
RESAMPLE_FREQ = 200  # Hz - This is the definitive sampling rate
WINDOW_LENGTH_SEC = 0.4  # 400ms windows
WINDOW_STEP_SEC = 0.2   # 100ms steps (CHANGED from 0.01)
DELAY_START = 2.0  # Start of delay period
DELAY_END = 4.5    # End of delay period
EPOCH_TMIN = 1.6   # Epoch start (with buffer)
EPOCH_TMAX = 4.9   # Epoch end (with buffer)

# Cross-validation parameters
OUTER_CV_SPLITS = 5
INNER_CV_SPLITS = 3
PCA_VARIANCE_THRESHOLD = 0.90

# Hyperparameter grid
PARAM_GRID = {'svc__C': [0.001, 0.01, 0.1]}

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

# Define the four corners (including both S1 and S2)
CORNERS = {
    'Corner1_Sp12L12': [111, 112, 121, 122, 211, 212, 221, 222],
    'Corner2_Sp12L34': [113, 114, 123, 124, 213, 214, 223, 224],
    'Corner3_Sp34L12': [131, 132, 141, 142, 231, 232, 241, 242],
    'Corner4_Sp34L34': [133, 134, 143, 144, 233, 234, 243, 244]
}

# Pseudotrial parameters
TRIALS_PER_CONDITION = 3
TOTAL_TRIALS = 12

def get_subject_mapping():
    """Create mapping between MEG subject numbers and behavioral VP codes"""
    metaInfo = pd.read_excel(META_FILE)
    
    # Get MEG subjects (raw numbers) 
    meg_subjects = np.unique(metaInfo.loc[metaInfo.FinalSample==1, 'Subject']).astype(int)
    
    # Check what behavioral files actually exist
    behavioral_files = [f for f in os.listdir(BEHAVIORAL_PATH) if f.startswith('Output_AWM4_Exp1_MEG_VP') and f.endswith('.txt')]
    actual_vp_codes = []
    for f in behavioral_files:
        vp_code = f.split('_')[-1].replace('.txt', '')  # Extract 'VP53' etc.
        actual_vp_codes.append(vp_code)
    actual_vp_codes = sorted(list(set(actual_vp_codes)))
    
    # Create subject mapping dictionary
    subject_mapping = {}
    
    # Direct mapping from metaInfo
    for _, row in metaInfo.iterrows():
        if pd.notna(row['Subject']) and pd.notna(row['SubjectCode']):
            meg_subject = int(row['Subject'])
            if 'S' in str(row['SubjectCode']):
                vp_code = 'VP' + str(row['SubjectCode']).replace('S', '')
                if vp_code in actual_vp_codes and meg_subject in meg_subjects:
                    subject_mapping[meg_subject] = vp_code
    
    return subject_mapping

def load_and_clean_behavioral_data(meg_subject):
    """
    Load and clean behavioral data for one subject
    Handle block restarts and extract correctness information
    meg_subject: MEG subject number (int)
    """
    # Get the correct VP code for this MEG subject
    subject_mapping = get_subject_mapping()
    
    if meg_subject not in subject_mapping:
        write_log(f"ERROR: MEG subject {meg_subject} not found in subject mapping")
        write_log(f"Available MEG subjects: {sorted(subject_mapping.keys())}")
        return None
    
    vp_code = subject_mapping[meg_subject]
    write_log(f"\nLoading behavioral data: MEG {meg_subject} → {vp_code}...")
    
    try:
        file_path = os.path.join(BEHAVIORAL_PATH, f'Output_AWM4_Exp1_MEG_{vp_code}.txt')
        if not os.path.exists(file_path):
            write_log(f"WARNING: Behavioral file not found for {vp_code}: {file_path}")
            return None
            
        # Read space-separated file, skip header line  
        data = pd.read_csv(file_path, sep=' ', skiprows=1, header=None)
        
        write_log(f"Behavioral data shape: {data.shape}")
        
        if data.shape[1] < 14:
            write_log(f"ERROR: Expected at least 14 columns, found {data.shape[1]}")
            return None
        
        # Extract key columns (0-indexed)
        blocks = data.iloc[:, 0].values      # Block
        trials = data.iloc[:, 1].values      # Trial within block 
        cumulative = data.iloc[:, 2].values  # Cumulative trial counter
        correct = data.iloc[:, 13].values    # Correct(1)/Incorrect(0)/TooSlow(2)
        
        # Clean data by removing pre-restart sequences
        cleaned_indices = []
        restart_info = {}
        
        for block_num in np.unique(blocks):
            block_mask = blocks == block_num
            block_trials = trials[block_mask]
            block_indices = np.where(block_mask)[0]
            
            # Find restart points: where trial number decreases
            restart_points = []
            for i in range(1, len(block_trials)):
                if block_trials[i] < block_trials[i-1]:
                    restart_points.append(i)
            
            if restart_points:
                # Keep only after the LAST restart
                last_restart = restart_points[-1]
                keep_indices = block_indices[last_restart:]
                restart_info[block_num] = {
                    'had_restart': True,
                    'trials_dropped': last_restart,
                    'trials_kept': len(keep_indices)
                }
                write_log(f"  Block {block_num}: restart detected, keeping {len(keep_indices)} trials")
            else:
                # No restart, keep all
                keep_indices = block_indices
                restart_info[block_num] = {'had_restart': False}
            
            cleaned_indices.extend(keep_indices)
        
        # Apply cleaning
        cleaned_indices = np.array(cleaned_indices)
        clean_correct = correct[cleaned_indices]
        
        # Log statistics
        n_total = len(clean_correct)
        n_correct = np.sum(clean_correct == 1)
        n_incorrect = np.sum(clean_correct == 0)
        n_too_slow = np.sum(clean_correct == 2)
        
        write_log(f"Cleaned behavioral data: {n_total} trials")
        write_log(f"Correct: {n_correct}, Incorrect: {n_incorrect}, Too slow: {n_too_slow}")
        write_log(f"Accuracy (excl. too slow): {n_correct/(n_correct+n_incorrect):.3f}")
        
        if any(info.get('had_restart', False) for info in restart_info.values()):
            write_log(f"Had block restarts: {sum(info.get('trials_dropped', 0) for info in restart_info.values())} trials dropped")
        
        return clean_correct
        
    except Exception as e:
        write_log(f"Error loading behavioral data: {str(e)}")
        import traceback
        write_log(traceback.format_exc())
        return None

def extract_maintained_information(subject, metaInfo):
    """Extract maintained information using retro-cue approach"""
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

def create_balanced_pseudotrials(epochs_data, events, corner_conditions, trials_per_condition):
    """
    Create balanced pseudotrials using systematic selection
    Ensures no trial is used twice and balances across conditions
    """
    # Organize trials by condition
    condition_trials = defaultdict(list)
    for idx, event in enumerate(events):
        if event in corner_conditions:
            condition_trials[event].append(idx)
    
    # Check available conditions
    valid_conditions = [cond for cond in corner_conditions 
                       if cond in condition_trials and len(condition_trials[cond]) >= trials_per_condition]
    
    if len(valid_conditions) == 0:
        return None
    
    # Balance across conditions - use minimum available trials
    min_available = min(len(condition_trials[cond]) for cond in valid_conditions)
    max_pseudotrials_per_condition = min_available // trials_per_condition
    
    if max_pseudotrials_per_condition == 0:
        return None
    
    # Create pseudotrials
    pseudotrials = []
    
    for pseudo_idx in range(max_pseudotrials_per_condition):
        # Collect trials from each valid condition
        trials_to_average = []
        
        for condition in valid_conditions:
            # Take systematic slice of trials (no random sampling)
            start_idx = pseudo_idx * trials_per_condition
            end_idx = start_idx + trials_per_condition
            condition_trial_indices = condition_trials[condition][start_idx:end_idx]
            
            # Add to averaging pool
            for trial_idx in condition_trial_indices:
                trials_to_average.append(epochs_data[trial_idx])
        
        # Average across all collected trials
        if len(trials_to_average) > 0:
            pseudotrial = np.mean(trials_to_average, axis=0)
            pseudotrials.append(pseudotrial)
    
    return np.array(pseudotrials) if pseudotrials else None

def extract_wolff_features(window_data):
    """
    Extract Wolff-style features from sliding window:
    1. Downsample by averaging every 10ms 
    2. Mean-center per trial per channel within the window
    3. Flatten to feature vector
    """
    n_trials, n_channels, n_timepoints = window_data.shape
    
    # Calculate samples per 10ms bin using the consistent sampling rate
    samples_per_bin = int(RESAMPLE_FREQ * 0.01)  # 10ms in samples (should be 5 at 500 Hz)
    n_bins = n_timepoints // samples_per_bin
    
    # Use all complete bins available (flexible for different window sizes)
    if n_bins < 1:
        raise ValueError(f"Window too short: only {n_timepoints} samples, need at least {samples_per_bin}")
    
    # Truncate to exact number of complete bins we can create
    usable_samples = n_bins * samples_per_bin
    if usable_samples < n_timepoints:
        window_data = window_data[:, :, :usable_samples]
    
    # Downsample by averaging within 10ms bins
    downsampled_data = np.zeros((n_trials, n_channels, n_bins))
    
    for bin_idx in range(n_bins):
        start_sample = bin_idx * samples_per_bin
        end_sample = start_sample + samples_per_bin
        downsampled_data[:, :, bin_idx] = np.mean(window_data[:, :, start_sample:end_sample], axis=2)
    
    # Mean-center per trial per channel within the window
    mean_centered_data = downsampled_data.copy()
    for trial_idx in range(n_trials):
        for channel_idx in range(n_channels):
            channel_mean = np.mean(downsampled_data[trial_idx, channel_idx, :])
            mean_centered_data[trial_idx, channel_idx, :] -= channel_mean
    
    # Flatten to feature vectors (channels × timepoints)
    features = mean_centered_data.reshape(n_trials, n_channels * n_bins)
    
    return features

def decode_sliding_window_with_pseudotrials(epochs_data, events, RESAMPLE_FREQ, feature_name, n_iterations):
    """
    Perform sliding window decoding with pseudotrials and nested cross-validation
    FIXED VERSION: Ensures sampling rate consistency and proper window calculation
    """
    write_log(f"\n  Decoding {feature_name} with sliding windows...")
        
    # Calculate window parameters using consistent sampling rate
    window_samples = int(RESAMPLE_FREQ * WINDOW_LENGTH_SEC)  # 200 samples at 500 Hz
    step_samples = int(RESAMPLE_FREQ * WINDOW_STEP_SEC)      # 50 samples at 500 Hz for 100ms steps

    write_log(f"  DEBUG: Window parameters:")
    write_log(f"    Window length: {WINDOW_LENGTH_SEC}s = {window_samples} samples")
    write_log(f"    Step size: {WINDOW_STEP_SEC}s = {step_samples} samples")
    
    delay_start_sample = int((DELAY_START - EPOCH_TMIN) * RESAMPLE_FREQ + 0.5)  # 200 (rounded)
    delay_end_sample = int((DELAY_END - EPOCH_TMIN) * RESAMPLE_FREQ + 0.5)      # 1450


    # Right-aligned windows: each window covers the previous 400 ms, ending at the timepoint
    first_window_end = delay_start_sample   
    last_window_end = delay_end_sample                       # 1450

    window_ends = list(range(first_window_end, last_window_end + 1, step_samples))
    
    # VERIFICATION: Check window calculation
    n_windows = len(window_ends)
    expected_duration = DELAY_END - DELAY_START  # 2.5s
    expected_windows = int(expected_duration / WINDOW_STEP_SEC) + 1  # 26 windows for 100ms steps
    actual_last_time = EPOCH_TMIN + (window_ends[-1] / RESAMPLE_FREQ)
    actual_first_time = EPOCH_TMIN + (window_ends[0] / RESAMPLE_FREQ)

    write_log(f"    Expected duration: {expected_duration}s")
    write_log(f"    Expected windows: {expected_windows}")
    write_log(f"    Actual windows: {n_windows}")
    write_log(f"    First window ends at: {EPOCH_TMIN + (window_ends[0] / RESAMPLE_FREQ):.3f}s")
    write_log(f"    Last window ends at: {actual_last_time:.3f}s")
    write_log(f"    Target end: {DELAY_END:.3f}s")
    
    # Verify we end at the right time
    time_error = abs(actual_last_time - DELAY_END)
    if time_error > 0.01:  # More than 10ms error
        write_log(f"  WARNING: Window calculation error! Ends at {actual_last_time:.3f}s, not {DELAY_END:.3f}s")
    else:
        write_log(f"  ✓ Window calculation verified: ends exactly at {DELAY_END:.3f}s")
    
    write_log(f"    Number of windows: {n_windows}")
    write_log(f"    Window coverage: {DELAY_START:.3f}s to {DELAY_END:.3f}s")
    
    # Storage for results across iterations
    iteration_results = []
    
    for iteration in range(n_iterations):
        write_log(f"      Iteration {iteration + 1}/{n_iterations}")
        
        window_scores = np.zeros(n_windows)
        window_pca_ratios = np.zeros(n_windows)
        
        # Process each window
        for window_idx in tqdm(range(n_windows), desc=f"{feature_name} windows"):
            win_end = window_ends[window_idx]
            win_start = win_end - window_samples
            
            if window_idx < 3:  # Debug first 3 windows
                write_log(f"    DEBUG Window {window_idx}:")
                write_log(f"      win_start: {win_start}")
                write_log(f"      win_end: {win_end}")
                write_log(f"      epochs_data.shape: {epochs_data.shape}")
                write_log(f"      window_samples: {window_samples}")
                write_log(f"      win_start >= 0: {win_start >= 0}")
                write_log(f"      win_end <= data_length: {win_end <= epochs_data.shape[2]}")
		
            # Extract window data
            window_data = epochs_data[:, :, win_start:win_end]
            
            # Extract Wolff-style features (uses RESAMPLE_FREQ internally)
            features = extract_wolff_features(window_data)
            
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
                window_pca_ratios[window_idx] = 0.0
                continue
            
            # Outer cross-validation
            outer_cv = StratifiedKFold(n_splits=OUTER_CV_SPLITS, shuffle=True)
            outer_scores = []
            pca_ratios = []
            
            for train_idx, test_idx in outer_cv.split(features, original_labels):
                # Split original trials
                train_data = features[train_idx]
                test_data = features[test_idx]
                train_events = events[train_idx]
                test_events = events[test_idx]
                
                # Create pseudotrials separately for train and test
                train_pseudo_data = []
                train_pseudo_labels = []
                test_pseudo_data = []
                test_pseudo_labels = []
                
                for corner_name, corner_conditions in CORNERS.items():
                    # Create training pseudotrials
                    train_indices = np.arange(len(train_events))
                    train_corner_data = train_data[np.isin(train_events, corner_conditions)]
                    train_corner_events = train_events[np.isin(train_events, corner_conditions)]
                    
                    if len(train_corner_data) > 0:
                        train_pseudo = create_balanced_pseudotrials(
                            train_corner_data, train_corner_events,
                            corner_conditions, TRIALS_PER_CONDITION
                        )
                        
                        if train_pseudo is not None:
                            train_pseudo_data.append(train_pseudo)
                            # Assign labels based on corner
                            if feature_name == 'maintained_voice':
                                labels = [0 if 'Sp12' in corner_name else 1] * len(train_pseudo)
                            else:
                                labels = [0 if 'L12' in corner_name else 1] * len(train_pseudo)
                            train_pseudo_labels.extend(labels)
                    
                    # Create test pseudotrials
                    test_indices = np.arange(len(test_events))
                    test_corner_data = test_data[np.isin(test_events, corner_conditions)]
                    test_corner_events = test_events[np.isin(test_events, corner_conditions)]
                    
                    if len(test_corner_data) > 0:
                        test_pseudo = create_balanced_pseudotrials(
                            test_corner_data, test_corner_events,
                            corner_conditions, TRIALS_PER_CONDITION
                        )
                        
                        if test_pseudo is not None:
                            test_pseudo_data.append(test_pseudo)
                            # Assign labels based on corner
                            if feature_name == 'maintained_voice':
                                labels = [0 if 'Sp12' in corner_name else 1] * len(test_pseudo)
                            else:
                                labels = [0 if 'L12' in corner_name else 1] * len(test_pseudo)
                            test_pseudo_labels.extend(labels)
                
                # Skip if insufficient data
                if not train_pseudo_data or not test_pseudo_data:
                    continue
                
                # Combine pseudotrials
                X_train = np.vstack(train_pseudo_data)
                y_train = np.array(train_pseudo_labels)
                X_test = np.vstack(test_pseudo_data)
                y_test = np.array(test_pseudo_labels)
                
                # Skip if insufficient classes
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    continue
                
                # Z-scoring (fit on train, apply to both)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # PCA (fit on train, apply to both)
                pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, svd_solver='full')
                try:
                    X_train_pca = pca.fit_transform(X_train_scaled)
                    X_test_pca = pca.transform(X_test_scaled)
                    
                    # Log PCA ratio
                    n_components_kept = pca.n_components_
                    n_original_features = X_train.shape[1]
                    pca_ratio = n_components_kept / n_original_features
                    pca_ratios.append(pca_ratio)
                    
                except Exception as e:
                    # PCA failed, skip this fold
                    continue
                
                # Inner CV for hyperparameter selection
                inner_cv = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True)
                grid_search = GridSearchCV(
                    SVC(kernel='linear', probability=True),
                    {'C': PARAM_GRID['svc__C']},
                    cv=inner_cv,
                    scoring='accuracy'
                )
                
                try:
                    grid_search.fit(X_train_pca, y_train)
                    test_score = grid_search.score(X_test_pca, y_test)
                    outer_scores.append(test_score)
                except Exception as e:
                    continue
            
            # Store results for this window
            if outer_scores and pca_ratios:
                window_scores[window_idx] = np.mean(outer_scores)
                window_pca_ratios[window_idx] = np.mean(pca_ratios)
            else:
                window_scores[window_idx] = 0.5
                window_pca_ratios[window_idx] = 0.0
        
        # Store iteration results
        iteration_results.append({
            'scores': window_scores,
            'pca_ratios': window_pca_ratios
        })
    
    # Aggregate results across iterations
    all_scores = np.array([r['scores'] for r in iteration_results])
    all_pca_ratios = np.array([r['pca_ratios'] for r in iteration_results])
    
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)
    mean_pca_ratios = np.mean(all_pca_ratios, axis=0)
    
    # FIXED: Create time points using consistent sampling rate
    timepoints = np.array([EPOCH_TMIN + (end / RESAMPLE_FREQ) for end in window_ends])
    
    # Log results
    write_log(f"    Mean accuracy: {np.mean(mean_scores):.3f}")
    write_log(f"    Peak accuracy: {np.max(mean_scores):.3f} at {timepoints[np.argmax(mean_scores)]:.3f}s")
    write_log(f"    Mean PCA ratio: {np.mean(mean_pca_ratios):.3f}")
    
    return mean_scores, std_scores, timepoints, all_scores, mean_pca_ratios, iteration_results

def load_subject_data(subject, meta_info):
    """Load and preprocess subject data for delay period analysis """
    write_log(f"\nLoading data for subject {subject}...")
    
    try:
        # Load behavioral correctness data
        behavioral_correctness = load_and_clean_behavioral_data(subject)
        if behavioral_correctness is None:
            write_log("Failed to load behavioral data")
            return None, None, None, None
        
        # Extract maintained information
        memorized, all_events = extract_maintained_information(subject, meta_info)
        
        if memorized is None:
            write_log("Could not extract maintained information")
            return None, None, None, None
        
        # Ensure behavioral correctness aligns with memorized trials
        if len(behavioral_correctness) != len(memorized):
            min_length = min(len(behavioral_correctness), len(memorized))
            write_log(f"WARNING: Length mismatch - behavioral: {len(behavioral_correctness)}, memorized: {len(memorized)}")
            write_log(f"Truncating both to {min_length} trials")
            behavioral_correctness = behavioral_correctness[:min_length]
            memorized = memorized[:min_length]
        
        write_log(f"Behavioral correctness aligned: {len(behavioral_correctness)} trials")
        
        # Load cleaned epochs
        clean_trials = mne.read_epochs(
            f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif",
            preload=True,
            verbose='ERROR'
        )
        
        if subject == 23:
            drop_idx = 64 * 7
            if drop_idx < len(clean_trials):
                memorized = np.delete(memorized, drop_idx)
                behavioral_correctness = np.delete(behavioral_correctness, drop_idx)

        if subject == 28:
            drop_idx = 63
            if drop_idx < len(clean_trials):
                clean_trials.drop(drop_idx)
                        
        # Handle jump artifacts (apply same drops to both arrays)
        jname = f"{PROCESSED_DIR}/ICAs/Jumps{subject}.npy"
        if os.path.isfile(jname):
            jump_inds = np.load(jname)
            if len(jump_inds) > 0:
                jump_inds = np.array(jump_inds, dtype=int)
                valid_jump_inds = jump_inds[jump_inds < len(clean_trials)]
                
                if len(valid_jump_inds) > 0:
                    clean_trials.drop(valid_jump_inds, reason='jump')
                    memorized = np.delete(memorized, valid_jump_inds)
                    behavioral_correctness = np.delete(behavioral_correctness, valid_jump_inds)
        
        # Final length check
        final_length = min(len(memorized), len(behavioral_correctness), len(clean_trials.events))
        memorized = memorized[:final_length]
        behavioral_correctness = behavioral_correctness[:final_length]
        
        # Update event codes to reflect maintained information
        clean_trials.events[:, 2] = memorized
        clean_trials.event_id = EVENT_DICT
        
        # Apply behavioral correctness filter if requested
        if args.correct_only:
            correct_mask = behavioral_correctness == 1
            write_log(f"Filtering to correct trials only: {np.sum(correct_mask)}/{len(correct_mask)} trials")
            
            if np.sum(correct_mask) == 0:
                write_log("ERROR: No correct trials found!")
                return None, None, None, None
            
            # Apply filter to epochs
            clean_trials = clean_trials[correct_mask]
            memorized = memorized[correct_mask]
            behavioral_correctness = behavioral_correctness[correct_mask]
        
        # Log behavioral correctness statistics
        n_correct = np.sum(behavioral_correctness == 1)
        n_incorrect = np.sum(behavioral_correctness == 0)
        n_too_slow = np.sum(behavioral_correctness == 2)
        write_log(f"Final trial correctness: Correct={n_correct}, Incorrect={n_incorrect}, TooSlow={n_too_slow}")
        
        # Crop to analysis window with buffer
        delay_epochs = clean_trials.copy()
        delay_epochs.crop(tmin=EPOCH_TMIN, tmax=EPOCH_TMAX)
        
        # Select magnetometers and resample
        mag_epochs = delay_epochs.copy().pick_types(meg='mag')
        mag_epochs = mag_epochs.resample(RESAMPLE_FREQ, npad='auto')
        
        # CRITICAL FIX: Log sampling rate info for debugging
        original_sfreq = mag_epochs.info['sfreq']
        write_log(f"DEBUG: Original sfreq from epochs.info: {original_sfreq} Hz")
        write_log(f"DEBUG: Expected RESAMPLE_FREQ: {RESAMPLE_FREQ} Hz")
        final_sfreq = mag_epochs.info['sfreq']
        write_log(f"DEBUG: Final sfreq after resampling: {final_sfreq} Hz")

        # Get data
        epochs_data = mag_epochs.get_data(copy=False)
        maintained_events = mag_epochs.events[:, 2]
	
        write_log(f"DEBUG: epochs_data.shape = {epochs_data.shape}")
        write_log(f"DEBUG: mag_epochs.tmin = {mag_epochs.tmin}")
        write_log(f"DEBUG: mag_epochs.tmax = {mag_epochs.tmax}")
        write_log(f"DEBUG: Expected time range: {EPOCH_TMIN} to {EPOCH_TMAX}")
        write_log(f"DEBUG: Time duration: {mag_epochs.tmax - mag_epochs.tmin} seconds")
        write_log(f"DEBUG: Expected samples at {RESAMPLE_FREQ}Hz: {int((mag_epochs.tmax - mag_epochs.tmin) * RESAMPLE_FREQ)}")
	
        expected_samples = int((EPOCH_TMAX - EPOCH_TMIN) * RESAMPLE_FREQ)
        write_log(f"DEBUG: Data shape: {epochs_data.shape}")
        write_log(f"DEBUG: Expected time samples: {expected_samples}")
        write_log(f"DEBUG: Actual time samples: {epochs_data.shape[2]}")
        
        write_log(f"Data loaded successfully. Shape: {epochs_data.shape}")
        write_log(f"Maintained events: {len(maintained_events)}")
        write_log(f"Time range: {mag_epochs.tmin:.2f}s to {mag_epochs.tmax:.2f}s")
        
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
        
        # CRITICAL FIX: Return RESAMPLE_FREQ instead of mag_epochs.info['sfreq']
        return epochs_data, maintained_events, RESAMPLE_FREQ, behavioral_correctness
        
    except Exception as e:
        write_log(f"Error loading data: {str(e)}")
        import traceback
        write_log(traceback.format_exc())
        return None, None, None, None

def plot_results(all_results):
    """Create summary plot of results"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    suffix = " (Correct Trials Only)" if args.correct_only else " (All Trials)"
    fig.suptitle(f'Subject {args.subject} - Delay Period Pseudotrial Analysis{suffix}', fontsize=16)
    
    features = ['maintained_voice', 'maintained_location']
    colors = ['#e41a1c', '#377eb8']
    
    for idx, feature_name in enumerate(features):
        ax = axes[idx]
        
        if feature_name in all_results:
            data = all_results[feature_name]
            
            # Plot time course
            ax.plot(data['timepoints'], data['mean_scores'], 
                   color=colors[idx], linewidth=2.5, label=f'{feature_name.replace("maintained_", "").title()}')
            
            # Add standard error
            ax.fill_between(data['timepoints'],
                          data['mean_scores'] - data['std_scores']/np.sqrt(args.n_iterations),
                          data['mean_scores'] + data['std_scores']/np.sqrt(args.n_iterations),
                          color=colors[idx], alpha=0.3)
        
        # Add reference lines
        ax.axvline(x=3.5, color='red', linestyle='--', alpha=0.7, label='Ping')
        ax.axvline(x=2.0, color='green', linestyle='--', alpha=0.7, label='Delay Start')
        ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Decoding Accuracy')
        ax.set_title(f'{feature_name.replace("maintained_", "").title()} Decoding')
        ax.set_ylim(0.45, 0.8)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/results_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    write_log(f"\nSummary plot saved to {OUTPUT_DIR}/results_summary.png")

def main():
    """Main processing function"""
    # Load metadata
    meta_info = pd.read_excel(META_FILE)
    
    # Load subject data
    epochs_data, events, actual_sfreq, behavioral_correctness = load_subject_data(args.subject, meta_info)
    
    if epochs_data is None:
        write_log("Failed to load data. Exiting.")
        sys.exit(1)
    
    write_log("\n=== DELAY PERIOD PSEUDOTRIAL SLIDING WINDOW ANALYSIS ===")
    write_log(f"Window: {WINDOW_LENGTH_SEC*1000:.0f}ms, Step: {WINDOW_STEP_SEC*1000:.0f}ms")
    write_log(f"Analysis period: {DELAY_START}s to {DELAY_END}s")
    write_log(f"Pseudotrial scheme: {TRIALS_PER_CONDITION} trials per condition")
    write_log(f"PCA variance threshold: {PCA_VARIANCE_THRESHOLD*100:.0f}%")
    write_log(f"Using correct trials only: {args.correct_only}")
    write_log(f"Sampling rate consistency: FIXED")
    
    # Verify expected window count
    expected_windows = int((DELAY_END - DELAY_START) / WINDOW_STEP_SEC) + 1
    write_log(f"Expected number of windows: {expected_windows}")
    
    # Save behavioral correctness info
    np.save(f'{OUTPUT_DIR}/behavioral_correctness.npy', behavioral_correctness)
    write_log(f"Behavioral correctness saved to: {OUTPUT_DIR}/behavioral_correctness.npy")
    
    # Results storage
    all_results = {}
    
    # Process each feature
    for feature_name in ['maintained_voice', 'maintained_location']:
        write_log(f"\n{'='*50}")
        write_log(f"Processing {feature_name}")
        write_log(f"{'='*50}")
        
        # Run sliding window decoding
        mean_scores, std_scores, timepoints, all_scores, mean_pca_ratios, iteration_results = decode_sliding_window_with_pseudotrials(
            epochs_data, events, RESAMPLE_FREQ, feature_name, args.n_iterations
        )
        
        # Store results
        all_results[feature_name] = {
            'mean_scores': mean_scores,
            'std_scores': std_scores,
            'timepoints': timepoints,
            'all_scores': all_scores,
            'mean_pca_ratios': mean_pca_ratios,
            'iteration_results': iteration_results
        }
        
        write_log(f"Mean accuracy: {np.mean(mean_scores):.3f}")
        write_log(f"Peak accuracy: {np.max(mean_scores):.3f}")
    
    # Save results
    write_log(f"\nSaving results...")
    
    for feature_name, results in all_results.items():
        # Save arrays
        np.save(f'{OUTPUT_DIR}/{feature_name}_mean_scores.npy', results['mean_scores'])
        np.save(f'{OUTPUT_DIR}/{feature_name}_std_scores.npy', results['std_scores'])
        np.save(f'{OUTPUT_DIR}/{feature_name}_timepoints.npy', results['timepoints'])
        np.save(f'{OUTPUT_DIR}/{feature_name}_all_scores.npy', results['all_scores'])
        np.save(f'{OUTPUT_DIR}/{feature_name}_pca_ratios.npy', results['mean_pca_ratios'])
    
    # Create summary plot
    plot_results(all_results)
    
    # Save comprehensive summary
    summary = {
        'subject': args.subject,
        'n_iterations': args.n_iterations,
        'analysis_window': f"{DELAY_START}-{DELAY_END}s",
        'window_length_ms': WINDOW_LENGTH_SEC * 1000,
        'window_step_ms': WINDOW_STEP_SEC * 1000,
        'pseudotrial_scheme': f"{TRIALS_PER_CONDITION} trials per condition",
        'pca_variance_threshold': PCA_VARIANCE_THRESHOLD,
        'correct_only': args.correct_only,
        'features': list(all_results.keys()),
        'processing_time': str(datetime.now()),
        'method': 'delay_pseudotrial_sliding_window_fixed',
        'sampling_rate_hz': RESAMPLE_FREQ,
        'version': 'FIXED - sampling rate consistency ensured',
        'results': {}
    }
    
    for feature_name, results in all_results.items():
        summary['results'][feature_name] = {
            'mean_accuracy': float(np.mean(results['mean_scores'])),
            'peak_accuracy': float(np.max(results['mean_scores'])),
            'peak_time': float(results['timepoints'][np.argmax(results['mean_scores'])]),
            'mean_pca_ratio': float(np.mean(results['mean_pca_ratios'])),
            'n_windows': len(results['timepoints'])
        }
    
    with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    write_log(f"\nProcessing completed at: {datetime.now()}")
    
    print(f"Subject {args.subject} delay period analysis completed successfully!")
    print(f"FIXED VERSION: Sampling rate consistency ensured, 100ms step size implemented")

if __name__ == "__main__":
    main()