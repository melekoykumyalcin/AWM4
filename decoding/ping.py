#!/usr/bin/env python
"""
Ping Period Analysis Script
Modified for:
1. Feature-wise normalization (StandardScaler)
2. Separate pre-ping (2.5-3.5s) and post-ping (3.5-4.5s) analysis
3. Flatten approach with separate classifiers for each period
4. Voice identity and location decoding within each period

Usage: python ping_period_analysis.py --subject 23
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

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run ping period analysis with separate pre/post periods')
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
OUTPUT_DIR = PROCESSED_DIR + f'pingAnalysis/subject_{args.subject}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log file path
LOG_FILE_PATH = f'{OUTPUT_DIR}/processing_log.txt'

def write_log(message):
    """Write message to log file"""
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(message + '\n')

# Initialize log
write_log(f"Ping period analysis started at: {datetime.now()}")
write_log(f"Subject: {args.subject}")
write_log(f"Using separate pre-ping and post-ping analysis with feature-wise normalization")

# Decoding parameters
RESAMPLE_FREQ = 100  # Hz
WINDOW_LENGTHS_SEC = [0.05, 0.1, 0.2, 0.5]  # 50ms, 100ms, 200ms, 500ms
WINDOW_STEP_SEC = 0.01  # seconds

# Define ping periods - cue at 2s, ping at 3s
PING_PERIODS = {
    'pre_ping': {
        'tmin': 2.5,  # 500ms after cue
        'tmax': 3.5,  # until ping
        'description': 'Pre-ping (maintenance period)'
    },
    'post_ping': {
        'tmin': 3.5,  # after ping
        'tmax': 4.5,  # end of trial
        'description': 'Post-ping (retrieval/comparison period)'
    }
}

# Delay Period Configuration for data loading
DELAY_CONFIG = {
    'tmin': 2.0,
    'tmax': 4.7,
    'timepoints': np.linspace(2.0, 4.7, int((4.7-2.0)*RESAMPLE_FREQ))
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

# Simplified averaging scheme - using avg12 as default
AVERAGING_SCHEME = {'trials_per_condition': 3, 'total_trials': 12} #each pseudo-trial average 3 trials from 4 conditions in a corner

# Conservative CV parameters
NUM_JOBS = args.n_jobs
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

def decode_period_with_sliding_window(epochs_data, events, sfreq, period_config, 
                                    feature_name, n_iterations, window_length_sec):
    """
    Decode a specific period using sliding window analysis with feature-wise normalization
    """
    write_log(f"\n  Decoding {feature_name} in {period_config['description']}")
    write_log(f"  Time window: {period_config['tmin']}s to {period_config['tmax']}s")
    write_log(f"  Window length: {WINDOW_LENGTHS_SEC*1000}ms")  # Add this
    
    # Calculate time window samples
    period_start_sample = int((period_config['tmin'] - DELAY_CONFIG['tmin']) * sfreq)
    period_end_sample = int((period_config['tmax'] - DELAY_CONFIG['tmin']) * sfreq)
    
    # Extract period data
    period_data = epochs_data[:, :, period_start_sample:period_end_sample]
    
    write_log(f"  Period data shape: {period_data.shape}")
    
    # Sliding window parameters
    window_length = int(sfreq * window_length_sec)  # Use parameter instead of global
    window_step = int(sfreq * WINDOW_STEP_SEC)
    n_times = period_data.shape[2]
    n_windows = int((n_times - window_length) / window_step) + 1
    
    write_log(f"  Number of sliding windows: {n_windows}")
    
    # Storage for results across iterations
    iteration_results = []
    
    for iteration in range(n_iterations):
        write_log(f"    Iteration {iteration + 1}/{n_iterations}")
        
        # Results for this iteration
        window_scores = np.zeros(n_windows)
        window_c_values = []
        
        # Process each sliding window
        for window_idx in tqdm(range(n_windows), desc=f"{feature_name} {period_config['description']} windows"):
            win_start = window_idx * window_step
            win_end = win_start + window_length
            
            # Extract window data and flatten
            window_data = period_data[:, :, win_start:win_end]
            n_trials, n_channels, n_times = window_data.shape
            flattened_data = window_data.reshape(n_trials, n_channels * n_times)
            
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
            for train_idx, test_idx in outer_cv.split(flattened_data, original_labels):
                
                # Step 1: Split original trials
                train_data = flattened_data[train_idx]
                test_data = flattened_data[test_idx]
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
    
    # Create time points (absolute time)
    timepoints = np.array([period_config['tmin'] + i * WINDOW_STEP_SEC 
                          for i in range(n_windows)])
    
    write_log(f"    Mean accuracy: {np.mean(mean_scores):.3f}")
    write_log(f"    Peak accuracy: {np.max(mean_scores):.3f}")
    write_log(f"    Peak time: {timepoints[np.argmax(mean_scores)]:.3f}s")
    
    return mean_scores, std_scores, timepoints, all_scores, iteration_results

def load_subject_data(subject, meta_info):
    """Load and preprocess data for the ping period analysis"""
    write_log(f"\nLoading ping period data for subject {subject}...")
    
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
        
        # only drop from the labels 
        if subject == 23:
            drop_idx = 64 * 7
            if drop_idx < len(clean_trials):
                #clean_trials.drop(drop_idx)
                memorized = np.delete(memorized, drop_idx)

        if subject == 28:
            drop_idx = 63
            if drop_idx < len(clean_trials):
                #clean_trials.drop(drop_idx)
                memorized = np.delete(memorized, drop_idx)
        
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
        
        # Crop to delay period
        delay_epochs = clean_trials.copy()
        delay_epochs.crop(tmin=DELAY_CONFIG['tmin'], tmax=DELAY_CONFIG['tmax'])
        
        # Select magnetometers and resample
        mag_epochs = delay_epochs.copy().pick_types(meg='mag')
        mag_epochs = mag_epochs.resample(RESAMPLE_FREQ, npad='auto')
        
        # Get data
        epochs_data = mag_epochs.get_data(copy=False)
        maintained_events = mag_epochs.events[:, 2]
        
        write_log(f"Data loaded successfully. Shape: {epochs_data.shape}")
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

def main():
    """Main processing function"""
    # Load metadata
    meta_info = pd.read_excel(META_FILE)
    
    # Load subject data
    epochs_data, events, sfreq = load_subject_data(args.subject, meta_info)
    
    if epochs_data is None:
        write_log("Failed to load data. Exiting.")
        sys.exit(1)
    
    write_log("\n=== Ping Period Analysis with Feature-wise Normalization ===")
    write_log("=== Separate Pre-ping and Post-ping Analysis ===")
    write_log(f"=== Testing window lengths: {[w*1000 for w in WINDOW_LENGTHS_SEC]}ms ===")
    
    # Results storage for all window lengths
    all_window_results = {}
    
    # Iterate through different window lengths
    for window_length_sec in WINDOW_LENGTHS_SEC:
        window_length_ms = int(window_length_sec * 1000)
        write_log(f"\n{'='*60}")
        write_log(f"Processing with window length: {window_length_ms}ms")
        write_log(f"{'='*60}")
        
        # Results storage for this window length
        all_results = {}
        
        # Process each ping period
        for period_name, period_config in PING_PERIODS.items():
            write_log(f"\n--- Processing {period_name}: {period_config['description']} ---")
            
            period_results = {}
            
            # Process each feature (voice and location)
            for feature_name in ['maintained_voice', 'maintained_location']:
                
                # Run decoding with current window length
                mean_scores, std_scores, timepoints, all_scores, iteration_results = decode_period_with_sliding_window(
                    epochs_data, events, sfreq, period_config, feature_name, 
                    args.n_iterations, window_length_sec  # Pass window length
                )
                
                # Store results
                period_results[feature_name] = {
                    'mean_scores': mean_scores,
                    'std_scores': std_scores,
                    'timepoints': timepoints,
                    'all_scores': all_scores,
                    'iteration_results': iteration_results,
                    'period_config': period_config,
                    'window_length_ms': window_length_ms
                }
                
                write_log(f"  {feature_name}: Mean={np.mean(mean_scores):.3f}, Peak={np.max(mean_scores):.3f}")
            
            all_results[period_name] = period_results
        
        # Store results for this window length
        all_window_results[window_length_ms] = all_results
        
        # Save results for this window length
        write_log(f"\nSaving results for {window_length_ms}ms window...")
        
        # Save temporal results with window length in filename
        for period_name, period_results in all_results.items():
            for feature_name, results in period_results.items():
                # Save temporal data
                np.save(f'{OUTPUT_DIR}/{period_name}_{feature_name}_{window_length_ms}ms_mean_scores.npy', 
                       results['mean_scores'])
                np.save(f'{OUTPUT_DIR}/{period_name}_{feature_name}_{window_length_ms}ms_std_scores.npy', 
                       results['std_scores'])
                np.save(f'{OUTPUT_DIR}/{period_name}_{feature_name}_{window_length_ms}ms_timepoints.npy', 
                       results['timepoints'])
                np.save(f'{OUTPUT_DIR}/{period_name}_{feature_name}_{window_length_ms}ms_all_scores.npy', 
                       results['all_scores'])
                np.save(f'{OUTPUT_DIR}/{period_name}_{feature_name}_{window_length_ms}ms_c_values.npy', 
                       [r['c_values'] for r in results['iteration_results']])
    
    # Save comprehensive summary including all window lengths
    summary = {
        'subject': args.subject,
        'n_iterations': args.n_iterations,
        'periods': list(PING_PERIODS.keys()),
        'features': ['maintained_voice', 'maintained_location'],
        'window_lengths_ms': [w*1000 for w in WINDOW_LENGTHS_SEC],
        'window_step_ms': WINDOW_STEP_SEC * 1000,
        'averaging_scheme': AVERAGING_SCHEME,
        'processing_time': str(datetime.now()),
        'method': 'ping_period_sliding_window_feature_normalization',
        'approach': 'sliding_window_with_feature_wise_normalization',
        'normalization': 'StandardScaler (feature-wise)',
        'results': {}
    }
    
    # Add detailed results for each window length
    for window_length_ms, window_results in all_window_results.items():
        summary['results'][f'window_{window_length_ms}ms'] = {}
        
        for period_name, period_results in window_results.items():
            summary['results'][f'window_{window_length_ms}ms'][period_name] = {}
            for feature_name, results in period_results.items():
                summary['results'][f'window_{window_length_ms}ms'][period_name][feature_name] = {
                    'mean_accuracy': float(np.mean(results['mean_scores'])),
                    'peak_accuracy': float(np.max(results['mean_scores'])),
                    'peak_time': float(results['timepoints'][np.argmax(results['mean_scores'])]),
                    'time_window': f"{results['period_config']['tmin']}-{results['period_config']['tmax']}s",
                    'n_timepoints': len(results['timepoints'])
                }
        
        # Compare periods for this window length
        write_log(f"\n=== COMPARISON BETWEEN PERIODS ({window_length_ms}ms window) ===")
        for feature_name in ['maintained_voice', 'maintained_location']:
            pre_mean = np.mean(window_results['pre_ping'][feature_name]['mean_scores'])
            post_mean = np.mean(window_results['post_ping'][feature_name]['mean_scores'])
            difference = post_mean - pre_mean
            
            write_log(f"{feature_name}:")
            write_log(f"  Pre-ping:  {pre_mean:.3f}")
            write_log(f"  Post-ping: {post_mean:.3f}")
            write_log(f"  Difference: {difference:+.3f}")
            
            summary['results'][f'window_{window_length_ms}ms'][f'{feature_name}_comparison'] = {
                'pre_ping_mean_accuracy': float(pre_mean),
                'post_ping_mean_accuracy': float(post_mean),
                'difference': float(difference)
            }
    
    # Add comparison across window lengths
    write_log(f"\n=== COMPARISON ACROSS WINDOW LENGTHS ===")
    for feature_name in ['maintained_voice', 'maintained_location']:
        write_log(f"\n{feature_name}:")
        for window_length_ms in sorted(all_window_results.keys()):
            pre_mean = np.mean(all_window_results[window_length_ms]['pre_ping'][feature_name]['mean_scores'])
            post_mean = np.mean(all_window_results[window_length_ms]['post_ping'][feature_name]['mean_scores'])
            write_log(f"  {window_length_ms}ms: Pre={pre_mean:.3f}, Post={post_mean:.3f}, Diff={post_mean-pre_mean:+.3f}")
    
    with open(f'{OUTPUT_DIR}/summary_all_windows.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    write_log(f"\nProcessing completed at: {datetime.now()}")
    
    print(f"Subject {args.subject} ping period analysis completed successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Tested window lengths: {[w*1000 for w in WINDOW_LENGTHS_SEC]}ms")

if __name__ == "__main__":
    main()