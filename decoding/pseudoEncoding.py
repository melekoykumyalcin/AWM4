#!/usr/bin/env python
"""
HPC script for S1 and S2 analysis with avg12 scheme only
Uses proper nested CV with pseudo-trials created within folds
Usage: python hpc_s1s2_avg12.py --subject 23
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
parser = argparse.ArgumentParser(description='Run S1 and S2 analysis with avg12 scheme')
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
OUTPUT_DIR = PROCESSED_DIR + f's1s2_avg12/subject_{args.subject}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log file path
LOG_FILE_PATH = f'{OUTPUT_DIR}/processing_log.txt'

def write_log(message):
    """Write message to log file"""
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(message + '\n')

# Initialize log
write_log(f"S1 and S2 processing started at: {datetime.now()}")
write_log(f"Subject: {args.subject}")
write_log(f"Using avg12 scheme only with proper nested CV")

# Decoding parameters
RESAMPLE_FREQ = 100  # Hz
WINDOW_LENGTH_SEC = 0.1  # seconds
WINDOW_STEP_SEC = 0.01  # seconds
NUM_JOBS = args.n_jobs

# Conservative hyperparameter grid
PARAM_GRID = [0.1, 1.0, 10.0]

# Event dictionary for both S1 and S2
EVENT_DICT = {
    'S1/Sp1/L1': 111, 'S1/Sp1/L2': 112, 'S1/Sp1/L3': 113, 'S1/Sp1/L4': 114,
    'S1/Sp2/L1': 121, 'S1/Sp2/L2': 122, 'S1/Sp2/L3': 123, 'S1/Sp2/L4': 124,
    'S1/Sp3/L1': 131, 'S1/Sp3/L2': 132, 'S1/Sp3/L3': 133, 'S1/Sp3/L4': 134,
    'S1/Sp4/L1': 141, 'S1/Sp4/L2': 142, 'S1/Sp4/L3': 143, 'S1/Sp4/L4': 144,
    'S2/Sp1/L1': 211, 'S2/Sp1/L2': 212, 'S2/Sp1/L3': 213, 'S2/Sp1/L4': 214,
    'S2/Sp2/L1': 221, 'S2/Sp2/L2': 222, 'S2/Sp2/L3': 223, 'S2/Sp2/L4': 224,
    'S2/Sp3/L1': 231, 'S2/Sp3/L2': 232, 'S2/Sp3/L3': 233, 'S2/Sp3/L4': 234,
    'S2/Sp4/L1': 241, 'S2/Sp4/L2': 242, 'S2/Sp4/L3': 243, 'S2/Sp4/L4': 244
}

# Define the four corners for each stimulus type
CORNERS = {
    'S1': {
        'Corner1_Sp12L12': [111, 112, 121, 122],
        'Corner2_Sp12L34': [113, 114, 123, 124],
        'Corner3_Sp34L12': [131, 132, 141, 142],
        'Corner4_Sp34L34': [133, 134, 143, 144]
    },
    'S2': {
        'Corner1_Sp12L12': [211, 212, 221, 222],
        'Corner2_Sp12L34': [213, 214, 223, 224],
        'Corner3_Sp34L12': [231, 232, 241, 242],
        'Corner4_Sp34L34': [233, 234, 243, 244]
    }
}

# Stimulus configurations
STIMULI = {
    'S1': {
        'name': 'S1',
        'tmin': 0.0,
        'tmax': 1.0,
        'time_start': 0.0,  # Analysis window start
        'time_end': 1.0     # Analysis window end
    },
    'S2': {
        'name': 'S2',
        'tmin': 1.0,
        'tmax': 2.0,
        'time_start': 1.0,  # Relative to S2 start (1.1s absolute)
        'time_end': 2.0     # Relative to S2 start (1.5s absolute)
    }
}

# Only avg12 scheme
AVERAGING_SCHEME = {'trials_per_condition': 3, 'total_trials': 12}

def create_pseudo_trials_from_indices(epochs_data, events, trial_indices, corner_conditions, n_trials_per_condition):
    """
    Create pseudo-trials from specific trial indices only
    This ensures no data leakage between train/test sets
    """
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
        
    min_trials = min(len(trials) for trials in condition_trials.values() if trials)
    if min_trials < n_trials_per_condition:
        return None
    
    # Calculate how many pseudo-trials we can create
    n_pseudo_trials = min_trials // n_trials_per_condition
    
    pseudo_trials = []
    
    # Create pseudo-trials
    for _ in range(n_pseudo_trials):
        # Sample trials from each condition
        sampled_data = []
        for condition in corner_conditions:
            if condition in condition_trials and len(condition_trials[condition]) >= n_trials_per_condition:
                # Random sample without replacement
                indices = np.random.choice(
                    condition_trials[condition], 
                    size=n_trials_per_condition, 
                    replace=False
                )
                sampled_data.extend(subset_data[indices])
                
                # Remove used indices to avoid resampling
                condition_trials[condition] = [i for i in condition_trials[condition] if i not in indices]
        
        if len(sampled_data) == len(corner_conditions) * n_trials_per_condition:
            # Average the sampled trials
            pseudo_trial = np.mean(sampled_data, axis=0)
            pseudo_trials.append(pseudo_trial)
    
    return np.array(pseudo_trials) if pseudo_trials else None

def decode_with_proper_nested_cv(epochs_data, events, sfreq, feature_name, stimulus_name, scheme_params, n_iterations):
    """
    Perform decoding with proper nested CV where pseudo-trials are created within each fold
    """
    # Get stimulus configuration
    stim_config = STIMULI[stimulus_name]
    
    # Calculate time window samples
    start_sample = int(stim_config['time_start'] * sfreq)
    end_sample = int(stim_config['time_end'] * sfreq)
    
    # Prepare for sliding window
    window_length = int(sfreq * WINDOW_LENGTH_SEC)
    window_step = int(sfreq * WINDOW_STEP_SEC)
    n_windows = int((end_sample - start_sample - window_length) / window_step) + 1
    
    # Get appropriate corners for this stimulus
    stimulus_corners = CORNERS[stimulus_name]
    
    # Storage for results across iterations
    iteration_results = []
    
    for iteration in range(n_iterations):
        write_log(f"  Iteration {iteration + 1}/{n_iterations} for {stimulus_name} {feature_name}")
        
        # Results for this iteration
        window_scores = np.zeros(n_windows)
        window_c_values = []
        c_consistency_warnings = 0
        
        # Process each time window
        for window_idx in tqdm(range(n_windows), desc=f"{stimulus_name} {feature_name} windows"):
            win_start = start_sample + window_idx * window_step
            win_end = win_start + window_length
            
            # Extract window data
            window_data = epochs_data[:, :, win_start:win_end]
            
            # Outer cross-validation
            outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
            outer_scores = []
            best_cs = []
            
            # Create labels based on feature type
            if feature_name == 'voice':
                # Speaker 1+2 vs 3+4
                original_labels = np.array([0 if (e//10)%10 in [1,2] else 1 for e in events])
            else:  # location
                # Location 1+2 vs 3+4
                original_labels = np.array([0 if e%10 in [1,2] else 1 for e in events])
            
            # Outer CV loop
            fold_idx = 0
            for train_idx, test_idx in outer_cv.split(window_data, original_labels):
                fold_idx += 1
                
                # Step 1: Split original trials
                train_data = window_data[train_idx]
                test_data = window_data[test_idx]
                train_events = events[train_idx]
                test_events = events[test_idx]
                
                # Step 2: Create pseudo-trials separately for train and test
                train_pseudo_data = []
                train_pseudo_labels = []
                test_pseudo_data = []
                test_pseudo_labels = []
                
                for corner_name, corner_conditions in stimulus_corners.items():
                    # Create training pseudo-trials
                    train_pseudo = create_pseudo_trials_from_indices(
                        train_data, train_events, 
                        np.arange(len(train_events)),
                        corner_conditions,
                        scheme_params['trials_per_condition']
                    )
                    
                    if train_pseudo is not None:
                        train_pseudo_data.append(train_pseudo)
                        # Assign labels based on corner
                        if feature_name == 'voice':
                            labels = [0 if 'Sp12' in corner_name else 1] * len(train_pseudo)
                        else:
                            labels = [0 if 'L12' in corner_name else 1] * len(train_pseudo)
                        train_pseudo_labels.extend(labels)
                    
                    # Create test pseudo-trials
                    test_pseudo = create_pseudo_trials_from_indices(
                        test_data, test_events,
                        np.arange(len(test_events)),
                        corner_conditions,
                        scheme_params['trials_per_condition']
                    )
                    
                    if test_pseudo is not None:
                        test_pseudo_data.append(test_pseudo)
                        # Assign labels based on corner
                        if feature_name == 'voice':
                            labels = [0 if 'Sp12' in corner_name else 1] * len(test_pseudo)
                        else:
                            labels = [0 if 'L12' in corner_name else 1] * len(test_pseudo)
                        test_pseudo_labels.extend(labels)
                
                # Skip if not enough data
                if not train_pseudo_data or not test_pseudo_data:
                    continue
                
                # Combine pseudo-trials
                X_train = np.vstack(train_pseudo_data)
                y_train = np.array(train_pseudo_labels)
                X_test = np.vstack(test_pseudo_data)
                y_test = np.array(test_pseudo_labels)
                
                # Step 3: Inner CV for hyperparameter selection
                best_score = -1
                best_c = 1.0
                
                # Use simple 3-fold CV for inner loop
                inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=None)
                
                for c_value in PARAM_GRID:
                    # Create pipeline
                    clf = make_pipeline(
                        Vectorizer(),
                        StandardScaler(),
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
                    Vectorizer(),
                    StandardScaler(),
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
                
                # Check consistency of C values
                unique_cs = list(set(best_cs))
                if len(unique_cs) > 2:
                    c_consistency_warnings += 1
                
                # Use most common C
                c_counter = Counter(best_cs)
                most_common_c = c_counter.most_common(1)[0][0]
                window_c_values.append(most_common_c)
            else:
                window_scores[window_idx] = 0.5
                window_c_values.append(1.0)
        
        # Log consistency warnings
        if c_consistency_warnings > 0:
            write_log(f"    Warning: {c_consistency_warnings} windows had inconsistent C values")
        
        # Store iteration results
        iteration_results.append({
            'scores': window_scores,
            'c_values': window_c_values
        })
    
    # Aggregate results across iterations
    all_scores = np.array([r['scores'] for r in iteration_results])
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)
    
    # Create time points (relative to stimulus start)
    timepoints = np.array([stim_config['tmin'] + stim_config['time_start'] + i * WINDOW_STEP_SEC 
                          for i in range(n_windows)])
    
    return mean_scores, std_scores, timepoints, all_scores, iteration_results

def load_subject_data(subject, meta_info):
    """Load and preprocess data for a single subject"""
    write_log(f"\nLoading data for subject {subject}...")
    
    try:
        # Load metadata and raw data (same as before)
        actInd = (meta_info.Subject==subject) & (meta_info.Valid==1)
        
        # Prepare file lists
        allFiles = meta_info['MEG_Name']
        corrected_files = [f.split('.')[0] + '_correct_triggers.fif' for f in allFiles]
        corrected_files_series = pd.Series(corrected_files)
        
        # Early subjects list
        early_subs = sorted(np.unique(meta_info.loc[meta_info.FinalSample==1,'Subject']))[:7]
        
        if subject in early_subs:
            actFiles = corrected_files_series[actInd]
        else:
            actFiles = allFiles[actInd]
        
        # Load raw data and events
        all_events = None
        reference_dev_head_t_ref = None
        
        for ff in range(actFiles.count()):
            if subject in early_subs:
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
        
        # Get S1 and S2 events
        S1_idx = [i - 1 for i in range(len(all_events[:,2])) if all_events[i,2] == 100]
        S1_values = all_events[S1_idx,2]
        
        S2_idx = [i - 1 for i in range(len(all_events[:,2])) if all_events[i,2] == 200]
        S2_values = all_events[S2_idx,2]
        
        write_log(f"Found {len(S1_values)} S1 events and {len(S2_values)} S2 events")
        
        # Handle special cases
        if subject == 23:
            drop_idx = 64 * 7
            S1_values = np.delete(S1_values, drop_idx)
            S2_values = np.delete(S2_values, drop_idx)
        
        # Load cleaned epochs
        clean_trials = mne.read_epochs(
            f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif",
            preload=True,
            verbose='ERROR'
        )
        
        if subject == 28:
            clean_trials.drop(63)
            if 63 < len(S1_values):
                S1_values = np.delete(S1_values, 63)
                S2_values = np.delete(S2_values, 63)
        
        # We need to handle both S1 and S2, so we'll process them separately
        # But use the same cleaned epochs
        
        write_log(f"Data loaded successfully. Shape: {clean_trials.get_data().shape}")
        
        return clean_trials, S1_values, S2_values, clean_trials.info['sfreq']
        
    except Exception as e:
        write_log(f"Error loading data: {str(e)}")
        import traceback
        write_log(traceback.format_exc())
        return None, None, None, None

def process_stimulus(clean_trials, stimulus_values, stimulus_name, sfreq):
    """Process a specific stimulus (S1 or S2)"""
    # Update events
    epochs = clean_trials.copy()
    epochs.events[:, 2] = stimulus_values
    epochs.event_id = EVENT_DICT
    
    # Check for jump artifacts
    jname = f"{PROCESSED_DIR}/ICAs/Jumps{args.subject}.npy"
    if os.path.isfile(jname):
        jump_inds = np.load(jname)
        if len(jump_inds) > 0:
            jump_inds = np.array(jump_inds, dtype=int)
            valid_jump_inds = jump_inds[jump_inds < len(epochs)]
            if len(valid_jump_inds) > 0:
                epochs.drop(valid_jump_inds, reason='jump')
    
    # Crop to stimulus period
    stim_config = STIMULI[stimulus_name]
    epochs.crop(tmin=stim_config['tmin'], tmax=stim_config['tmax'])
    
    # Select magnetometers and resample
    mag_epochs = epochs.copy().pick_types(meg='mag')
    mag_epochs = mag_epochs.resample(RESAMPLE_FREQ, npad='auto')
    
    # Get data
    epochs_data = mag_epochs.get_data(copy=False)
    events = mag_epochs.events[:, 2]
    
    write_log(f"{stimulus_name} data prepared. Shape: {epochs_data.shape}")
    
    return epochs_data, events

def main():
    """Main processing function"""
    # Load metadata
    meta_info = pd.read_excel(META_FILE)
    
    # Load subject data
    clean_trials, S1_values, S2_values, sfreq = load_subject_data(args.subject, meta_info)
    
    if clean_trials is None:
        write_log("Failed to load data. Exiting.")
        sys.exit(1)
    
    write_log("\n=== Using avg12 scheme with proper nested CV ===")
    
    # Results storage
    all_results = {}
    
    # Process each stimulus
    for stimulus_name in ['S1', 'S2']:
        write_log(f"\n--- Processing {stimulus_name} ---")
        
        # Get appropriate stimulus values
        stimulus_values = S1_values if stimulus_name == 'S1' else S2_values
        
        # Process this stimulus
        epochs_data, events = process_stimulus(clean_trials, stimulus_values, stimulus_name, sfreq)
        
        stimulus_results = {}
        
        # Process each feature
        for feature_name in ['voice', 'location']:
            write_log(f"\n  Decoding {feature_name} for {stimulus_name}...")
            
            # Run decoding with proper nested CV
            mean_scores, std_scores, timepoints, all_scores, iteration_results = decode_with_proper_nested_cv(
                epochs_data, events, sfreq, feature_name, stimulus_name, 
                AVERAGING_SCHEME, args.n_iterations
            )
            
            # Store results
            stimulus_results[feature_name] = {
                'mean': mean_scores,
                'std': std_scores,
                'timepoints': timepoints,
                'scores': all_scores,
                'iterations': iteration_results
            }
            
            # Log summary
            write_log(f"    Mean accuracy: {np.mean(mean_scores):.3f}")
            write_log(f"    Peak accuracy: {np.max(mean_scores):.3f}")
            write_log(f"    Peak time: {timepoints[np.argmax(mean_scores)]:.3f}s")
        
        all_results[stimulus_name] = stimulus_results
    
    # Save results
    write_log(f"\nSaving results...")
    
    # Save as numpy arrays
    for stimulus_name in ['S1', 'S2']:
        for feature in ['voice', 'location']:
            # Save scores
            np.save(f'{OUTPUT_DIR}/avg12_{stimulus_name}_{feature}_scores.npy', 
                   all_results[stimulus_name][feature]['scores'])
            np.save(f'{OUTPUT_DIR}/avg12_{stimulus_name}_{feature}_mean.npy', 
                   all_results[stimulus_name][feature]['mean'])
            np.save(f'{OUTPUT_DIR}/avg12_{stimulus_name}_{feature}_std.npy', 
                   all_results[stimulus_name][feature]['std'])
            np.save(f'{OUTPUT_DIR}/avg12_{stimulus_name}_{feature}_timepoints.npy', 
                   all_results[stimulus_name][feature]['timepoints'])
            
            # Save C values
            all_c_values = [r['c_values'] for r in all_results[stimulus_name][feature]['iterations']]
            np.save(f'{OUTPUT_DIR}/avg12_{stimulus_name}_{feature}_c_values.npy', all_c_values)
    
    # Save summary
    summary = {
        'subject': args.subject,
        'n_iterations': args.n_iterations,
        'scheme': 'avg12',
        'stimuli': ['S1', 'S2'],
        'processing_time': str(datetime.now()),
        'method': 'proper_nested_cv_s1s2',
        'results_summary': {}
    }
    
    for stimulus_name in ['S1', 'S2']:
        summary['results_summary'][stimulus_name] = {
            'voice': {
                'mean_accuracy': float(np.mean(all_results[stimulus_name]['voice']['mean'])),
                'max_accuracy': float(np.max(all_results[stimulus_name]['voice']['mean']))
            },
            'location': {
                'mean_accuracy': float(np.mean(all_results[stimulus_name]['location']['mean'])),
                'max_accuracy': float(np.max(all_results[stimulus_name]['location']['mean']))
            }
        }
    
    with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    write_log(f"\nProcessing completed at: {datetime.now()}")
    
    print(f"Subject {args.subject} S1 and S2 analysis completed successfully!")

if __name__ == "__main__":
    main()