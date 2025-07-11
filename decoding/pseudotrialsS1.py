#!/usr/bin/env python
"""
HPC script for single subject pseudo-trial averaging analysis
Usage: python hpc_pseudo_trial_subject.py --subject 23
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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mne.decoding import Vectorizer
from collections import defaultdict, Counter
from joblib import Parallel, delayed
import json
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run pseudo-trial analysis for a single subject')
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
OUTPUT_DIR = PROCESSED_DIR + f'pseudo_trial_S1(1)/subject_{args.subject}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log file path - global to avoid pickling issues
LOG_FILE_PATH = f'{OUTPUT_DIR}/processing_log.txt'

def write_log(message):
    """Write message to log file"""
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(message + '\n')

# Initialize log
write_log(f"Processing started at: {datetime.now()}")
write_log(f"Subject: {args.subject}")
write_log(f"Iterations: {args.n_iterations}")

# Decoding parameters
RESAMPLE_FREQ = 100  # Hz
WINDOW_LENGTH_SEC = 0.1  # seconds
WINDOW_STEP_SEC = 0.01  # seconds
TIME_START = 0.1  # Start of time window
TIME_END = 0.5   # End of time window
CV_SPLITS = 5
CV_REPEATS = 100  
NUM_JOBS = args.n_jobs

# Event dictionary for S1
S1_EVENT_DICT = {
    'S1/Sp1/L1': 111, 'S1/Sp1/L2': 112, 'S1/Sp1/L3': 113, 'S1/Sp1/L4': 114,
    'S1/Sp2/L1': 121, 'S1/Sp2/L2': 122, 'S1/Sp2/L3': 123, 'S1/Sp2/L4': 124,
    'S1/Sp3/L1': 131, 'S1/Sp3/L2': 132, 'S1/Sp3/L3': 133, 'S1/Sp3/L4': 134,
    'S1/Sp4/L1': 141, 'S1/Sp4/L2': 142, 'S1/Sp4/L3': 143, 'S1/Sp4/L4': 144
}

# Define the four corners
CORNERS = {
    'Corner1_Sp12L12': [111, 112, 121, 122],
    'Corner2_Sp12L34': [113, 114, 123, 124],
    'Corner3_Sp34L12': [131, 132, 141, 142],
    'Corner4_Sp34L34': [133, 134, 143, 144]
}

# Averaging schemes
AVERAGING_SCHEMES = {
    'avg4': {'trials_per_condition': 1, 'total_trials': 4},
    'avg8': {'trials_per_condition': 2, 'total_trials': 8},
    'avg12': {'trials_per_condition': 3, 'total_trials': 12}
}

def create_pseudo_trials(epochs_data, events, corner_conditions, n_trials_per_condition, n_iterations):
    """Create pseudo-trials by averaging trials from conditions within a corner."""
    condition_trials = defaultdict(list)
    for idx, event in enumerate(events):
        if event in corner_conditions:
            condition_trials[event].append(idx)
    
    min_trials = min(len(trials) for trials in condition_trials.values())
    if min_trials < n_trials_per_condition:
        write_log(f"Warning: Not enough trials. Min available: {min_trials}, requested: {n_trials_per_condition}")
        return None
    
    n_pseudo_trials = min_trials // n_trials_per_condition
    all_pseudo_trials = []
    
    for iteration in range(n_iterations):
        iteration_pseudo_trials = []
        
        for pseudo_idx in range(n_pseudo_trials):
            sampled_trials = []
            for condition in corner_conditions:
                available_indices = condition_trials[condition].copy()
                selected_indices = np.random.choice(
                    available_indices, 
                    size=n_trials_per_condition, 
                    replace=False
                )
                sampled_trials.extend(epochs_data[selected_indices])
            
            pseudo_trial = np.mean(sampled_trials, axis=0)
            iteration_pseudo_trials.append(pseudo_trial)
        
        all_pseudo_trials.append(np.array(iteration_pseudo_trials))
    
    return all_pseudo_trials

def decode_sliding_window_nested(data, y, sfreq, feature_name):
    """Perform sliding window decoding with nested cross-validation"""
    # Calculate sample indices for the time window
    start_sample = int(TIME_START * sfreq)
    end_sample = int(TIME_END * sfreq)
    
    # Crop data to time window
    data_windowed = data[:, :, start_sample:end_sample]
    
    n_trials, n_channels, n_timepoints = data_windowed.shape
    window_length = int(sfreq * WINDOW_LENGTH_SEC)
    window_step = int(sfreq * WINDOW_STEP_SEC)
    n_windows = int((n_timepoints - window_length) / window_step) + 1
    
    # Define parameter grid for SVM
    param_grid = {'svc__C': [0.001, 0.01, 0.1, 1, 10]}  # Note: svc__C for pipeline
    
    def process_window(window_idx):
        """Process a single window with nested cross-validation"""
        try:
            win_start = window_idx * window_step
            win_end = win_start + window_length
            window_center = (win_start + win_end) / 2 / sfreq + TIME_START
            
            # Extract window data - keep 3D shape for Vectorizer
            X_win = data_windowed[:, :, win_start:win_end]
            
            # Create pipeline matching original scripts
            from mne.decoding import Vectorizer
            clf_pipeline = make_pipeline(
                Vectorizer(),  # Converts from (trials, channels, times) to (trials, features)
                StandardScaler(),  # Standardize features
                SVC(kernel='linear', probability=True)
            )
            
            # Setup cross-validation with more conservative parameters
            outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
            inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

            nested_scores = []
            best_c_values = []
            
            for train_idx, test_idx in outer_cv.split(X_win, y):
                X_train, X_test = X_win[train_idx], X_win[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Grid search with pipeline
                grid_search = GridSearchCV(
                    estimator=clf_pipeline,
                    param_grid=param_grid,
                    cv=inner_cv,
                    scoring='accuracy',
                    n_jobs=1  # Avoid nested parallelization
                )
                
                grid_search.fit(X_train, y_train)
                best_c = grid_search.best_params_['svc__C']
                best_c_values.append(best_c)
                
                # Evaluate on test set
                score = grid_search.score(X_test, y_test)
                nested_scores.append(score)
            
            mean_score = np.mean(nested_scores)
            std_score = np.std(nested_scores)
            c_counter = Counter(best_c_values)
            most_common_c = c_counter.most_common(1)[0][0]
            
            return {
                'accuracy': mean_score,
                'std': std_score,
                'best_c': most_common_c,
                'window_center': window_center,
                'window_idx': window_idx,
                'error': None
            }
        except Exception as e:
            return {
                'accuracy': 0.5,
                'std': 0.0,
                'best_c': 1.0,
                'window_center': TIME_START + window_idx * WINDOW_STEP_SEC,
                'window_idx': window_idx,
                'error': str(e)
            }
    
    print(f"Decoding {feature_name} with {n_windows} windows...")
    results = Parallel(n_jobs=NUM_JOBS)(
        delayed(process_window)(window_idx) for window_idx in tqdm(range(n_windows))
    )
    
    # Check for errors and log them
    errors = [r for r in results if r['error'] is not None]
    if errors:
        write_log(f"Warning: {len(errors)} windows had errors during {feature_name} processing")
        for err in errors[:5]:  # Log first 5 errors
            write_log(f"  Window {err['window_idx']}: {err['error']}")
    
    # Sort results by window index
    results.sort(key=lambda x: x['window_idx'])
    
    window_scores = np.array([r['accuracy'] for r in results])
    window_centers = np.array([r['window_center'] for r in results])
    best_c_values = [r['best_c'] for r in results]
    
    return window_scores, window_centers, best_c_values

def load_subject_data(subject, meta_info):
    """Load and preprocess data for a single subject"""
    write_log(f"\nLoading data for subject {subject}...")
    
    try:
        # Load metadata
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
        
        # Get S1 events
        S1_idx = [i - 1 for i in range(len(all_events[:,2])) if all_events[i,2] == 100]
        S1_values = all_events[S1_idx,2]
        
        # Handle special cases
        if subject == 23:
            drop_idx = 64 * 7
            S1_values = np.delete(S1_values, drop_idx)
        
        # Load cleaned epochs
        clean_trials = mne.read_epochs(
            f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif",
            preload=True,
            verbose='ERROR'
        )
        
        if subject == 28:
            clean_trials.drop(63)
        
        clean_trials.events[:,2] = S1_values
        clean_trials.event_id = S1_EVENT_DICT
        
        # Crop to S1 period
        S1_epochs = clean_trials.copy()
        S1_epochs.crop(tmin=0, tmax=1)
        
        # Select magnetometers and resample
        mag_epochs = S1_epochs.copy().pick_types(meg='mag')
        mag_epochs = mag_epochs.resample(RESAMPLE_FREQ, npad='auto')
        
        # Get data with copy=False explicitly to avoid FutureWarning
        epochs_data = mag_epochs.get_data(copy=False)
        write_log(f"Data loaded successfully. Shape: {epochs_data.shape}")
        
        return epochs_data, mag_epochs.events[:, 2], mag_epochs.info['sfreq']
        
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
    
    # Results storage
    all_results = {}
    
    # Process each averaging scheme
    for scheme_name, scheme_params in AVERAGING_SCHEMES.items():
        write_log(f"\n--- Processing {scheme_name} ---")
        
        scheme_results = {
            'voice': {'scores': [], 'timepoints': None, 'best_c': []},
            'location': {'scores': [], 'timepoints': None, 'best_c': []}
        }
        
        # Create pseudo-trials for each corner
        corner_pseudo_trials = {}
        for corner_name, corner_conditions in CORNERS.items():
            pseudo_trials = create_pseudo_trials(
                epochs_data, 
                events, 
                corner_conditions,
                scheme_params['trials_per_condition'],
                args.n_iterations
            )
            
            if pseudo_trials is not None:
                corner_pseudo_trials[corner_name] = pseudo_trials
            else:
                write_log(f"Skipping {corner_name} - insufficient trials")
        
        # Run decoding for each iteration
        for iteration in range(args.n_iterations):
            write_log(f"Iteration {iteration+1}/{args.n_iterations}")
            
            # Combine pseudo-trials from all corners
            all_pseudo_data = []
            all_pseudo_labels_voice = []
            all_pseudo_labels_location = []
            
            for corner_name, pseudo_trials_list in corner_pseudo_trials.items():
                if pseudo_trials_list is None:
                    continue
                
                pseudo_data = pseudo_trials_list[iteration]
                n_trials = pseudo_data.shape[0]
                
                # Assign labels
                if 'Sp12' in corner_name:
                    all_pseudo_labels_voice.extend([0] * n_trials)
                else:
                    all_pseudo_labels_voice.extend([1] * n_trials)
                
                if 'L12' in corner_name:
                    all_pseudo_labels_location.extend([0] * n_trials)
                else:
                    all_pseudo_labels_location.extend([1] * n_trials)
                
                all_pseudo_data.append(pseudo_data)
            
            # Stack all data
            X = np.vstack(all_pseudo_data)
            y_voice = np.array(all_pseudo_labels_voice)
            y_location = np.array(all_pseudo_labels_location)
            
            write_log(f"  Created {X.shape[0]} pseudo-trials")
            
            # Decode voice identity
            scores_voice, timepoints, best_c_voice = decode_sliding_window_nested(X, y_voice, sfreq, 'voice')
            scheme_results['voice']['scores'].append(scores_voice)
            scheme_results['voice']['timepoints'] = timepoints
            scheme_results['voice']['best_c'].append(best_c_voice)
            
            # Decode location
            scores_location, timepoints, best_c_location = decode_sliding_window_nested(X, y_location, sfreq, 'location')
            scheme_results['location']['scores'].append(scores_location)
            scheme_results['location']['timepoints'] = timepoints
            scheme_results['location']['best_c'].append(best_c_location)
        
        # Convert to arrays and compute statistics
        for feature in ['voice', 'location']:
            scheme_results[feature]['scores'] = np.array(scheme_results[feature]['scores'])
            scheme_results[feature]['mean'] = np.mean(scheme_results[feature]['scores'], axis=0)
            scheme_results[feature]['std'] = np.std(scheme_results[feature]['scores'], axis=0)
            scheme_results[feature]['best_c'] = np.array(scheme_results[feature]['best_c'])
        
        all_results[scheme_name] = scheme_results
    
    # Save results
    write_log(f"\nSaving results...")
    
    # Save as numpy arrays
    for scheme_name, scheme_results in all_results.items():
        for feature in ['voice', 'location']:
            # Save scores
            np.save(f'{OUTPUT_DIR}/{scheme_name}_{feature}_scores.npy', 
                   scheme_results[feature]['scores'])
            np.save(f'{OUTPUT_DIR}/{scheme_name}_{feature}_mean.npy', 
                   scheme_results[feature]['mean'])
            np.save(f'{OUTPUT_DIR}/{scheme_name}_{feature}_std.npy', 
                   scheme_results[feature]['std'])
            np.save(f'{OUTPUT_DIR}/{scheme_name}_{feature}_timepoints.npy', 
                   scheme_results[feature]['timepoints'])
            
            # Save best C values
            np.save(f'{OUTPUT_DIR}/{scheme_name}_{feature}_best_c.npy',
                   scheme_results[feature]['best_c'])
    
    # Save summary as JSON
    summary = {
        'subject': args.subject,
        'n_iterations': args.n_iterations,
        'schemes': list(all_results.keys()),
        'processing_time': str(datetime.now()),
        'results_summary': {}
    }
    
    for scheme_name, scheme_results in all_results.items():
        summary['results_summary'][scheme_name] = {
            'voice': {
                'mean_accuracy': float(np.mean(scheme_results['voice']['mean'])),
                'max_accuracy': float(np.max(scheme_results['voice']['mean']))
            },
            'location': {
                'mean_accuracy': float(np.mean(scheme_results['location']['mean'])),
                'max_accuracy': float(np.max(scheme_results['location']['mean']))
            }
        }
    
    with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    write_log(f"Processing completed at: {datetime.now()}")
    
    print(f"Subject {args.subject} completed successfully!")

if __name__ == "__main__":
    main()