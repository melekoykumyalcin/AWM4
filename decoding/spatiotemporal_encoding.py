#!/usr/bin/env python
"""
S1 Spatiotemporal Analysis with Pseudotrials - Based on proven pseudo.py structure
- Uses working pseudo.py pipeline with spatiotemporal feature extraction
- Only avg12 scheme (3 trials per condition)
- Wolff-style feature extraction (10ms averaging, per-trial mean-centering)
- 100ms sliding windows with 10ms steps
- Analysis window: 0.0-1.0s (S1 encoding period)

Usage: python spatiotemporal_pseudo_s1.py --subject 23
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from collections import defaultdict, Counter
import json
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run S1 spatiotemporal analysis with pseudotrials')
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
OUTPUT_DIR = PROCESSED_DIR + f'spatiotemporal_S1/subject_{args.subject}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log file path
LOG_FILE_PATH = f'{OUTPUT_DIR}/processing_log.txt'

def write_log(message):
    """Write message to log file"""
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(message + '\n')

# Initialize log
write_log(f"S1 spatiotemporal processing started at: {datetime.now()}")
write_log(f"Subject: {args.subject}")
write_log(f"Using PROPER nested CV with spatiotemporal features")
write_log(f"Based on proven pseudo.py pipeline structure")

# Decoding parameters
RESAMPLE_FREQ = 100  # Hz
WINDOW_LENGTH_SEC = 0.1  # 100ms windows
WINDOW_STEP_SEC = 0.01  # 10ms steps
TIME_START = 0.0  # Start of S1 analysis window
TIME_END = 0.5    # End of S1 analysis window

# Cross-validation parameters
OUTER_CV_SPLITS = 5
INNER_CV_SPLITS = 3
PCA_VARIANCE_THRESHOLD = 0.90  # 90% variance for 100ms windows
NUM_JOBS = args.n_jobs

# Hyperparameter grid - stronger regularization for higher dimensions
PARAM_GRID = {'svc__C': [0.001, 0.01, 0.1]}

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

# Only avg12 scheme
AVERAGING_SCHEME = {'trials_per_condition': 3, 'total_trials': 12}

class SpatiotemporalVectorizer:
    """
    Custom vectorizer that implements Wolff-style spatiotemporal feature extraction
    Integrates seamlessly with sklearn pipeline like MNE's Vectorizer
    """
    
    def __init__(self, sfreq=100):
        self.sfreq = sfreq
        
    def fit(self, X, y=None):
        """Fit method (no-op for this transformer)"""
        return self
    
    def transform(self, X):
        """
        Transform window data to Wolff-style spatiotemporal features
        
        Parameters:
        X : array, shape (n_trials, n_channels, n_timepoints)
        
        Returns:
        features : array, shape (n_trials, n_channels * n_bins)
        """
        n_trials, n_channels, n_timepoints = X.shape
        
        # Calculate samples per 10ms bin
        samples_per_bin = int(self.sfreq * 0.01)  # 10ms in samples
        n_bins = n_timepoints // samples_per_bin
        
        if n_bins < 1:
            raise ValueError(f"Window too short: {n_timepoints} samples, need at least {samples_per_bin}")
        
        # Use complete bins only
        usable_samples = n_bins * samples_per_bin
        if usable_samples < n_timepoints:
            X = X[:, :, :usable_samples]
        
        # Downsample by averaging within 10ms bins
        downsampled_data = np.zeros((n_trials, n_channels, n_bins))
        
        for bin_idx in range(n_bins):
            start_sample = bin_idx * samples_per_bin
            end_sample = start_sample + samples_per_bin
            downsampled_data[:, :, bin_idx] = np.mean(X[:, :, start_sample:end_sample], axis=2)
        
        # Mean-center per trial per channel within the window
        mean_centered_data = downsampled_data.copy()
        for trial_idx in range(n_trials):
            for channel_idx in range(n_channels):
                channel_mean = np.mean(downsampled_data[trial_idx, channel_idx, :])
                mean_centered_data[trial_idx, channel_idx, :] -= channel_mean
        
        # Flatten to feature vectors (channels × timepoints)
        features = mean_centered_data.reshape(n_trials, n_channels * n_bins)
        
        return features
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)

def create_pseudo_trials_from_indices(epochs_data, events, trial_indices, corner_conditions, n_trials_per_condition):
    """
    Create pseudo-trials from specific trial indices only
    This ensures no data leakage between train/test sets
    EXACT COPY from working pseudo.py
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

def decode_with_proper_nested_cv(epochs_data, events, sfreq, feature_name, scheme_params, n_iterations):
    """
    Perform decoding with proper nested CV using spatiotemporal features
    ADAPTED from pseudo.py to include spatiotemporal feature extraction
    """
    # Calculate time window samples
    start_sample = int(TIME_START * sfreq)
    end_sample = int(TIME_END * sfreq)
    
    # Prepare for sliding window
    window_length = int(sfreq * WINDOW_LENGTH_SEC)
    window_step = int(sfreq * WINDOW_STEP_SEC)
    n_windows = int((end_sample - start_sample - window_length) / window_step) + 1
    
    # Storage for results across iterations
    iteration_results = []
    
    for iteration in range(n_iterations):
        write_log(f"\n  Iteration {iteration + 1}/{n_iterations} for {feature_name}")
        
        # Results for this iteration
        window_scores = np.zeros(n_windows)
        window_c_values = []
        window_pca_ratios = []
        
        # Process each time window
        for window_idx in tqdm(range(n_windows), desc=f"{feature_name} windows"):
            win_start = start_sample + window_idx * window_step
            win_end = win_start + window_length
            
            # Extract window data
            window_data = epochs_data[:, :, win_start:win_end]
            
            # Outer cross-validation
            outer_cv = StratifiedKFold(n_splits=OUTER_CV_SPLITS, shuffle=True)
            outer_scores = []
            best_cs = []
            fold_pca_ratios = []
            
            # Create labels based on feature type
            if feature_name == 'voice':
                # Speaker 1+2 vs 3+4
                original_labels = np.array([0 if (e//10)%10 in [1,2] else 1 for e in events])
            else:  # location
                # Location 1+2 vs 3+4
                original_labels = np.array([0 if e%10 in [1,2] else 1 for e in events])
            
            # Outer CV loop - SAME STRUCTURE AS pseudo.py
            for train_idx, test_idx in outer_cv.split(window_data, original_labels):
                # Step 1: Split original trials (SAME AS pseudo.py)
                train_data = window_data[train_idx]
                test_data = window_data[test_idx]
                train_events = events[train_idx]
                test_events = events[test_idx]
                
                # Step 2: Create pseudo-trials separately for train and test (SAME AS pseudo.py)
                train_pseudo_data = []
                train_pseudo_labels = []
                test_pseudo_data = []
                test_pseudo_labels = []
                
                for corner_name, corner_conditions in CORNERS.items():
                    # Create training pseudo-trials
                    train_pseudo = create_pseudo_trials_from_indices(
                        train_data, train_events, 
                        np.arange(len(train_events)),  # All training indices
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
                        np.arange(len(test_events)),  # All test indices
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
                
                # Step 3: Apply spatiotemporal feature extraction + PCA pipeline
                # (MODIFICATION: Replace simple Vectorizer with SpatiotemporalVectorizer + PCA)
                
                # Create spatiotemporal pipeline with PCA
                spatiotemporal_pipeline = make_pipeline(
                    SpatiotemporalVectorizer(sfreq=sfreq),
                    StandardScaler(),
                    PCA(n_components=PCA_VARIANCE_THRESHOLD, svd_solver='full')
                )
                
                # Fit pipeline on training data
                try:
                    X_train_transformed = spatiotemporal_pipeline.fit_transform(X_train)
                    X_test_transformed = spatiotemporal_pipeline.transform(X_test)
                    
                    # Log PCA ratio
                    pca_step = spatiotemporal_pipeline.named_steps['pca']
                    n_components_kept = pca_step.n_components_
                    # Calculate original features: n_channels * n_bins
                    n_bins = int(WINDOW_LENGTH_SEC * sfreq * 0.1)  # 10ms bins
                    n_original_features = X_train.shape[1] * n_bins  # Estimate
                    pca_ratio = n_components_kept / n_original_features if n_original_features > 0 else 0
                    fold_pca_ratios.append(pca_ratio)
                    
                except Exception as e:
                    write_log(f"    PCA failed for window {window_idx}, fold: {str(e)}")
                    continue
                
                # Step 4: Inner CV for hyperparameter selection (SAME STRUCTURE AS pseudo.py)
                best_score = -1
                best_c = 1.0
                
                inner_cv = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True)
                
                for c_value in PARAM_GRID['svc__C']:
                    # Create SVM classifier
                    svm = SVC(kernel='linear', C=c_value, probability=True)
                    
                    # Evaluate on inner CV (SAME AS pseudo.py)
                    try:
                        inner_scores = cross_val_score(
                            svm, X_train_transformed, y_train, 
                            cv=inner_cv, 
                            scoring='accuracy',
                            n_jobs=1  # Keep low to avoid memory issues
                        )
                        mean_inner_score = np.mean(inner_scores)
                        
                        if mean_inner_score > best_score:
                            best_score = mean_inner_score
                            best_c = c_value
                    except Exception as e:
                        continue
                
                # Step 5: Train final model with best C and evaluate on test set (SAME AS pseudo.py)
                final_svm = SVC(kernel='linear', C=best_c, probability=True)
                
                try:
                    final_svm.fit(X_train_transformed, y_train)
                    test_score = final_svm.score(X_test_transformed, y_test)
                    outer_scores.append(test_score)
                    best_cs.append(best_c)
                except Exception as e:
                    continue
            
            # Store results for this window (SAME AS pseudo.py)
            if outer_scores:
                window_scores[window_idx] = np.mean(outer_scores)
                window_pca_ratios.append(np.mean(fold_pca_ratios) if fold_pca_ratios else 0.0)
                
                # Check consistency of C values
                c_counter = Counter(best_cs)
                most_common_c = c_counter.most_common(1)[0][0]
                window_c_values.append(most_common_c)
                
                # Log if C values are inconsistent
                if len(set(best_cs)) > 2:
                    write_log(f"    Warning: Inconsistent C values at window {window_idx}: {best_cs}")
            else:
                window_scores[window_idx] = 0.5
                window_c_values.append(1.0)
                window_pca_ratios.append(0.0)
        
        # Store iteration results
        iteration_results.append({
            'scores': window_scores,
            'c_values': window_c_values,
            'pca_ratios': window_pca_ratios
        })
    
    # Aggregate results across iterations (SAME AS pseudo.py)
    all_scores = np.array([r['scores'] for r in iteration_results])
    all_pca_ratios = np.array([r['pca_ratios'] for r in iteration_results])
    
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)
    mean_pca_ratios = np.mean(all_pca_ratios, axis=0)
    
    # Create time points (SAME AS pseudo.py)
    timepoints = np.array([TIME_START + i * WINDOW_STEP_SEC for i in range(n_windows)])
    
    return mean_scores, std_scores, timepoints, all_scores, mean_pca_ratios, iteration_results

def load_subject_data(subject, meta_info):
    """
    Load and preprocess data for a single subject
    EXACT COPY from pseudo.py with minor modifications for S1 analysis
    """
    write_log(f"\nLoading S1 data for subject {subject}...")
    
    try:
        # Load metadata (SAME AS pseudo.py)
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
        
        # Load raw data and events (SAME AS pseudo.py)
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
        
        # Get S1 events (SAME AS pseudo.py)
        S1_idx = [i - 1 for i in range(len(all_events[:,2])) if all_events[i,2] == 100]
        S1_values = all_events[S1_idx,2]
        
        # Handle special cases (SAME AS pseudo.py)
        if subject == 23:
            drop_idx = 64 * 7
            S1_values = np.delete(S1_values, drop_idx)
        
        # Load cleaned epochs (SAME AS pseudo.py)
        clean_trials = mne.read_epochs(
            f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif",
            preload=True,
            verbose='ERROR'
        )
        
        if subject == 28:
            clean_trials.drop(63)
        
        clean_trials.events[:,2] = S1_values
        clean_trials.event_id = S1_EVENT_DICT
        
        # Crop to S1 period (SAME AS pseudo.py)
        S1_epochs = clean_trials.copy()
        S1_epochs.crop(tmin=0, tmax=1)
        
        # Select magnetometers and resample (MODIFIED: 500Hz instead of 100Hz)
        mag_epochs = S1_epochs.copy().pick_types(meg='mag')
        mag_epochs = mag_epochs.resample(RESAMPLE_FREQ, npad='auto')
        
        # Get data
        epochs_data = mag_epochs.get_data(copy=False)
        write_log(f"Data loaded successfully. Shape: {epochs_data.shape}")
        write_log(f"Sampling frequency: {mag_epochs.info['sfreq']} Hz")
        
        return epochs_data, mag_epochs.events[:, 2], mag_epochs.info['sfreq']
        
    except Exception as e:
        write_log(f"Error loading data: {str(e)}")
        import traceback
        write_log(traceback.format_exc())
        return None, None, None

def main():
    """Main processing function - ADAPTED from pseudo.py structure"""
    # Load metadata
    meta_info = pd.read_excel(META_FILE)
    
    # Load subject data
    epochs_data, events, sfreq = load_subject_data(args.subject, meta_info)
    
    if epochs_data is None:
        write_log("Failed to load data. Exiting.")
        sys.exit(1)
    
    # Results storage
    all_results = {}
    
    # Process ONLY avg12 scheme (SIMPLIFIED from pseudo.py)
    scheme_name = 'avg12'
    scheme_params = AVERAGING_SCHEME
    
    write_log(f"\n--- Processing {scheme_name} with spatiotemporal features ---")
    
    scheme_results = {}
    
    # Process each feature
    for feature_name in ['voice', 'location']:
        write_log(f"\n  Decoding {feature_name}...")
        
        # Run decoding with proper nested CV + spatiotemporal features
        mean_scores, std_scores, timepoints, all_scores, mean_pca_ratios, iteration_results = decode_with_proper_nested_cv(
            epochs_data, events, sfreq, feature_name, scheme_params, args.n_iterations
        )
        
        # Store results
        scheme_results[feature_name] = {
            'mean': mean_scores,
            'std': std_scores,
            'timepoints': timepoints,
            'all_scores': all_scores,
            'pca_ratios': mean_pca_ratios,
            'iterations': iteration_results
        }
        
        # Log summary
        write_log(f"    Mean accuracy: {np.mean(mean_scores):.3f}")
        write_log(f"    Peak accuracy: {np.max(mean_scores):.3f}")
        write_log(f"    Mean PCA ratio: {np.mean(mean_pca_ratios):.3f}")
        
    all_results[scheme_name] = scheme_results
    
    # Save results (ADAPTED from pseudo.py)
    write_log(f"\nSaving results...")
    
    # Save as numpy arrays
    for feature in ['voice', 'location']:
        np.save(f'{OUTPUT_DIR}/avg12_{feature}_mean.npy', 
               scheme_results[feature]['mean'])
        np.save(f'{OUTPUT_DIR}/avg12_{feature}_std.npy', 
               scheme_results[feature]['std'])
        np.save(f'{OUTPUT_DIR}/avg12_{feature}_timepoints.npy', 
               scheme_results[feature]['timepoints'])
        np.save(f'{OUTPUT_DIR}/avg12_{feature}_pca_ratios.npy', 
               scheme_results[feature]['pca_ratios'])
        
        # Save individual iteration scores
        iteration_scores = np.array([r['scores'] for r in scheme_results[feature]['iterations']])
        np.save(f'{OUTPUT_DIR}/avg12_{feature}_scores.npy', iteration_scores)
    
    # Save summary (ADAPTED from pseudo.py)
    summary = {
        'subject': args.subject,
        'n_iterations': args.n_iterations,
        'scheme': 'avg12',
        'processing_time': str(datetime.now()),
        'method': 'spatiotemporal_proper_nested_cv',
        'feature_extraction': 'wolff_style_spatiotemporal',
        'pca_variance_threshold': PCA_VARIANCE_THRESHOLD,
        'analysis_window': f'{TIME_START}-{TIME_END}s',
        'results_summary': {}
    }
    
    summary['results_summary']['avg12'] = {
        'voice': {
            'mean_accuracy': float(np.mean(scheme_results['voice']['mean'])),
            'max_accuracy': float(np.max(scheme_results['voice']['mean'])),
            'mean_pca_ratio': float(np.mean(scheme_results['voice']['pca_ratios']))
        },
        'location': {
            'mean_accuracy': float(np.mean(scheme_results['location']['mean'])),
            'max_accuracy': float(np.max(scheme_results['location']['mean'])),
            'mean_pca_ratio': float(np.mean(scheme_results['location']['pca_ratios']))
        }
    }
    
    with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    write_log(f"\nProcessing completed at: {datetime.now()}")
    
    print(f"Subject {args.subject} S1 spatiotemporal analysis completed successfully!")

if __name__ == "__main__":
    main()