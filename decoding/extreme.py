#!/usr/bin/env python
"""
Extreme Contrast Decoding Script
Decodes Sp1 vs Sp4 (voice) and L1 vs L4 (location) using 100ms windows
Uses pseudo-trial averaging (12 trials per pseudo-trial)

Usage: python extreme_contrast_decoding.py --subject 23
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
parser = argparse.ArgumentParser(description='Run extreme contrast decoding (Sp1 vs Sp4, L1 vs L4)')
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
OUTPUT_DIR = PROCESSED_DIR + f'extremeContrastAnalysis/subject_{args.subject}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log file path
LOG_FILE_PATH = f'{OUTPUT_DIR}/processing_log.txt'

def write_log(message):
    """Write message to log file"""
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(message + '\n')

# Initialize log
write_log(f"Extreme contrast decoding started at: {datetime.now()}")
write_log(f"Subject: {args.subject}")
write_log(f"Decoding: Sp1 vs Sp4 (voice), L1 vs L4 (location)")
write_log(f"Window length: 100ms, Step: 10ms")

# Decoding parameters
RESAMPLE_FREQ = 100  # Hz
WINDOW_LENGTH_SEC = 0.1  # 100ms fixed
WINDOW_STEP_SEC = 0.01  # 10ms

# Define ping periods
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

# Define extreme groupings
EXTREME_GROUPINGS = {
    'voice_extreme': {
        'Sp1_only': [111, 112, 113, 114, 211, 212, 213, 214],  # All Sp1 trials
        'Sp4_only': [141, 142, 143, 144, 241, 242, 243, 244],  # All Sp4 trials
        'label_names': ['Sp1', 'Sp4']
    },
    'location_extreme': {
        'L1_only': [111, 121, 131, 141, 211, 221, 231, 241],   # All L1 trials
        'L4_only': [114, 124, 134, 144, 214, 224, 234, 244],   # All L4 trials
        'label_names': ['L1', 'L4']
    }
}

# Averaging parameters 
# For extreme contrasts: 6 trials per condition × 2 conditions = 12 total
AVERAGING_SCHEME = {'trials_per_condition': 6, 'total_trials': 12}

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

def load_subject_data(subject, meta_info):
    """Load and preprocess data for extreme contrast analysis"""
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
                clean_trials.drop(drop_idx)
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
        
        # Log distribution for extreme contrasts
        sp1_count = sum(1 for v in maintained_events if v in EXTREME_GROUPINGS['voice_extreme']['Sp1_only'])
        sp4_count = sum(1 for v in maintained_events if v in EXTREME_GROUPINGS['voice_extreme']['Sp4_only'])
        l1_count = sum(1 for v in maintained_events if v in EXTREME_GROUPINGS['location_extreme']['L1_only'])
        l4_count = sum(1 for v in maintained_events if v in EXTREME_GROUPINGS['location_extreme']['L4_only'])
        
        write_log(f"Extreme contrast distribution:")
        write_log(f"  Sp1: {sp1_count}, Sp4: {sp4_count}")
        write_log(f"  L1: {l1_count}, L4: {l4_count}")
        
        return epochs_data, maintained_events, mag_epochs.info['sfreq']
        
    except Exception as e:
        write_log(f"Error loading data: {str(e)}")
        import traceback
        write_log(traceback.format_exc())
        return None, None, None

def create_pseudo_trials_extreme(data, labels, n_trials_per_pseudo):
    """Create pseudo-trials by averaging n_trials_per_pseudo trials from same class"""
    pseudo_trials = []
    pseudo_labels = []
    
    # Process each class
    for class_label in np.unique(labels):
        class_indices = np.where(labels == class_label)[0]
        
        # Shuffle indices
        np.random.shuffle(class_indices)
        
        # Create pseudo-trials
        n_pseudo = len(class_indices) // n_trials_per_pseudo
        
        for i in range(n_pseudo):
            start_idx = i * n_trials_per_pseudo
            end_idx = start_idx + n_trials_per_pseudo
            
            if end_idx <= len(class_indices):
                selected_indices = class_indices[start_idx:end_idx]
                pseudo_trial = np.mean(data[selected_indices], axis=0)
                pseudo_trials.append(pseudo_trial)
                pseudo_labels.append(class_label)
    
    if len(pseudo_trials) > 0:
        return np.array(pseudo_trials), np.array(pseudo_labels)
    else:
        return None, None

def decode_extreme_period(epochs_data, events, sfreq, period_config, comparison_name, groupings, n_iterations):
    """
    Decode extreme contrasts for a specific period using 100ms sliding windows
    """
    write_log(f"\n  Decoding {comparison_name} in {period_config['description']}")
    write_log(f"  Time window: {period_config['tmin']}s to {period_config['tmax']}s")
    
    # Calculate time window samples
    period_start_sample = int((period_config['tmin'] - DELAY_CONFIG['tmin']) * sfreq)
    period_end_sample = int((period_config['tmax'] - DELAY_CONFIG['tmin']) * sfreq)
    
    # Extract period data
    period_data = epochs_data[:, :, period_start_sample:period_end_sample]
    
    write_log(f"  Period data shape: {period_data.shape}")
    
    # Sliding window parameters
    window_length = int(sfreq * WINDOW_LENGTH_SEC)
    window_step = int(sfreq * WINDOW_STEP_SEC)
    n_times = period_data.shape[2]
    n_windows = int((n_times - window_length) / window_step) + 1
    
    write_log(f"  Number of sliding windows: {n_windows}")
    
    # Get groupings
    if comparison_name == 'voice_extreme':
        group1_conditions = groupings['Sp1_only']
        group2_conditions = groupings['Sp4_only']
    else:  # location_extreme
        group1_conditions = groupings['L1_only']
        group2_conditions = groupings['L4_only']
    
    # Filter data to only include relevant trials
    valid_indices = []
    labels = []
    
    for idx, event in enumerate(events):
        if event in group1_conditions:
            valid_indices.append(idx)
            labels.append(0)
        elif event in group2_conditions:
            valid_indices.append(idx)
            labels.append(1)
    
    if len(valid_indices) < 20:  # Need reasonable number of trials
        write_log(f"  Not enough trials for {comparison_name}: {len(valid_indices)}")
        return None
    
    # Filter data
    filtered_data = period_data[valid_indices]
    filtered_labels = np.array(labels)
    
    write_log(f"  Using {len(valid_indices)} trials: {sum(filtered_labels==0)} class 0, {sum(filtered_labels==1)} class 1")
    
    # Storage for results across iterations
    iteration_results = []
    
    for iteration in range(n_iterations):
        write_log(f"    Iteration {iteration + 1}/{n_iterations}")
        
        # Results for this iteration
        window_scores = np.zeros(n_windows)
        window_c_values = []
        
        # Process each sliding window
        for window_idx in tqdm(range(n_windows), desc=f"{comparison_name} iter {iteration+1}"):
            win_start = window_idx * window_step
            win_end = win_start + window_length
            
            # Extract window data and flatten
            window_data = filtered_data[:, :, win_start:win_end]
            n_trials, n_channels, n_times_win = window_data.shape
            flattened_data = window_data.reshape(n_trials, n_channels * n_times_win)
            
            # Outer cross-validation
            outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
            outer_scores = []
            best_cs = []
            
            # Outer CV loop
            for train_idx, test_idx in outer_cv.split(flattened_data, filtered_labels):
                
                # Create pseudo-trials for train and test separately
                train_pseudo_data, train_pseudo_labels = create_pseudo_trials_extreme(
                    flattened_data[train_idx], 
                    filtered_labels[train_idx],
                    AVERAGING_SCHEME['trials_per_condition']
                )
                
                test_pseudo_data, test_pseudo_labels = create_pseudo_trials_extreme(
                    flattened_data[test_idx], 
                    filtered_labels[test_idx],
                    AVERAGING_SCHEME['trials_per_condition']
                )
                
                # Skip if not enough pseudo-trials
                if (train_pseudo_data is None or test_pseudo_data is None or
                    len(train_pseudo_data) < 4 or len(test_pseudo_data) < 2 or
                    len(np.unique(train_pseudo_labels)) < 2 or len(np.unique(test_pseudo_labels)) < 2):
                    continue
                
                # Inner CV for hyperparameter selection
                best_score = -1
                best_c = 1.0
                
                inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=None)
                
                for c_value in PARAM_GRID:
                    clf = make_pipeline(
                        StandardScaler(),
                        SVC(kernel='linear', C=c_value, probability=True)
                    )
                    
                    try:
                        inner_scores = cross_val_score(
                            clf, train_pseudo_data, train_pseudo_labels, 
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
                
                # Train final model
                final_clf = make_pipeline(
                    StandardScaler(),
                    SVC(kernel='linear', C=best_c, probability=True)
                )
                
                try:
                    final_clf.fit(train_pseudo_data, train_pseudo_labels)
                    test_score = final_clf.score(test_pseudo_data, test_pseudo_labels)
                    outer_scores.append(test_score)
                    best_cs.append(best_c)
                except:
                    continue
            
            # Store results for this window
            if outer_scores:
                window_scores[window_idx] = np.mean(outer_scores)
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
    
    return {
        'mean_scores': mean_scores,
        'std_scores': std_scores,
        'timepoints': timepoints,
        'all_scores': all_scores,
        'iteration_results': iteration_results,
        'groupings': groupings,
        'comparison_name': comparison_name
    }

def main():
    """Main processing function"""
    # Load metadata
    meta_info = pd.read_excel(META_FILE)
    
    # Load subject data
    epochs_data, events, sfreq = load_subject_data(args.subject, meta_info)
    
    if epochs_data is None:
        write_log("Failed to load data. Exiting.")
        sys.exit(1)
    
    write_log("\n=== EXTREME CONTRAST DECODING ANALYSIS ===")
    write_log("=== Sp1 vs Sp4 (voice) and L1 vs L4 (location) ===")
    
    # Results storage
    all_results = {}
    
    # Process each ping period
    for period_name, period_config in PING_PERIODS.items():
        write_log(f"\n--- Processing {period_name}: {period_config['description']} ---")
        
        period_results = {}
        
        # Process each extreme comparison
        for comparison_name, groupings in EXTREME_GROUPINGS.items():
            
            results = decode_extreme_period(
                epochs_data, events, sfreq, period_config, 
                comparison_name, groupings, args.n_iterations
            )
            
            if results is not None:
                period_results[comparison_name] = results
            else:
                write_log(f"  Skipping {comparison_name} due to insufficient data")
        
        all_results[period_name] = period_results
    
    # Save results
    write_log(f"\nSaving results...")
    
    # Save temporal results
    for period_name, period_results in all_results.items():
        for comparison_name, results in period_results.items():
            # Save temporal data
            np.save(f'{OUTPUT_DIR}/{period_name}_{comparison_name}_mean_scores.npy', 
                   results['mean_scores'])
            np.save(f'{OUTPUT_DIR}/{period_name}_{comparison_name}_std_scores.npy', 
                   results['std_scores'])
            np.save(f'{OUTPUT_DIR}/{period_name}_{comparison_name}_timepoints.npy', 
                   results['timepoints'])
            np.save(f'{OUTPUT_DIR}/{period_name}_{comparison_name}_all_scores.npy', 
                   results['all_scores'])
            np.save(f'{OUTPUT_DIR}/{period_name}_{comparison_name}_c_values.npy', 
                   [r['c_values'] for r in results['iteration_results']])
    
    # Save comprehensive summary
    summary = {
        'subject': args.subject,
        'n_iterations': args.n_iterations,
        'periods': list(all_results.keys()),
        'comparisons': list(EXTREME_GROUPINGS.keys()),
        'processing_time': str(datetime.now()),
        'method': 'extreme_contrast_decoding',
        'window_length_ms': WINDOW_LENGTH_SEC * 1000,
        'window_step_ms': WINDOW_STEP_SEC * 1000,
        'averaging_scheme': AVERAGING_SCHEME,
        'normalization': 'StandardScaler (feature-wise)',
        'results': {}
    }
    
    # Add detailed results
    for period_name, period_results in all_results.items():
        summary['results'][period_name] = {}
        for comparison_name, results in period_results.items():
            if results is not None:
                summary['results'][period_name][comparison_name] = {
                    'mean_accuracy': float(np.mean(results['mean_scores'])),
                    'peak_accuracy': float(np.max(results['mean_scores'])),
                    'peak_time': float(results['timepoints'][np.argmax(results['mean_scores'])]),
                    'time_window': f"{period_config['tmin']}-{period_config['tmax']}s",
                    'n_timepoints': len(results['timepoints']),
                    'groupings': results['groupings']
                }
    
    # Compare periods
    write_log(f"\n=== COMPARISON BETWEEN PERIODS ===")
    for comparison_name in EXTREME_GROUPINGS.keys():
        if (comparison_name in all_results['pre_ping'] and 
            comparison_name in all_results['post_ping']):
            
            pre_mean = np.mean(all_results['pre_ping'][comparison_name]['mean_scores'])
            post_mean = np.mean(all_results['post_ping'][comparison_name]['mean_scores'])
            difference = post_mean - pre_mean
            
            write_log(f"{comparison_name}:")
            write_log(f"  Pre-ping:  {pre_mean:.3f}")
            write_log(f"  Post-ping: {post_mean:.3f}")
            write_log(f"  Difference: {difference:+.3f}")
            
            summary['results'][f'{comparison_name}_comparison'] = {
                'pre_ping_mean_accuracy': float(pre_mean),
                'post_ping_mean_accuracy': float(post_mean),
                'difference': float(difference)
            }
    
    with open(f'{OUTPUT_DIR}/extreme_contrast_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    write_log(f"\nProcessing completed at: {datetime.now()}")
    
    print(f"Subject {args.subject} extreme contrast analysis completed successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()