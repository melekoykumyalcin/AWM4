#!/usr/bin/env python
"""
Comprehensive Decoding Analysis Script
Compares three pseudo-trial approaches:
1. Extreme Contrast: Sp1 vs Sp4, L1 vs L4
2. Feature-Pure: Ignoring orthogonal dimension
3. Condition-Specific: Maintaining all 16 conditions

Usage: python comprehensive_decoding_analysis.py --subject 23
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
parser = argparse.ArgumentParser(description='Run comprehensive decoding analysis with three approaches')
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
OUTPUT_DIR = PROCESSED_DIR + f'extremesDecoding/subject_{args.subject}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log file path
LOG_FILE_PATH = f'{OUTPUT_DIR}/processing_log.txt'

def write_log(message):
    """Write message to log file"""
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(message + '\n')

# Initialize log
write_log(f"Comprehensive decoding analysis started at: {datetime.now()}")
write_log(f"Subject: {args.subject}")
write_log(f"Three approaches: Extreme Contrast, Feature-Pure, Condition-Specific")
write_log(f"Window length: 100ms, Step: 10ms")

# Decoding parameters
RESAMPLE_FREQ = 100  # Hz
WINDOW_LENGTH_SEC = 0.1  # 100ms fixed
WINDOW_STEP_SEC = 0.01  # 10ms

# Delay Period Configuration for data loading
DELAY_CONFIG = {
    'tmin': 2.0,
    'tmax': 4.7,
    'timepoints': np.linspace(2.0, 4.7, int((4.7-2.0)*RESAMPLE_FREQ)),
    'description': 'full delay period'
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

# Define extreme groupings for Approach 1
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

# Define feature-pure groupings for Approach 2
FEATURE_PURE_GROUPINGS = {
    'voice_pure': {
        'groups': {
            'Sp1': [111, 112, 113, 114, 211, 212, 213, 214],
            'Sp2': [121, 122, 123, 124, 221, 222, 223, 224],
            'Sp3': [131, 132, 133, 134, 231, 232, 233, 234],
            'Sp4': [141, 142, 143, 144, 241, 242, 243, 244]
        },
        'labels': {'Sp1': 0, 'Sp2': 0, 'Sp3': 1, 'Sp4': 1}
    },
    'location_pure': {
        'groups': {
            'L1': [111, 121, 131, 141, 211, 221, 231, 241],
            'L2': [112, 122, 132, 142, 212, 222, 232, 242],
            'L3': [113, 123, 133, 143, 213, 223, 233, 243],
            'L4': [114, 124, 134, 144, 214, 224, 234, 244]
        },
        'labels': {'L1': 0, 'L2': 0, 'L3': 1, 'L4': 1}
    }
}

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
    """Load and preprocess data for comprehensive analysis"""
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
        
        return epochs_data, maintained_events, mag_epochs.info['sfreq']
        
    except Exception as e:
        write_log(f"Error loading data: {str(e)}")
        import traceback
        write_log(traceback.format_exc())
        return None, None, None

def create_pseudo_trials_extreme(data, labels, n_trials_per_pseudo=6):
    """Create pseudo-trials for extreme contrast (Approach 1)"""
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

def create_feature_pure_pseudo_trials(data, events, feature='voice'):
    """
    Simplified feature-pure pseudo-trial creation
    """
    if feature == 'voice':
        # Group by speaker: Sp1,2 vs Sp3,4
        low_conditions = [111, 112, 113, 114, 121, 122, 123, 124,  # Sp1, Sp2
                         211, 212, 213, 214, 221, 222, 223, 224]
        high_conditions = [131, 132, 133, 134, 141, 142, 143, 144,  # Sp3, Sp4
                          231, 232, 233, 234, 241, 242, 243, 244]
    else:  # location
        # Group by location: L1,2 vs L3,4
        low_conditions = [111, 112, 121, 122, 131, 132, 141, 142,  # L1, L2
                         211, 212, 221, 222, 231, 232, 241, 242]
        high_conditions = [113, 114, 123, 124, 133, 134, 143, 144,  # L3, L4
                          213, 214, 223, 224, 233, 234, 243, 244]
    
    # Create labels
    labels = []
    for event in events:
        if event in low_conditions:
            labels.append(0)
        elif event in high_conditions:
            labels.append(1)
        else:
            labels.append(-1)  # Invalid
    
    # Filter valid trials
    valid_indices = np.array(labels) != -1
    if np.sum(valid_indices) < 12:
        return None, None
    
    filtered_data = data[valid_indices]
    filtered_labels = np.array(labels)[valid_indices]
    
    # Create pseudo-trials
    return create_pseudo_trials_extreme(filtered_data, filtered_labels, n_trials_per_pseudo=3)

def create_condition_specific_pseudo_trials(data, events, feature='voice'):
    """
    Simplified condition-specific pseudo-trial creation
    """
    # Create labels based on feature
    labels = []
    for event in events:
        if event >= 111 and event <= 244:  # Valid event codes
            if feature == 'voice':
                speaker = (event // 10) % 10  # Extract speaker (1-4)
                labels.append(0 if speaker <= 2 else 1)  # Sp1,2 vs Sp3,4
            else:  # location
                location = event % 10  # Extract location (1-4)
                labels.append(0 if location <= 2 else 1)  # L1,2 vs L3,4
        else:
            labels.append(-1)  # Invalid
    
    # Filter valid trials
    valid_indices = np.array(labels) != -1
    if np.sum(valid_indices) < 12:
        return None, None
    
    filtered_data = data[valid_indices]
    filtered_labels = np.array(labels)[valid_indices]
    
    # Create pseudo-trials
    return create_pseudo_trials_extreme(filtered_data, filtered_labels, n_trials_per_pseudo=3)

def decode_sliding_window_fixed(period_data, events, sfreq, period_config, approach_name, 
                               feature_name, n_iterations, pseudo_trial_params):
    """
    Fixed sliding window decoding - clean and simple approach
    """
    write_log(f"\n  {approach_name} - {feature_name} in {period_config['description']}")
    
    # Sliding window parameters
    window_length = int(sfreq * WINDOW_LENGTH_SEC)
    window_step = int(sfreq * WINDOW_STEP_SEC)
    n_times = period_data.shape[2]
    n_windows = int((n_times - window_length) / window_step) + 1
    
    # Storage for results across iterations
    all_scores = []
    
    for iteration in range(n_iterations):
        window_scores = []
        
        # Process each sliding window
        for window_idx in tqdm(range(n_windows), desc=f"{approach_name} {feature_name} iter {iteration+1}", 
                              disable=True):
            win_start = window_idx * window_step
            win_end = win_start + window_length
            
            # Extract window data and flatten
            window_data = period_data[:, :, win_start:win_end]
            n_trials, n_channels, n_times_win = window_data.shape
            flattened_data = window_data.reshape(n_trials, n_channels * n_times_win)
            
            # Create pseudo-trials and labels based on approach
            if approach_name == "Extreme Contrast":
                # For extreme contrast, we already have filtered data and simple labels
                valid_indices = pseudo_trial_params['valid_indices']
                labels = pseudo_trial_params['labels']
                
                filtered_data = flattened_data[valid_indices]
                filtered_labels = np.array(labels)
                
                # Create pseudo-trials
                pseudo_data, pseudo_labels = create_pseudo_trials_extreme(
                    filtered_data, filtered_labels, n_trials_per_pseudo=6
                )
                
            elif approach_name == "Feature-Pure":
                # For feature-pure, create pseudo-trials ignoring orthogonal dimension
                pseudo_data, pseudo_labels = create_feature_pure_pseudo_trials(
                    flattened_data, events, pseudo_trial_params['feature']
                )
                
            elif approach_name == "Condition-Specific":
                # For condition-specific, create pseudo-trials within each condition
                pseudo_data, pseudo_labels = create_condition_specific_pseudo_trials(
                    flattened_data, events, pseudo_trial_params['feature']
                )
            
            # Check if we have valid data
            if (pseudo_data is None or len(pseudo_data) < 6 or 
                len(np.unique(pseudo_labels)) < 2):
                window_scores.append(0.5)
                continue
            
            # Simple cross-validation on pseudo-trials
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
            
            # Create classifier
            clf = make_pipeline(
                StandardScaler(),
                SVC(kernel='linear', C=1.0, probability=True)
            )
            
            try:
                # Cross-validate directly on pseudo-trials
                cv_scores = cross_val_score(clf, pseudo_data, pseudo_labels, 
                                          cv=cv, scoring='accuracy', n_jobs=1)
                window_scores.append(np.mean(cv_scores))
            except:
                window_scores.append(0.5)
        
        all_scores.append(window_scores)
    
    # Aggregate results
    all_scores = np.array(all_scores)
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)
    
    # Create time points
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
        'approach': approach_name,
        'feature': feature_name
    }

def run_extreme_contrast_decoding(epochs_data, events, sfreq, period_config, n_iterations):
    """Fixed extreme contrast decoding"""
    write_log(f"\n--- Approach 1: Extreme Contrast Decoding ---")
    
    results = {}
    
    for comparison_name, groupings in EXTREME_GROUPINGS.items():
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
        
        if len(valid_indices) < 20:
            write_log(f"  Not enough trials for {comparison_name}: {len(valid_indices)}")
            continue
        
        # Extract period data
        period_start_sample = int((period_config['tmin'] - DELAY_CONFIG['tmin']) * sfreq)
        period_end_sample = int((period_config['tmax'] - DELAY_CONFIG['tmin']) * sfreq)
        period_data = epochs_data[:, :, period_start_sample:period_end_sample]
        
        # Prepare parameters for the decoding function
        pseudo_trial_params = {
            'valid_indices': valid_indices,
            'labels': labels
        }
        
        # Run decoding
        result = decode_sliding_window_fixed(
            period_data, events, sfreq, period_config,
            "Extreme Contrast", comparison_name, n_iterations,
            pseudo_trial_params
        )
        
        results[comparison_name] = result
    
    return results

def run_feature_pure_decoding(epochs_data, events, sfreq, period_config, n_iterations):
    """Fixed feature-pure decoding"""
    write_log(f"\n--- Approach 2: Feature-Pure Decoding ---")
    
    results = {}
    
    # Extract period data
    period_start_sample = int((period_config['tmin'] - DELAY_CONFIG['tmin']) * sfreq)
    period_end_sample = int((period_config['tmax'] - DELAY_CONFIG['tmin']) * sfreq)
    period_data = epochs_data[:, :, period_start_sample:period_end_sample]
    
    # Voice decoding
    pseudo_trial_params = {'feature': 'voice'}
    result = decode_sliding_window_fixed(
        period_data, events, sfreq, period_config,
        "Feature-Pure", "voice_pure", n_iterations,
        pseudo_trial_params
    )
    results['voice_pure'] = result
    
    # Location decoding  
    pseudo_trial_params = {'feature': 'location'}
    result = decode_sliding_window_fixed(
        period_data, events, sfreq, period_config,
        "Feature-Pure", "location_pure", n_iterations,
        pseudo_trial_params
    )
    results['location_pure'] = result
    
    return results

def run_condition_specific_decoding(epochs_data, events, sfreq, period_config, n_iterations):
    """Fixed condition-specific decoding"""
    write_log(f"\n--- Approach 3: Condition-Specific Decoding ---")
    
    results = {}
    
    # Extract period data
    period_start_sample = int((period_config['tmin'] - DELAY_CONFIG['tmin']) * sfreq)
    period_end_sample = int((period_config['tmax'] - DELAY_CONFIG['tmin']) * sfreq)
    period_data = epochs_data[:, :, period_start_sample:period_end_sample]
    
    # Voice decoding
    pseudo_trial_params = {'feature': 'voice'}
    result = decode_sliding_window_fixed(
        period_data, events, sfreq, period_config,
        "Condition-Specific", "voice_specific", n_iterations,
        pseudo_trial_params
    )
    results['voice_specific'] = result
    
    # Location decoding
    pseudo_trial_params = {'feature': 'location'}
    result = decode_sliding_window_fixed(
        period_data, events, sfreq, period_config,
        "Condition-Specific", "location_specific", n_iterations,
        pseudo_trial_params
    )
    results['location_specific'] = result
    
    return results

def main():
    """Main processing function"""
    # Load metadata
    meta_info = pd.read_excel(META_FILE)
    
    # Load subject data
    epochs_data, events, sfreq = load_subject_data(args.subject, meta_info)
    
    if epochs_data is None:
        write_log("Failed to load data. Exiting.")
        sys.exit(1)
    
    write_log("\n=== COMPREHENSIVE DECODING ANALYSIS ===")
    
    # Results storage
    all_results = {}
    
    # Run all three decoding approaches over the full delay period
    period_results = {}
    
    # Approach 1: Extreme Contrast
    extreme_results = run_extreme_contrast_decoding(
        epochs_data, events, sfreq, DELAY_CONFIG, args.n_iterations
    )
    period_results['extreme_contrast'] = extreme_results
    
    # Approach 2: Feature-Pure
    feature_pure_results = run_feature_pure_decoding(
        epochs_data, events, sfreq, DELAY_CONFIG, args.n_iterations
    )
    period_results['feature_pure'] = feature_pure_results
    
    # Approach 3: Condition-Specific
    condition_specific_results = run_condition_specific_decoding(
        epochs_data, events, sfreq, DELAY_CONFIG, args.n_iterations
    )
    period_results['condition_specific'] = condition_specific_results

    all_results['full_delay'] = period_results
    
    # Save results
    write_log(f"\nSaving results...")
    
    # Save temporal results for each approach
    for period_name, period_results in all_results.items():
        for approach_name, approach_results in period_results.items():
            for feature_name, results in approach_results.items():
                # Save temporal data
                np.save(f'{OUTPUT_DIR}/{period_name}_{approach_name}_{feature_name}_mean_scores.npy', 
                       results['mean_scores'])
                np.save(f'{OUTPUT_DIR}/{period_name}_{approach_name}_{feature_name}_std_scores.npy', 
                       results['std_scores'])
                np.save(f'{OUTPUT_DIR}/{period_name}_{approach_name}_{feature_name}_timepoints.npy', 
                       results['timepoints'])
                np.save(f'{OUTPUT_DIR}/{period_name}_{approach_name}_{feature_name}_all_scores.npy', 
                       results['all_scores'])
    
    # Create comprehensive summary
    summary = {
        'subject': args.subject,
        'n_iterations': args.n_iterations,
        'periods': list(all_results.keys()),
        'approaches': ['extreme_contrast', 'feature_pure', 'condition_specific'],
        'processing_time': str(datetime.now()),
        'window_length_ms': WINDOW_LENGTH_SEC * 1000,
        'window_step_ms': WINDOW_STEP_SEC * 1000,
        'normalization': 'StandardScaler (feature-wise)',
        'results': {}
    }
    
    # Add detailed results and comparisons
    for period_name, period_results in all_results.items():
        summary['results'][period_name] = {}
        
        # Store results for each approach
        for approach_name, approach_results in period_results.items():
            summary['results'][period_name][approach_name] = {}
            for feature_name, results in approach_results.items():
                summary['results'][period_name][approach_name][feature_name] = {
                    'mean_accuracy': float(np.mean(results['mean_scores'])),
                    'peak_accuracy': float(np.max(results['mean_scores'])),
                    'peak_time': float(results['timepoints'][np.argmax(results['mean_scores'])]),
                    'n_timepoints': len(results['timepoints'])
                }
    
    # Compare approaches
    write_log(f"\n=== APPROACH COMPARISONS ===")
    
    for period_name in all_results.keys():
        write_log(f"\n{period_name}:")
        
        # Voice comparisons
        write_log(f"\nVoice decoding:")
        if 'voice_extreme' in all_results[period_name]['extreme_contrast']:
            extreme_voice = np.mean(all_results[period_name]['extreme_contrast']['voice_extreme']['mean_scores'])
            write_log(f"  Extreme (Sp1 vs Sp4):     {extreme_voice:.3f}")
        
        if 'voice_pure' in all_results[period_name]['feature_pure']:
            pure_voice = np.mean(all_results[period_name]['feature_pure']['voice_pure']['mean_scores'])
            write_log(f"  Feature-Pure:             {pure_voice:.3f}")
        
        if 'voice_specific' in all_results[period_name]['condition_specific']:
            specific_voice = np.mean(all_results[period_name]['condition_specific']['voice_specific']['mean_scores'])
            write_log(f"  Condition-Specific:       {specific_voice:.3f}")
        
        # Location comparisons
        write_log(f"\nLocation decoding:")
        if 'location_extreme' in all_results[period_name]['extreme_contrast']:
            extreme_location = np.mean(all_results[period_name]['extreme_contrast']['location_extreme']['mean_scores'])
            write_log(f"  Extreme (L1 vs L4):       {extreme_location:.3f}")
        
        if 'location_pure' in all_results[period_name]['feature_pure']:
            pure_location = np.mean(all_results[period_name]['feature_pure']['location_pure']['mean_scores'])
            write_log(f"  Feature-Pure:             {pure_location:.3f}")
        
        if 'location_specific' in all_results[period_name]['condition_specific']:
            specific_location = np.mean(all_results[period_name]['condition_specific']['location_specific']['mean_scores'])
            write_log(f"  Condition-Specific:       {specific_location:.3f}")
    
    # Save summary
    with open(f'{OUTPUT_DIR}/comprehensive_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    write_log(f"\nProcessing completed at: {datetime.now()}")
    
    print(f"Subject {args.subject} comprehensive analysis completed successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
