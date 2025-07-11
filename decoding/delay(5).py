#!/usr/bin/env python
"""
Optimal Pseudotrial Delay Period Decoding Analysis
Based on Scrivener et al. 2023 best practices and trial count analysis
"""

import os
import sys
import locale
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.ioff()
plt.rcParams['figure.figsize'] = [10, 8]

import mne
mne.set_log_level('WARNING')  # Reduce verbosity for HPC logs
import pandas as pd
import pickle

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from mne.decoding import cross_val_multiscore, SlidingEstimator
from joblib import Parallel, delayed
from collections import Counter


# Constants and Configuration
HOME_DIR = '/mnt/hpc/projects/awm4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
DATA_PATH = HOME_DIR + '/AWM4_data/raw/'
CORRECTED_DATA = HOME_DIR + '/AWM4_data/raw/correctTriggers/'

# Optimal Parameters (from trial count analysis)
TRIALS_PER_PSEUDOTRIAL = 5      # Optimal for your dataset
PSEUDOTRIALS_PER_CONDITION = 4  # Maximizes data usage
N_ITERATIONS = 10              
RESAMPLING_FACTOR = 1           # No resampling needed with good trial counts

# Decoding Parameters
RESAMPLE_FREQ = 100  # Hz
WINDOW_LENGTH_SEC = 0.1  # 100ms windows
WINDOW_STEP_SEC = 0.01   # 10ms steps
CV_SPLITS = 5
CV_REPEATS = 20
NUM_JOBS = 20

# Delay Period Configuration
DELAY_CONFIG = {
    'tmin': 2.0,
    'tmax': 4.7,
    'timepoints': np.linspace(2.0, 4.7, int((4.7-2.0)*RESAMPLE_FREQ))
}

# Critical time points to mark on plots
CRITICAL_TIMEPOINTS = [3.5, 4.5]

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

# All valid conditions (16 total)
ALL_CONDITIONS = [111, 112, 113, 114, 121, 122, 123, 124,
                 131, 132, 133, 134, 141, 142, 143, 144]

# Feature Definitions
FEATURES = {
    'maintained_voice_identity': {
        'name': 'Maintained Voice Identity',
        'color': '#1c686b',
        'description': 'Voice identity information maintained in working memory'
    },
    'maintained_location': {
        'name': 'Maintained Location',
        'color': '#cb6a3e',
        'description': 'Location information maintained in working memory'
    }
}

def extract_maintained_information(subject, metaInfo):
    """
    Extract which information is maintained in working memory based on retro-cues
    """
    print(f'\n=== Extracting maintained information for subject {subject} ===')
    
    try:
        # Get file information
        actInd = (metaInfo.Subject==subject) & (metaInfo.Valid==1)
        
        # Determine if subject is in early subjects (with corrected files)
        early_subject = subject in np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])[:7]
        
        if early_subject:
            actFiles = pd.Series([f.split('.')[0] + '_correct_triggers.fif' for f in metaInfo['MEG_Name']])[actInd]
        else:
            actFiles = metaInfo['MEG_Name'][actInd]
        
        # Load and concatenate events from raw files
        all_events = None
        reference_dev_head_t_ref = None
        
        # Process each file
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
                
            if np.any(events[:, 1] < 0):
                raise ValueError('Faulty trigger found, please inspect manually.')
                
            if ff == 0:
                all_events = events
            else:
                all_events = np.concatenate((all_events, events), axis=0)
            del raw
        
        print(f'Raw data loaded for subject {subject}')
        
        # Find all retro-cue events (101 for S1, 201 for S2)
        S1_cue_indices = np.where(all_events[:,2] == 101)[0]
        S2_cue_indices = np.where(all_events[:,2] == 201)[0]
        
        print(f"Found {len(S1_cue_indices)} S1 retro-cues and {len(S2_cue_indices)} S2 retro-cues")
        
        # Create memorized array to track which stimulus is in working memory
        memorized = np.zeros(len(all_events[:,2]))
        
        # For S1 cues, the maintained stimulus is 4 positions before
        for i in S1_cue_indices:
            if i >= 4:
                memorized[i - 4] = all_events[i - 4, 2]
        
        # For S2 cues, the maintained stimulus is 2 positions before
        for i in S2_cue_indices:
            if i >= 2:
                memorized[i - 2] = all_events[i - 2, 2]
        
        # Keep only the non-zero values (corresponding to maintained stimuli)
        memorized_values = memorized[memorized != 0]
        
        print(f"Maintained stimuli: {len(memorized_values)} total trials")
        
        return memorized_values, all_events
        
    except Exception as e:
        print(f"Error extracting maintained information for subject {subject}: {str(e)}")
        return None, None

def create_optimal_pseudotrials(epochs, memorized_values):
    """
    Create optimal pseudotrials using subcategory approach
    Based on trial count analysis: 5 trials per pseudotrial, 4 pseudotrials per condition
    """
    print(f'\n=== Creating optimal pseudotrials ===')
    print(f"Parameters: {TRIALS_PER_PSEUDOTRIAL} trials per pseudotrial, {PSEUDOTRIALS_PER_CONDITION} pseudotrials per condition")
    
    data = epochs.get_data()
    events = memorized_values
    
    print(f"Input data shape: {data.shape}")
    print(f"Total events: {len(events)}")
    
    # Initialize storage for all iterations
    pseudotrial_iterations = []
    
    for iteration in range(N_ITERATIONS):
        print(f"\nProcessing iteration {iteration + 1}/{N_ITERATIONS}")
        
        all_pseudotrials = []
        all_voice_labels = []
        all_location_labels = []
        condition_success_count = 0
        
        # Create pseudotrials for each of the 16 conditions
        for condition in ALL_CONDITIONS:
            # Find trials for this specific condition
            condition_indices = [i for i, event in enumerate(events) if event == condition]
            
            if len(condition_indices) < TRIALS_PER_PSEUDOTRIAL:
                print(f"  Warning: Condition {condition} has only {len(condition_indices)} trials, skipping")
                continue
            
            print(f"  Condition {condition}: {len(condition_indices)} trials available")
            
            # Create pseudotrials for this condition
            condition_pseudotrials = create_condition_pseudotrials(
                data, condition_indices, TRIALS_PER_PSEUDOTRIAL, PSEUDOTRIALS_PER_CONDITION
            )
            
            if not condition_pseudotrials:
                print(f"  Warning: Failed to create pseudotrials for condition {condition}")
                continue
            
            # Determine binary labels for this condition
            speaker_group = get_speaker_group(condition)    # 0 for Sp1+2, 1 for Sp3+4
            location_group = get_location_group(condition)  # 0 for L1+2, 1 for L3+4
            
            # Add pseudotrials and labels
            for pseudotrial in condition_pseudotrials:
                all_pseudotrials.append(pseudotrial)
                all_voice_labels.append(speaker_group)
                all_location_labels.append(location_group)
            
            condition_success_count += 1
        
        print(f"  Successfully processed {condition_success_count}/16 conditions")
        print(f"  Created {len(all_pseudotrials)} total pseudotrials")
        
        # Check if we have enough pseudotrials and both classes
        if len(all_pseudotrials) < 20:  # Minimum threshold
            print(f"  Warning: Only {len(all_pseudotrials)} pseudotrials created, skipping iteration")
            continue
        
        # Verify we have both classes for both features
        unique_voice = np.unique(all_voice_labels)
        unique_location = np.unique(all_location_labels)
        
        if len(unique_voice) < 2 or len(unique_location) < 2:
            print(f"  Warning: Missing classes (Voice: {unique_voice}, Location: {unique_location}), skipping iteration")
            continue
        
        # Shuffle the order to randomize
        indices = np.random.permutation(len(all_pseudotrials))
        
        iteration_data = {
            'data': np.array([all_pseudotrials[i] for i in indices]),
            'voice_labels': np.array([all_voice_labels[i] for i in indices]),
            'location_labels': np.array([all_location_labels[i] for i in indices])
        }
        
        # Report class balance
        voice_counts = np.bincount(iteration_data['voice_labels'])
        location_counts = np.bincount(iteration_data['location_labels'])
        
        print(f"  Voice classes: Sp1+2={voice_counts[0]}, Sp3+4={voice_counts[1]}")
        print(f"  Location classes: L1+2={location_counts[0]}, L3+4={location_counts[1]}")
        
        pseudotrial_iterations.append(iteration_data)
    
    print(f"\nPseudotrial creation completed: {len(pseudotrial_iterations)} valid iterations")
    return pseudotrial_iterations

def create_condition_pseudotrials(data, condition_indices, trials_per_pseudotrial, n_pseudotrials):
    """
    Create pseudotrials for a single condition by averaging randomly selected trials
    """
    if len(condition_indices) < trials_per_pseudotrial:
        return []
    
    pseudotrials = []
    used_indices = set()
    
    for _ in range(n_pseudotrials):
        # Find available indices (not overused)
        available_indices = [idx for idx in condition_indices if idx not in used_indices]
        
        # If we don't have enough unused indices, allow some resampling
        if len(available_indices) < trials_per_pseudotrial:
            if RESAMPLING_FACTOR > 1:
                # Add back some indices for resampling
                additional_indices = [idx for idx in condition_indices 
                                    if used_indices.count(idx) < RESAMPLING_FACTOR]
                available_indices.extend(additional_indices)
            
            if len(available_indices) < trials_per_pseudotrial:
                break  # Not enough trials available
        
        # Randomly select trials for this pseudotrial
        selected_indices = np.random.choice(
            available_indices, 
            size=trials_per_pseudotrial, 
            replace=False
        )
        
        # Average the selected trials
        selected_trials = [data[idx] for idx in selected_indices]
        pseudotrial = np.mean(selected_trials, axis=0)
        pseudotrials.append(pseudotrial)
        
        # Mark indices as used
        for idx in selected_indices:
            used_indices.add(idx)
    
    return pseudotrials

def get_speaker_group(condition):
    """Get speaker group (0 for Sp1+2, 1 for Sp3+4) from condition code"""
    speaker = (condition // 10) % 10
    return 0 if speaker in [1, 2] else 1

def get_location_group(condition):
    """Get location group (0 for L1+2, 1 for L3+4) from condition code"""
    location = condition % 10
    return 0 if location in [1, 2] else 1

def load_subject_data(subject, metaInfo):
    """
    Load and preprocess MEG data, create optimal pseudotrials
    """
    print(f'\n=== Preparing data for subject {subject} ===')
    
    # Check if there's already pseudotrial data available
    save_path = f"{PROCESSED_DIR}/optimal_pseudotrials/sub-{subject}"
    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_path}/optimal_pseudotrials.pkl"
    
    force_recompute = True  # Set to False to use cached data
    
    if os.path.exists(save_file) and not force_recompute:
        print(f"Loading pre-computed pseudotrials for subject {subject}...")
        with open(save_file, 'rb') as f:
            pseudotrial_data = pickle.load(f)
        print(f"Loaded {len(pseudotrial_data)} iterations of pseudotrial data")
        return pseudotrial_data, RESAMPLE_FREQ
    
    # Process from raw data
    print(f"Processing from raw data...")
    
    try:
        # Extract maintained information based on retro-cues
        memorized, all_events = extract_maintained_information(subject, metaInfo)
        
        if memorized is None:
            print(f"Could not extract maintained information for subject {subject}")
            return None, None
        
        # Load cleaned epochs
        clean_trials = mne.read_epochs(
            f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif",
            preload=True,
            verbose='WARNING'
        )
        print(f"Initial epochs loaded: {len(clean_trials)}")
        
        # Special case handling
        if subject == 28:
            clean_trials.drop(63)
        if subject == 23:
            clean_trials.drop(64*7)
        
        # Update event codes to reflect maintained information
        if len(memorized) == len(clean_trials.events):
            clean_trials.events[:, 2] = memorized
        else:
            print(f"Warning: Mismatch in trial counts. Memorized: {len(memorized)}, Epochs: {len(clean_trials.events)}")
            # Try to match by taking the first len(clean_trials) memorized events
            clean_trials.events[:, 2] = memorized[:len(clean_trials)]
        
        clean_trials.event_id = EVENT_DICT
        
        # Handle jump artifacts
        jname = f"{PROCESSED_DIR}/ICAs/Jumps{subject}.npy"
        if os.path.isfile(jname):
            jump_inds = np.load(jname)
            if len(jump_inds) > 0:
                jump_inds = np.array(jump_inds, dtype=int)
                valid_jump_inds = jump_inds[jump_inds < len(clean_trials)]
                if len(valid_jump_inds) > 0:
                    clean_trials.drop(valid_jump_inds, reason='jump')
                    print(f"Dropped {len(valid_jump_inds)} trials due to jump artifacts")
        
        # Crop to delay period
        delay_epochs = clean_trials.copy()
        delay_epochs.crop(tmin=DELAY_CONFIG['tmin'], tmax=DELAY_CONFIG['tmax'])
        
        # Select magnetometer channels
        mag_epochs = delay_epochs.copy().pick_types(meg='mag')
        
        # Resample
        mag_epochs = mag_epochs.resample(RESAMPLE_FREQ, npad='auto')
        print(f"Final epochs for pseudotrial creation: {len(mag_epochs)}")
        
        # Create optimal pseudotrials
        memorized_values = mag_epochs.events[:, 2]
        pseudotrial_data = create_optimal_pseudotrials(mag_epochs, memorized_values)
        
        # Save pseudotrial data
        if pseudotrial_data:
            with open(save_file, 'wb') as f:
                pickle.dump(pseudotrial_data, f)
            print(f"Saved {len(pseudotrial_data)} iterations of pseudotrial data")
        
        return pseudotrial_data, RESAMPLE_FREQ
    
    except Exception as e:
        print(f"Error processing subject {subject}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def decode_sliding_window(data, y, sfreq, subject=None):
    """
    Perform sliding window decoding with nested cross-validation
    """
    if data.size == 0 or len(np.unique(y)) < 2:
        print("Invalid data for decoding")
        return None, None
    
    n_trials, n_channels, n_timepoints = data.shape
    
    # Convert window parameters to samples
    window_length = int(WINDOW_LENGTH_SEC * sfreq)
    window_step = int(WINDOW_STEP_SEC * sfreq)
    
    # Calculate number of windows
    n_windows = int((n_timepoints - window_length) / window_step) + 1
    
    window_scores = np.zeros(n_windows)
    window_centers = np.zeros(n_windows)
    
    # Parameter grid for SVM
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    
    def process_window_nested(window_idx):
        """Process a single window with nested CV"""
        try:
            # Calculate window boundaries
            win_start = window_idx * window_step
            win_end = win_start + window_length
            window_center = (win_start + win_end) / 2 / sfreq + DELAY_CONFIG['tmin']
            
            # Extract and reshape data
            X_win = data[:, :, win_start:win_end].reshape(n_trials, -1)
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_win)
            
            # Cross-validation setup
            outer_cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS // 10)
            inner_cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=2)
            
            nested_scores = []
            
            # Nested CV loop
            for train_idx, test_idx in outer_cv.split(X_scaled, y):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Inner grid search
                grid_search = GridSearchCV(
                    estimator=SVC(kernel='linear', probability=True),
                    param_grid=param_grid,
                    cv=inner_cv,
                    scoring='accuracy'
                )
                
                grid_search.fit(X_train, y_train)
                
                # Test with best parameters
                best_clf = SVC(kernel='linear', C=grid_search.best_params_['C'], probability=True)
                best_clf.fit(X_train, y_train)
                score = best_clf.score(X_test, y_test)
                nested_scores.append(score)
            
            return {
                'accuracy': np.mean(nested_scores),
                'window_center': window_center,
                'success': True
            }
        except Exception as e:
            return {
                'accuracy': 0.5,
                'window_center': (window_idx * window_step + window_length/2) / sfreq + DELAY_CONFIG['tmin'],
                'success': False
            }
    
    print(f"\nPerforming sliding window decoding across {n_windows} windows...")
    results = Parallel(n_jobs=NUM_JOBS)(
        delayed(process_window_nested)(window_idx) for window_idx in tqdm(range(n_windows))
    )
    
    # Extract results
    for i, r in enumerate(results):
        window_scores[i] = r['accuracy']
        window_centers[i] = r['window_center']
    
    return window_scores, window_centers

def decode_pseudotrial_iterations(pseudotrial_data, sfreq, feature_name, subject=None):
    """
    Perform decoding across multiple pseudotrial iterations
    """
    if not pseudotrial_data:
        print("No pseudotrial data available")
        return None, None, None
    
    all_window_scores = []
    window_centers = None
    
    for iter_idx, iteration_data in enumerate(pseudotrial_data):
        print(f"\nDecoding iteration {iter_idx + 1}/{len(pseudotrial_data)}")
        
        X = iteration_data['data']
        
        # Select labels based on feature
        if feature_name == 'maintained_voice_identity':
            y = iteration_data['voice_labels']
            print(f"Voice decoding: Sp1+2 vs Sp3+4")
        else:  # maintained_location
            y = iteration_data['location_labels']
            print(f"Location decoding: L1+2 vs L3+4")
        
        # Check class balance
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        
        if len(unique_classes) < 2:
            print(f"Skipping iteration {iter_idx + 1}: insufficient classes")
            continue
        
        try:
            # Decode this iteration
            iter_scores, iter_centers = decode_sliding_window(X, y, sfreq, 
                                                            subject=f"{subject}_iter{iter_idx}" if subject else None)
            
            if iter_scores is not None and iter_centers is not None:
                all_window_scores.append(iter_scores)
                if window_centers is None:
                    window_centers = iter_centers
            else:
                print(f"Iteration {iter_idx + 1} failed")
        except Exception as e:
            print(f"Error in iteration {iter_idx + 1}: {e}")
            continue
    
    if not all_window_scores:
        print("No valid decoding results")
        return None, None, None
    
    # Calculate statistics across iterations
    all_window_scores = np.array(all_window_scores)
    mean_scores = np.mean(all_window_scores, axis=0)
    std_scores = np.std(all_window_scores, axis=0)
    
    return mean_scores, std_scores, window_centers

def process_subject(subject):
    """
    Process a single subject with optimal pseudotrial approach
    """
    print(f"\n==== Processing subject {subject} with optimal pseudotrials ====")
    print(f"Parameters: {TRIALS_PER_PSEUDOTRIAL} trials/pseudotrial, {PSEUDOTRIALS_PER_CONDITION} pseudotrials/condition")
    
    # Load metadata
    metaInfo = pd.read_excel(META_FILE)
    
    try:
        # Load subject data and create pseudotrials
        pseudotrial_data, sfreq = load_subject_data(subject, metaInfo)
        
        if pseudotrial_data is None:
            print(f"Failed to create pseudotrials for subject {subject}")
            return
        
        # Process both features
        for feature_name in FEATURES.keys():
            print(f"\n--- Processing {feature_name} ---")
            
            # Create results directory
            results_path = f"{PROCESSED_DIR}/timepoints/optimal_pseudotrials/{feature_name}"
            os.makedirs(results_path, exist_ok=True)
            
            try:
                # Perform decoding
                mean_scores, std_scores, window_centers = decode_pseudotrial_iterations(
                    pseudotrial_data, sfreq, feature_name, subject
                )
                
                if mean_scores is None:
                    print(f"No valid decoding results for {feature_name}")
                    continue
                
                # Plot results
                plt.figure(figsize=(12, 7))
                plt.plot(window_centers, mean_scores, 
                        color=FEATURES[feature_name]['color'], 
                        linewidth=2, label=f'Subject {subject}')
                plt.fill_between(window_centers, 
                               mean_scores - std_scores, 
                               mean_scores + std_scores, 
                               alpha=0.2,
                               color=FEATURES[feature_name]['color'])
                
                # Add critical timepoints
                for tp in CRITICAL_TIMEPOINTS:
                    plt.axvline(x=tp, color='gray', linestyle='--', alpha=0.7)
                
                plt.title(f'{FEATURES[feature_name]["name"]} - Subject {subject}\n'
                         f'Optimal Pseudotrials ({TRIALS_PER_PSEUDOTRIAL} trials/pseudotrial)')
                plt.xlabel('Time (s)')
                plt.ylabel('Decoding Accuracy')
                plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
                plt.ylim(0.45, 0.75)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'{results_path}/sub{subject}_optimal_pseudotrial_decoding.png', dpi=150)
                plt.close()
                
                # Save results
                subject_result = {
                    'mean_scores': mean_scores,
                    'std_scores': std_scores,
                    'window_centers': window_centers,
                    'subject': subject,
                    'parameters': {
                        'trials_per_pseudotrial': TRIALS_PER_PSEUDOTRIAL,
                        'pseudotrials_per_condition': PSEUDOTRIALS_PER_CONDITION,
                        'n_iterations': N_ITERATIONS
                    }
                }
                
                with open(f'{results_path}/sub{subject}_optimal_results.pkl', 'wb') as f:
                    pickle.dump(subject_result, f)
                
                print(f'Completed {feature_name} decoding for subject {subject}')
                print(f'Mean accuracy: {np.mean(mean_scores):.3f} ± {np.mean(std_scores):.3f}')
                
            except Exception as e:
                print(f"Error during {feature_name} decoding: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"Error processing subject {subject}: {e}")
        import traceback
        traceback.print_exc()

# Main execution
if __name__ == "__main__":
    # Get subject ID from command line
    if len(sys.argv) != 2:
        print("Usage: python optimal_pseudotrial_delay.py SUBJECT_ID")
        sys.exit(1)
    
    try:
        subject = int(sys.argv[1])
    except ValueError:
        print("Error: SUBJECT_ID must be an integer")
        sys.exit(1)
    
    # Set locale
    try:
        locale.setlocale(locale.LC_ALL, "en_US.utf8")
    except:
        pass
    
    # Create output directories
    os.makedirs(f"{PROCESSED_DIR}/timepoints/optimal_pseudotrials", exist_ok=True)
    os.makedirs(f"{PROCESSED_DIR}/optimal_pseudotrials", exist_ok=True)
    
    print("="*80)
    print("OPTIMAL PSEUDOTRIAL DELAY PERIOD DECODING")
    print("="*80)
    print(f"Based on Scrivener et al. 2023 best practices")
    print(f"Optimal parameters from trial count analysis:")
    print(f"  - {TRIALS_PER_PSEUDOTRIAL} trials per pseudotrial")
    print(f"  - {PSEUDOTRIALS_PER_CONDITION} pseudotrials per condition")
    print(f"  - {N_ITERATIONS} iterations per subject")
    print(f"  - Expected ~32 pseudotrials per binary class")
    print("="*80)
    
    # Process the subject
    process_subject(subject)
    
    print(f"\nSubject {subject} processing completed!")