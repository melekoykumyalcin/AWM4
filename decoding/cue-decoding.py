#!/usr/bin/env python
"""
Retrocue Decoding Analysis (S1 vs S2)
Sanity check decoding of retrocue presentation
Using already cleaned epochs, cutting to retrocue period
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

# Decoding Parameters
RESAMPLE_FREQ = 100  # Hz
WINDOW_LENGTH_SEC = 0.1  # 100ms windows
WINDOW_STEP_SEC = 0.01   # 10ms steps
CV_SPLITS = 5
CV_REPEATS = 20
NUM_JOBS = 20

# Retrocue Period Configuration (within trial time)
RETROCUE_CONFIG = {
    'tmin': 1.9,   # 100ms before retrocue onset at 2s
    'tmax': 2.5,   # Covers full 500ms retrocue presentation
    'timepoints': np.linspace(1.9, 2.5, int((2.5-1.9)*RESAMPLE_FREQ))
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

def extract_maintained_information(subject, metaInfo):
    """
    Extract which stimulus is maintained based on retro-cues
    Returns labels indicating whether each trial has S1 or S2 cue
    """
    print(f'\n=== Extracting retrocue information for subject {subject} ===')
    
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
        
        # Create array to track which cue type follows each stimulus
        retrocue_labels = np.zeros(len(all_events[:,2]))
        
        # Mark trials that are followed by S1 cue
        for i in S1_cue_indices:
            if i >= 4:  # S1 is 4 positions before its cue
                retrocue_labels[i - 4] = 101
        
        # Mark trials that are followed by S2 cue
        for i in S2_cue_indices:
            if i >= 2:  # S2 is 2 positions before its cue
                retrocue_labels[i - 2] = 201
        
        # Keep only the stimulus trials (not the cue events themselves)
        stimulus_mask = np.isin(all_events[:,2], list(range(111, 145)) + list(range(211, 245)))
        retrocue_labels = retrocue_labels[stimulus_mask]
        
        # Remove zeros (trials without identified cues)
        valid_trials = retrocue_labels != 0
        
        print(f"Identified retrocue type for {np.sum(valid_trials)} stimulus trials")
        
        return retrocue_labels, valid_trials
        
    except Exception as e:
        print(f"Error extracting retrocue information for subject {subject}: {str(e)}")
        return None, None

def load_subject_data_retrocue(subject, metaInfo):
    """
    Load cleaned epochs and prepare for retrocue decoding
    """
    print(f'\n=== Loading cleaned epochs for retrocue decoding - subject {subject} ===')
    
    try:
        # Extract retrocue labels
        retrocue_labels, valid_trials = extract_maintained_information(subject, metaInfo)
        
        if retrocue_labels is None:
            print(f"Could not extract retrocue information for subject {subject}")
            return None, None, None
        
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
        
        # Check if we have the right number of labels
        if len(retrocue_labels) != len(clean_trials):
            print(f"Warning: Mismatch in counts. Labels: {len(retrocue_labels)}, Epochs: {len(clean_trials)}")
            # Try to match sizes
            min_len = min(len(retrocue_labels), len(clean_trials))
            retrocue_labels = retrocue_labels[:min_len]
            valid_trials = valid_trials[:min_len]
            clean_trials = clean_trials[:min_len]
        
        # Keep only trials with identified retrocue
        clean_trials = clean_trials[valid_trials]
        retrocue_labels = retrocue_labels[valid_trials]
        
        print(f"Epochs with retrocue labels: {len(clean_trials)}")
        print(f"  - S1 cue trials: {np.sum(retrocue_labels == 101)}")
        print(f"  - S2 cue trials: {np.sum(retrocue_labels == 201)}")
        
        # Handle jump artifacts
        jname = f"{PROCESSED_DIR}/ICAs/Jumps{subject}.npy"
        if os.path.isfile(jname):
            jump_inds = np.load(jname)
            if len(jump_inds) > 0:
                jump_inds = np.array(jump_inds, dtype=int)
                # Map jump indices to our subset
                original_indices = np.where(valid_trials)[0]
                jump_mask = np.isin(original_indices, jump_inds)
                
                if np.any(jump_mask):
                    valid_jump_inds = np.where(jump_mask)[0]
                    valid_jump_inds = valid_jump_inds[valid_jump_inds < len(clean_trials)]
                    if len(valid_jump_inds) > 0:
                        clean_trials.drop(valid_jump_inds, reason='jump')
                        # Also remove from labels
                        retrocue_labels = np.delete(retrocue_labels, valid_jump_inds)
                        print(f"Dropped {len(valid_jump_inds)} trials due to jump artifacts")
        
        # Crop to retrocue period
        retrocue_epochs = clean_trials.copy()
        retrocue_epochs.crop(tmin=RETROCUE_CONFIG['tmin'], tmax=RETROCUE_CONFIG['tmax'])
        
        # Select magnetometer channels
        mag_epochs = retrocue_epochs.copy().pick_types(meg='mag')
        
        # Resample
        mag_epochs = mag_epochs.resample(RESAMPLE_FREQ, npad='auto')
        print(f"Final epochs for decoding: {len(mag_epochs)}")
        
        # Get data
        X = mag_epochs.get_data()
        
        # Convert labels to binary (0 for S1, 1 for S2)
        y_binary = (retrocue_labels == 201).astype(int)
        
        # Final check
        unique_labels, counts = np.unique(y_binary, return_counts=True)
        print(f"Final class distribution:")
        for label, count in zip(unique_labels, counts):
            cue_type = 'S2' if label == 1 else 'S1'
            print(f"  - {cue_type}: {count} trials")
        
        return X, y_binary, RESAMPLE_FREQ
    
    except Exception as e:
        print(f"Error processing subject {subject}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def decode_sliding_window_retrocue(data, y, sfreq):
    """
    Perform sliding window decoding for retrocue
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
    
    def process_window(window_idx):
        """Process a single window"""
        try:
            # Calculate window boundaries
            win_start = window_idx * window_step
            win_end = win_start + window_length
            window_center = (win_start + win_end) / 2 / sfreq + RETROCUE_CONFIG['tmin']
            
            # Extract and reshape data
            X_win = data[:, :, win_start:win_end].reshape(n_trials, -1)
            
            # Create pipeline
            clf = make_pipeline(
                StandardScaler(),
                SVC(kernel='linear', C=1.0, probability=True)
            )
            
            # Cross-validation
            cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS)
            scores = cross_val_multiscore(clf, X_win, y, cv=cv, n_jobs=1)
            
            return {
                'accuracy': np.mean(scores),
                'window_center': window_center,
                'success': True
            }
        except Exception as e:
            return {
                'accuracy': 0.5,
                'window_center': (window_idx * window_step + window_length/2) / sfreq + RETROCUE_CONFIG['tmin'],
                'success': False
            }
    
    print(f"\nPerforming sliding window decoding across {n_windows} windows...")
    results = Parallel(n_jobs=NUM_JOBS)(
        delayed(process_window)(window_idx) for window_idx in tqdm(range(n_windows))
    )
    
    # Extract results
    for i, r in enumerate(results):
        window_scores[i] = r['accuracy']
        window_centers[i] = r['window_center']
    
    return window_scores, window_centers

def process_subject(subject):
    """
    Process a single subject for retrocue decoding
    """
    print(f"\n==== Processing subject {subject} - Retrocue Decoding ====")
    print(f"Decoding S1 (101) vs S2 (201) retrocues from cleaned epochs")
    print(f"Time window: {RETROCUE_CONFIG['tmin']} to {RETROCUE_CONFIG['tmax']}s")
    
    # Load metadata
    metaInfo = pd.read_excel(META_FILE)
    
    try:
        # Load subject data
        X, y, sfreq = load_subject_data_retrocue(subject, metaInfo)
        
        if X is None:
            print(f"Failed to load data for subject {subject}")
            return
        
        # Create results directory
        results_path = f"{PROCESSED_DIR}/retrocue_decoding"
        os.makedirs(results_path, exist_ok=True)
        
        # Perform decoding
        scores, window_centers = decode_sliding_window_retrocue(X, y, sfreq)
        
        if scores is None:
            print(f"Decoding failed for subject {subject}")
            return
        
        # Calculate statistics
        mean_score = np.mean(scores)
        max_score = np.max(scores)
        max_time = window_centers[np.argmax(scores)]
        
        print(f"\nDecoding results for subject {subject}:")
        print(f"  - Mean accuracy: {mean_score:.3f}")
        print(f"  - Peak accuracy: {max_score:.3f} at {max_time:.3f}s")
        
        # Plot results
        plt.figure(figsize=(12, 7))
        plt.plot(window_centers, scores, 'b-', linewidth=2)
        plt.fill_between(window_centers, 0.5, scores, 
                        where=(scores > 0.5), alpha=0.3, color='blue')
        
        # Mark key timepoints
        plt.axvline(x=2.0, color='red', linestyle='--', alpha=0.7, label='Retrocue onset')
        plt.axvline(x=2.5, color='red', linestyle='--', alpha=0.7, label='Retrocue offset')
        
        # Add significance threshold (binomial test)
        n_trials = len(y)
        sig_threshold = 0.5 + 1.96 * np.sqrt(0.25 / n_trials)
        plt.axhline(y=sig_threshold, color='gray', linestyle=':', alpha=0.7, 
                   label=f'p<0.05 threshold')
        
        plt.title(f'Retrocue Decoding (S1 vs S2) - Subject {subject}\n'
                 f'Peak: {max_score:.3f} at {max_time:.3f}s')
        plt.xlabel('Time (s)')
        plt.ylabel('Decoding Accuracy')
        plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Chance')
        plt.ylim(0.4, 0.9)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{results_path}/sub{subject}_retrocue_decoding.png', dpi=150)
        plt.close()
        
        # Save results
        results_dict = {
            'scores': scores,
            'window_centers': window_centers,
            'subject': subject,
            'mean_accuracy': mean_score,
            'peak_accuracy': max_score,
            'peak_time': max_time,
            'n_trials_per_class': np.bincount(y),
            'parameters': {
                'window_length': WINDOW_LENGTH_SEC,
                'window_step': WINDOW_STEP_SEC,
                'cv_splits': CV_SPLITS,
                'cv_repeats': CV_REPEATS,
                'resample_freq': RESAMPLE_FREQ
            }
        }
        
        with open(f'{results_path}/sub{subject}_retrocue_results.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
        
        print(f'\nCompleted retrocue decoding for subject {subject}')
        
        # Generate temporal generalization analysis
        print("\nPerforming temporal generalization analysis...")
        from mne.decoding import GeneralizingEstimator
        
        # Create generalizing estimator
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel='linear', C=1.0)
        )
        gen = GeneralizingEstimator(clf, scoring='accuracy', n_jobs=NUM_JOBS)
        
        # Fit and score
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
        scores_gen = cross_val_multiscore(gen, X, y, cv=cv, n_jobs=1)
        
        # Plot temporal generalization
        fig, ax = plt.subplots(figsize=(10, 8))
        times = np.linspace(RETROCUE_CONFIG['tmin'], RETROCUE_CONFIG['tmax'], X.shape[2])
        
        im = ax.imshow(
            np.mean(scores_gen, axis=0),
            origin='lower',
            extent=[times[0], times[-1], times[0], times[-1]],
            aspect='auto',
            cmap='RdBu_r',
            vmin=0.3, vmax=0.8
        )
        
        ax.set_xlabel('Testing Time (s)')
        ax.set_ylabel('Training Time (s)')
        ax.set_title(f'Temporal Generalization: Retrocue Decoding - Subject {subject}')
        ax.axvline(2.0, color='k', linestyle='--', alpha=0.5)
        ax.axhline(2.0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(2.5, color='k', linestyle='--', alpha=0.5)
        ax.axhline(2.5, color='k', linestyle='--', alpha=0.5)
        
        plt.colorbar(im, ax=ax, label='Accuracy')
        plt.tight_layout()
        plt.savefig(f'{results_path}/sub{subject}_retrocue_temporal_gen.png', dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"Error processing subject {subject}: {e}")
        import traceback
        traceback.print_exc()

# Main execution
if __name__ == "__main__":
    # Get subject ID from command line
    if len(sys.argv) != 2:
        print("Usage: python retrocue_decoding.py SUBJECT_ID")
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
    os.makedirs(f"{PROCESSED_DIR}/retrocue_decoding", exist_ok=True)
    
    print("="*80)
    print("RETROCUE DECODING ANALYSIS")
    print("="*80)
    print(f"Sanity check: Decoding S1 (101) vs S2 (201) retrocues")
    print(f"Using cleaned epochs, cropped to retrocue period")
    print(f"Time window: {RETROCUE_CONFIG['tmin']} to {RETROCUE_CONFIG['tmax']} seconds")
    print(f"Expected: High accuracy shortly after retrocue onset (2.0s)")
    print("="*80)
    
    # Process the subject
    process_subject(subject)
    
    print(f"\nSubject {subject} retrocue decoding completed!")