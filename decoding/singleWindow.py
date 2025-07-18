#!/usr/bin/env python
"""
Single Window PCA Delay Period Analysis Script
Implements whole delay period (2500ms) analysis with PCA preprocessing:
- Single window 2.0-4.5s → Z-score → MinMax → PCA → SVM decode

Usage: python pca_single_window_analysis.py --subject 23
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from collections import defaultdict, Counter
from joblib import Parallel, delayed
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run single window PCA delay period analysis')
parser.add_argument('--subject', type=int, required=True, help='Subject ID')
parser.add_argument('--n_iterations', type=int, default=10, help='Number of iterations')
parser.add_argument('--n_jobs', type=int, default=20, help='Number of parallel jobs')
parser.add_argument('--explained_variance', type=float, default=0.95, help='PCA explained variance (0.95 = 95%)')
args = parser.parse_args()

# Set paths for HPC
HOME_DIR = '/mnt/hpc/projects/awm4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
DATA_PATH = HOME_DIR + '/AWM4_data/raw/'
CORRECTED_DATA = HOME_DIR + '/AWM4_data/raw/correctTriggers/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'

# Create output directory
OUTPUT_DIR = PROCESSED_DIR + f'pcaSingleWindow/subject_{args.subject}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log file path
LOG_FILE_PATH = f'{OUTPUT_DIR}/processing_log.txt'

def write_log(message):
    """Write message to log file"""
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(message + '\n')

# Initialize log
write_log(f"Single Window PCA analysis started at: {datetime.now()}")
write_log(f"Subject: {args.subject}")
write_log(f"PCA explained variance: {args.explained_variance:.1%}")
write_log(f"Using single 2500ms window: 2.0-4.5s")

# Single window parameters
TARGET_SAMPLING_RATE = 50  # Hz
WINDOW_START = 2.0  # Start at 2.0s
WINDOW_END = 4.5    # End at 4.5s  
WINDOW_DURATION = WINDOW_END - WINDOW_START  # 2.5s = 2500ms

# Important time markers
CUE_TIME = 2.0  # Cue presentation
PING_TIME = 3.5  # Ping occurs at 3.5s
PROBE_TIME = 4.7  # Probe presentation (outside our window)

# Analysis configuration
ANALYSIS_CONFIG = {
    'tmin': WINDOW_START,
    'tmax': WINDOW_END,
    'duration': WINDOW_DURATION,
    'description': f'Single window ({WINDOW_START}-{WINDOW_END}s, {WINDOW_DURATION*1000:.0f}ms)'
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

# Updated parameter grid with smaller C values
PARAM_GRID = [0.001, 0.01, 0.1, 1.0]

def apply_preprocessing_pipeline(train_data, test_data, explained_variance_ratio=0.95):
    """
    Apply preprocessing pipeline: Z-score → MinMax → PCA
    Fit on training data, apply to both train and test
    
    Parameters:
    train_data: shape (n_train_trials, n_channels * n_timepoints)
    test_data: shape (n_test_trials, n_channels * n_timepoints)
    explained_variance_ratio: 0.95 for 95%, 0.99 for 99%
    
    Returns:
    train_processed: PCA-transformed training data
    test_processed: PCA-transformed test data  
    pipeline_info: dict with component info
    """
    
    # Step 1: Z-score normalization
    # Each feature (channel×timepoint) normalized across trials
    z_scaler = StandardScaler()
    train_z_scored = z_scaler.fit_transform(train_data)  # Fit on train
    test_z_scored = z_scaler.transform(test_data)        # Apply to test
    
    # Step 2: MinMax scaling to 0-1 range
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = minmax_scaler.fit_transform(train_z_scored)  # Fit on train
    test_scaled = minmax_scaler.transform(test_z_scored)        # Apply to test
    
    # Step 3: PCA with explained variance
    pca = PCA(n_components=explained_variance_ratio)
    train_pca = pca.fit_transform(train_scaled)  # Fit on train
    test_pca = pca.transform(test_scaled)        # Apply to test
    
    # Calculate actual explained variance
    actual_variance = np.sum(pca.explained_variance_ratio_)
    n_components = pca.n_components_
    
    pipeline_info = {
        'original_features': train_data.shape[1],
        'pca_components': n_components,
        'explained_variance': actual_variance,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'z_scaler': z_scaler,
        'minmax_scaler': minmax_scaler, 
        'pca': pca
    }
    
    return train_pca, test_pca, pipeline_info

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

def decode_single_window_with_pca(epochs_data, events, feature_name, n_iterations, explained_variance_ratio):
    """
    Decode using single window with PCA preprocessing
    """
    write_log(f"\nSingle window PCA decoding: {feature_name}")
    write_log(f"Window: {ANALYSIS_CONFIG['description']}")
    write_log(f"PCA explained variance: {explained_variance_ratio:.1%}")
    write_log(f"Original data shape: {epochs_data.shape}")
    
    # Flatten spatio-temporal data (single window!)
    n_trials, n_channels, n_timepoints = epochs_data.shape
    X_flattened = epochs_data.reshape(n_trials, n_channels * n_timepoints)
    
    write_log(f"Flattened features: {n_channels} channels × {n_timepoints} timepoints = {X_flattened.shape[1]}")
    
    # Storage for results across iterations
    iteration_results = []
    
    for iteration in range(n_iterations):
        write_log(f"  Iteration {iteration + 1}/{n_iterations}")
        
        # Create labels based on feature type
        if feature_name == 'maintained_voice':
            # Speaker 1+2 vs 3+4
            original_labels = np.array([0 if (e//10)%10 in [1,2] else 1 for e in events])
        else:  # maintained_location
            # Location 1+2 vs 3+4
            original_labels = np.array([0 if e%10 in [1,2] else 1 for e in events])
        
        # Skip if only one class
        if len(np.unique(original_labels)) < 2:
            write_log(f"  Skipping - only one class present")
            continue
        
        # Outer cross-validation
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
        outer_scores = []
        best_cs = []
        pca_info_list = []
        
        # Outer CV loop
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_flattened, original_labels)):
            
            # Step 1: Split original trials
            train_data = X_flattened[train_idx]
            test_data = X_flattened[test_idx]
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
            X_train_pseudo = np.vstack(train_pseudo_data)
            y_train = np.array(train_pseudo_labels)
            X_test_pseudo = np.vstack(test_pseudo_data)
            y_test = np.array(test_pseudo_labels)
            
            # Verify both classes present
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            
            # Step 3: Apply PCA preprocessing pipeline
            try:
                X_train_processed, X_test_processed, pipeline_info = apply_preprocessing_pipeline(
                    X_train_pseudo, X_test_pseudo, explained_variance_ratio
                )
                pca_info_list.append(pipeline_info)
                
                write_log(f"    Fold {fold_idx+1}: {pipeline_info['original_features']} → {pipeline_info['pca_components']} features ({pipeline_info['explained_variance']:.1%} variance)")
                
            except Exception as e:
                write_log(f"    Fold {fold_idx+1}: PCA preprocessing failed - {str(e)}")
                continue
            
            # Step 4: Inner CV for hyperparameter selection
            best_score = -1
            best_c = 1.0
            
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=None)
            
            for c_value in PARAM_GRID:
                # No additional scaling needed - data already preprocessed
                clf = SVC(kernel='linear', C=c_value, probability=True)
                
                try:
                    inner_scores = cross_val_score(
                        clf, X_train_processed, y_train,
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
            
            # Step 5: Train final model with best C and evaluate on test set
            final_clf = SVC(kernel='linear', C=best_c, probability=True)
            
            try:
                final_clf.fit(X_train_processed, y_train)
                test_score = final_clf.score(X_test_processed, y_test)
                outer_scores.append(test_score)
                best_cs.append(best_c)
                
                write_log(f"    Fold {fold_idx+1}: Accuracy = {test_score:.3f}, Best C = {best_c}")
                
            except Exception as e:
                write_log(f"    Fold {fold_idx+1}: Final training failed - {str(e)}")
                continue
        
        # Store iteration results
        if outer_scores:
            iteration_accuracy = np.mean(outer_scores)
            iteration_results.append({
                'accuracy': iteration_accuracy,
                'scores': outer_scores,
                'best_cs': best_cs,
                'pca_info': pca_info_list
            })
            
            write_log(f"  Iteration {iteration + 1}: Accuracy = {iteration_accuracy:.3f}")
        else:
            write_log(f"  Iteration {iteration + 1}: No valid folds")
    
    # Aggregate results across iterations
    if iteration_results:
        all_accuracies = [r['accuracy'] for r in iteration_results]
        mean_accuracy = np.mean(all_accuracies)
        std_accuracy = np.std(all_accuracies)
        
        # Average PCA info across iterations
        avg_components = np.mean([np.mean([p['pca_components'] for p in r['pca_info']]) 
                                 for r in iteration_results])
        avg_explained_var = np.mean([np.mean([p['explained_variance'] for p in r['pca_info']]) 
                                    for r in iteration_results])
        
        write_log(f"  Final Results:")
        write_log(f"    Mean accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
        write_log(f"    Average PCA components: {avg_components:.1f}")
        write_log(f"    Average explained variance: {avg_explained_var:.1%}")
        
        # Most common C value
        all_cs = [c for r in iteration_results for c in r['best_cs']]
        c_counter = Counter(all_cs)
        most_common_c = c_counter.most_common(1)[0][0] if all_cs else 1.0
        write_log(f"    Most common C: {most_common_c}")
        
        return {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'all_accuracies': all_accuracies,
            'avg_pca_components': avg_components,
            'avg_explained_variance': avg_explained_var,
            'most_common_c': most_common_c,
            'iteration_results': iteration_results
        }
    else:
        write_log(f"  No valid results for {feature_name}")
        return None

def load_subject_data(subject, meta_info):
    """Load and preprocess data for the single window analysis"""
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
        
        # Crop to single window
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
        write_log(f"Timepoints: {epochs_data.shape[2]} (at {TARGET_SAMPLING_RATE}Hz)")
        write_log(f"Total features after flattening: {epochs_data.shape[1] * epochs_data.shape[2]}")
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

def plot_single_window_results(results):
    """Create a summary plot of single window PCA results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Subject {args.subject} - Single Window PCA Analysis\n{ANALYSIS_CONFIG["description"]}', fontsize=16)
    
    features = ['maintained_voice', 'maintained_location']
    colors = ['#e41a1c', '#377eb8']
    
    for idx, (feature_name, color) in enumerate(zip(features, colors)):
        if feature_name in results:
            data = results[feature_name]
            
            # Plot 1: Accuracy distribution across iterations
            ax1 = axes[idx, 0]
            ax1.hist(data['all_accuracies'], bins=10, alpha=0.7, color=color, edgecolor='black')
            ax1.axvline(data['mean_accuracy'], color='red', linestyle='--', 
                       label=f'Mean: {data["mean_accuracy"]:.3f}')
            ax1.axhline(y=0, color='black', linestyle=':', alpha=0.5)
            ax1.set_xlabel('Accuracy')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'{feature_name.replace("maintained_", "").title()} - Accuracy Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: PCA components vs accuracy
            ax2 = axes[idx, 1]
            # Extract PCA components and accuracies for each iteration
            pca_components = []
            iteration_accuracies = []
            for iter_result in data['iteration_results']:
                avg_components = np.mean([p['pca_components'] for p in iter_result['pca_info']])
                pca_components.append(avg_components)
                iteration_accuracies.append(iter_result['accuracy'])
            
            ax2.scatter(pca_components, iteration_accuracies, color=color, alpha=0.7, s=50)
            ax2.set_xlabel('Average PCA Components')
            ax2.set_ylabel('Accuracy')
            ax2.set_title(f'PCA Components vs Accuracy')
            ax2.grid(True, alpha=0.3)
            
            # Add correlation
            if len(pca_components) > 1:
                corr = np.corrcoef(pca_components, iteration_accuracies)[0, 1]
                ax2.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax2.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/single_window_pca_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    write_log(f"\nSummary plot saved to {OUTPUT_DIR}/single_window_pca_results.png")

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
    
    write_log("\n=== Single Window PCA Analysis ===")
    write_log(f"=== Window: {ANALYSIS_CONFIG['description']} ===")
    write_log(f"=== PCA explained variance: {args.explained_variance:.1%} ===")
    write_log(f"=== Parameter grid: {PARAM_GRID} ===")
    
    # Results storage
    all_results = {}
    
    # Process each feature (voice and location)
    for feature_name in ['maintained_voice', 'maintained_location']:
        write_log(f"\n{'='*60}")
        write_log(f"Processing feature: {feature_name}")
        write_log(f"{'='*60}")
        
        # Run single window PCA decoding
        results = decode_single_window_with_pca(
            epochs_data, events, feature_name, 
            args.n_iterations, args.explained_variance
        )
        
        if results is not None:
            all_results[feature_name] = results
            write_log(f"  {feature_name}: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
        else:
            write_log(f"  {feature_name}: No valid results")
    
    # Create summary plot
    if all_results:
        plot_single_window_results(all_results)
    
    # Save results
    write_log(f"\nSaving results...")
    
    for feature_name, results in all_results.items():
        # Save individual results
        np.save(f'{OUTPUT_DIR}/single_window_pca_{feature_name}_accuracies.npy', 
               results['all_accuracies'])
        np.save(f'{OUTPUT_DIR}/single_window_pca_{feature_name}_pca_components.npy', 
               [r['avg_pca_components'] for r in [results]])
        np.save(f'{OUTPUT_DIR}/single_window_pca_{feature_name}_explained_variance.npy', 
               [r['avg_explained_variance'] for r in [results]])
    
    # Save comprehensive summary
    summary = {
        'subject': args.subject,
        'n_iterations': args.n_iterations,
        'analysis_window': ANALYSIS_CONFIG['description'],
        'pca_explained_variance': args.explained_variance,
        'parameter_grid': PARAM_GRID,
        'features': ['maintained_voice', 'maintained_location'],
        'processing_time': str(datetime.now()),
        'method': 'single_window_pca_analysis',
        'preprocessing_pipeline': 'Z-score → MinMax(0,1) → PCA → SVM',
        'sensor_type': 'magnetometers_only',
        'n_channels': epochs_data.shape[1],
        'n_timepoints': epochs_data.shape[2],
        'total_features_before_pca': epochs_data.shape[1] * epochs_data.shape[2],
        'results': {}
    }
    
    # Add detailed results for each feature
    for feature_name, results in all_results.items():
        summary['results'][feature_name] = {
            'mean_accuracy': float(results['mean_accuracy']),
            'std_accuracy': float(results['std_accuracy']),
            'avg_pca_components': float(results['avg_pca_components']),
            'avg_explained_variance': float(results['avg_explained_variance']),
            'most_common_c': float(results['most_common_c']),
            'n_iterations': len(results['all_accuracies']),
            'above_chance': float(results['mean_accuracy'] > 0.5)
        }
    
    # Add comparison
    if len(all_results) == 2:
        voice_acc = all_results['maintained_voice']['mean_accuracy']
        location_acc = all_results['maintained_location']['mean_accuracy']
        summary['comparison'] = {
            'voice_vs_location_diff': float(voice_acc - location_acc),
            'better_feature': 'voice' if voice_acc > location_acc else 'location'
        }
    
    with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    write_log(f"\n=== FINAL RESULTS ===")
    for feature_name, results in all_results.items():
        write_log(f"{feature_name}: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
        write_log(f"  PCA components: {results['avg_pca_components']:.1f}")
        write_log(f"  Explained variance: {results['avg_explained_variance']:.1%}")
        write_log(f"  Most common C: {results['most_common_c']}")
    
    write_log(f"\nProcessing completed at: {datetime.now()}")
    
    print(f"Subject {args.subject} single window PCA analysis completed successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")
    
    # Print final summary
    print("\n=== FINAL RESULTS ===")
    for feature_name, results in all_results.items():
        print(f"{feature_name}: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
        print(f"  PCA components: {results['avg_pca_components']:.1f}")
        print(f"  Explained variance: {results['avg_explained_variance']:.1%}")

if __name__ == "__main__":
    main()