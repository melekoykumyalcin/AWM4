#!/usr/bin/env python
"""
Pre-Post Ping PCA Analysis Script
Implements separate PCA analysis for pre-ping (2.5-3.5s) and post-ping (3.5-4.5s) windows:
- Pre-ping: Pure maintenance period (1000ms)
- Post-ping: Post-distractor maintenance period (1000ms)
- Separate preprocessing pipelines for each window
- Tests ping interference effects on voice and location maintenance

Usage: python pre_post_ping_analysis.py --subject 23
"""

import os
import sys
import argparse
import numpy as np
import mne
mne.set_log_level('WARNING')
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from collections import defaultdict, Counter
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run pre-post ping PCA analysis')
parser.add_argument('--subject', type=int, required=True, help='Subject ID')
parser.add_argument('--n_iterations', type=int, default=10, help='Number of iterations')
parser.add_argument('--explained_variance', type=float, default=0.95, help='PCA explained variance')
args = parser.parse_args()

# Set paths
HOME_DIR = '/mnt/hpc/projects/awm4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
DATA_PATH = HOME_DIR + '/AWM4_data/raw/'
CORRECTED_DATA = HOME_DIR + '/AWM4_data/raw/correctTriggers/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'

# Create output directory
OUTPUT_DIR = PROCESSED_DIR + f'prePostPingPCA/subject_{args.subject}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log file
LOG_FILE_PATH = f'{OUTPUT_DIR}/processing_log.txt'

def write_log(message):
    """Write message to log file"""
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(message + '\n')

# Initialize log
write_log(f"Pre-Post Ping PCA analysis started at: {datetime.now()}")
write_log(f"Subject: {args.subject}")
write_log(f"PCA explained variance: {args.explained_variance:.1%}")

# Window parameters
TARGET_SAMPLING_RATE = 50  # Hz
PRE_PING_START = 2.5   # Start after cue presentation
PRE_PING_END = 3.5     # End at ping
POST_PING_START = 3.5  # Start at ping
POST_PING_END = 4.5    # End before probe

# Window configurations
WINDOWS = {
    'pre_ping': {
        'tmin': PRE_PING_START,
        'tmax': PRE_PING_END,
        'duration': PRE_PING_END - PRE_PING_START,
        'description': 'Pre-ping maintenance (2.5-3.5s)'
    },
    'post_ping': {
        'tmin': POST_PING_START,
        'tmax': POST_PING_END,
        'duration': POST_PING_END - POST_PING_START,
        'description': 'Post-ping maintenance (3.5-4.5s)'
    }
}

# Time markers
CUE_TIME = 2.0
PING_TIME = 3.5
PROBE_TIME = 4.7

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

# Corners and parameters
CORNERS = {
    'Corner1_Sp12L12': [111, 112, 121, 122, 211, 212, 221, 222],
    'Corner2_Sp12L34': [113, 114, 123, 124, 213, 214, 223, 224],
    'Corner3_Sp34L12': [131, 132, 141, 142, 231, 232, 241, 242],
    'Corner4_Sp34L34': [133, 134, 143, 144, 233, 234, 243, 244]
}

AVERAGING_SCHEME = {'trials_per_condition': 3, 'total_trials': 12}
PARAM_GRID = [0.001, 0.01, 0.1, 1.0]

def apply_preprocessing_pipeline(train_data, test_data, window_name, explained_variance_ratio=0.95):
    """
    Apply preprocessing pipeline: Z-score → MinMax → PCA
    Fit on training data, apply to both train and test
    """
    
    # Step 1: Z-score normalization
    z_scaler = StandardScaler()
    train_z_scored = z_scaler.fit_transform(train_data)
    test_z_scored = z_scaler.transform(test_data)
    
    # Step 2: MinMax scaling
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = minmax_scaler.fit_transform(train_z_scored)
    test_scaled = minmax_scaler.transform(test_z_scored)
    
    # Step 3: PCA
    pca = PCA(n_components=explained_variance_ratio)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)
    
    # Calculate actual explained variance
    actual_variance = np.sum(pca.explained_variance_ratio_)
    n_components = pca.n_components_
    
    pipeline_info = {
        'window': window_name,
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
    """Extract maintained information (same as before)"""
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
    """Create pseudo-trials from specific trial indices"""
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
        
    # Count valid conditions
    valid_conditions = [cond for cond in corner_conditions 
                       if cond in condition_trials and len(condition_trials[cond]) >= n_trials_per_condition]
    
    if len(valid_conditions) < 2:
        return None
    
    # Calculate pseudo-trials
    min_trials = min(len(condition_trials[cond]) for cond in valid_conditions)
    n_pseudo_trials = min_trials // n_trials_per_condition
    
    if n_pseudo_trials == 0:
        return None
    
    pseudo_trials = []
    
    # Create pseudo-trials
    for _ in range(n_pseudo_trials):
        sampled_data = []
        for condition in valid_conditions:
            indices = np.random.choice(
                condition_trials[condition], 
                size=n_trials_per_condition, 
                replace=False
            )
            sampled_data.extend(subset_data[indices])
            
            # Remove used indices
            condition_trials[condition] = [i for i in condition_trials[condition] if i not in indices]
        
        pseudo_trial = np.mean(sampled_data, axis=0)
        pseudo_trials.append(pseudo_trial)
    
    return np.array(pseudo_trials) if pseudo_trials else None

def decode_window_with_pca(epochs_data, events, feature_name, window_name, n_iterations, explained_variance_ratio):
    """
    Decode single window with PCA preprocessing
    """
    write_log(f"\nDecoding {window_name}: {feature_name}")
    write_log(f"Window: {WINDOWS[window_name]['description']}")
    write_log(f"Data shape: {epochs_data.shape}")
    
    # Flatten spatio-temporal data
    n_trials, n_channels, n_timepoints = epochs_data.shape
    X_flattened = epochs_data.reshape(n_trials, n_channels * n_timepoints)
    
    write_log(f"Flattened features: {n_channels} channels × {n_timepoints} timepoints = {X_flattened.shape[1]}")
    
    # Storage for results
    iteration_results = []
    
    for iteration in range(n_iterations):
        write_log(f"  Iteration {iteration + 1}/{n_iterations}")
        
        # Create labels
        if feature_name == 'maintained_voice':
            original_labels = np.array([0 if (e//10)%10 in [1,2] else 1 for e in events])
        else:
            original_labels = np.array([0 if e%10 in [1,2] else 1 for e in events])
        
        if len(np.unique(original_labels)) < 2:
            write_log(f"  Skipping - only one class present")
            continue
        
        # Outer CV
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
        outer_scores = []
        best_cs = []
        pca_info_list = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_flattened, original_labels)):
            
            # Split data
            train_data = X_flattened[train_idx]
            test_data = X_flattened[test_idx]
            train_events = events[train_idx]
            test_events = events[test_idx]
            
            # Create pseudo-trials
            train_pseudo_data = []
            train_pseudo_labels = []
            test_pseudo_data = []
            test_pseudo_labels = []
            
            for corner_name, corner_conditions in CORNERS.items():
                # Training pseudo-trials
                train_pseudo = create_pseudo_trials_from_indices(
                    train_data, train_events, 
                    np.arange(len(train_events)),
                    corner_conditions,
                    AVERAGING_SCHEME['trials_per_condition']
                )
                
                if train_pseudo is not None:
                    train_pseudo_data.append(train_pseudo)
                    if feature_name == 'maintained_voice':
                        labels = [0 if 'Sp12' in corner_name else 1] * len(train_pseudo)
                    else:
                        labels = [0 if 'L12' in corner_name else 1] * len(train_pseudo)
                    train_pseudo_labels.extend(labels)
                
                # Test pseudo-trials
                test_pseudo = create_pseudo_trials_from_indices(
                    test_data, test_events,
                    np.arange(len(test_events)),
                    corner_conditions,
                    AVERAGING_SCHEME['trials_per_condition']
                )
                
                if test_pseudo is not None:
                    test_pseudo_data.append(test_pseudo)
                    if feature_name == 'maintained_voice':
                        labels = [0 if 'Sp12' in corner_name else 1] * len(test_pseudo)
                    else:
                        labels = [0 if 'L12' in corner_name else 1] * len(test_pseudo)
                    test_pseudo_labels.extend(labels)
            
            # Check data availability
            if not train_pseudo_data or not test_pseudo_data:
                continue
            
            if len(train_pseudo_data) < 2 or len(test_pseudo_data) < 2:
                continue
            
            # Combine pseudo-trials
            X_train_pseudo = np.vstack(train_pseudo_data)
            y_train = np.array(train_pseudo_labels)
            X_test_pseudo = np.vstack(test_pseudo_data)
            y_test = np.array(test_pseudo_labels)
            
            # Verify classes
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            
            # Apply preprocessing pipeline
            try:
                X_train_processed, X_test_processed, pipeline_info = apply_preprocessing_pipeline(
                    X_train_pseudo, X_test_pseudo, window_name, explained_variance_ratio
                )
                pca_info_list.append(pipeline_info)
                
                write_log(f"    Fold {fold_idx+1}: {pipeline_info['original_features']} → {pipeline_info['pca_components']} features")
                
            except Exception as e:
                write_log(f"    Fold {fold_idx+1}: PCA failed - {str(e)}")
                continue
            
            # Inner CV for hyperparameter selection
            best_score = -1
            best_c = 1.0
            
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=None)
            
            for c_value in PARAM_GRID:
                clf = SVC(kernel='linear', C=c_value, probability=True)
                
                try:
                    inner_scores = cross_val_score(
                        clf, X_train_processed, y_train,
                        cv=inner_cv, scoring='accuracy', n_jobs=1
                    )
                    mean_inner_score = np.mean(inner_scores)
                    
                    if mean_inner_score > best_score:
                        best_score = mean_inner_score
                        best_c = c_value
                except:
                    continue
            
            # Final model
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
    
    # Aggregate results
    if iteration_results:
        all_accuracies = [r['accuracy'] for r in iteration_results]
        mean_accuracy = np.mean(all_accuracies)
        std_accuracy = np.std(all_accuracies)
        
        # Average PCA info
        avg_components = np.mean([np.mean([p['pca_components'] for p in r['pca_info']]) 
                                 for r in iteration_results])
        avg_explained_var = np.mean([np.mean([p['explained_variance'] for p in r['pca_info']]) 
                                    for r in iteration_results])
        
        write_log(f"  Final Results:")
        write_log(f"    Mean accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
        write_log(f"    Average PCA components: {avg_components:.1f}")
        write_log(f"    Average explained variance: {avg_explained_var:.1%}")
        
        return {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'all_accuracies': all_accuracies,
            'avg_pca_components': avg_components,
            'avg_explained_variance': avg_explained_var,
            'iteration_results': iteration_results
        }
    else:
        write_log(f"  No valid results for {feature_name} in {window_name}")
        return None

def load_subject_data(subject, meta_info):
    """Load and preprocess data for both windows"""
    write_log(f"\nLoading data for subject {subject}...")
    
    try:
        # Extract maintained information
        memorized, all_events = extract_maintained_information(subject, meta_info)
        
        if memorized is None:
            write_log("Could not extract maintained information")
            return None, None, None, None
        
        # Load cleaned epochs
        clean_trials = mne.read_epochs(
            f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif",
            preload=True, verbose='ERROR'
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
        
        # Update event codes
        clean_trials.events[:, 2] = memorized[:len(clean_trials.events)]
        clean_trials.event_id = EVENT_DICT
        
        # Select magnetometers and resample
        mag_epochs = clean_trials.copy().pick_types(meg='mag')
        mag_epochs = mag_epochs.resample(TARGET_SAMPLING_RATE, npad='auto')
        
        # Extract both windows
        pre_ping_epochs = mag_epochs.copy().crop(tmin=PRE_PING_START, tmax=PRE_PING_END)
        post_ping_epochs = mag_epochs.copy().crop(tmin=POST_PING_START, tmax=POST_PING_END)
        
        # Get data
        pre_ping_data = pre_ping_epochs.get_data(copy=False)
        post_ping_data = post_ping_epochs.get_data(copy=False)
        maintained_events = mag_epochs.events[:, 2]
        
        write_log(f"Data loaded successfully:")
        write_log(f"  Pre-ping shape: {pre_ping_data.shape}")
        write_log(f"  Post-ping shape: {post_ping_data.shape}")
        write_log(f"  Maintained events: {len(maintained_events)}")
        
        return pre_ping_data, post_ping_data, maintained_events, mag_epochs.info['sfreq']
        
    except Exception as e:
        write_log(f"Error loading data: {str(e)}")
        import traceback
        write_log(traceback.format_exc())
        return None, None, None, None

def plot_pre_post_ping_results(all_results):
    """Create comparison plots for pre-post ping results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Subject {args.subject} - Pre-Post Ping PCA Analysis', fontsize=16)
    
    features = ['maintained_voice', 'maintained_location']
    feature_labels = ['Voice Identity', 'Location']
    windows = ['pre_ping', 'post_ping']
    window_labels = ['Pre-Ping', 'Post-Ping']
    colors = {'pre_ping': '#2E8B57', 'post_ping': '#DC143C'}  # Green for pre, red for post
    
    for feat_idx, (feature, feature_label) in enumerate(zip(features, feature_labels)):
        
        # Plot 1: Accuracy comparison
        ax1 = axes[feat_idx, 0]
        
        pre_accuracy = all_results['pre_ping'][feature]['mean_accuracy'] if feature in all_results['pre_ping'] else np.nan
        post_accuracy = all_results['post_ping'][feature]['mean_accuracy'] if feature in all_results['post_ping'] else np.nan
        pre_std = all_results['pre_ping'][feature]['std_accuracy'] if feature in all_results['pre_ping'] else 0
        post_std = all_results['post_ping'][feature]['std_accuracy'] if feature in all_results['post_ping'] else 0
        
        x_pos = [0, 1]
        accuracies = [pre_accuracy, post_accuracy]
        stds = [pre_std, post_std]
        
        bars = ax1.bar(x_pos, accuracies, yerr=stds, capsize=5,
                      color=[colors['pre_ping'], colors['post_ping']], alpha=0.7)
        
        ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Chance')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(window_labels)
        ax1.set_ylabel('Decoding Accuracy')
        ax1.set_title(f'{feature_label} - Pre vs Post Ping')
        ax1.set_ylim(0.45, 0.75)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add accuracy values as text
        for i, (acc, std) in enumerate(zip(accuracies, stds)):
            if not np.isnan(acc):
                ax1.text(i, acc + std + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Ping effect (difference)
        ax2 = axes[feat_idx, 1]
        
        ping_effect = post_accuracy - pre_accuracy
        
        if not np.isnan(ping_effect):
            color = colors['post_ping'] if ping_effect > 0 else colors['pre_ping']
            bar = ax2.bar([0], [ping_effect], color=color, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
            ax2.set_ylabel('Ping Effect (Post - Pre)')
            ax2.set_title(f'{feature_label} - Ping Effect')
            ax2.set_xticks([0])
            ax2.set_xticklabels([feature_label])
            ax2.grid(True, alpha=0.3)
            
            # Add effect value as text
            ax2.text(0, ping_effect + 0.005 if ping_effect > 0 else ping_effect - 0.005, 
                    f'{ping_effect:.3f}', ha='center', 
                    va='bottom' if ping_effect > 0 else 'top', fontweight='bold')
            
            # Add interpretation
            if ping_effect > 0:
                ax2.text(0, ping_effect/2, 'Improved\nafter ping', ha='center', va='center', 
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            else:
                ax2.text(0, ping_effect/2, 'Disrupted\nby ping', ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/pre_post_ping_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    write_log(f"\nComparison plot saved to {OUTPUT_DIR}/pre_post_ping_comparison.png")

def main():
    """Main processing function"""
    # Load metadata
    meta_info = pd.read_excel(META_FILE)
    
    # Load subject data
    pre_ping_data, post_ping_data, events, sfreq = load_subject_data(args.subject, meta_info)
    
    if pre_ping_data is None:
        write_log("Failed to load data. Exiting.")
        sys.exit(1)
    
    write_log(f"Loaded data sampling rate: {sfreq}Hz")
    
    write_log("\n=== PRE-POST PING PCA ANALYSIS ===")
    write_log(f"=== Pre-ping: {WINDOWS['pre_ping']['description']} ===")
    write_log(f"=== Post-ping: {WINDOWS['post_ping']['description']} ===")
    write_log(f"=== PCA explained variance: {args.explained_variance:.1%} ===")
    
    # Results storage
    all_results = {
        'pre_ping': {},
        'post_ping': {}
    }
    
    # Process each window
    for window_name, window_data in [('pre_ping', pre_ping_data), ('post_ping', post_ping_data)]:
        write_log(f"\n{'='*60}")
        write_log(f"Processing {window_name.upper()} window")
        write_log(f"{'='*60}")
        
        # Process each feature
        for feature_name in ['maintained_voice', 'maintained_location']:
            write_log(f"\nProcessing {feature_name} in {window_name}")
            
            results = decode_window_with_pca(
                window_data, events, feature_name, window_name,
                args.n_iterations, args.explained_variance
            )
            
            if results is not None:
                all_results[window_name][feature_name] = results
                write_log(f"  {feature_name} in {window_name}: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
            else:
                write_log(f"  {feature_name} in {window_name}: No valid results")
    
    # Create comparison plots
    if any(all_results['pre_ping']) and any(all_results['post_ping']):
        plot_pre_post_ping_results(all_results)
    
    # Calculate ping effects
    ping_effects = {}
    for feature_name in ['maintained_voice', 'maintained_location']:
        if (feature_name in all_results['pre_ping'] and 
            feature_name in all_results['post_ping']):
            
            pre_acc = all_results['pre_ping'][feature_name]['mean_accuracy']
            post_acc = all_results['post_ping'][feature_name]['mean_accuracy']
            ping_effect = post_acc - pre_acc
            ping_effects[feature_name] = ping_effect
    
    # Save results
    write_log(f"\nSaving results...")
    
    # Save individual results
    for window_name, window_results in all_results.items():
        for feature_name, results in window_results.items():
            np.save(f'{OUTPUT_DIR}/{window_name}_{feature_name}_accuracies.npy', 
                   results['all_accuracies'])
    
    # Save comprehensive summary
    summary = {
        'subject': args.subject,
        'n_iterations': args.n_iterations,
        'windows': WINDOWS,
        'pca_explained_variance': args.explained_variance,
        'parameter_grid': PARAM_GRID,
        'features': ['maintained_voice', 'maintained_location'],
        'processing_time': str(datetime.now()),
        'method': 'pre_post_ping_pca_analysis',
        'preprocessing_pipeline': 'Z-score → MinMax(0,1) → PCA → SVM (separate for each window)',
        'results': {},
        'ping_effects': ping_effects
    }
    
    # Add detailed results
    for window_name, window_results in all_results.items():
        summary['results'][window_name] = {}
        for feature_name, results in window_results.items():
            summary['results'][window_name][feature_name] = {
                'mean_accuracy': float(results['mean_accuracy']),
                'std_accuracy': float(results['std_accuracy']),
                'avg_pca_components': float(results['avg_pca_components']),
                'avg_explained_variance': float(results['avg_explained_variance']),
                'n_iterations': len(results['all_accuracies'])
            }
    
    with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Final summary
    write_log(f"\n=== PING EFFECTS ANALYSIS ===")
    for feature_name, ping_effect in ping_effects.items():
        write_log(f"{feature_name}: {ping_effect:+.3f} (Post - Pre)")
        if ping_effect > 0:
            write_log(f"  → {feature_name} IMPROVED after ping")
        else:
            write_log(f"  → {feature_name} DISRUPTED by ping")
    
    write_log(f"\nProcessing completed at: {datetime.now()}")
    
    print(f"Subject {args.subject} pre-post ping analysis completed!")
    print(f"Results saved to: {OUTPUT_DIR}")
    
    # Print summary
    print("\n=== PING EFFECTS ===")
    for feature_name, ping_effect in ping_effects.items():
        print(f"{feature_name}: {ping_effect:+.3f}")

if __name__ == "__main__":
    main()