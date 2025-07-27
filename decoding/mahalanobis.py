#!/usr/bin/env python
"""
Mahalanobis Distance Decoding and RSA Analysis Script
Implements 4-way classification and representational similarity analysis
For auditory working memory with 4 speakers × 4 locations

Usage: python mahalanobis_rsa_analysis.py --subject 23
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
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from collections import defaultdict, Counter
from joblib import Parallel, delayed
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run Mahalanobis distance decoding and RSA analysis')
parser.add_argument('--subject', type=int, required=True, help='Subject ID')
parser.add_argument('--n_iterations', type=int, default=5, help='Number of iterations')
parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
parser.add_argument('--n_jobs', type=int, default=20, help='Number of parallel jobs')
parser.add_argument('--tmin', type=float, default=2.0, help='Start time of analysis window')
parser.add_argument('--tmax', type=float, default=4.5, help='End time of analysis window')
args = parser.parse_args()

# Set paths for HPC (using your existing paths)
HOME_DIR = '/media/headmodel/Elements/AWM4/'
#HOME_DIR = '/mnt/hpc/projects/awm4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
DATA_PATH = HOME_DIR + '/AWM4_data/raw/'
CORRECTED_DATA = HOME_DIR + '/AWM4_data/raw/correctTriggers/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'

# Create output directory
OUTPUT_DIR = PROCESSED_DIR + f'mahalanobis_analysis/subject_{args.subject}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log file path
LOG_FILE_PATH = f'{OUTPUT_DIR}/processing_log.txt'

def write_log(message):
    """Write message to log file"""
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(message + '\n')

# Initialize log
write_log(f"Mahalanobis distance decoding and RSA analysis started at: {datetime.now()}")
write_log(f"Subject: {args.subject}")
write_log(f"Analysis window: {args.tmin}s to {args.tmax}s")

# Decoding parameters
RESAMPLE_FREQ = 100  # Hz
WINDOW_LENGTH_SEC = 0.1  # 100ms as in the paper
WINDOW_STEP_SEC = 0.01  # 10ms step

# Important time markers
CUE_TIME = 2.0  # Cue presentation
PING_TIME = 3.5  # Ping occurs at 3.5s
PROBE_TIME = 4.5  # Probe presentation

# Stimulus parameters
FREQUENCIES_HZ = [114.33, 136.16, 162.16, 193.12]
ITD_MS = [-0.35, -0.12, 0.12, 0.35]
AZIMUTH_DEG = [-39, -13, 13, 39]

# Event dictionary (using your existing one)
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

class MahalanobisDecoder:
    """
    Implements Mahalanobis distance decoding for multi-class problems
    Following Wolff et al. approach
    """
    def __init__(self, n_repetitions=10, n_folds=5):
        self.n_repetitions = n_repetitions
        self.n_folds = n_folds
        
    def decode_timepoint(self, X, y):
        """
        Decode a single timepoint using repeated stratified CV
        Returns: accuracy and detailed results
        """
        repetition_accuracies = []
        all_confusion_matrices = []
        
        for rep in range(self.n_repetitions):
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=rep)
            fold_accuracies = []
            fold_confusions = []
            
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Ensure balanced training set by subsampling
                unique_classes = np.unique(y_train)
                min_samples = min([np.sum(y_train == c) for c in unique_classes])
                
                balanced_idx = []
                for class_label in unique_classes:
                    class_indices = np.where(y_train == class_label)[0]
                    if len(class_indices) >= min_samples:
                        selected = np.random.choice(class_indices, min_samples, replace=False)
                        balanced_idx.extend(train_idx[selected])
                
                if len(balanced_idx) < len(unique_classes) * 2:  # Need at least 2 samples per class
                    continue
                
                X_train_balanced = X[balanced_idx]
                y_train_balanced = y[balanced_idx]
                
                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_balanced)
                X_test_scaled = scaler.transform(X_test)
                
                # Compute class prototypes (mean patterns)
                prototypes = {}
                for class_label in unique_classes:
                    class_mask = y_train_balanced == class_label
                    if np.sum(class_mask) > 0:
                        prototypes[class_label] = np.mean(X_train_scaled[class_mask], axis=0)
                
                # Estimate covariance using Ledoit-Wolf shrinkage
                try:
                    cov_estimator = LedoitWolf()
                    cov_estimator.fit(X_train_scaled)
                    precision = cov_estimator.precision_
                except:
                    # Fallback to simple covariance if LedoitWolf fails
                    cov = np.cov(X_train_scaled.T)
                    precision = np.linalg.pinv(cov)
                
                # Compute distances and predict for each test trial
                predictions = []
                for test_sample in X_test_scaled:
                    distances = {}
                    for class_label, prototype in prototypes.items():
                        diff = test_sample - prototype
                        # Mahalanobis distance
                        dist = np.sqrt(np.abs(diff @ precision @ diff))
                        distances[class_label] = dist
                    
                    # Predict class with minimum distance
                    if distances:
                        pred_label = min(distances, key=distances.get)
                        predictions.append(pred_label)
                    else:
                        predictions.append(y_test[0])  # Fallback
                
                # Calculate accuracy
                if predictions:
                    accuracy = np.mean(np.array(predictions) == y_test)
                    fold_accuracies.append(accuracy)
                    
                    # Store confusion matrix
                    confusion = np.zeros((len(unique_classes), len(unique_classes)))
                    for true_label, pred_label in zip(y_test, predictions):
                        if true_label in unique_classes and pred_label in unique_classes:
                            true_idx = np.where(unique_classes == true_label)[0][0]
                            pred_idx = np.where(unique_classes == pred_label)[0][0]
                            confusion[true_idx, pred_idx] += 1
                    fold_confusions.append(confusion)
            
            if fold_accuracies:
                repetition_accuracies.append(np.mean(fold_accuracies))
                all_confusion_matrices.extend(fold_confusions)
        
        if repetition_accuracies:
            return {
                'accuracy': np.mean(repetition_accuracies),
                'accuracy_std': np.std(repetition_accuracies),
                'all_accuracies': repetition_accuracies,
                'confusion_matrix': np.mean(all_confusion_matrices, axis=0) if all_confusion_matrices else None
            }
        else:
            return {
                'accuracy': 1.0 / len(np.unique(y)),  # Chance level
                'accuracy_std': 0.0,
                'all_accuracies': [],
                'confusion_matrix': None
            }

class RSAAnalyzer:
    """
    Implements Representational Similarity Analysis
    """
    def __init__(self, n_iterations=5):
        self.n_iterations = n_iterations
        
    def compute_neural_rdm(self, X, conditions):
        """
        Compute neural RDM using Mahalanobis distances
        """
        unique_conditions = np.unique(conditions)
        n_conditions = len(unique_conditions)
        
        # Check if we have enough trials per condition
        min_trials_per_condition = 3
        valid_conditions = []
        for cond in unique_conditions:
            if np.sum(conditions == cond) >= min_trials_per_condition:
                valid_conditions.append(cond)
        
        if len(valid_conditions) < 2:
            return None, None, None
        
        rdm_iterations = []
        
        for iteration in range(self.n_iterations):
            # Balance trials across conditions
            min_trials = min([np.sum(conditions == c) for c in valid_conditions])
            balanced_idx = []
            
            for cond in valid_conditions:
                cond_idx = np.where(conditions == cond)[0]
                selected = np.random.choice(cond_idx, min_trials, replace=False)
                balanced_idx.extend(selected)
            
            X_balanced = X[balanced_idx]
            conditions_balanced = conditions[balanced_idx]
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_balanced)
            
            # Compute condition averages
            condition_patterns = {}
            for cond in valid_conditions:
                mask = conditions_balanced == cond
                condition_patterns[cond] = np.mean(X_scaled[mask], axis=0)
            
            # Estimate covariance
            try:
                cov_est = LedoitWolf()
                cov_est.fit(X_scaled)
                precision = cov_est.precision_
            except:
                cov = np.cov(X_scaled.T)
                precision = np.linalg.pinv(cov)
            
            # Compute pairwise distances
            rdm = np.zeros((len(valid_conditions), len(valid_conditions)))
            for i, cond1 in enumerate(valid_conditions):
                for j, cond2 in enumerate(valid_conditions):
                    if i != j:
                        diff = condition_patterns[cond1] - condition_patterns[cond2]
                        dist = np.sqrt(np.abs(diff @ precision @ diff))
                        rdm[i, j] = dist
            
            rdm_iterations.append(rdm)
        
        if rdm_iterations:
            mean_rdm = np.mean(rdm_iterations, axis=0)
            std_rdm = np.std(rdm_iterations, axis=0)
            return mean_rdm, std_rdm, valid_conditions
        else:
            return None, None, None
    
    def create_model_rdms(self):
        """
        Create the hypothesis-based model RDMs we discussed
        """
        models = {}
        
        # SPEAKER MODELS
        # 1. Harmonic (log frequency) model
        log_freq_dist = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                log_freq_dist[i, j] = abs(np.log2(FREQUENCIES_HZ[i]) - 
                                         np.log2(FREQUENCIES_HZ[j]))
        models['speaker_harmonic'] = log_freq_dist
        
        # 2. Categorical pitch model (low vs high)
        categorical_pitch = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0]
        ])
        models['speaker_categorical'] = categorical_pitch
        
        # LOCATION MODELS
        # 3. ITD magnitude (ignoring side)
        itd_magnitude = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                itd_magnitude[i, j] = abs(abs(ITD_MS[i]) - abs(ITD_MS[j]))
        models['location_magnitude'] = itd_magnitude
        
        # 4. Hemispheric model
        hemispheric = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0]
        ])
        models['location_hemispheric'] = hemispheric
        
        # Z-score all models for fair comparison
        for name, rdm in models.items():
            # Only z-score if there's variance
            if np.std(rdm[np.triu_indices(4, k=1)]) > 0:
                rdm_vec = rdm[np.triu_indices(4, k=1)]
                rdm_vec_z = (rdm_vec - np.mean(rdm_vec)) / np.std(rdm_vec)
                rdm_z = np.zeros_like(rdm)
                rdm_z[np.triu_indices(4, k=1)] = rdm_vec_z
                rdm_z = rdm_z + rdm_z.T
                models[name] = rdm_z
        
        return models
    
    def compare_rdms(self, neural_rdm, model_rdms, method='regression'):
        """
        Compare neural RDM to model RDMs using multiple regression
        """
        if neural_rdm is None:
            return None
            
        # Vectorize upper triangle
        n = neural_rdm.shape[0]
        mask = np.triu(np.ones_like(neural_rdm), k=1).astype(bool)
        neural_vec = neural_rdm[mask]
        
        results = {}
        
        if method == 'regression':
            # Prepare predictors
            model_matrix = []
            model_names = []
            for name, rdm in model_rdms.items():
                if rdm.shape[0] == n:  # Make sure dimensions match
                    model_vec = rdm[mask]
                    model_matrix.append(model_vec)
                    model_names.append(name)
            
            if len(model_matrix) > 0:
                X = np.array(model_matrix).T
                
                # Fit regression
                try:
                    reg = LinearRegression()
                    reg.fit(X, neural_vec)
                    
                    # Store results
                    for i, name in enumerate(model_names):
                        r, p = pearsonr(X[:, i], neural_vec)
                        results[name] = {
                            'beta': reg.coef_[i],
                            'correlation': r,
                            'p_value': p
                        }
                    
                    results['r_squared'] = reg.score(X, neural_vec)
                except:
                    results['r_squared'] = 0.0
        
        return results

# Your existing helper functions with modifications
def extract_labels(events, feature='speaker'):
    """
    Extract labels from event codes for 4-way classification
    """
    labels = []
    valid_idx = []
    
    for i, event in enumerate(events):
        if 111 <= event <= 244:
            if feature == 'speaker':
                label = (event // 10) % 10 - 1  # 0-3 for speakers 1-4
            elif feature == 'location':
                label = event % 10 - 1  # 0-3 for locations 1-4
            else:  # 'full' - all 16 conditions
                speaker = (event // 10) % 10 - 1
                location = event % 10 - 1
                label = speaker * 4 + location  # 0-15
            
            labels.append(label)
            valid_idx.append(i)
    
    return np.array(labels), np.array(valid_idx)

def mahalanobis_sliding_window(epochs_data, events, sfreq, feature='speaker', 
                              n_repetitions=10, n_folds=5):
    """
    Run Mahalanobis distance decoding with sliding window
    """
    write_log(f"\nRunning Mahalanobis distance decoding for {feature}")
    
    # Extract labels
    labels, valid_idx = extract_labels(events, feature)
    if len(valid_idx) == 0:
        write_log(f"No valid trials found for {feature}")
        return None
    
    data = epochs_data[valid_idx]
    
    # Check class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    write_log(f"Class distribution: {dict(zip(unique_labels, counts))}")
    
    # Initialize decoder
    decoder = MahalanobisDecoder(n_repetitions=n_repetitions, n_folds=n_folds)
    
    # Sliding window parameters
    window_length = int(sfreq * WINDOW_LENGTH_SEC)
    window_step = int(sfreq * WINDOW_STEP_SEC)
    n_times = data.shape[2]
    n_windows = (n_times - window_length) // window_step + 1
    
    # Storage
    accuracies = []
    accuracy_stds = []
    timepoints = []
    all_confusion_matrices = []
    
    write_log(f"Number of windows: {n_windows}, Window length: {window_length} samples, Step: {window_step} samples")
    write_log(f"Total time points: {n_times}, Sampling frequency: {sfreq} Hz")
    
    # Process each window
    for win_idx in tqdm(range(n_windows), desc=f"{feature} decoding"):
        win_start = win_idx * window_step
        win_end = win_start + window_length
        
        # Extract and flatten window
        X = data[:, :, win_start:win_end]
        X = X.reshape(data.shape[0], -1)
        
        # Decode
        result = decoder.decode_timepoint(X, labels)
        
        accuracies.append(result['accuracy'])
        accuracy_stds.append(result['accuracy_std'])
        all_confusion_matrices.append(result['confusion_matrix'])
        
        # Time point (center of window)
        time = args.tmin + (win_start / sfreq) + (WINDOW_LENGTH_SEC / 2)
        timepoints.append(time)
    
    # Log results
    write_log(f"Mean accuracy: {np.mean(accuracies):.3f}")
    write_log(f"Peak accuracy: {np.max(accuracies):.3f} at {timepoints[np.argmax(accuracies)]:.3f}s")
    
    # Find accuracy at ping time
    ping_idx = np.argmin(np.abs(np.array(timepoints) - PING_TIME))
    write_log(f"Accuracy at ping ({PING_TIME}s): {accuracies[ping_idx]:.3f}")
    
    return {
        'accuracies': np.array(accuracies),
        'accuracy_stds': np.array(accuracy_stds),
        'timepoints': np.array(timepoints),
        'confusion_matrices': all_confusion_matrices,
        'n_classes': len(unique_labels)
    }

def rsa_sliding_window(epochs_data, events, sfreq, feature='speaker', n_iterations=5):
    """
    Run RSA with sliding window
    """
    write_log(f"\nRunning RSA for {feature}")
    
    # Extract labels
    labels, valid_idx = extract_labels(events, feature)
    if len(valid_idx) == 0:
        write_log(f"No valid trials found for {feature}")
        return None
    
    data = epochs_data[valid_idx]
    
    # Initialize RSA analyzer
    rsa = RSAAnalyzer(n_iterations=n_iterations)
    
    # Create model RDMs
    model_rdms_all = rsa.create_model_rdms()
    # Select relevant models
    if feature == 'speaker':
        model_rdms = {k: v for k, v in model_rdms_all.items() if 'speaker' in k}
    else:
        model_rdms = {k: v for k, v in model_rdms_all.items() if 'location' in k}
    
    # Sliding window parameters
    window_length = int(sfreq * WINDOW_LENGTH_SEC)
    window_step = int(sfreq * WINDOW_STEP_SEC)
    n_times = data.shape[2]
    n_windows = (n_times - window_length) // window_step + 1
    
    # Storage
    results = {
        'timepoints': [],
        'neural_rdms': [],
        'model_fits': defaultdict(list),
        'r_squared': []
    }
    
    # Process each window
    for win_idx in tqdm(range(n_windows), desc=f"{feature} RSA"):
        win_start = win_idx * window_step
        win_end = win_start + window_length
        
        # Extract and flatten window
        X = data[:, :, win_start:win_end]
        X = X.reshape(data.shape[0], -1)
        
        # Compute neural RDM
        neural_rdm, rdm_std, valid_conditions = rsa.compute_neural_rdm(X, labels)
        
        if neural_rdm is not None:
            # Compare to models
            comparison = rsa.compare_rdms(neural_rdm, model_rdms, method='regression')
            
            if comparison is not None:
                results['neural_rdms'].append(neural_rdm)
                results['r_squared'].append(comparison.get('r_squared', 0))
                
                for model_name in model_rdms.keys():
                    if model_name in comparison:
                        results['model_fits'][model_name].append({
                            'beta': comparison[model_name]['beta'],
                            'correlation': comparison[model_name]['correlation'],
                            'p_value': comparison[model_name]['p_value']
                        })
                    else:
                        results['model_fits'][model_name].append({
                            'beta': 0,
                            'correlation': 0,
                            'p_value': 1
                        })
            else:
                # Append zeros if comparison failed
                results['neural_rdms'].append(None)
                results['r_squared'].append(0)
                for model_name in model_rdms.keys():
                    results['model_fits'][model_name].append({
                        'beta': 0,
                        'correlation': 0,
                        'p_value': 1
                    })
        else:
            # Append None if RDM computation failed
            results['neural_rdms'].append(None)
            results['r_squared'].append(0)
            for model_name in model_rdms.keys():
                results['model_fits'][model_name].append({
                    'beta': 0,
                    'correlation': 0,
                    'p_value': 1
                })
        
        # Time point
        time = args.tmin + (win_start / sfreq) + (WINDOW_LENGTH_SEC / 2)
        results['timepoints'].append(time)
    
    return results

def load_subject_data(subject, meta_info):
    """Load and preprocess data - using your existing function"""
    write_log(f"\nLoading data for subject {subject}...")
    
    try:
        # Extract maintained information
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
        delay_epochs.crop(tmin=args.tmin, tmax=args.tmax)
        
        # Select magnetometers and resample
        mag_epochs = delay_epochs.copy().pick_types(meg='mag')
        mag_epochs = mag_epochs.resample(RESAMPLE_FREQ, npad='auto')
        
        # Get data
        epochs_data = mag_epochs.get_data(copy=False)
        maintained_events = mag_epochs.events[:, 2]
        
        write_log(f"Data loaded successfully. Shape: {epochs_data.shape}")
        write_log(f"Maintained events: {len(maintained_events)}")
        
        # Log distribution for 4-way classification
        speaker_dist = Counter((e//10)%10 for e in maintained_events if 111 <= e <= 244)
        location_dist = Counter(e%10 for e in maintained_events if 111 <= e <= 244)
        
        write_log(f"Speaker distribution: {dict(sorted(speaker_dist.items()))}")
        write_log(f"Location distribution: {dict(sorted(location_dist.items()))}")
        
        return epochs_data, maintained_events, mag_epochs.info['sfreq']
        
    except Exception as e:
        write_log(f"Error loading data: {str(e)}")
        import traceback
        write_log(traceback.format_exc())
        return None, None, None

def extract_maintained_information(subject, metaInfo):
    """Using your existing function"""
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

def visualize_results(speaker_decoding, location_decoding, speaker_rsa, location_rsa):
    """Create comprehensive visualization of results"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Decoding time courses
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    
    # Speaker decoding
    if speaker_decoding is not None:
        ax1.plot(speaker_decoding['timepoints'], speaker_decoding['accuracies'], 
                'b-', linewidth=2, label='4-way decoding')
        ax1.fill_between(speaker_decoding['timepoints'],
                        speaker_decoding['accuracies'] - speaker_decoding['accuracy_stds']/np.sqrt(args.n_iterations),
                        speaker_decoding['accuracies'] + speaker_decoding['accuracy_stds']/np.sqrt(args.n_iterations),
                        alpha=0.3, color='b')
        ax1.axhline(0.25, color='k', linestyle='--', alpha=0.5, label='Chance')
        ax1.axvline(PING_TIME, color='r', linestyle='--', alpha=0.7, label='Ping')
        ax1.set_ylabel('Decoding Accuracy')
        ax1.set_xlabel('Time (s)')
        ax1.set_title('Speaker Identity Decoding (4-way Mahalanobis)')
        ax1.set_ylim([0.15, max(0.6, np.max(speaker_decoding['accuracies']) + 0.1)])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Location decoding
    if location_decoding is not None:
        ax2.plot(location_decoding['timepoints'], location_decoding['accuracies'], 
                'r-', linewidth=2, label='4-way decoding')
        ax2.fill_between(location_decoding['timepoints'],
                        location_decoding['accuracies'] - location_decoding['accuracy_stds']/np.sqrt(args.n_iterations),
                        location_decoding['accuracies'] + location_decoding['accuracy_stds']/np.sqrt(args.n_iterations),
                        alpha=0.3, color='r')
        ax2.axhline(0.25, color='k', linestyle='--', alpha=0.5, label='Chance')
        ax2.axvline(PING_TIME, color='r', linestyle='--', alpha=0.7, label='Ping')
        ax2.set_ylabel('Decoding Accuracy')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('Location Decoding (4-way Mahalanobis)')
        ax2.set_ylim([0.15, max(0.6, np.max(location_decoding['accuracies']) + 0.1)])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 2. RSA model fits
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2:])
    
    # Speaker RSA
    if speaker_rsa is not None and 'model_fits' in speaker_rsa:
        for model_name, fits in speaker_rsa['model_fits'].items():
            betas = [f['beta'] for f in fits]
            label = model_name.replace('speaker_', '').capitalize()
            ax3.plot(speaker_rsa['timepoints'], betas, linewidth=2, label=label)
        
        ax3.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax3.axvline(PING_TIME, color='r', linestyle='--', alpha=0.7)
        ax3.set_ylabel('Model Beta Weight')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Speaker Model Fits (Multiple Regression)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Location RSA
    if location_rsa is not None and 'model_fits' in location_rsa:
        for model_name, fits in location_rsa['model_fits'].items():
            betas = [f['beta'] for f in fits]
            label = model_name.replace('location_', '').capitalize()
            ax4.plot(location_rsa['timepoints'], betas, linewidth=2, label=label)
        
        ax4.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax4.axvline(PING_TIME, color='r', linestyle='--', alpha=0.7)
        ax4.set_ylabel('Model Beta Weight')
        ax4.set_xlabel('Time (s)')
        ax4.set_title('Location Model Fits (Multiple Regression)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 3. R-squared over time
    ax5 = fig.add_subplot(gs[2, :2])
    ax6 = fig.add_subplot(gs[2, 2:])
    
    if speaker_rsa is not None and 'r_squared' in speaker_rsa:
        ax5.plot(speaker_rsa['timepoints'], speaker_rsa['r_squared'], 
                'k-', linewidth=2)
        ax5.set_ylabel('R²')
        ax5.set_xlabel('Time (s)')
        ax5.set_title('Speaker Models Variance Explained')
        ax5.set_ylim([0, 0.5])
        ax5.axvline(PING_TIME, color='r', linestyle='--', alpha=0.7)
        ax5.grid(True, alpha=0.3)
    
    if location_rsa is not None and 'r_squared' in location_rsa:
        ax6.plot(location_rsa['timepoints'], location_rsa['r_squared'], 
                'k-', linewidth=2)
        ax6.set_ylabel('R²')
        ax6.set_xlabel('Time (s)')
        ax6.set_title('Location Models Variance Explained')
        ax6.set_ylim([0, 0.5])
        ax6.axvline(PING_TIME, color='r', linestyle='--', alpha=0.7)
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Subject {args.subject} - Mahalanobis Decoding & RSA Results', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/mahalanobis_rsa_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    write_log(f"\nResults visualization saved to {OUTPUT_DIR}/mahalanobis_rsa_results.png")

def visualize_rdms_at_peak(speaker_decoding, location_decoding, speaker_rsa, location_rsa):
    """Visualize RDMs at peak decoding times"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Speaker RDMs at peak
    if speaker_decoding is not None and speaker_rsa is not None:
        peak_idx = np.argmax(speaker_decoding['accuracies'])
        peak_time = speaker_decoding['timepoints'][peak_idx]
        
        # Find closest RSA timepoint
        rsa_idx = np.argmin(np.abs(np.array(speaker_rsa['timepoints']) - peak_time))
        
        if rsa_idx < len(speaker_rsa['neural_rdms']) and speaker_rsa['neural_rdms'][rsa_idx] is not None:
            # Neural RDM
            im1 = axes[0, 0].imshow(speaker_rsa['neural_rdms'][rsa_idx], cmap='hot')
            axes[0, 0].set_title(f'Neural RDM at {peak_time:.2f}s')
            axes[0, 0].set_xticks([0, 1, 2, 3])
            axes[0, 0].set_yticks([0, 1, 2, 3])
            axes[0, 0].set_xticklabels(['Sp1', 'Sp2', 'Sp3', 'Sp4'])
            axes[0, 0].set_yticklabels(['Sp1', 'Sp2', 'Sp3', 'Sp4'])
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Model RDMs
            rsa_analyzer = RSAAnalyzer()
            model_rdms = rsa_analyzer.create_model_rdms()
            
            # Harmonic model
            if 'speaker_harmonic' in model_rdms:
                im2 = axes[0, 1].imshow(model_rdms['speaker_harmonic'], cmap='hot')
                axes[0, 1].set_title('Harmonic Model')
                axes[0, 1].set_xticks([0, 1, 2, 3])
                axes[0, 1].set_yticks([0, 1, 2, 3])
                axes[0, 1].set_xticklabels(['Sp1', 'Sp2', 'Sp3', 'Sp4'])
                axes[0, 1].set_yticklabels(['Sp1', 'Sp2', 'Sp3', 'Sp4'])
                plt.colorbar(im2, ax=axes[0, 1])
            
            # Categorical model
            if 'speaker_categorical' in model_rdms:
                im3 = axes[0, 2].imshow(model_rdms['speaker_categorical'], cmap='hot')
                axes[0, 2].set_title('Categorical Model')
                axes[0, 2].set_xticks([0, 1, 2, 3])
                axes[0, 2].set_yticks([0, 1, 2, 3])
                axes[0, 2].set_xticklabels(['Sp1', 'Sp2', 'Sp3', 'Sp4'])
                axes[0, 2].set_yticklabels(['Sp1', 'Sp2', 'Sp3', 'Sp4'])
                plt.colorbar(im3, ax=axes[0, 2])
    
    # Location RDMs at peak
    if location_decoding is not None and location_rsa is not None:
        peak_idx = np.argmax(location_decoding['accuracies'])
        peak_time = location_decoding['timepoints'][peak_idx]
        
        # Find closest RSA timepoint
        rsa_idx = np.argmin(np.abs(np.array(location_rsa['timepoints']) - peak_time))
        
        if rsa_idx < len(location_rsa['neural_rdms']) and location_rsa['neural_rdms'][rsa_idx] is not None:
            # Neural RDM
            im4 = axes[1, 0].imshow(location_rsa['neural_rdms'][rsa_idx], cmap='hot')
            axes[1, 0].set_title(f'Neural RDM at {peak_time:.2f}s')
            axes[1, 0].set_xticks([0, 1, 2, 3])
            axes[1, 0].set_yticks([0, 1, 2, 3])
            axes[1, 0].set_xticklabels(['L1', 'L2', 'L3', 'L4'])
            axes[1, 0].set_yticklabels(['L1', 'L2', 'L3', 'L4'])
            plt.colorbar(im4, ax=axes[1, 0])
            
            # Model RDMs
            rsa_analyzer = RSAAnalyzer()
            model_rdms = rsa_analyzer.create_model_rdms()
            
            # ITD Magnitude model
            if 'location_magnitude' in model_rdms:
                im5 = axes[1, 1].imshow(model_rdms['location_magnitude'], cmap='hot')
                axes[1, 1].set_title('ITD Magnitude Model')
                axes[1, 1].set_xticks([0, 1, 2, 3])
                axes[1, 1].set_yticks([0, 1, 2, 3])
                axes[1, 1].set_xticklabels(['L1', 'L2', 'L3', 'L4'])
                axes[1, 1].set_yticklabels(['L1', 'L2', 'L3', 'L4'])
                plt.colorbar(im5, ax=axes[1, 1])
            
            # Hemispheric model
            if 'location_hemispheric' in model_rdms:
                im6 = axes[1, 2].imshow(model_rdms['location_hemispheric'], cmap='hot')
                axes[1, 2].set_title('Hemispheric Model')
                axes[1, 2].set_xticks([0, 1, 2, 3])
                axes[1, 2].set_yticks([0, 1, 2, 3])
                axes[1, 2].set_xticklabels(['L1', 'L2', 'L3', 'L4'])
                axes[1, 2].set_yticklabels(['L1', 'L2', 'L3', 'L4'])
                plt.colorbar(im6, ax=axes[1, 2])
    
    plt.suptitle(f'Subject {args.subject} - RDMs at Peak Decoding Times', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/rdms_at_peak.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    write_log(f"\nRDM visualization saved to {OUTPUT_DIR}/rdms_at_peak.png")

def main():
    """Main processing function"""
    # Load metadata
    meta_info = pd.read_excel(META_FILE)
    
    # Load subject data
    epochs_data, events, sfreq = load_subject_data(args.subject, meta_info)
    
    if epochs_data is None:
        write_log("Failed to load data. Exiting.")
        sys.exit(1)
    
    write_log("\n=== MAHALANOBIS DISTANCE DECODING & RSA ANALYSIS ===")
    
    # 1. Run Mahalanobis distance decoding
    write_log("\n--- Part 1: Mahalanobis Distance Decoding (4-way) ---")
    
    speaker_decoding = mahalanobis_sliding_window(
        epochs_data, events, sfreq, feature='speaker',
        n_repetitions=args.n_iterations, n_folds=args.n_folds
    )
    
    location_decoding = mahalanobis_sliding_window(
        epochs_data, events, sfreq, feature='location',
        n_repetitions=args.n_iterations, n_folds=args.n_folds
    )
    
    # 2. Run RSA
    write_log("\n--- Part 2: Representational Similarity Analysis ---")
    
    speaker_rsa = rsa_sliding_window(
        epochs_data, events, sfreq, feature='speaker',
        n_iterations=args.n_iterations
    )
    
    location_rsa = rsa_sliding_window(
        epochs_data, events, sfreq, feature='location',
        n_iterations=args.n_iterations
    )
    
    # 3. Save results
    write_log("\n--- Saving Results ---")
    
    # Save decoding results
    if speaker_decoding is not None:
        np.save(f'{OUTPUT_DIR}/speaker_decoding_accuracies.npy', speaker_decoding['accuracies'])
        np.save(f'{OUTPUT_DIR}/speaker_decoding_stds.npy', speaker_decoding['accuracy_stds'])
        np.save(f'{OUTPUT_DIR}/speaker_decoding_timepoints.npy', speaker_decoding['timepoints'])
        if speaker_decoding['confusion_matrices'][0] is not None:
            np.save(f'{OUTPUT_DIR}/speaker_confusion_matrices.npy', speaker_decoding['confusion_matrices'])
    
    if location_decoding is not None:
        np.save(f'{OUTPUT_DIR}/location_decoding_accuracies.npy', location_decoding['accuracies'])
        np.save(f'{OUTPUT_DIR}/location_decoding_stds.npy', location_decoding['accuracy_stds'])
        np.save(f'{OUTPUT_DIR}/location_decoding_timepoints.npy', location_decoding['timepoints'])
        if location_decoding['confusion_matrices'][0] is not None:
            np.save(f'{OUTPUT_DIR}/location_confusion_matrices.npy', location_decoding['confusion_matrices'])
    
    # Save RSA results
    if speaker_rsa is not None:
        np.save(f'{OUTPUT_DIR}/speaker_rsa_results.npy', speaker_rsa)
    
    if location_rsa is not None:
        np.save(f'{OUTPUT_DIR}/location_rsa_results.npy', location_rsa)
    
    # 4. Create visualizations
    write_log("\n--- Creating Visualizations ---")
    
    visualize_results(speaker_decoding, location_decoding, speaker_rsa, location_rsa)
    visualize_rdms_at_peak(speaker_decoding, location_decoding, speaker_rsa, location_rsa)
    
    # 5. Create summary
    summary = {
        'subject': args.subject,
        'n_iterations': args.n_iterations,
        'n_folds': args.n_folds,
        'analysis_window': f"{args.tmin}-{args.tmax}s",
        'window_length_ms': WINDOW_LENGTH_SEC * 1000,
        'window_step_ms': WINDOW_STEP_SEC * 1000,
        'method': 'Mahalanobis distance decoding + RSA',
        'processing_time': str(datetime.now()),
        'results': {}
    }
    
    # Add speaker results
    if speaker_decoding is not None:
        peak_idx = np.argmax(speaker_decoding['accuracies'])
        ping_idx = np.argmin(np.abs(speaker_decoding['timepoints'] - PING_TIME))
        
        summary['results']['speaker_decoding'] = {
            'mean_accuracy': float(np.mean(speaker_decoding['accuracies'])),
            'peak_accuracy': float(speaker_decoding['accuracies'][peak_idx]),
            'peak_time': float(speaker_decoding['timepoints'][peak_idx]),
            'ping_accuracy': float(speaker_decoding['accuracies'][ping_idx]),
            'n_classes': speaker_decoding['n_classes']
        }
    
    # Add location results
    if location_decoding is not None:
        peak_idx = np.argmax(location_decoding['accuracies'])
        ping_idx = np.argmin(np.abs(location_decoding['timepoints'] - PING_TIME))
        
        summary['results']['location_decoding'] = {
            'mean_accuracy': float(np.mean(location_decoding['accuracies'])),
            'peak_accuracy': float(location_decoding['accuracies'][peak_idx]),
            'peak_time': float(location_decoding['timepoints'][peak_idx]),
            'ping_accuracy': float(location_decoding['accuracies'][ping_idx]),
            'n_classes': location_decoding['n_classes']
        }
    
    # Add RSA summary
    if speaker_rsa is not None and 'model_fits' in speaker_rsa:
        # Find time of maximum model fit for each model
        summary['results']['speaker_rsa'] = {}
        for model_name, fits in speaker_rsa['model_fits'].items():
            betas = [f['beta'] for f in fits]
            correlations = [f['correlation'] for f in fits]
            max_idx = np.argmax(np.abs(betas))
            summary['results']['speaker_rsa'][model_name] = {
                'max_beta': float(betas[max_idx]),
                'max_beta_time': float(speaker_rsa['timepoints'][max_idx]),
                'max_correlation': float(np.max(correlations))
            }
    
    if location_rsa is not None and 'model_fits' in location_rsa:
        summary['results']['location_rsa'] = {}
        for model_name, fits in location_rsa['model_fits'].items():
            betas = [f['beta'] for f in fits]
            correlations = [f['correlation'] for f in fits]
            max_idx = np.argmax(np.abs(betas))
            summary['results']['location_rsa'][model_name] = {
                'max_beta': float(betas[max_idx]),
                'max_beta_time': float(location_rsa['timepoints'][max_idx]),
                'max_correlation': float(np.max(correlations))
            }
    
    # Save summary
    with open(f'{OUTPUT_DIR}/analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    write_log(f"\nProcessing completed at: {datetime.now()}")
    
    print(f"Subject {args.subject} Mahalanobis & RSA analysis completed successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()