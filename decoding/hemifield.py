#!/usr/bin/env python
"""
Hemifield-Specific Voice Identity Decoding Analysis
Includes:
1. Within-hemifield voice decoding (left and right separately)
2. Cross-hemifield generalization (train on one side, test on other)
3. Location invariance index calculation
4. Comprehensive visualization
"""

import numpy as np
import os
import mne
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import SVC
from joblib import Parallel, delayed
import pickle
import logging
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths
HOME_DIR = '/mnt/hpc/projects/awm4/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
TFR_DATA_DIR = PROCESSED_DIR + 'AllSensorsTFR/data/'
RESULTS_DIR = PROCESSED_DIR + 'HemifieldVoiceDecoding/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load metadata
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(META_FILE)
Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])

# Frequency bands
FREQ_BANDS = {
    'theta': (4, 7),
    'alpha': (8, 12),
    'beta': (13, 30),
    'gamma': (31, 100)
}

# Decoding parameters
N_FOLDS = 10
N_PERMUTATIONS = 1000
CLUSTER_THRESHOLD = 2.086  # t-value for p < 0.05, df=29

def load_raw_events_for_subject(subject):
    """Load raw events for a subject"""
    allFiles = metaInfo['MEG_Name']
    corrected_files = [f.split('.')[0] + '_correct_triggers.fif' for f in allFiles]
    corrected_files_series = pd.Series(corrected_files)
    
    actInd = (metaInfo.Subject==subject) & (metaInfo.Valid==1)
    if subject in Subs[:7]:
        actFiles = corrected_files_series[actInd]
    else:
        actFiles = allFiles[actInd]
    
    all_events = None
    
    for ff in range(actFiles.count()):
        if subject in Subs[:7]:
            fname = f"{PROCESSED_DIR}/../raw/correctTriggers/" + actFiles.iloc[ff]
            raw = mne.io.read_raw_fif(fname, preload=False)
        else:
            fname = f"{PROCESSED_DIR}/../raw/" + actFiles.iloc[ff]
            raw = mne.io.read_raw_ctf(fname, 'truncate', False)
            
        events = mne.find_events(raw, 'UPPT001', shortest_event=1)
        if ff != 0:
            events = events[events[:, 1] == 0, :]
            
        if ff == 0:
            all_events = events
        else:
            all_events = np.concatenate((all_events, events), axis=0)
        del raw
    
    return all_events

def extract_stimulus_events(all_events, stimulus_name):
    """Extract S1 or S2 condition codes"""
    if stimulus_name == 'S2':
        stim_idx = [i - 1 for i in range(len(all_events[:,2])) if all_events[i,2] == 200]
        stim_values = all_events[stim_idx,2]
    else:
        stim_idx = [i - 1 for i in range(len(all_events[:,2])) if all_events[i,2] == 100]
        stim_values = all_events[stim_idx,2]
    
    return stim_values

def map_aligned_events_to_s2(subject):
    """Map aligned events to S2 condition codes"""
    if subject == 28:
        events_file = f"{PROCESSED_DIR}/aligned_events_fixed/sub-{subject}_events.npy"
    else:
        events_file = f"{PROCESSED_DIR}/aligned_events_corrected/sub-{subject}_events.npy"
    
    if not os.path.exists(events_file):
        raise FileNotFoundError(f"Aligned events file not found: {events_file}")
    
    aligned_events = np.load(events_file)
    
    all_raw_events = load_raw_events_for_subject(subject)
    s1_codes_raw = extract_stimulus_events(all_raw_events, 'S1')
    s2_codes_raw = extract_stimulus_events(all_raw_events, 'S2')
    
    if len(s1_codes_raw) != len(s2_codes_raw):
        raise ValueError(f"S1/S2 code count mismatch: {len(s1_codes_raw)} vs {len(s2_codes_raw)}")
    
    kept_indices = []
    s1_codes_list = s1_codes_raw.tolist()
    
    for aligned_code in aligned_events:
        try:
            idx = s1_codes_list.index(aligned_code)
            kept_indices.append(idx)
            s1_codes_list[idx] = -1
        except ValueError:
            raise ValueError(f"Aligned code {aligned_code} not found in remaining S1 codes")
    
    s2_events_aligned = s2_codes_raw[kept_indices]
    
    logging.info(f"Mapped {len(aligned_events)} events: S1 [{np.min(aligned_events)}-{np.max(aligned_events)}], S2 [{np.min(s2_events_aligned)}-{np.max(s2_events_aligned)}]")
    
    return aligned_events, s2_events_aligned

def load_tfr_data(subject):
    """Load single-trial TFR data"""
    tfr_file = f"{TFR_DATA_DIR}/sub-{subject}_tfr-nobaselinecor.h5"
    
    if os.path.exists(tfr_file):
        tfr = mne.time_frequency.read_tfrs(tfr_file)[0]
        logging.info(f"Loaded TFR data: {tfr.data.shape}")
        return tfr
    else:
        logging.error(f"TFR file not found for subject {subject}")
        return None

def extract_band_power(tfr_data, freqs):
    """Extract power for each frequency band"""
    band_powers = {}
    
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        
        if not np.any(freq_mask):
            continue
        
        band_power = np.mean(tfr_data[:, :, freq_mask, :], axis=2)
        band_powers[band_name] = band_power
    
    return band_powers

def decode_window_nested_cv(X_window, y, param_grid=None):
    """Decode single time window with nested cross-validation"""
    if param_grid is None:
        param_grid = {
            'svc__C': [0.001, 0.01, 0.1, 1, 10, 100]
        }
    
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel='linear', probability=True)
    )
    
    outer_auc_scores = []
    outer_acc_scores = []
    best_params_list = []
    
    for train_idx, test_idx in outer_cv.split(X_window, y):
        X_train, X_test = X_window[train_idx], X_window[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=1
        )
        
        grid_search.fit(X_train, y_train)
        best_params_list.append(grid_search.best_params_)
        
        y_prob = grid_search.predict_proba(X_test)[:, 1]
        y_pred = grid_search.predict(X_test)
        
        auc = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        
        outer_auc_scores.append(auc)
        outer_acc_scores.append(accuracy)
    
    if best_params_list:
        param_name = list(param_grid.keys())[0]
        param_values = [p[param_name] for p in best_params_list]
        most_common = Counter(param_values).most_common(1)[0][0]
        best_params = {param_name: most_common}
    else:
        best_params = {}
    
    return np.mean(outer_auc_scores), np.mean(outer_acc_scores), best_params

def decode_sliding_window_nested(X, y, times, window_length_s=0.25, window_step_s=0.05, n_jobs=16):
    """Sliding window decoding with nested CV"""
    n_trials, n_channels, n_times = X.shape
    time_step = times[1] - times[0]
    window_samples = int(window_length_s / time_step)
    step_samples = max(1, int(window_step_s / time_step))
    
    window_starts = range(0, n_times - window_samples + 1, step_samples)
    
    def process_window(start):
        end = start + window_samples
        X_window = np.mean(X[:, :, start:end], axis=2)
        auc, accuracy, best_params = decode_window_nested_cv(X_window, y)
        window_time = times[start + window_samples // 2]
        
        return {
            'auc': auc,
            'accuracy': accuracy,
            'time': window_time,
            'best_params': best_params
        }
    
    print(f"Processing {len(window_starts)} windows with nested CV...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_window)(start) for start in tqdm(window_starts)
    )
    
    auc_scores = np.array([r['auc'] for r in results])
    accuracy_scores = np.array([r['accuracy'] for r in results])
    window_times = np.array([r['time'] for r in results])
    all_best_params = [r['best_params'] for r in results]
    
    return auc_scores, accuracy_scores, window_times, all_best_params

def decode_sliding_window_cross_hemifield(X_train, y_train, X_test, y_test, times, 
                                         window_length_s=0.25, window_step_s=0.05, n_jobs=16):
    """Cross-hemifield decoding: train on one hemifield, test on another"""
    n_times = X_train.shape[2]
    time_step = times[1] - times[0]
    window_samples = int(window_length_s / time_step)
    step_samples = max(1, int(window_step_s / time_step))
    
    window_starts = range(0, n_times - window_samples + 1, step_samples)
    
    def process_window(start):
        end = start + window_samples
        
        X_train_window = np.mean(X_train[:, :, start:end], axis=2)
        X_test_window = np.mean(X_test[:, :, start:end], axis=2)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_window)
        X_test_scaled = scaler.transform(X_test_window)
        
        clf = SVC(kernel='linear', probability=True, C=1.0)
        clf.fit(X_train_scaled, y_train)
        
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        y_pred = clf.predict(X_test_scaled)
        
        auc = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        
        window_time = times[start + window_samples // 2]
        
        return {
            'auc': auc,
            'accuracy': accuracy,
            'time': window_time
        }
    
    print(f"Processing {len(window_starts)} windows for cross-hemifield decoding...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_window)(start) for start in tqdm(window_starts)
    )
    
    auc_scores = np.array([r['auc'] for r in results])
    accuracy_scores = np.array([r['accuracy'] for r in results])
    window_times = np.array([r['time'] for r in results])
    
    return auc_scores, accuracy_scores, window_times

def calculate_location_invariance(results):
    """Calculate location invariance index"""
    times = results['within_left']['combined']['times']
    
    within_left_auc = results['within_left']['combined']['auc']
    within_right_auc = results['within_right']['combined']['auc']
    within_avg_auc = (within_left_auc + within_right_auc) / 2
    
    cross_lr_auc = results['cross_hemifield']['train_left_test_right']['combined']['auc']
    cross_rl_auc = results['cross_hemifield']['train_right_test_left']['combined']['auc']
    cross_avg_auc = (cross_lr_auc + cross_rl_auc) / 2
    
    invariance_index = cross_avg_auc / within_avg_auc
    invariance_index = np.minimum(invariance_index, 1.0)
    
    return {
        'times': times,
        'invariance_index': invariance_index,
        'within_avg_auc': within_avg_auc,
        'cross_avg_auc': cross_avg_auc,
        'within_left_auc': within_left_auc,
        'within_right_auc': within_right_auc,
        'cross_lr_auc': cross_lr_auc,
        'cross_rl_auc': cross_rl_auc
    }

def process_hemifield_decoding(subject, stimulus='S1'):
    """Main function for hemifield-specific voice decoding analysis"""
    
    # Load TFR data
    tfr = load_tfr_data(subject)
    if tfr is None:
        return None
    
    # Apply baseline
    tfr_baseline = tfr.copy()
    if subject in [13, 30]:
        tfr_baseline.apply_baseline((None, 0), mode='logratio')
    else:
        tfr_baseline.apply_baseline((-0.4, -0.25), mode='logratio')
    
    # Crop to stimulus period
    tmin, tmax = (0, 1) if stimulus == 'S1' else (1, 2)
    tfr_stim = tfr_baseline.copy().crop(tmin=tmin, tmax=tmax)
    
    # Get events
    s1_events, s2_events = map_aligned_events_to_s2(subject)
    events = s1_events if stimulus == 'S1' else s2_events
    
    # # Apply subject-specific corrections
    # original_length = len(events)
    # if subject == 28 and len(events) > 63:
    #     events = np.delete(events, 63)
    #     logging.info(f"Applied subject 28 correction: {original_length} -> {len(events)}")
    # elif subject == 23:
    #     drop_idx = 64 * 7
    #     if len(events) > drop_idx:
    #         events = np.delete(events, drop_idx)
    #         logging.info(f"Applied subject 23 correction: {original_length} -> {len(events)}")
    
    # Verify alignment
    if len(events) != tfr_stim.data.shape[0]:
        logging.error(f"Event/TFR mismatch: {len(events)} vs {tfr_stim.data.shape[0]}")
        return None
    
    # Define hemifield groups
    if stimulus == 'S1':
        left_hemifield_events = [111, 112, 121, 122, 131, 132, 141, 142]  # L1, L2
        right_hemifield_events = [113, 114, 123, 124, 133, 134, 143, 144]  # L3, L4
        
        left_voice1 = [111, 112, 121, 122]  # Speakers 1-2
        left_voice2 = [131, 132, 141, 142]  # Speakers 3-4
        
        right_voice1 = [113, 114, 123, 124]
        right_voice2 = [133, 134, 143, 144]
    else:  # S2
        left_hemifield_events = [211, 212, 221, 222, 231, 232, 241, 242]
        right_hemifield_events = [213, 214, 223, 224, 233, 234, 243, 244]
        
        left_voice1 = [211, 212, 221, 222]
        left_voice2 = [231, 232, 241, 242]
        
        right_voice1 = [213, 214, 223, 224]
        right_voice2 = [233, 234, 243, 244]
    
    # Collect data for both hemifields
    hemifield_data = {}
    
    for hemifield in ['left', 'right']:
        if hemifield == 'left':
            hemi_events = left_hemifield_events
            voice_group1 = left_voice1
            voice_group2 = left_voice2
        else:
            hemi_events = right_hemifield_events
            voice_group1 = right_voice1
            voice_group2 = right_voice2
        
        hemi_mask = np.isin(events, hemi_events)
        hemi_indices = np.where(hemi_mask)[0]
        
        if len(hemi_indices) < 20:
            logging.warning(f"Not enough trials for {hemifield} hemifield: {len(hemi_indices)}")
            continue
        
        hemi_data = tfr_stim.data[hemi_indices]
        hemi_events_subset = events[hemi_indices]
        
        y_voice = []
        valid_idx = []
        
        for i, event in enumerate(hemi_events_subset):
            if event in voice_group1:
                y_voice.append(0)
                valid_idx.append(i)
            elif event in voice_group2:
                y_voice.append(1)
                valid_idx.append(i)
        
        if len(y_voice) < 10:
            logging.warning(f"Not enough valid trials for {hemifield} hemifield")
            continue
        
        y_voice = np.array(y_voice)
        hemi_data_valid = hemi_data[valid_idx]
        
        class_counts = np.bincount(y_voice)
        logging.info(f"{hemifield} hemifield: Voice1={class_counts[0]}, Voice2={class_counts[1]} trials")
        
        band_powers = extract_band_power(hemi_data_valid, tfr_stim.freqs)
        
        hemifield_data[hemifield] = {
            'data': hemi_data_valid,
            'labels': y_voice,
            'band_powers': band_powers,
            'n_trials': len(y_voice)
        }
    
    if 'left' not in hemifield_data or 'right' not in hemifield_data:
        logging.error("Need both hemifields for complete analysis")
        return None
    
    results = {}
    
    # 1. WITHIN-HEMIFIELD DECODING
    for hemifield in ['left', 'right']:
        hemi_results = {}
        data = hemifield_data[hemifield]
        
        for band_name, band_data in data['band_powers'].items():
            print(f"\nWithin-hemifield decoding: {band_name} band for {hemifield} hemifield...")
            
            auc_scores, accuracy_scores, times, best_params = decode_sliding_window_nested(
                band_data, data['labels'], tfr_stim.times, n_jobs=16
            )
            
            hemi_results[band_name] = {
                'auc': auc_scores,
                'accuracy': accuracy_scores,
                'times': times,
                'best_params': best_params
            }
        
        all_aucs = [hemi_results[band]['auc'] for band in hemi_results]
        all_accs = [hemi_results[band]['accuracy'] for band in hemi_results]
        
        hemi_results['combined'] = {
            'auc': np.mean(all_aucs, axis=0),
            'accuracy': np.mean(all_accs, axis=0),
            'times': times
        }
        
        results[f'within_{hemifield}'] = hemi_results
    
    # 2. CROSS-HEMIFIELD GENERALIZATION
    cross_results = {}
    
    # Train on left, test on right
    print("\nCross-hemifield: Training on LEFT, testing on RIGHT...")
    cross_results['train_left_test_right'] = {}
    
    for band_name in hemifield_data['left']['band_powers'].keys():
        left_band_data = hemifield_data['left']['band_powers'][band_name]
        right_band_data = hemifield_data['right']['band_powers'][band_name]
        
        auc_scores, accuracy_scores, times = decode_sliding_window_cross_hemifield(
            left_band_data, hemifield_data['left']['labels'],
            right_band_data, hemifield_data['right']['labels'],
            tfr_stim.times, n_jobs=16
        )
        
        cross_results['train_left_test_right'][band_name] = {
            'auc': auc_scores,
            'accuracy': accuracy_scores,
            'times': times
        }
    
    all_cross_aucs = [cross_results['train_left_test_right'][band]['auc'] 
                      for band in cross_results['train_left_test_right']]
    all_cross_accs = [cross_results['train_left_test_right'][band]['accuracy'] 
                      for band in cross_results['train_left_test_right']]
    
    cross_results['train_left_test_right']['combined'] = {
        'auc': np.mean(all_cross_aucs, axis=0),
        'accuracy': np.mean(all_cross_accs, axis=0),
        'times': times
    }
    
    # Train on right, test on left
    print("\nCross-hemifield: Training on RIGHT, testing on LEFT...")
    cross_results['train_right_test_left'] = {}
    
    for band_name in hemifield_data['right']['band_powers'].keys():
        right_band_data = hemifield_data['right']['band_powers'][band_name]
        left_band_data = hemifield_data['left']['band_powers'][band_name]
        
        auc_scores, accuracy_scores, times = decode_sliding_window_cross_hemifield(
            right_band_data, hemifield_data['right']['labels'],
            left_band_data, hemifield_data['left']['labels'],
            tfr_stim.times, n_jobs=16
        )
        
        cross_results['train_right_test_left'][band_name] = {
            'auc': auc_scores,
            'accuracy': accuracy_scores,
            'times': times
        }
    
    all_cross_aucs = [cross_results['train_right_test_left'][band]['auc'] 
                      for band in cross_results['train_right_test_left']]
    all_cross_accs = [cross_results['train_right_test_left'][band]['accuracy'] 
                      for band in cross_results['train_right_test_left']]
    
    cross_results['train_right_test_left']['combined'] = {
        'auc': np.mean(all_cross_aucs, axis=0),
        'accuracy': np.mean(all_cross_accs, axis=0),
        'times': times
    }
    
    results['cross_hemifield'] = cross_results
    
    # 3. CALCULATE LOCATION INVARIANCE
    results['location_invariance'] = calculate_location_invariance(results)

    # Store hemifield data info in results dictionary
    results['hemifield_data_info'] = {
        'left_n_trials': hemifield_data['left']['n_trials'],
        'right_n_trials': hemifield_data['right']['n_trials']
    }

    # Save results
    save_dir = f"{RESULTS_DIR}/{stimulus}/individual_results/"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/sub-{subject}_hemifield_results.pkl", 'wb') as f:
        pickle.dump({
            'subject': subject,
            'stimulus': stimulus,
            'within_left': results.get('within_left', None),
            'within_right': results.get('within_right', None),
            'cross_hemifield': results.get('cross_hemifield', None),
            'location_invariance': results.get('location_invariance', None),
            'hemifield_data_info': results['hemifield_data_info']
        }, f)

    # Create visualizations
    plot_hemifield_results(results, subject, stimulus)

    return results
def plot_hemifield_results(results, subject, stimulus):
    """Create comprehensive plots for hemifield analysis"""
    
    save_dir = f"{RESULTS_DIR}/{stimulus}/individual_plots/"
    os.makedirs(save_dir, exist_ok=True)
    
    times = results['within_left']['combined']['times']
    
    # Plot 1: Within vs Cross-hemifield comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # AUC comparison
    ax1.plot(times, results['within_left']['combined']['auc'], 'b-', 
             label='Within Left', linewidth=2)
    ax1.plot(times, results['within_right']['combined']['auc'], 'r-', 
             label='Within Right', linewidth=2)
    ax1.plot(times, results['cross_hemifield']['train_left_test_right']['combined']['auc'], 
             'b--', label='Train Left → Test Right', linewidth=2, alpha=0.7)
    ax1.plot(times, results['cross_hemifield']['train_right_test_left']['combined']['auc'], 
             'r--', label='Train Right → Test Left', linewidth=2, alpha=0.7)
    
    ax1.axhline(0.5, color='k', linestyle=':', alpha=0.5)
    ax1.set_ylabel('Voice Decoding AUC')
    ax1.set_title(f'Subject {subject} - Within vs Cross-Hemifield Voice Decoding ({stimulus})')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.35, 0.75)
    
    # Location invariance index
    invariance = results['location_invariance']['invariance_index']
    ax2.plot(times, invariance, 'purple', linewidth=2)
    ax2.fill_between(times, invariance, 0, alpha=0.3, color='purple')
    ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Perfect invariance')
    ax2.axhline(0.5, color='k', linestyle=':', alpha=0.5, label='Chance ratio')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Location Invariance Index')
    ax2.set_title('Voice Representation Location Invariance (Cross/Within Ratio)')
    ax2.set_ylim(0, 1.2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sub-{subject}_within_vs_cross_hemifield.png", dpi=300)
    plt.close()
    
    # Plot 2: Generalization matrix
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top left: Train Left, Test Left (within)
    axes[0,0].plot(times, results['within_left']['combined']['auc'], 'b-', linewidth=2)
    axes[0,0].fill_between(times, 0.5, results['within_left']['combined']['auc'], 
                           alpha=0.3, color='blue')
    axes[0,0].axhline(0.5, color='k', linestyle='--', alpha=0.5)
    axes[0,0].set_title('Train Left, Test Left (Within)')
    axes[0,0].set_ylabel('AUC')
    axes[0,0].set_ylim(0.35, 0.75)
    axes[0,0].grid(True, alpha=0.3)
    
    # Top right: Train Left, Test Right (cross)
    axes[0,1].plot(times, results['cross_hemifield']['train_left_test_right']['combined']['auc'], 
                   'purple', linewidth=2)
    axes[0,1].fill_between(times, 0.5, results['cross_hemifield']['train_left_test_right']['combined']['auc'], 
                           alpha=0.3, color='purple')
    axes[0,1].axhline(0.5, color='k', linestyle='--', alpha=0.5)
    axes[0,1].set_title('Train Left, Test Right (Cross)')
    axes[0,1].set_ylim(0.35, 0.75)
    axes[0,1].grid(True, alpha=0.3)
    
    # Bottom left: Train Right, Test Left (cross)
    axes[1,0].plot(times, results['cross_hemifield']['train_right_test_left']['combined']['auc'], 
                   'orange', linewidth=2)
    axes[1,0].fill_between(times, 0.5, results['cross_hemifield']['train_right_test_left']['combined']['auc'], 
                           alpha=0.3, color='orange')
    axes[1,0].axhline(0.5, color='k', linestyle='--', alpha=0.5)
    axes[1,0].set_title('Train Right, Test Left (Cross)')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('AUC')
    axes[1,0].set_ylim(0.35, 0.75)
    axes[1,0].grid(True, alpha=0.3)
    
    # Bottom right: Train Right, Test Right (within)
    axes[1,1].plot(times, results['within_right']['combined']['auc'], 'r-', linewidth=2)
    axes[1,1].fill_between(times, 0.5, results['within_right']['combined']['auc'], 
                           alpha=0.3, color='red')
    axes[1,1].axhline(0.5, color='k', linestyle='--', alpha=0.5)
    axes[1,1].set_title('Train Right, Test Right (Within)')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylim(0.35, 0.75)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Subject {subject} - Cross-Hemifield Generalization Matrix ({stimulus})')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sub-{subject}_generalization_matrix.png", dpi=300)
    plt.close()
    
    # Plot 3: Summary statistics
    create_summary_report(results, subject, stimulus, save_dir)

def create_summary_report(results, subject, stimulus, save_dir):
    """Create text summary of key findings"""
    
    with open(f"{save_dir}/sub-{subject}_summary.txt", 'w') as f:
        f.write(f"Hemifield Voice Decoding Summary\n")
        f.write(f"Subject: {subject}, Stimulus: {stimulus}\n")
        f.write("="*60 + "\n\n")
        
        # Peak within-hemifield performance
        times = results['within_left']['combined']['times']
        left_auc = results['within_left']['combined']['auc']
        right_auc = results['within_right']['combined']['auc']
        
        left_peak_idx = np.argmax(left_auc)
        right_peak_idx = np.argmax(right_auc)
        
        f.write("WITHIN-HEMIFIELD PERFORMANCE:\n")
        f.write(f"Left:  Peak AUC = {left_auc[left_peak_idx]:.3f} at {times[left_peak_idx]:.2f}s\n")
        f.write(f"Right: Peak AUC = {right_auc[right_peak_idx]:.3f} at {times[right_peak_idx]:.2f}s\n")
        f.write(f"Trials: Left={results['hemifield_data_info']['left_n_trials']}, "
                f"Right={results['hemifield_data_info']['right_n_trials']}\n\n")
        
        # Cross-hemifield performance
        cross_lr = results['cross_hemifield']['train_left_test_right']['combined']['auc']
        cross_rl = results['cross_hemifield']['train_right_test_left']['combined']['auc']
        
        cross_lr_peak_idx = np.argmax(cross_lr)
        cross_rl_peak_idx = np.argmax(cross_rl)
        
        f.write("CROSS-HEMIFIELD GENERALIZATION:\n")
        f.write(f"Train L→Test R: Peak AUC = {cross_lr[cross_lr_peak_idx]:.3f} at {times[cross_lr_peak_idx]:.2f}s\n")
        f.write(f"Train R→Test L: Peak AUC = {cross_rl[cross_rl_peak_idx]:.3f} at {times[cross_rl_peak_idx]:.2f}s\n\n")
        
        # Location invariance
        invariance = results['location_invariance']['invariance_index']
        peak_invariance_idx = np.argmax(invariance)
        
        f.write("LOCATION INVARIANCE:\n")
        f.write(f"Peak invariance = {invariance[peak_invariance_idx]:.3f} at {times[peak_invariance_idx]:.2f}s\n")
        f.write(f"Mean invariance = {np.mean(invariance):.3f}\n\n")
        
        # Interpretation
        f.write("INTERPRETATION:\n")
        if invariance[peak_invariance_idx] > 0.8:
            f.write("High location invariance: Voice identity generalizes well across hemifields\n")
        elif invariance[peak_invariance_idx] > 0.6:
            f.write("Moderate location invariance: Partial generalization across hemifields\n")
        else:
            f.write("Low location invariance: Voice identity is largely location-specific\n")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hemifield Voice Decoding Analysis')
    parser.add_argument('--subject', type=int, required=True,
                       help='Subject ID to process')
    parser.add_argument('--stimulus', type=str, default='S1',
                       choices=['S1', 'S2'],
                       help='Stimulus period to analyze')
    
    args = parser.parse_args()
    
    logging.info(f"Processing hemifield analysis for subject {args.subject}, stimulus {args.stimulus}")
    
    try:
        results = process_hemifield_decoding(args.subject, args.stimulus)
        
        if results is not None:
            logging.info(f"✅ Hemifield analysis complete for subject {args.subject}")
            
            # Print summary
            invariance = results['location_invariance']['invariance_index']
            peak_invariance = np.max(invariance)
            
            print(f"\nSUMMARY:")
            print(f"Peak location invariance: {peak_invariance:.3f}")
            print(f"Mean location invariance: {np.mean(invariance):.3f}")
            
        else:
            logging.error(f"❌ Failed to process subject {args.subject}")
            
    except Exception as e:
        logging.error(f"Error processing subject {args.subject}: {str(e)}")
        raise

if __name__ == "__main__":
    main()