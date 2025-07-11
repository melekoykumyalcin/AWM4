#!/usr/bin/env python
"""
Time-Frequency Power Decoding Analysis for Delay Period
Based on maintained information in working memory

This script performs decoding on time-frequency power data during the delay period:
1. Loads TFR data computed by time-frequency.py
2. Extracts maintained information based on retro-cues
3. Averages power within frequency bands
4. Performs SVM decoding on frequency band power during delay
5. Implements cluster-based permutation testing
"""

import numpy as np
import os
import mne
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import interpolate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from joblib import Parallel, delayed
import pickle
import logging

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC
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
RESULTS_DIR = PROCESSED_DIR + 'TFPowerDecodingSVMDelay/'
os.makedirs(RESULTS_DIR, exist_ok=True)
CORRECTED_DATA = HOME_DIR + '/AWM4_data/raw/correctTriggers/'
DATA_PATH = HOME_DIR + '/AWM4_data/raw/'

# Load metadata
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(META_FILE)
Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])

# Frequency bands (following Kaiser 2025)
FREQ_BANDS = {
    'theta': (4, 7),
    'alpha': (8, 12),
    'beta': (13, 30),
    'gamma': (31, 100)
}

# Delay period configuration
DELAY_CONFIG = {
    'tmin': 2.0,
    'tmax': 5.0
}

# Decoding parameters 
N_FOLDS = 10  # 10-fold cross-validation
N_PERMUTATIONS = 1000  # For cluster-based permutation test
CLUSTER_THRESHOLD = 2.086  # t-value for p < 0.05, df=29 (30 subjects)

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
    Extract which information is maintained in working memory based on retro-cues
    This function is adapted from the delay script
    """
    logging.info(f'Extracting maintained information for subject {subject}')
    
    try:
        # Get file information
        actInd = (metaInfo.Subject==subject) & (metaInfo.Valid==1)
        
        # Determine if subject is in early subjects (with corrected files)
        early_subject = subject in Subs[:7]
        
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
                raw = mne.io.read_raw_fif(fname, preload=False)
            else:
                fname = DATA_PATH + actFiles.iloc[ff]
                raw = mne.io.read_raw_ctf(fname, 'truncate', False)
                
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
        
        # Find all retro-cue events
        S1_cue_indices = np.where(all_events[:,2] == 101)[0]
        S2_cue_indices = np.where(all_events[:,2] == 201)[0]
        
        logging.info(f"Found {len(S1_cue_indices)} S1 retro-cues and {len(S2_cue_indices)} S2 retro-cues")
        
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
        
        logging.info(f"Maintained stimuli: {len(memorized_values)} total trials")
        
        return memorized_values, all_events
        
    except Exception as e:
        logging.error(f"Error extracting maintained information: {str(e)}")
        return None, None

def load_tfr_data(subject):
    """Load single-trial TFR data"""
    tfr_file = f"{TFR_DATA_DIR}/sub-{subject}_tfr-nobaselinecor.h5"
    
    if os.path.exists(tfr_file):
        tfr = mne.time_frequency.read_tfrs(tfr_file)[0]
        logging.info(f"TFR data shape: {tfr.data.shape}")
        return tfr
    else:
        logging.error(f"TFR file not found for subject {subject}")
        return None

def extract_band_power(tfr_data, freqs):
    """Extract power for each frequency band from single-trial TFR"""
    band_powers = {}
    
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        
        if not np.any(freq_mask):
            continue
        
        # Average power across frequencies in band
        band_power = np.mean(tfr_data[:, :, freq_mask, :], axis=2)
        band_powers[band_name] = band_power
    
    return band_powers

def process_subject_tf_decoding_delay(subject, feature):
    """Process TF decoding for delay period"""
    
    # Load single-trial TFR
    tfr = load_tfr_data(subject)
    if tfr is None:
        return None
    
    # Apply baseline correction
    tfr_baseline = tfr.copy()
    if subject in [13, 30]:
        tfr_baseline.apply_baseline((None, 0), mode='logratio')
    else:
        tfr_baseline.apply_baseline((-0.4, -0.25), mode='logratio')
    
    # Crop to DELAY period instead of S1/S2
    tfr_delay = tfr_baseline.copy().crop(tmin=DELAY_CONFIG['tmin'], tmax=DELAY_CONFIG['tmax'])
    
    # Extract maintained information
    memorized_values, all_events = extract_maintained_information(subject, metaInfo)
    
    if memorized_values is None:
        logging.error(f"Failed to extract maintained information for subject {subject}")
        return None
    
    # Get aligned events for this subject
    if subject == 28:
        events_file = f"{PROCESSED_DIR}/aligned_events_fixed/sub-{subject}_events.npy"
    else:
        events_file = f"{PROCESSED_DIR}/aligned_events_corrected/sub-{subject}_events.npy"
    
    if not os.path.exists(events_file):
        logging.error(f"Aligned events file not found: {events_file}")
        return None
    
    aligned_events = np.load(events_file)
    
    # Map aligned trials to maintained values
    # This assumes the same trial structure as in the original scripts
    if len(memorized_values) != len(aligned_events):
        logging.warning(f"Mismatch: {len(memorized_values)} memorized vs {len(aligned_events)} aligned")
        # Truncate to the shorter length
        min_len = min(len(memorized_values), len(aligned_events))
        memorized_values = memorized_values[:min_len]
        aligned_events = aligned_events[:min_len]
    
    # # Apply subject-specific corrections
    # if subject == 28 and len(memorized_values) > 63:
    #     memorized_values = np.delete(memorized_values, 63)
    #     logging.info(f"Applied subject 28 correction")
    # elif subject == 23:
    #     drop_idx = 64 * 7
    #     if len(memorized_values) > drop_idx:
    #         memorized_values = np.delete(memorized_values, drop_idx)
    #         logging.info(f"Applied subject 23 correction")
    
    # Verify alignment with TFR data
    if len(memorized_values) != tfr_delay.data.shape[0]:
        logging.error(f"Final mismatch: {len(memorized_values)} vs {tfr_delay.data.shape[0]}")
        return None
    
    # Use memorized values as events for delay period
    events = memorized_values
    
    # Create binary labels based on maintained information
    if feature == 'maintained_location':
        # Location decoding: L1+L2 vs L3+L4
        # Extract location from event code (last digit)
        low_group = []
        high_group = []
        for code in EVENT_DICT.values():
            if isinstance(code, int) and code > 100 and code < 300:
                location = code % 10
                if location in [1, 2]:
                    low_group.append(code)
                elif location in [3, 4]:
                    high_group.append(code)
    else:  # maintained_voice_identity
        # Voice decoding: Sp1+Sp2 vs Sp3+Sp4
        # Extract speaker from event code (second digit)
        low_group = []
        high_group = []
        for code in EVENT_DICT.values():
            if isinstance(code, int) and code > 100 and code < 300:
                speaker = (code // 10) % 10
                if speaker in [1, 2]:
                    low_group.append(code)
                elif speaker in [3, 4]:
                    high_group.append(code)
    
    # Create labels and keep only valid trials
    y = []
    valid_idx = []
    for i, event in enumerate(events):
        if event in low_group:
            y.append(0)
            valid_idx.append(i)
        elif event in high_group:
            y.append(1)
            valid_idx.append(i)
    
    y = np.array(y)
    
    # Check if we have enough trials
    if len(y) == 0:
        logging.error(f"No valid trials found for {feature} in subject {subject}")
        return None
    
    # Check class balance
    class_counts = np.bincount(y)
    if len(class_counts) < 2 or np.min(class_counts) < 5:
        logging.error(f"Insufficient trials for {feature} in subject {subject}")
        return None
    
    logging.info(f"Found {len(y)} valid trials: Class 0={class_counts[0]}, Class 1={class_counts[1]}")
    
    # Filter TFR data to valid trials only
    tfr_data = tfr_delay.data[valid_idx]
    
    # Extract band powers
    band_powers = extract_band_power(tfr_data, tfr_delay.freqs)
    
    # Decode each band with nested CV
    band_results = {}
    for band_name, band_data in band_powers.items():
        print(f"\nDecoding {band_name} band with nested CV...")
        
        try:
            auc_scores, accuracy_scores, times, best_params = decode_sliding_window_nested(
                band_data, y, tfr_delay.times, n_jobs=16
            )
            
            band_results[band_name] = {
                'auc': auc_scores,
                'accuracy': accuracy_scores,
                'times': times,
                'best_params': best_params
            }
            
        except Exception as e:
            logging.error(f"Error decoding {band_name} band: {str(e)}")
            return None
    
    # Calculate combined bands
    all_band_aucs = [band_results[band]['auc'] for band in band_results]
    all_band_accuracies = [band_results[band]['accuracy'] for band in band_results]
    
    combined_aucs = np.mean(all_band_aucs, axis=0)
    combined_accuracies = np.mean(all_band_accuracies, axis=0)
    
    # Return structured results
    result = {
        'subject': subject,
        'band_results': band_results,
        'combined_bands': {
            'auc': combined_aucs,
            'accuracy': combined_accuracies
        },
        'times': times + DELAY_CONFIG['tmin']  # Adjust times to actual delay period
    }
    
    # Save results
    save_individual_results_delay(result, feature)
    
    return result

def decode_sliding_window_nested(X, y, times, window_length_s=0.25, window_step_s=0.05, n_jobs=16):
    """
    Sliding window decoding with nested CV - same as original but adjusted for delay timing
    """
    n_trials, n_channels, n_times = X.shape
    
    # Calculate window parameters
    time_step = times[1] - times[0]
    window_samples = int(window_length_s / time_step)
    step_samples = max(1, int(window_step_s / time_step))
    
    # Prepare windows
    window_starts = range(0, n_times - window_samples + 1, step_samples)
    
    def process_window(start):
        """Process single window"""
        end = start + window_samples
        
        # Extract and average window
        X_window = np.mean(X[:, :, start:end], axis=2)
        
        # Decode with nested CV
        auc, accuracy, best_params = decode_window_nested_cv(X_window, y)
        
        window_time = times[start + window_samples // 2]
        
        return {
            'auc': auc,
            'accuracy': accuracy,
            'time': window_time,
            'best_params': best_params
        }
    
    # Parallel processing across windows
    print(f"Processing {len(window_starts)} windows with nested CV...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_window)(start) for start in tqdm(window_starts)
    )
    
    # Extract results
    auc_scores = np.array([r['auc'] for r in results])
    accuracy_scores = np.array([r['accuracy'] for r in results])
    window_times = np.array([r['time'] for r in results])
    all_best_params = [r['best_params'] for r in results]
    
    return auc_scores, accuracy_scores, window_times, all_best_params

def decode_window_nested_cv(X_window, y, param_grid=None):
    """Decode single time window with nested cross-validation - same as original"""
    if param_grid is None:
        param_grid = {
            'svc__C': [0.001, 0.01, 0.1, 1, 10, 100]
        }

    # Setup nested CV
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # SVM pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel='linear', probability=True)
    )

    outer_auc_scores = []
    outer_acc_scores = []
    best_params_list = []
    
    # Outer CV loop
    for train_idx, test_idx in outer_cv.split(X_window, y):
        X_train, X_test = X_window[train_idx], X_window[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner CV for hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=1
        )
        
        # Fit on training data
        grid_search.fit(X_train, y_train)
        
        # Store best params
        best_params_list.append(grid_search.best_params_)
        
        # Evaluate on test set
        y_prob = grid_search.predict_proba(X_test)[:, 1]
        y_pred = grid_search.predict(X_test)
        
        auc = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        
        outer_auc_scores.append(auc)
        outer_acc_scores.append(accuracy)
    
    # Get most common best parameters
    if best_params_list:
        param_name = list(param_grid.keys())[0]
        param_values = [p[param_name] for p in best_params_list]
        most_common = Counter(param_values).most_common(1)[0][0]
        best_params = {param_name: most_common}
    else:
        best_params = {}
    
    return np.mean(outer_auc_scores), np.mean(outer_acc_scores), best_params

def save_individual_results_delay(result, feature):
    """Save individual subject results for delay period"""
    save_dir = f"{RESULTS_DIR}/{feature}/individual_results/"
    os.makedirs(save_dir, exist_ok=True)
    
    subject = result['subject']
    
    # Save as pickle
    with open(f"{save_dir}/sub-{subject}_results.pkl", 'wb') as f:
        pickle.dump(result, f)
    
    # Save as numpy arrays
    np.save(f"{save_dir}/sub-{subject}_times.npy", result['times'])
    np.save(f"{save_dir}/sub-{subject}_combined_auc.npy", result['combined_bands']['auc'])
    np.save(f"{save_dir}/sub-{subject}_combined_accuracy.npy", result['combined_bands']['accuracy'])
    
    # Save each frequency band
    for band_name, band_data in result['band_results'].items():
        np.save(f"{save_dir}/sub-{subject}_{band_name}_auc.npy", band_data['auc'])
        np.save(f"{save_dir}/sub-{subject}_{band_name}_accuracy.npy", band_data['accuracy'])

def cluster_based_permutation_test(data1, data2=None, n_permutations=1000, threshold=2.086):
    """Cluster-based permutation test - same as original"""
    from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test
    
    if data2 is None:
        # Test against chance (0.5 for AUC)
        data_centered = data1 - 0.5
        t_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
            data_centered, 
            n_permutations=n_permutations,
            threshold=threshold,
            tail=1
        )
    else:
        # Compare two conditions
        t_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
            [data1, data2],
            n_permutations=n_permutations,
            threshold=threshold
        )
    
    return clusters, cluster_p_values

def process_subject_all_delay_features(subject):
    """Process all delay features for a single subject"""
    
    features = ['maintained_location', 'maintained_voice_identity']
    results_summary = {}
    
    for feature in features:
        logging.info(f"Processing subject {subject}: {feature}")
        
        try:
            result = process_subject_tf_decoding_delay(subject, feature)
            
            if result is not None:
                results_summary[feature] = {
                    'status': 'success',
                    'peak_auc': np.max(result['combined_bands']['auc']),
                    'peak_accuracy': np.max(result['combined_bands']['accuracy'])
                }
                logging.info(f"✅ {feature}: Peak AUC={np.max(result['combined_bands']['auc']):.3f}")
            else:
                results_summary[feature] = {'status': 'failed'}
                logging.error(f"❌ {feature}: Failed")
                
        except Exception as e:
            logging.error(f"❌ {feature}: Error - {str(e)}")
            results_summary[feature] = {'status': 'error', 'error': str(e)}
    
    # Save summary
    summary_dir = f"{RESULTS_DIR}/subject_summaries/"
    os.makedirs(summary_dir, exist_ok=True)
    
    with open(f"{summary_dir}/sub-{subject}_summary.json", 'w') as f:
        import json
        json.dump(results_summary, f, indent=2)
    
    return results_summary

def main():
    """Main function to handle delay period TF decoding"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TF Power Decoding Analysis - Delay Period')
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                       help='Subject IDs to process (default: all subjects)')
    parser.add_argument('--feature', type=str, default='maintained_location',
                       choices=['maintained_location', 'maintained_voice_identity'],
                       help='Feature to decode')
    parser.add_argument('--single', type=int, default=None,
                       help='Process single subject')
    parser.add_argument('--aggregate', action='store_true',
                       help='Aggregate individual results into group analysis')
    
    args = parser.parse_args()
    
    if args.single is not None:
        # Process single subject
        logging.info(f"Processing delay period for subject {args.single}")
        summary = process_subject_all_delay_features(args.single)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SUBJECT {args.single} DELAY PROCESSING COMPLETE")
        print(f"{'='*60}")
        
        for feature, result in summary.items():
            if result['status'] == 'success':
                print(f"✅ {feature}: AUC={result['peak_auc']:.3f}, Acc={result['peak_accuracy']:.3f}")
            else:
                print(f"❌ {feature}: {result['status']}")
                
    elif args.aggregate:
        # Aggregate results
        if args.subjects is None:
            subjects = Subs
        else:
            subjects = args.subjects
            
        # TODO: Implement aggregation function similar to aggregate_group_results
        logging.info("Aggregation not yet implemented for delay period")

if __name__ == "__main__":
    main()