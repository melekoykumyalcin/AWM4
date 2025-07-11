#!/usr/bin/env python
"""
Time-Frequency Power Decoding Analysis
Based on Kaiser et al. 2025 (Journal of Neurophysiology)

This script performs decoding on time-frequency power data:
1. Loads TFR data computed by time-frequency.py
2. Averages power within frequency bands
3. 200ms resolution
4. Performs SVM decoding on frequency band power
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
RESULTS_DIR = PROCESSED_DIR + 'TFPowerDecodingSVMhemifield/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load metadata
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(META_FILE)
Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])

# Frequency bands (following Kaiser 2025)
FREQ_BANDS = {
    'theta': (4, 7),
    'alpha': (8, 12),
    'beta': (13, 30),
    'gamma': (31, 100)  # Note: paper uses 31-70 Hz
}

# Decoding parameters 
N_FOLDS = 10  # 10-fold cross-validation
N_PERMUTATIONS = 1000  # For cluster-based permutation test
CLUSTER_THRESHOLD = 2.086  # t-value for p < 0.05, df=29 (30 subjects)

# Event dictionary (from your allEncoding.py)
EVENT_DICT = {
    'S1/Sp1/L1': 111, 'S1/Sp1/L2': 112, 'S1/Sp1/L3': 113, 'S1/Sp1/L4': 114,
    'S1/Sp2/L1': 121, 'S1/Sp2/L2': 122, 'S1/Sp2/L3': 123, 'S1/Sp2/L4': 124,
    'S1/Sp3/L1': 131, 'S1/Sp3/L2': 132, 'S1/Sp3/L3': 133, 'S1/Sp3/L4': 134,
    'S1/Sp4/L1': 141, 'S1/Sp4/L2': 142, 'S1/Sp4/L3': 143, 'S1/Sp4/L4': 144,
    'S2/Sp1/L1': 211, 'S2/Sp1/L2': 212, 'S2/Sp1/L3': 213, 'S2/Sp1/L4': 214,
    'S2/Sp2/L1': 221, 'S2/Sp2/L2': 222, 'S2/Sp2/L3': 223, 'S2/Sp2/L4': 224,
    'S2/Sp3/L1': 231, 'S2/Sp3/L2': 232, 'S2/Sp3/L3': 233, 'S2/Sp3/L4': 234,
    'S2/Sp4/L1': 241, 'S2/Sp4/L2': 242, 'S2/Sp4/L3': 243, 'S2/Sp4/L4': 244
}

def load_raw_events_for_subject(subject):
    """
    Load raw events for a subject to extract S2 condition codes
    Based on the method from allEncoding.py
    """
    
    # Load metadata
    metaInfo = pd.read_excel(META_FILE)
    Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
    
    # Prepare corrected files list for early subjects
    allFiles = metaInfo['MEG_Name']
    corrected_files = [f.split('.')[0] + '_correct_triggers.fif' for f in allFiles]
    corrected_files_series = pd.Series(corrected_files)
    
    # Get files for this subject
    actInd = (metaInfo.Subject==subject) & (metaInfo.Valid==1)
    if subject in Subs[:7]:
        actFiles = corrected_files_series[actInd]
    else:
        actFiles = allFiles[actInd]
    
    # Load and concatenate events from raw files
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
    """
    Extract S1 or S2 condition codes from raw events
    Exactly following the allEncoding.py method
    """
    
    if stimulus_name == 'S2':
        # S2: Find trigger 200, get event code from BEFORE it
        stim_idx = [i - 1 for i in range(len(all_events[:,2])) if all_events[i,2] == 200]
        stim_values = all_events[stim_idx,2]
    else:
        # S1: Find trigger 100, get event code from BEFORE it  
        stim_idx = [i - 1 for i in range(len(all_events[:,2])) if all_events[i,2] == 100]
        stim_values = all_events[stim_idx,2]
    
    return stim_values

def load_tfr_data(subject):
    """Load single-trial TFR data"""
    # Update filename to your single-trial TFR
    tfr_file = f"{TFR_DATA_DIR}/sub-{subject}_tfr-nobaselinecor.h5"
    
    if os.path.exists(tfr_file):
        # This loads an EpochsTFR object (not AverageTFR)
        tfr = mne.time_frequency.read_tfrs(tfr_file)[0]
        print(f"TFR data shape: {tfr.data.shape}")  # Should be 4D: [trials, channels, freqs, times]
        
        return tfr
    else:
        logging.error(f"TFR file not found for subject {subject}")
        return None

def extract_band_power(tfr_data, freqs):
    """
    Extract power for each frequency band from single-trial TFR
    
    Parameters:
    -----------
    tfr_data : array [n_trials, n_channels, n_frequencies, n_timepoints]
    freqs : array of frequencies
    
    Returns:
    --------
    band_powers : dict with arrays [n_trials, n_channels, n_timepoints] for each band
    """
    band_powers = {}
    
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        # Find frequencies in this band
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        
        if not np.any(freq_mask):
            continue
        
        # Average power across frequencies in band
        # From: [n_trials, n_channels, n_frequencies, n_timepoints]
        # To:   [n_trials, n_channels, n_timepoints]
        band_power = np.mean(tfr_data[:, :, freq_mask, :], axis=2)
        band_powers[band_name] = band_power
    
    return band_powers

def process_subject_all_combinations(subject):
    """
    Process ALL feature/stimulus combinations for a single subject
    NOW ALL 4 COMBINATIONS SHOULD WORK
    """
    
    # All 4 combinations
    combinations = [
        ('location', 'S1'),
        ('location', 'S2'),         # ✅ Now working!
        ('voice_identity', 'S1'),
        ('voice_identity', 'S2')    # ✅ Now working!
    ]
    
    results_summary = {}
    
    for feature, stimulus in combinations:
        logging.info(f"Processing subject {subject}: {feature}_{stimulus}")
        
        try:
            result = process_subject_tf_decoding_nested_single(subject, feature, stimulus)
            
            if result is not None:
                results_summary[f"{feature}_{stimulus}"] = {
                    'status': 'success',
                    'peak_auc': np.max(result['combined_bands']['auc']),
                    'peak_accuracy': np.max(result['combined_bands']['accuracy'])
                }
                logging.info(f"✅ {feature}_{stimulus}: Peak AUC={np.max(result['combined_bands']['auc']):.3f}, Peak Acc={np.max(result['combined_bands']['accuracy']):.3f}")
            else:
                results_summary[f"{feature}_{stimulus}"] = {'status': 'failed'}
                logging.error(f"❌ {feature}_{stimulus}: Failed")
                
        except Exception as e:
            logging.error(f"❌ {feature}_{stimulus}: Error - {str(e)}")
            results_summary[f"{feature}_{stimulus}"] = {'status': 'error', 'error': str(e)}
    
    # Process hemifield-specific voice decoding for both S1 and S2
    for stimulus in ['S1', 'S2']:
        logging.info(f"Processing hemifield-specific voice decoding for {stimulus}")
        
        try:
            hemi_results = process_hemifield_specific_decoding(subject, stimulus=stimulus)
            
            if hemi_results is not None and 'left' in hemi_results and 'right' in hemi_results:
                # Calculate peak performance for each hemifield
                left_peak_auc = np.max(hemi_results['left']['combined']['auc'])
                right_peak_auc = np.max(hemi_results['right']['combined']['auc'])
                left_peak_acc = np.max(hemi_results['left']['combined']['accuracy'])
                right_peak_acc = np.max(hemi_results['right']['combined']['accuracy'])
                
                results_summary[f"hemifield_voice_{stimulus}"] = {
                    'status': 'success',
                    'left_peak_auc': left_peak_auc,
                    'right_peak_auc': right_peak_auc,
                    'left_peak_accuracy': left_peak_acc,
                    'right_peak_accuracy': right_peak_acc,
                    'hemifield_difference_auc': right_peak_auc - left_peak_auc,
                    'hemifield_difference_acc': right_peak_acc - left_peak_acc
                }
                
                logging.info(f"✅ Hemifield voice {stimulus}: Left AUC={left_peak_auc:.3f}, Right AUC={right_peak_auc:.3f}")
                
                # Create comparison plots
                plot_hemifield_comparison(hemi_results, subject, stimulus)
                
            else:
                results_summary[f"hemifield_voice_{stimulus}"] = {'status': 'failed'}
                logging.error(f"❌ Hemifield voice {stimulus}: Failed - insufficient data")
                
        except Exception as e:
            logging.error(f"❌ Hemifield voice {stimulus}: Error - {str(e)}")
            results_summary[f"hemifield_voice_{stimulus}"] = {'status': 'error', 'error': str(e)}

    # Save summary for this subject
    summary_dir = f"{RESULTS_DIR}/subject_summaries/"
    os.makedirs(summary_dir, exist_ok=True)
    
    with open(f"{summary_dir}/sub-{subject}_summary.json", 'w') as f:
        import json
        json.dump(results_summary, f, indent=2)
    
    return results_summary

def plot_group_hemifield_results(group_results, save_dir, stimulus):
    """Create group-level plots for hemifield-specific voice decoding"""
    
    times = group_results['times']
    left_aucs = group_results['left_aucs']
    right_aucs = group_results['right_aucs']
    left_accs = group_results['left_accuracies']
    right_accs = group_results['right_accuracies']
    n_subjects = len(group_results['subjects'])
    
    # Calculate means and SEMs
    mean_left_auc = np.mean(left_aucs, axis=0)
    sem_left_auc = np.std(left_aucs, axis=0) / np.sqrt(n_subjects)
    mean_right_auc = np.mean(right_aucs, axis=0)
    sem_right_auc = np.std(right_aucs, axis=0) / np.sqrt(n_subjects)
    
    mean_left_acc = np.mean(left_accs, axis=0)
    sem_left_acc = np.std(left_accs, axis=0) / np.sqrt(n_subjects)
    mean_right_acc = np.mean(right_accs, axis=0)
    sem_right_acc = np.std(right_accs, axis=0) / np.sqrt(n_subjects)
    
    # ===== PLOT 1: Group comparison of hemifields =====
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # AUC comparison
    ax1.plot(times, mean_left_auc, 'b-', label='Left Hemifield', linewidth=2)
    ax1.fill_between(times, mean_left_auc - sem_left_auc, 
                     mean_left_auc + sem_left_auc, alpha=0.3, color='blue')
    
    ax1.plot(times, mean_right_auc, 'r-', label='Right Hemifield', linewidth=2)
    ax1.fill_between(times, mean_right_auc - sem_right_auc, 
                     mean_right_auc + sem_right_auc, alpha=0.3, color='red')
    
    # Add significance markers if available
    if 'statistical_tests' in group_results:
        # Mark where left is significantly above chance
        left_clusters, left_p = group_results['statistical_tests']['left_vs_chance_auc']
        for i, (cluster, p_val) in enumerate(zip(left_clusters, left_p)):
            if p_val < 0.05:
                cluster_times = times[cluster[0]]
                ax1.axhspan(0.51, 0.52, xmin=cluster_times[0], xmax=cluster_times[-1],
                           color='blue', alpha=0.2)
        
        # Mark where right is significantly above chance
        right_clusters, right_p = group_results['statistical_tests']['right_vs_chance_auc']
        for i, (cluster, p_val) in enumerate(zip(right_clusters, right_p)):
            if p_val < 0.05:
                cluster_times = times[cluster[0]]
                ax1.axhspan(0.52, 0.53, xmin=cluster_times[0], xmax=cluster_times[-1],
                           color='red', alpha=0.2)
    
    ax1.axhline(0.5, color='k', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Voice Decoding AUC')
    ax1.set_title(f'Group Voice Identity Decoding by Hemifield ({stimulus}, N={n_subjects})')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.4, 0.65)
    
    # Accuracy comparison
    ax2.plot(times, mean_left_acc, 'b-', label='Left Hemifield', linewidth=2)
    ax2.fill_between(times, mean_left_acc - sem_left_acc, 
                     mean_left_acc + sem_left_acc, alpha=0.3, color='blue')
    
    ax2.plot(times, mean_right_acc, 'r-', label='Right Hemifield', linewidth=2)
    ax2.fill_between(times, mean_right_acc - sem_right_acc, 
                     mean_right_acc + sem_right_acc, alpha=0.3, color='red')
    
    ax2.axhline(0.5, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Voice Decoding Accuracy')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.4, 0.65)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/group_hemifield_comparison.png", dpi=300)
    plt.close()
    
    # ===== PLOT 2: Difference plot (Right - Left) =====
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate differences
    mean_diff_auc = mean_right_auc - mean_left_auc
    mean_diff_acc = mean_right_acc - mean_left_acc
    
    # Propagate error for difference
    sem_diff_auc = np.sqrt(sem_left_auc**2 + sem_right_auc**2)
    sem_diff_acc = np.sqrt(sem_left_acc**2 + sem_right_acc**2)
    
    ax.plot(times, mean_diff_auc, 'purple', label='AUC Difference (R-L)', linewidth=2)
    ax.fill_between(times, mean_diff_auc - sem_diff_auc, 
                   mean_diff_auc + sem_diff_auc, alpha=0.3, color='purple')
    
    ax.plot(times, mean_diff_acc, 'orange', label='Accuracy Difference (R-L)', linewidth=2)
    ax.fill_between(times, mean_diff_acc - sem_diff_acc, 
                   mean_diff_acc + sem_diff_acc, alpha=0.3, color='orange')
    
    # Add significance for difference
    if 'statistical_tests' in group_results:
        diff_clusters, diff_p = group_results['statistical_tests']['right_vs_left_auc']
        for i, (cluster, p_val) in enumerate(zip(diff_clusters, diff_p)):
            if p_val < 0.05:
                cluster_times = times[cluster[0]]
                ax.axvspan(cluster_times[0], cluster_times[-1], 
                          alpha=0.1, color='gray', label='p<0.05' if i==0 else '')
    
    ax.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Performance Difference (Right - Left)')
    ax.set_title(f'Hemifield Advantage for Voice Decoding ({stimulus}, N={n_subjects})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 0.1)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/group_hemifield_difference.png", dpi=300)
    plt.close()
    
    # ===== PLOT 3: Individual subject scatter plot =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Find peak times
    peak_time_idx = np.argmax(mean_left_auc + mean_right_auc)  # Time of best overall performance
    
    # AUC scatter at peak time
    left_peak_aucs = left_aucs[:, peak_time_idx]
    right_peak_aucs = right_aucs[:, peak_time_idx]
    
    ax1.scatter(left_peak_aucs, right_peak_aucs, alpha=0.6, s=50)
    ax1.plot([0.4, 0.7], [0.4, 0.7], 'k--', alpha=0.5, label='Equal performance')
    
    # Add subject labels
    for i, subj in enumerate(group_results['subjects']):
        ax1.annotate(str(subj), (left_peak_aucs[i], right_peak_aucs[i]), 
                    fontsize=8, alpha=0.7)
    
    ax1.set_xlabel('Left Hemifield AUC')
    ax1.set_ylabel('Right Hemifield AUC')
    ax1.set_title(f'Peak Voice Decoding at {times[peak_time_idx]:.2f}s')
    ax1.set_xlim(0.4, 0.7)
    ax1.set_ylim(0.4, 0.7)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Bar plot of mean performance
    hemifields = ['Left', 'Right']
    mean_peak_aucs = [np.mean(left_peak_aucs), np.mean(right_peak_aucs)]
    sem_peak_aucs = [np.std(left_peak_aucs)/np.sqrt(n_subjects), 
                     np.std(right_peak_aucs)/np.sqrt(n_subjects)]
    
    bars = ax2.bar(hemifields, mean_peak_aucs, yerr=sem_peak_aucs, 
                   color=['blue', 'red'], alpha=0.7, capsize=10)
    ax2.axhline(0.5, color='k', linestyle='--', alpha=0.5, label='Chance')
    ax2.set_ylabel('Peak AUC')
    ax2.set_title(f'Average Peak Performance (at {times[peak_time_idx]:.2f}s)')
    ax2.set_ylim(0.4, 0.65)
    ax2.legend()
    
    # Add significance stars if one is better
    from scipy import stats
    t_stat, p_val = stats.ttest_rel(right_peak_aucs, left_peak_aucs)
    if p_val < 0.05:
        y_max = max(mean_peak_aucs) + max(sem_peak_aucs) + 0.02
        ax2.plot([0, 1], [y_max, y_max], 'k-')
        ax2.text(0.5, y_max + 0.005, '*', ha='center', va='bottom', fontsize=16)
        ax2.text(0.5, y_max + 0.015, f'p={p_val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/group_individual_scatter.png", dpi=300)
    plt.close()
    
    # ===== Save summary statistics =====
    with open(f"{save_dir}/group_hemifield_summary.txt", 'w') as f:
        f.write(f"Hemifield-Specific Voice Decoding Results\n")
        f.write(f"Stimulus: {stimulus}\n")
        f.write(f"Number of subjects: {n_subjects}\n")
        f.write(f"Subjects: {group_results['subjects']}\n")
        f.write("="*60 + "\n\n")
        
        # Peak performance
        left_peak_idx = np.argmax(mean_left_auc)
        right_peak_idx = np.argmax(mean_right_auc)
        
        f.write("Peak Performance:\n")
        f.write("-"*30 + "\n")
        f.write(f"Left hemifield:  AUC = {mean_left_auc[left_peak_idx]:.3f} ± {sem_left_auc[left_peak_idx]:.3f} at {times[left_peak_idx]:.2f}s\n")
        f.write(f"Right hemifield: AUC = {mean_right_auc[right_peak_idx]:.3f} ± {sem_right_auc[right_peak_idx]:.3f} at {times[right_peak_idx]:.2f}s\n")
        
        f.write(f"\nLeft hemifield:  Acc = {mean_left_acc[left_peak_idx]:.3f} ± {sem_left_acc[left_peak_idx]:.3f} at {times[left_peak_idx]:.2f}s\n")
        f.write(f"Right hemifield: Acc = {mean_right_acc[right_peak_idx]:.3f} ± {sem_right_acc[right_peak_idx]:.3f} at {times[right_peak_idx]:.2f}s\n")
        
        # Overall statistics
        f.write(f"\n\nPaired t-test at peak ({times[peak_time_idx]:.2f}s):\n")
        f.write(f"t({n_subjects-1}) = {t_stat:.3f}, p = {p_val:.3f}\n")
        
        if p_val < 0.05:
            if t_stat > 0:
                f.write("Right hemifield significantly better than left\n")
            else:
                f.write("Left hemifield significantly better than right\n")
        else:
            f.write("No significant difference between hemifields\n")
    
    logging.info(f"Group hemifield plots saved to {save_dir}")

def aggregate_hemifield_results(subjects, stimulus='S1'):
    """Aggregate hemifield-specific results across subjects"""
    
    save_dir = f"{RESULTS_DIR}/hemifield_voice_{stimulus}/"
    individual_dir = f"{save_dir}/individual_results/"
    
    # Load all individual results
    left_aucs = []
    right_aucs = []
    left_accs = []
    right_accs = []
    successful_subjects = []
    
    for subject in subjects:
        result_file = f"{individual_dir}/sub-{subject}_results.pkl"
        if os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                result = pickle.load(f)
                
            if result['left_results'] is not None and result['right_results'] is not None:
                left_aucs.append(result['left_results']['combined']['auc'])
                right_aucs.append(result['right_results']['combined']['auc'])
                left_accs.append(result['left_results']['combined']['accuracy'])
                right_accs.append(result['right_results']['combined']['accuracy'])
                successful_subjects.append(subject)
        else:
            logging.warning(f"Missing hemifield results for subject {subject}")
    
    if len(successful_subjects) == 0:
        logging.error("No hemifield results found!")
        return None
    
    logging.info(f"Loaded hemifield results for {len(successful_subjects)} subjects")
    
    # Convert to arrays
    left_aucs = np.array(left_aucs)
    right_aucs = np.array(right_aucs)
    left_accs = np.array(left_accs)
    right_accs = np.array(right_accs)
    
    # Get times from first subject
    times = result['left_results']['combined']['times']
    
    # Perform statistical tests
    # Test if each hemifield is above chance
    left_clusters_auc, left_p_auc = cluster_based_permutation_test(left_aucs, n_permutations=N_PERMUTATIONS)
    right_clusters_auc, right_p_auc = cluster_based_permutation_test(right_aucs, n_permutations=N_PERMUTATIONS)
    
    # Test difference between hemifields
    diff_clusters_auc, diff_p_auc = cluster_based_permutation_test(
        right_aucs, left_aucs, n_permutations=N_PERMUTATIONS
    )
    
    # Save results
    group_results = {
        'times': times,
        'left_aucs': left_aucs,
        'right_aucs': right_aucs,
        'left_accuracies': left_accs,
        'right_accuracies': right_accs,
        'subjects': successful_subjects,
        'statistical_tests': {
            'left_vs_chance_auc': (left_clusters_auc, left_p_auc),
            'right_vs_chance_auc': (right_clusters_auc, right_p_auc),
            'right_vs_left_auc': (diff_clusters_auc, diff_p_auc)
        }
    }
    
    with open(f"{save_dir}/group_hemifield_results.pkl", 'wb') as f:
        pickle.dump(group_results, f)
    
    # Create group plots
    plot_group_hemifield_results(group_results, save_dir, stimulus)
    
    return group_results

def decode_window_nested_cv(X_window, y, param_grid=None):
    """
    Decode single time window with nested cross-validation
    NOW RETURNS BOTH AUC AND ACCURACY
    
    Parameters:
    -----------
    X_window : array [n_trials, n_features]
    y : array [n_trials] - class labels
    param_grid : dict - parameters to search
    
    Returns:
    --------
    mean_auc : float - unbiased AUC estimate
    mean_accuracy : float - unbiased accuracy estimate
    best_params : dict - most common best parameters
    """
    
    # for LDA 
    # if param_grid is None:
    #     param_grid = {
    #         'lineardiscriminantanalysis__solver': ['eigen', 'lsqr'],
    #         'lineardiscriminantanalysis__shrinkage': [None, 0.1, 0.3, 0.5, 0.7, 0.9]
    #     }
    
    if param_grid is None:
        # For SVM, search over C parameter
        param_grid = {
            'svc__C': [0.001, 0.01, 0.1, 1, 10, 100]
        }

    # Setup nested CV
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # For LDA pipeline
    # pipeline = make_pipeline(
    #     StandardScaler(),
    #     LinearDiscriminantAnalysis()
    # )
    
    # For SVM pipeline
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
            scoring='roc_auc',  # Use AUC for hyperparameter selection
            n_jobs=1
        )
        
        # Fit on training data
        grid_search.fit(X_train, y_train)
        
        # Store best params
        best_params_list.append(grid_search.best_params_)
        
        # Evaluate on test set - GET BOTH METRICS
        y_prob = grid_search.predict_proba(X_test)[:, 1]
        y_pred = grid_search.predict(X_test)
        
        auc = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        
        outer_auc_scores.append(auc)
        outer_acc_scores.append(accuracy)
    
    # Get most common best parameters
    if best_params_list:
        if len(param_grid) == 1:
            param_name = list(param_grid.keys())[0]
            param_values = [p[param_name] for p in best_params_list]
            most_common = Counter(param_values).most_common(1)[0][0]
            best_params = {param_name: most_common}
        else:
            param_tuples = [tuple(p.items()) for p in best_params_list]
            most_common_tuple = Counter(param_tuples).most_common(1)[0][0]
            best_params = dict(most_common_tuple)
    else:
        best_params = {}
    
    return np.mean(outer_auc_scores), np.mean(outer_acc_scores), best_params

def decode_sliding_window_nested(X, y, times, window_length_s=0.25, window_step_s=0.05, n_jobs=16):
    """
    Sliding window decoding with nested CV
    
    Parameters:
    -----------
    X : array [n_trials, n_channels, n_timepoints]
    y : array [n_trials] - class labels
    times : array - time points
    n_jobs : int - number of parallel jobs
    """
    
    n_trials, n_channels, n_times = X.shape
    window_step_s = 0.05
    window_length_s = 0.25    
    # Calculate window parameters
    time_step = times[1] - times[0]
    window_samples = int(window_length_s / time_step)
    step_samples = max(1, int(window_step_s / time_step))
    
    print(f"Time resolution: {time_step:.4f}s")
    print(f"Actual time step: {time_step}")  # Debug
    print(f"Window length (s): {window_length_s}")
    print(f"Window length (samples): {window_samples}")
    print(f"Step size (samples): {step_samples}")
    print(f"Total time points: {n_times}")
    print(f"Time range: {times[0]:.3f} to {times[-1]:.3f}")

    # Prepare windows
    window_starts = range(0, n_times - window_samples + 1, step_samples)
    
    def process_window(start):
        """Process single window - NOW RETURNS BOTH METRICS"""
        end = start + window_samples
        
        # Extract and average window
        X_window = np.mean(X[:, :, start:end], axis=2)
        
        # Decode with nested CV - GET BOTH METRICS
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
    
    # Extract results - BOTH METRICS
    auc_scores = np.array([r['auc'] for r in results])
    accuracy_scores = np.array([r['accuracy'] for r in results])
    window_times = np.array([r['time'] for r in results])
    all_best_params = [r['best_params'] for r in results]
    
    return auc_scores, accuracy_scores, window_times, all_best_params

def cluster_based_permutation_test(data1, data2=None, n_permutations=1000, threshold=2.086):
    """
    Cluster-based permutation test
    
    Parameters:
    -----------
    data1 : array (n_subjects, n_times) - condition 1 or data to test against chance
    data2 : array (n_subjects, n_times) - condition 2 (optional, if None test against 0.5)
    n_permutations : int
    threshold : float - cluster-forming threshold
    
    Returns:
    --------
    clusters : list of significant clusters
    cluster_p_values : array of p-values for each cluster
    """
    from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test
    
    if data2 is None:
        # Test against chance (0.5 for AUC)
        data_centered = data1 - 0.5
        t_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
            data_centered, 
            n_permutations=n_permutations,
            threshold=threshold,
            tail=1  # One-tailed for above-chance
        )
    else:
        # Compare two conditions
        t_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
            [data1, data2],
            n_permutations=n_permutations,
            threshold=threshold
        )
    
    return clusters, cluster_p_values

def map_aligned_events_to_s2(subject):
    """
    Map aligned events to S2 condition codes using complex mapping
    Returns both S1 (aligned) and corresponding S2 events
    """
    
    # Load aligned events
    if subject == 28:
        events_file = f"{PROCESSED_DIR}/aligned_events_fixed/sub-{subject}_events.npy"
    else:
        events_file = f"{PROCESSED_DIR}/aligned_events_corrected/sub-{subject}_events.npy"
    
    if not os.path.exists(events_file):
        raise FileNotFoundError(f"Aligned events file not found: {events_file}")
    
    aligned_events = np.load(events_file)
    
    # Load raw events and extract S1/S2 codes
    all_raw_events = load_raw_events_for_subject(subject)
    s1_codes_raw = extract_stimulus_events(all_raw_events, 'S1')
    s2_codes_raw = extract_stimulus_events(all_raw_events, 'S2')
    
    if len(s1_codes_raw) != len(s2_codes_raw):
        raise ValueError(f"S1/S2 code count mismatch: {len(s1_codes_raw)} vs {len(s2_codes_raw)}")
    
    # Complex mapping: find which trials were kept
    kept_indices = []
    s1_codes_list = s1_codes_raw.tolist()
    
    for aligned_code in aligned_events:
        try:
            # Find next occurrence of this code
            idx = s1_codes_list.index(aligned_code)
            kept_indices.append(idx)
            s1_codes_list[idx] = -1  # Mark as used
        except ValueError:
            raise ValueError(f"Aligned code {aligned_code} not found in remaining S1 codes")
    
    # Get corresponding S2 codes
    s2_events_aligned = s2_codes_raw[kept_indices]
    
    logging.info(f"Complex mapping successful: {len(aligned_events)} events mapped")
    logging.info(f"S1 range: {np.min(aligned_events)}-{np.max(aligned_events)}")
    logging.info(f"S2 range: {np.min(s2_events_aligned)}-{np.max(s2_events_aligned)}")
    
    return aligned_events, s2_events_aligned

def process_subject_tf_decoding_nested_single(subject, feature, stimulus):
    """Process TF decoding with nested CV - FIXED with complex mapping"""
    
    # Load single-trial TFR
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
    if stimulus == 'S1':
        tmin, tmax = 0, 1
    else:  # S2
        tmin, tmax = 1, 2
    
    tfr_stim = tfr_baseline.copy().crop(tmin=tmin, tmax=tmax)
    
    # 🔥 NEW: Use complex mapping for perfect trial alignment
    try:
        s1_events_aligned, s2_events_aligned = map_aligned_events_to_s2(subject)
        
        # Select events based on stimulus
        if stimulus == 'S1':
            condition_codes = s1_events_aligned
        else:  # S2
            condition_codes = s2_events_aligned
        
        logging.info(f"Using {len(condition_codes)} trial-aligned condition codes for {stimulus}")
        logging.info(f"Condition code range: {np.min(condition_codes)} - {np.max(condition_codes)}")
        
    except Exception as e:
        logging.error(f"Failed to map events: {str(e)}")
        return None
    
    # Verify perfect alignment
    if len(condition_codes) != tfr_stim.data.shape[0]:
        logging.error(f"Event/TFR mismatch: {len(condition_codes)} vs {tfr_stim.data.shape[0]}")
        return None
    
    # Apply subject-specific corrections if needed
    original_length = len(condition_codes)
    if subject == 28 and len(condition_codes) > 63:
        condition_codes = np.delete(condition_codes, 63)
        logging.info(f"Applied subject 28 correction: {original_length} -> {len(condition_codes)}")
    elif subject == 23:
        drop_idx = 64 * 7
        if len(condition_codes) > drop_idx:
            condition_codes = np.delete(condition_codes, drop_idx)
            logging.info(f"Applied subject 23 correction: {original_length} -> {len(condition_codes)}")
    
    # Final verification after subject-specific corrections
    if len(condition_codes) != tfr_stim.data.shape[0]:
        logging.error(f"Final mismatch: {len(condition_codes)} vs {tfr_stim.data.shape[0]}")
        return None
    
    events = condition_codes
    
    # Create binary labels
    if feature == 'location':
        if stimulus == 'S1':
            low_group = [111, 121, 131, 141, 112, 122, 132, 142]
            high_group = [113, 123, 133, 143, 114, 124, 134, 144]
        else:  # S2
            low_group = [211, 221, 231, 241, 212, 222, 232, 242]
            high_group = [213, 223, 233, 243, 214, 224, 234, 244]
    else:  # voice_identity
        if stimulus == 'S1':
            low_group = [111, 112, 113, 114, 121, 122, 123, 124]
            high_group = [131, 132, 133, 134, 141, 142, 143, 144]
        else:  # S2
            low_group = [211, 212, 213, 214, 221, 222, 223, 224]
            high_group = [231, 232, 233, 234, 241, 242, 243, 244]
    
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
        logging.error(f"No valid trials found for {feature}_{stimulus} in subject {subject}")
        logging.error(f"Available events: {np.unique(events)}")
        logging.error(f"Looking for low_group: {low_group}")
        logging.error(f"Looking for high_group: {high_group}")
        return None
    
    # Check class balance
    class_counts = np.bincount(y)
    if len(class_counts) < 2 or np.min(class_counts) < 5:
        logging.error(f"Insufficient trials for {feature}_{stimulus} in subject {subject}")
        logging.error(f"Class 0: {class_counts[0] if len(class_counts) > 0 else 0} trials")
        logging.error(f"Class 1: {class_counts[1] if len(class_counts) > 1 else 0} trials")
        return None
    
    logging.info(f"Found {len(y)} valid trials: Class 0={class_counts[0]}, Class 1={class_counts[1]}")
    
    # Filter TFR data to valid trials only
    tfr_data = tfr_stim.data[valid_idx]
    
    # Extract band powers
    band_powers = extract_band_power(tfr_data, tfr_stim.freqs)
    
    # Decode each band with nested CV
    band_results = {}
    for band_name, band_data in band_powers.items():
        print(f"\nDecoding {band_name} band with nested CV...")
        
        try:
            auc_scores, accuracy_scores, times, best_params = decode_sliding_window_nested(
                band_data, y, tfr_stim.times, n_jobs=16
            )
            
            band_results[band_name] = {
                'auc': auc_scores,
                'accuracy': accuracy_scores,
                'times': times,
                'best_params': best_params
            }
            
            # Save best parameters
            save_dir = f"{RESULTS_DIR}/{feature}_{stimulus}/params/"
            os.makedirs(save_dir, exist_ok=True)
            
            with open(f"{save_dir}/sub-{subject}_{band_name}_params.txt", 'w') as f:
                f.write("Time\tBest_Parameters\n")
                for t, params in zip(times, best_params):
                    f.write(f"{t:.3f}\t{params}\n")
                    
        except Exception as e:
            logging.error(f"Error decoding {band_name} band: {str(e)}")
            return None
    
    # Calculate combined bands (average across all bands) - BOTH METRICS
    all_band_aucs = [band_results[band]['auc'] for band in band_results]
    all_band_accuracies = [band_results[band]['accuracy'] for band in band_results]
    
    combined_aucs = np.mean(all_band_aucs, axis=0)
    combined_accuracies = np.mean(all_band_accuracies, axis=0)
    
    # Return structured results - BOTH METRICS
    result = {
        'subject': subject,
        'band_results': band_results,
        'combined_bands': {
            'auc': combined_aucs,
            'accuracy': combined_accuracies
        },
        'times': times
    }
    
    # Save results
    save_individual_results(result, feature, stimulus)
    
    return result

def process_hemifield_specific_decoding(subject, stimulus):
    """
    Perform voice identity decoding separately for left and right hemifields
    """
    
    # Load and preprocess data as before
    tfr = load_tfr_data(subject)
    if tfr is None:
        return None
    
    # Apply baseline and crop
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
    
    # Define hemifield groups based on your event codes
    if stimulus == 'S1':
        # L1, L2 are left hemifield; L3, L4 are right hemifield
        left_hemifield_events = [111, 112, 121, 122, 131, 132, 141, 142]  # L1, L2
        right_hemifield_events = [113, 114, 123, 124, 133, 134, 143, 144]  # L3, L4
        
        # Voice groups within each hemifield
        left_voice1 = [111, 112, 121, 122]  # Left field, Speakers 1-2
        left_voice2 = [131, 132, 141, 142]  # Left field, Speakers 3-4
        
        right_voice1 = [113, 114, 123, 124]  # Right field, Speakers 1-2
        right_voice2 = [133, 134, 143, 144]  # Right field, Speakers 3-4
    else:  # S2
        left_hemifield_events = [211, 212, 221, 222, 231, 232, 241, 242]
        right_hemifield_events = [213, 214, 223, 224, 233, 234, 243, 244]
        
        left_voice1 = [211, 212, 221, 222]
        left_voice2 = [231, 232, 241, 242]
        
        right_voice1 = [213, 214, 223, 224]
        right_voice2 = [233, 234, 243, 244]
    
    results = {}
    
    # Process each hemifield separately
    for hemifield in ['left', 'right']:
        if hemifield == 'left':
            hemi_events = left_hemifield_events
            voice_group1 = left_voice1
            voice_group2 = left_voice2
        else:
            hemi_events = right_hemifield_events
            voice_group1 = right_voice1
            voice_group2 = right_voice2
        
        # Get trials for this hemifield only
        hemi_mask = np.isin(events, hemi_events)
        hemi_indices = np.where(hemi_mask)[0]
        
        if len(hemi_indices) < 20:  # Need enough trials
            logging.warning(f"Not enough trials for {hemifield} hemifield: {len(hemi_indices)}")
            continue
        
        # Extract data and labels for this hemifield
        hemi_data = tfr_stim.data[hemi_indices]
        hemi_events_subset = events[hemi_indices]
        
        # Create voice labels (0 for voice group 1, 1 for voice group 2)
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
            logging.warning(f"Not enough valid trials for voice decoding in {hemifield} hemifield")
            continue
        
        y_voice = np.array(y_voice)
        hemi_data_valid = hemi_data[valid_idx]
        
        # Check class balance
        class_counts = np.bincount(y_voice)
        logging.info(f"{hemifield} hemifield: Voice1={class_counts[0]}, Voice2={class_counts[1]} trials")
        
        # Extract band powers and decode
        band_powers = extract_band_power(hemi_data_valid, tfr_stim.freqs)
        
        hemi_results = {}
        for band_name, band_data in band_powers.items():
            print(f"\nDecoding {band_name} band for {hemifield} hemifield...")
            
            auc_scores, accuracy_scores, times, best_params = decode_sliding_window_nested(
                band_data, y_voice, tfr_stim.times, n_jobs=16
            )
            
            hemi_results[band_name] = {
                'auc': auc_scores,
                'accuracy': accuracy_scores,
                'times': times,
                'best_params': best_params
            }
        
        # Combined bands
        all_aucs = [hemi_results[band]['auc'] for band in hemi_results]
        all_accs = [hemi_results[band]['accuracy'] for band in hemi_results]
        
        hemi_results['combined'] = {
            'auc': np.mean(all_aucs, axis=0),
            'accuracy': np.mean(all_accs, axis=0),
            'times': times
        }
        
        results[hemifield] = hemi_results
    
    # Save individual results
    save_dir = f"{RESULTS_DIR}/hemifield_voice_{stimulus}/individual_results/"
    os.makedirs(save_dir, exist_ok=True)
    
    with open(f"{save_dir}/sub-{subject}_results.pkl", 'wb') as f:
        pickle.dump({
            'subject': subject,
            'stimulus': stimulus,
            'left_results': results.get('left', None),
            'right_results': results.get('right', None)
        }, f)
    
    return results

def plot_hemifield_comparison(results, subject, stimulus):
    """Plot comparison of voice decoding between hemifields"""
    
    if 'left' not in results or 'right' not in results:
        logging.error("Need both hemifield results for comparison")
        return
    
    save_dir = f"{RESULTS_DIR}/hemifield_voice_{stimulus}/individual_plots/"
    os.makedirs(save_dir, exist_ok=True)
    
    times = results['left']['combined']['times']
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # AUC comparison
    ax1.plot(times, results['left']['combined']['auc'], 'b-', 
             label='Left Hemifield', linewidth=2)
    ax1.plot(times, results['right']['combined']['auc'], 'r-', 
             label='Right Hemifield', linewidth=2)
    ax1.axhline(0.5, color='k', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Voice Decoding AUC')
    ax1.set_title(f'Subject {subject} - Voice Identity Decoding by Hemifield ({stimulus})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.35, 0.7)
    
    # Accuracy comparison
    ax2.plot(times, results['left']['combined']['accuracy'], 'b-', 
             label='Left Hemifield', linewidth=2)
    ax2.plot(times, results['right']['combined']['accuracy'], 'r-', 
             label='Right Hemifield', linewidth=2)
    ax2.axhline(0.5, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Voice Decoding Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.35, 0.7)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sub-{subject}_hemifield_comparison.png", dpi=300)
    plt.close()
    
    # Plot difference
    fig, ax = plt.subplots(figsize=(10, 6))
    
    auc_diff = results['right']['combined']['auc'] - results['left']['combined']['auc']
    acc_diff = results['right']['combined']['accuracy'] - results['left']['combined']['accuracy']
    
    ax.plot(times, auc_diff, 'purple', label='AUC Difference (R-L)', linewidth=2)
    ax.plot(times, acc_diff, 'orange', label='Accuracy Difference (R-L)', linewidth=2)
    ax.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax.fill_between(times, auc_diff, 0, alpha=0.3, color='purple')
    ax.fill_between(times, acc_diff, 0, alpha=0.3, color='orange')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Performance Difference (Right - Left)')
    ax.set_title(f'Hemifield Advantage for Voice Decoding ({stimulus})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations for peak differences
    peak_auc_diff_idx = np.argmax(np.abs(auc_diff))
    peak_auc_diff = auc_diff[peak_auc_diff_idx]
    ax.annotate(f'Peak AUC diff: {peak_auc_diff:.3f}\nat {times[peak_auc_diff_idx]:.2f}s',
                xy=(times[peak_auc_diff_idx], peak_auc_diff),
                xytext=(times[peak_auc_diff_idx] + 0.1, peak_auc_diff + 0.02),
                arrowprops=dict(arrowstyle='->', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sub-{subject}_hemifield_difference.png", dpi=300)
    plt.close()

def save_individual_results(result, feature, stimulus):
    """Save individual subject results - MODIFIED for dual metrics"""
    save_dir = f"{RESULTS_DIR}/{feature}_{stimulus}/individual_results/"
    os.makedirs(save_dir, exist_ok=True)
    
    subject = result['subject']
    
    # Save as pickle for easy loading
    with open(f"{save_dir}/sub-{subject}_results.pkl", 'wb') as f:
        pickle.dump(result, f)
    
    # Save as numpy arrays for quick access - BOTH METRICS
    np.save(f"{save_dir}/sub-{subject}_times.npy", result['times'])
    np.save(f"{save_dir}/sub-{subject}_combined_auc.npy", result['combined_bands']['auc'])
    np.save(f"{save_dir}/sub-{subject}_combined_accuracy.npy", result['combined_bands']['accuracy'])
    
    # Save each frequency band - BOTH METRICS
    for band_name, band_data in result['band_results'].items():
        np.save(f"{save_dir}/sub-{subject}_{band_name}_auc.npy", band_data['auc'])
        np.save(f"{save_dir}/sub-{subject}_{band_name}_accuracy.npy", band_data['accuracy'])

def plot_group_results_dual_metrics(results_dict, save_dir, feature, stimulus):
    """Create plots for group results - BOTH AUC AND ACCURACY"""
    times = results_dict['times']
    band_aucs = results_dict['band_aucs']
    band_accuracies = results_dict['band_accuracies']
    combined_aucs = results_dict['combined_aucs']
    combined_accuracies = results_dict['combined_accuracies']
    cluster_results_auc = results_dict['cluster_results_auc']
    cluster_results_accuracy = results_dict['cluster_results_accuracy']
    n_subjects = len(results_dict['subjects'])
    
    # ===================================================================
    # PLOT 1: AUC Results (4 frequency bands)
    # ===================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (band_name, aucs) in enumerate(band_aucs.items()):
        ax = axes[idx]
        
        # Calculate mean and SEM
        mean_auc = np.mean(aucs, axis=0)
        sem_auc = np.std(aucs, axis=0) / np.sqrt(n_subjects)
        
        # Plot
        ax.plot(times, mean_auc, label=band_name, linewidth=2, color='blue')
        ax.fill_between(times, mean_auc - sem_auc, mean_auc + sem_auc, alpha=0.3, color='blue')
        
        # Add significance markers
        clusters, p_values = cluster_results_auc[band_name]
        for cluster, p_val in zip(clusters, p_values):
            if p_val < 0.05:
                cluster_times = times[cluster[0]]
                ax.axhspan(0.52, 0.53, xmin=cluster_times[0], xmax=cluster_times[-1],
                          color='red', alpha=0.3)
        
        ax.axhline(0.5, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('AUC')
        ax.set_title(f'{band_name} band - AUC')
        ax.set_ylim(0.35, 0.65)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{feature.title()} Decoding - {stimulus} (AUC, N={n_subjects})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/frequency_bands_decoding_AUC.png", dpi=300)
    plt.close()
    
    # ===================================================================
    # PLOT 2: Accuracy Results (4 frequency bands)
    # ===================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (band_name, accuracies) in enumerate(band_accuracies.items()):
        ax = axes[idx]
        
        # Calculate mean and SEM
        mean_acc = np.mean(accuracies, axis=0)
        sem_acc = np.std(accuracies, axis=0) / np.sqrt(n_subjects)
        
        # Plot
        ax.plot(times, mean_acc, label=band_name, linewidth=2, color='green')
        ax.fill_between(times, mean_acc - sem_acc, mean_acc + sem_acc, alpha=0.3, color='green')
        
        # Add significance markers
        clusters, p_values = cluster_results_accuracy[band_name]
        for cluster, p_val in zip(clusters, p_values):
            if p_val < 0.05:
                cluster_times = times[cluster[0]]
                ax.axhspan(0.52, 0.53, xmin=cluster_times[0], xmax=cluster_times[-1],
                          color='red', alpha=0.3)
        
        ax.axhline(0.5, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{band_name} band - Accuracy')
        ax.set_ylim(0.35, 0.65)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{feature.title()} Decoding - {stimulus} (Accuracy, N={n_subjects})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/frequency_bands_decoding_Accuracy.png", dpi=300)
    plt.close()
    
    # ===================================================================
    # PLOT 3: Combined bands comparison - AUC vs Accuracy
    # ===================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # AUC Combined
    mean_combined_auc = np.mean(combined_aucs, axis=0)
    sem_combined_auc = np.std(combined_aucs, axis=0) / np.sqrt(n_subjects)
    
    ax1.plot(times, mean_combined_auc, 'b-', linewidth=2, label='AUC')
    ax1.fill_between(times, mean_combined_auc - sem_combined_auc, 
                     mean_combined_auc + sem_combined_auc, alpha=0.3, color='blue')
    
    # Add AUC significance
    clusters, p_values = cluster_results_auc['combined']
    for i, (cluster, p_val) in enumerate(zip(clusters, p_values)):
        if p_val < 0.05:
            cluster_times = times[cluster[0]]
            ax1.axhspan(0.52, 0.53, xmin=cluster_times[0], xmax=cluster_times[-1],
                    color='red', alpha=0.3, label='p < 0.05' if i == 0 else '')

    
    ax1.axhline(0.5, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('AUC')
    ax1.set_title('Combined Frequency Bands - AUC')
    ax1.set_ylim(0.35, 0.65)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy Combined
    mean_combined_acc = np.mean(combined_accuracies, axis=0)
    sem_combined_acc = np.std(combined_accuracies, axis=0) / np.sqrt(n_subjects)
    
    ax2.plot(times, mean_combined_acc, 'g-', linewidth=2, label='Accuracy')
    ax2.fill_between(times, mean_combined_acc - sem_combined_acc, 
                     mean_combined_acc + sem_combined_acc, alpha=0.3, color='green')
    
    # Add Accuracy significance
    clusters, p_values = cluster_results_accuracy['combined']
    for i, (cluster, p_val) in enumerate(zip(clusters, p_values)):
        if p_val < 0.05:
            cluster_times = times[cluster[0]]
            ax2.axhspan(0.52, 0.53, xmin=cluster_times[0], xmax=cluster_times[-1],
                    color='red', alpha=0.3, label='p < 0.05' if i == 0 else '')
    
    ax2.axhline(0.5, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Combined Frequency Bands - Accuracy')
    ax2.set_ylim(0.35, 0.65)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(f'{feature.title()} Decoding - {stimulus} (N={n_subjects})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/combined_AUC_vs_Accuracy.png", dpi=300)
    plt.close()
    
    # ===================================================================
    # PLOT 4: Direct AUC vs Accuracy comparison (overlaid)
    # ===================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot both metrics on same axis
    ax.plot(times, mean_combined_auc, 'b-', linewidth=2, label='AUC', alpha=0.8)
    ax.fill_between(times, mean_combined_auc - sem_combined_auc, 
                   mean_combined_auc + sem_combined_auc, alpha=0.2, color='blue')
    
    ax.plot(times, mean_combined_acc, 'g-', linewidth=2, label='Accuracy', alpha=0.8)
    ax.fill_between(times, mean_combined_acc - sem_combined_acc, 
                   mean_combined_acc + sem_combined_acc, alpha=0.2, color='green')
    
    ax.axhline(0.5, color='k', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Performance')
    ax.set_title(f'{feature.title()} Decoding - {stimulus} (AUC vs Accuracy)')
    ax.set_ylim(0.35, 0.65)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/AUC_vs_Accuracy_overlay.png", dpi=300)
    plt.close()
    
    # ===================================================================
    # PLOT 5: Heatmap comparison - AUC
    # ===================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # AUC Heatmap
    band_matrix_auc = []
    band_labels = []
    for band_name, aucs in band_aucs.items():
        band_matrix_auc.append(np.mean(aucs, axis=0))
        band_labels.append(band_name)
    band_matrix_auc.append(mean_combined_auc)
    band_labels.append('Combined')
    band_matrix_auc = np.array(band_matrix_auc)
    
    im1 = ax1.imshow(band_matrix_auc, aspect='auto', cmap='RdBu_r', 
                     vmin=0.45, vmax=0.55, origin='lower')
    ax1.set_xticks(np.arange(0, len(times), 2))
    ax1.set_xticklabels([f'{t:.1f}' for t in times[::2]])
    ax1.set_yticks(range(len(band_labels)))
    ax1.set_yticklabels(band_labels)
    ax1.set_title(f'{feature.title()} Decoding - AUC Heatmap')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('AUC')
    
    # Accuracy Heatmap
    band_matrix_acc = []
    for band_name, accuracies in band_accuracies.items():
        band_matrix_acc.append(np.mean(accuracies, axis=0))
    band_matrix_acc.append(mean_combined_acc)
    band_matrix_acc = np.array(band_matrix_acc)
    
    im2 = ax2.imshow(band_matrix_acc, aspect='auto', cmap='RdBu_r', 
                     vmin=0.45, vmax=0.55, origin='lower')
    ax2.set_xticks(np.arange(0, len(times), 2))
    ax2.set_xticklabels([f'{t:.1f}' for t in times[::2]])
    ax2.set_yticks(range(len(band_labels)))
    ax2.set_yticklabels(band_labels)
    ax2.set_xlabel('Time (s)')
    ax2.set_title(f'{feature.title()} Decoding - Accuracy Heatmap')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Accuracy')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/decoding_heatmaps_dual.png", dpi=300)
    plt.close()
    
    # ===================================================================
    # Save summary statistics - BOTH METRICS
    # ===================================================================
    summary = {
        'peak_performance_auc': {},
        'peak_performance_accuracy': {},
        'significant_windows_auc': {},
        'significant_windows_accuracy': {}
    }
    
    for band_name in list(band_aucs.keys()) + ['combined']:
        if band_name == 'combined':
            aucs = combined_aucs
            accuracies = combined_accuracies
            clusters_auc, p_values_auc = cluster_results_auc['combined']
            clusters_acc, p_values_acc = cluster_results_accuracy['combined']
        else:
            aucs = band_aucs[band_name]
            accuracies = band_accuracies[band_name]
            clusters_auc, p_values_auc = cluster_results_auc[band_name]
            clusters_acc, p_values_acc = cluster_results_accuracy[band_name]
        
        # AUC peak performance
        mean_auc = np.mean(aucs, axis=0)
        peak_idx_auc = np.argmax(mean_auc)
        peak_time_auc = times[peak_idx_auc]
        peak_auc = mean_auc[peak_idx_auc]
        
        summary['peak_performance_auc'][band_name] = {
            'time': peak_time_auc,
            'auc': peak_auc,
            'sem': np.std(aucs[:, peak_idx_auc]) / np.sqrt(n_subjects)
        }
        
        # Accuracy peak performance
        mean_acc = np.mean(accuracies, axis=0)
        peak_idx_acc = np.argmax(mean_acc)
        peak_time_acc = times[peak_idx_acc]
        peak_acc = mean_acc[peak_idx_acc]
        
        summary['peak_performance_accuracy'][band_name] = {
            'time': peak_time_acc,
            'accuracy': peak_acc,
            'sem': np.std(accuracies[:, peak_idx_acc]) / np.sqrt(n_subjects)
        }
        
        # Significant time windows - AUC
        sig_windows_auc = []
        for cluster, p_val in zip(clusters_auc, p_values_auc):
            if p_val < 0.05:
                start_time = times[cluster[0][0]]
                end_time = times[cluster[0][-1]]
                sig_windows_auc.append((start_time, end_time, p_val))
        summary['significant_windows_auc'][band_name] = sig_windows_auc
        
        # Significant time windows - Accuracy
        sig_windows_acc = []
        for cluster, p_val in zip(clusters_acc, p_values_acc):
            if p_val < 0.05:
                start_time = times[cluster[0][0]]
                end_time = times[cluster[0][-1]]
                sig_windows_acc.append((start_time, end_time, p_val))
        summary['significant_windows_accuracy'][band_name] = sig_windows_acc
    
    # Save summary
    with open(f"{save_dir}/summary_statistics_dual_metrics.txt", 'w') as f:
        f.write(f"Time-Frequency Power Decoding Results (AUC + Accuracy)\n")
        f.write(f"Feature: {feature}, Stimulus: {stimulus}\n")
        f.write(f"Number of subjects: {n_subjects}\n")
        f.write("="*60 + "\n\n")
        
        f.write("Peak Performance - AUC:\n")
        f.write("-"*30 + "\n")
        for band, stats in summary['peak_performance_auc'].items():
            f.write(f"{band}: AUC = {stats['auc']:.3f} ± {stats['sem']:.3f} at {stats['time']:.2f}s\n")
        
        f.write("\nPeak Performance - Accuracy:\n")
        f.write("-"*30 + "\n")
        for band, stats in summary['peak_performance_accuracy'].items():
            f.write(f"{band}: Accuracy = {stats['accuracy']:.3f} ± {stats['sem']:.3f} at {stats['time']:.2f}s\n")
        
        f.write("\n\nSignificant Time Windows - AUC (p < 0.05):\n")
        f.write("-"*30 + "\n")
        for band, windows in summary['significant_windows_auc'].items():
            f.write(f"{band}:\n")
            if windows:
                for start, end, p in windows:
                    f.write(f"  {start:.2f}s - {end:.2f}s (p = {p:.3f})\n")
            else:
                f.write("  No significant windows\n")
        
        f.write("\n\nSignificant Time Windows - Accuracy (p < 0.05):\n")
        f.write("-"*30 + "\n")
        for band, windows in summary['significant_windows_accuracy'].items():
            f.write(f"{band}:\n")
            if windows:
                for start, end, p in windows:
                    f.write(f"  {start:.2f}s - {end:.2f}s (p = {p:.3f})\n")
            else:
                f.write("  No significant windows\n")
    
    # Save data to Excel for easy analysis
    export_results_to_excel(results_dict, save_dir, feature, stimulus)

def export_results_to_excel(results_dict, save_dir, feature, stimulus):
    """Export results to Excel with separate sheets for AUC and Accuracy"""
    
    times = results_dict['times']
    band_aucs = results_dict['band_aucs']
    band_accuracies = results_dict['band_accuracies']
    combined_aucs = results_dict['combined_aucs']
    combined_accuracies = results_dict['combined_accuracies']
    subjects = results_dict['subjects']
    
    excel_file = f"{save_dir}/decoding_results_dual_metrics.xlsx"
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        
        # Sheet 1: AUC Group Averages
        auc_avg_data = {'Time (s)': times}
        for band_name, aucs in band_aucs.items():
            auc_avg_data[f'AUC_{band_name}_mean'] = np.mean(aucs, axis=0)
            auc_avg_data[f'AUC_{band_name}_sem'] = np.std(aucs, axis=0) / np.sqrt(len(subjects))
        auc_avg_data['AUC_Combined_mean'] = np.mean(combined_aucs, axis=0)
        auc_avg_data['AUC_Combined_sem'] = np.std(combined_aucs, axis=0) / np.sqrt(len(subjects))
        
        pd.DataFrame(auc_avg_data).to_excel(writer, sheet_name='AUC_GroupAverage', index=False)
        
        # Sheet 2: Accuracy Group Averages
        acc_avg_data = {'Time (s)': times}
        for band_name, accuracies in band_accuracies.items():
            acc_avg_data[f'Accuracy_{band_name}_mean'] = np.mean(accuracies, axis=0)
            acc_avg_data[f'Accuracy_{band_name}_sem'] = np.std(accuracies, axis=0) / np.sqrt(len(subjects))
        acc_avg_data['Accuracy_Combined_mean'] = np.mean(combined_accuracies, axis=0)
        acc_avg_data['Accuracy_Combined_sem'] = np.std(combined_accuracies, axis=0) / np.sqrt(len(subjects))
        
        pd.DataFrame(acc_avg_data).to_excel(writer, sheet_name='Accuracy_GroupAverage', index=False)
        
        # Sheet 3: Individual Subject AUC (Combined bands)
        ind_auc_data = {'Subject': subjects}
        for i, time_point in enumerate(times[::5]):  # Every 5th time point to reduce size
            ind_auc_data[f'AUC_{time_point:.2f}s'] = combined_aucs[:, i*5]
        
        pd.DataFrame(ind_auc_data).to_excel(writer, sheet_name='Individual_AUC', index=False)
        
        # Sheet 4: Individual Subject Accuracy (Combined bands)
        ind_acc_data = {'Subject': subjects}
        for i, time_point in enumerate(times[::5]):  # Every 5th time point to reduce size
            ind_acc_data[f'Accuracy_{time_point:.2f}s'] = combined_accuracies[:, i*5]
        
        pd.DataFrame(ind_acc_data).to_excel(writer, sheet_name='Individual_Accuracy', index=False)
    
    logging.info(f"Results exported to {excel_file}")

def aggregate_group_results(subjects, feature='location', stimulus='S1'):
    """
    Aggregate individual results into group analysis - MODIFIED for dual metrics
    """
    save_dir = f"{RESULTS_DIR}/{feature}_{stimulus}/"
    individual_dir = f"{save_dir}/individual_results/"
    
    # Load all individual results (same as before)
    all_results = []
    successful_subjects = []
    
    for subject in subjects:
        result_file = f"{individual_dir}/sub-{subject}_results.pkl"
        if os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                result = pickle.load(f)
                all_results.append(result)
                successful_subjects.append(subject)
        else:
            logging.warning(f"Missing results for subject {subject}")
    
    if len(all_results) == 0:
        logging.error("No individual results found!")
        return None
    
    logging.info(f"Loaded results for {len(all_results)} subjects: {successful_subjects}")
    
    # Aggregate results - BOTH METRICS
    times = all_results[0]['times']
    band_names = list(all_results[0]['band_results'].keys())
    
    # Collect scores for each band - BOTH METRICS
    band_aucs = {band: [] for band in band_names}
    band_accuracies = {band: [] for band in band_names}
    combined_aucs = []
    combined_accuracies = []

    for result in all_results:
        for band in band_names:
            band_aucs[band].append(result['band_results'][band]['auc'])
            band_accuracies[band].append(result['band_results'][band]['accuracy'])
        combined_aucs.append(result['combined_bands']['auc'])
        combined_accuracies.append(result['combined_bands']['accuracy'])

    # Convert to arrays
    for band in band_names:
        band_aucs[band] = np.array(band_aucs[band])
        band_accuracies[band] = np.array(band_accuracies[band])
    combined_aucs = np.array(combined_aucs)
    combined_accuracies = np.array(combined_accuracies)
    
    # Perform cluster-based permutation tests - BOTH METRICS
    logging.info("Performing cluster-based permutation tests...")
    
    cluster_results_auc = {}
    cluster_results_accuracy = {}
    
    for band in band_names:
        # AUC clusters (test against 0.5)
        clusters_auc, p_values_auc = cluster_based_permutation_test(
            band_aucs[band], 
            n_permutations=N_PERMUTATIONS
        )
        cluster_results_auc[band] = (clusters_auc, p_values_auc)
        
        # Accuracy clusters (test against 0.5)
        clusters_acc, p_values_acc = cluster_based_permutation_test(
            band_accuracies[band], 
            n_permutations=N_PERMUTATIONS
        )
        cluster_results_accuracy[band] = (clusters_acc, p_values_acc)
    
    # Test combined bands - BOTH METRICS
    clusters_combined_auc, p_values_combined_auc = cluster_based_permutation_test(
        combined_aucs, n_permutations=N_PERMUTATIONS
    )
    cluster_results_auc['combined'] = (clusters_combined_auc, p_values_combined_auc)
    
    clusters_combined_acc, p_values_combined_acc = cluster_based_permutation_test(
        combined_accuracies, n_permutations=N_PERMUTATIONS
    )
    cluster_results_accuracy['combined'] = (clusters_combined_acc, p_values_combined_acc)
    
    # Save aggregated results - BOTH METRICS
    results_dict = {
        'times': times,
        'band_aucs': band_aucs,
        'band_accuracies': band_accuracies,
        'combined_aucs': combined_aucs,
        'combined_accuracies': combined_accuracies,
        'cluster_results_auc': cluster_results_auc,
        'cluster_results_accuracy': cluster_results_accuracy,
        'subjects': successful_subjects
    }
    
    with open(f"{save_dir}/group_results.pkl", 'wb') as f:
        pickle.dump(results_dict, f)
    
    # Create plots (will need to modify this function)
    plot_group_results_dual_metrics(results_dict, save_dir, feature, stimulus)
    
    logging.info(f"Group analysis complete. Results saved to {save_dir}")
    return results_dict

def create_individual_combination_plots(subject, summary):
    """Create detailed plots for each combination for one subject"""
    
    successful_combinations = [k for k, v in summary.items() if v['status'] == 'success']
    
    for combination in successful_combinations:
        feature, stimulus = combination.split('_')
        result_file = f"{RESULTS_DIR}/{combination}/individual_results/sub-{subject}_results.pkl"
        
        if os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                result = pickle.load(f)
            
            create_single_combination_plots(subject, feature, stimulus, result)

def create_single_combination_plots(subject, feature, stimulus, result):
    """Create plots for a single combination"""
    
    save_dir = f"{RESULTS_DIR}/individual_plots/{feature}_{stimulus}/"
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: AUC and Accuracy separately
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    for band_name, band_data in result['band_results'].items():
        aucs = band_data['auc']
        plt.plot(result['times'], aucs, label=f'{band_name} AUC', alpha=0.7)
    plt.plot(result['times'], result['combined_bands']['auc'], 'k-', 
            label='Combined AUC', linewidth=2)
    plt.axhline(0.5, color='k', linestyle='--', alpha=0.5)
    plt.ylabel('AUC')
    plt.title(f'Subject {subject} - {feature} decoding during {stimulus} (AUC)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    plt.subplot(2, 1, 2)
    for band_name, band_data in result['band_results'].items():
        accuracies = band_data['accuracy']
        plt.plot(result['times'], accuracies, label=f'{band_name} Accuracy', alpha=0.7)
    plt.plot(result['times'], result['combined_bands']['accuracy'], 'k-', 
            label='Combined Accuracy', linewidth=2)
    plt.axhline(0.5, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.title(f'Subject {subject} - {feature} decoding during {stimulus} (Accuracy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sub_{subject}_decoding_dual_metrics.png", dpi=150)
    plt.close()
    
    # Comparison plot: AUC vs Accuracy (overlaid)
    plt.figure(figsize=(10, 6))
    plt.plot(result['times'], result['combined_bands']['auc'], 'b-', 
            label='AUC', linewidth=2, alpha=0.8)
    plt.plot(result['times'], result['combined_bands']['accuracy'], 'g-', 
            label='Accuracy', linewidth=2, alpha=0.8)
    plt.axhline(0.5, color='k', linestyle='--', alpha=0.5, label='Chance')
    plt.xlabel('Time (s)')
    plt.ylabel('Performance')
    plt.title(f'Subject {subject} - {feature} decoding during {stimulus} (AUC vs Accuracy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.3, 0.8)
    plt.savefig(f"{save_dir}/sub_{subject}_AUC_vs_Accuracy.png", dpi=150)
    plt.close()

    # Save to Excel
    save_individual_excel_dual_metrics(result, save_dir, subject)
    
    logging.info(f"Individual plots saved for subject {subject} - {feature}_{stimulus}")

def main():
    """Updated main function to handle all combinations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TF Power Decoding Analysis')
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                       help='Subject IDs to process (default: all subjects)')
    parser.add_argument('--feature', type=str, default='location',
                       choices=['location', 'voice_identity'],
                       help='Feature to decode (only used if --single-combination)')
    parser.add_argument('--stimulus', type=str, default='S1',
                       choices=['S1', 'S2'],
                       help='Stimulus period (only used if --single-combination)')
    parser.add_argument('--single', type=int, default=None,
                       help='Process single subject - ALL combinations (for SLURM arrays)')
    parser.add_argument('--single-combination', type=int, default=None,
                       help='Process single subject - ONE combination only')
    parser.add_argument('--aggregate', action='store_true',
                       help='Aggregate individual results into group analysis')
    parser.add_argument('--run-all-features', action='store_true',
                       help='Run aggregation for all feature/stimulus combinations')
    
    args = parser.parse_args()
    
    if args.single is not None:
        # Process ALL combinations for this subject
        logging.info(f"Processing ALL combinations for subject {args.single}")
        
        summary = process_subject_all_combinations(args.single)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SUBJECT {args.single} PROCESSING COMPLETE")
        print(f"{'='*60}")
        
        for combination, result in summary.items():
            if result['status'] == 'success':
                print(f"✅ {combination}: AUC={result['peak_auc']:.3f}, Acc={result['peak_accuracy']:.3f}")
            else:
                print(f"❌ {combination}: {result['status']}")
        
        # 🔥 FIX: Create individual plots for all combinations
        create_subject_overview_plots(args.single, summary)
        create_individual_combination_plots(args.single, summary)
        
    elif args.single_combination is not None:
        # Process only one specific combination (if needed for debugging)
        logging.info(f"Processing subject {args.single_combination}: {args.feature}_{args.stimulus}")
        
        result = process_subject_tf_decoding_nested_single(
            args.single_combination, args.feature, args.stimulus
        )
        
        if result is not None:
            # Create individual plots for this combination
            create_single_combination_plots(args.single_combination, args.feature, args.stimulus, result)
            logging.info(f"Subject {args.single_combination} processing complete")
        else:
            logging.error(f"Failed to process subject {args.single_combination}")
            
    elif args.aggregate:
        # Aggregate results for ALL combinations (S1 AND S2)
        if args.subjects is None:
            subjects = Subs
        else:
            subjects = args.subjects
            
        if args.run_all_features:
            # Process ALL 4 combinations
            for feature in ['location', 'voice_identity']:
                for stimulus in ['S1', 'S2']:  # ✅ S2 now included
                    logging.info(f"Aggregating {feature}_{stimulus}")
                    aggregate_group_results(subjects, feature, stimulus)
                            # ADD: Aggregate hemifield results
            for stimulus in ['S1', 'S2']:
                logging.info(f"Aggregating hemifield voice decoding for {stimulus}")
                aggregate_hemifield_results(subjects, stimulus)

        else:
            logging.info(f"Aggregating {args.feature}_{args.stimulus}")
            aggregate_group_results(subjects, args.feature, args.stimulus)

def create_subject_overview_plots(subject, summary):
    """Create overview plots showing all combinations for one subject"""
    
    successful_combinations = []
    for combo_name, result in summary.items():
        if result['status'] == 'success':
            successful_combinations.append(combo_name)
    
    if len(successful_combinations) == 0:
        logging.warning(f"No successful combinations for subject {subject}")
        return
    
    save_dir = f"{RESULTS_DIR}/subject_overviews/"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load results for plotting
    all_results = {}
    for combination in successful_combinations:
        # Handle both standard and hemifield combinations
        if combination.startswith('hemifield_voice_'):
            # Skip hemifield for now in overview (they have their own plots)
            continue
        else:
            # Standard combinations (location_S1, voice_identity_S2, etc.)
            parts = combination.split('_')
            if len(parts) == 2:
                feature, stimulus = parts
            else:
                # Handle unexpected format
                logging.warning(f"Unexpected combination format: {combination}")
                continue
        
        result_file = f"{RESULTS_DIR}/{combination}/individual_results/sub-{subject}_results.pkl"
        
        if os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                all_results[combination] = pickle.load(f)
    
    if len(all_results) == 0:
        logging.warning(f"No result files found for subject {subject}")
        return
    
    # Create 2x2 subplot for the 4 STANDARD combinations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    colors = ['blue', 'green', 'red', 'orange']
    combinations_order = ['location_S1', 'location_S2', 'voice_identity_S1', 'voice_identity_S2']
    
    for idx, combination in enumerate(combinations_order):
        if combination in all_results:
            result = all_results[combination]
            ax = axes[idx]
            
            # Plot combined AUC and accuracy
            times = result['times']
            auc = result['combined_bands']['auc']
            accuracy = result['combined_bands']['accuracy']
            
            ax.plot(times, auc, color=colors[idx], linewidth=2, label='AUC', alpha=0.8)
            ax.plot(times, accuracy, color=colors[idx], linewidth=2, linestyle='--', label='Accuracy', alpha=0.8)
            ax.axhline(0.5, color='k', linestyle=':', alpha=0.5)
            
            ax.set_title(f'{combination.replace("_", " ").title()}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Performance')
            ax.set_ylim(0.3, 0.8)
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            # Combination failed or missing
            axes[idx].text(0.5, 0.5, f'{combination}\nNOT AVAILABLE', 
                          ha='center', va='center', transform=axes[idx].transAxes,
                          fontsize=14, color='gray')
            axes[idx].set_title(f'{combination.replace("_", " ").title()}')
    
    plt.suptitle(f'Subject {subject} - Standard Decoding Results', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sub-{subject}_overview.png", dpi=150)
    plt.close()
    
    # Create separate summary plot for hemifield results if they exist
    hemifield_results = []
    for stimulus in ['S1', 'S2']:
        if f"hemifield_voice_{stimulus}" in summary and summary[f"hemifield_voice_{stimulus}"]['status'] == 'success':
            hemifield_results.append(stimulus)
    
    if hemifield_results:
        create_hemifield_summary_plot(subject, summary, hemifield_results)
    
    logging.info(f"Overview plot saved for subject {subject}")

def create_hemifield_summary_plot(subject, summary, hemifield_stimuli):
    """Create summary plot for hemifield results"""
    
    save_dir = f"{RESULTS_DIR}/subject_overviews/"
    
    fig, axes = plt.subplots(1, len(hemifield_stimuli), figsize=(8*len(hemifield_stimuli), 6))
    if len(hemifield_stimuli) == 1:
        axes = [axes]
    
    for idx, stimulus in enumerate(hemifield_stimuli):
        ax = axes[idx]
        key = f"hemifield_voice_{stimulus}"
        
        if key in summary and summary[key]['status'] == 'success':
            # Create bar plot
            hemifields = ['Left', 'Right']
            aucs = [summary[key]['left_peak_auc'], summary[key]['right_peak_auc']]
            accs = [summary[key]['left_peak_accuracy'], summary[key]['right_peak_accuracy']]
            
            x = np.arange(len(hemifields))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, aucs, width, label='AUC', color='blue', alpha=0.7)
            bars2 = ax.bar(x + width/2, accs, width, label='Accuracy', color='green', alpha=0.7)
            
            ax.axhline(0.5, color='k', linestyle='--', alpha=0.5, label='Chance')
            ax.set_ylabel('Peak Performance')
            ax.set_title(f'Hemifield Voice Decoding - {stimulus}')
            ax.set_xticks(x)
            ax.set_xticklabels(hemifields)
            ax.legend()
            ax.set_ylim(0.4, 0.7)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
            
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Subject {subject} - Hemifield-Specific Voice Decoding', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sub-{subject}_hemifield_summary.png", dpi=150)
    plt.close()

def save_individual_excel_dual_metrics(result, save_dir, subject):
    """Save individual subject results to Excel with both AUC and Accuracy"""
    
    # Create main dataframe with time and combined metrics
    excel_data = {
        'Time (s)': result['times'],
        'Combined_AUC': result['combined_bands']['auc'],
        'Combined_Accuracy': result['combined_bands']['accuracy']
    }
    
    # Add each frequency band - both metrics
    for band_name, band_data in result['band_results'].items():
        excel_data[f'AUC_{band_name}'] = band_data['auc']
        excel_data[f'Accuracy_{band_name}'] = band_data['accuracy']
    
    # Save to Excel
    excel_path = os.path.join(save_dir, f"sub_{subject}_dual_metrics.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # Main sheet with all data
        main_df = pd.DataFrame(excel_data)
        main_df.to_excel(writer, sheet_name='All_Results', index=False)
        
        # Separate sheet for AUC only
        auc_columns = ['Time (s)', 'Combined_AUC'] + [col for col in excel_data.keys() if col.startswith('AUC_')]
        auc_df = main_df[auc_columns]
        auc_df.to_excel(writer, sheet_name='AUC_Only', index=False)
        
        # Separate sheet for Accuracy only
        acc_columns = ['Time (s)', 'Combined_Accuracy'] + [col for col in excel_data.keys() if col.startswith('Accuracy_')]
        acc_df = main_df[acc_columns]
        acc_df.to_excel(writer, sheet_name='Accuracy_Only', index=False)
        
        # Summary sheet with peak values
        summary_data = {
            'Metric': [],
            'Band': [],
            'Peak_Value': [],
            'Peak_Time': []
        }
        
        # Find peak values for each metric and band
        times = result['times']
        
        # AUC peaks
        combined_auc_peak_idx = np.argmax(result['combined_bands']['auc'])
        summary_data['Metric'].append('AUC')
        summary_data['Band'].append('Combined')
        summary_data['Peak_Value'].append(result['combined_bands']['auc'][combined_auc_peak_idx])
        summary_data['Peak_Time'].append(times[combined_auc_peak_idx])
        
        for band_name, band_data in result['band_results'].items():
            auc_peak_idx = np.argmax(band_data['auc'])
            summary_data['Metric'].append('AUC')
            summary_data['Band'].append(band_name)
            summary_data['Peak_Value'].append(band_data['auc'][auc_peak_idx])
            summary_data['Peak_Time'].append(times[auc_peak_idx])
        
        # Accuracy peaks
        combined_acc_peak_idx = np.argmax(result['combined_bands']['accuracy'])
        summary_data['Metric'].append('Accuracy')
        summary_data['Band'].append('Combined')
        summary_data['Peak_Value'].append(result['combined_bands']['accuracy'][combined_acc_peak_idx])
        summary_data['Peak_Time'].append(times[combined_acc_peak_idx])
        
        for band_name, band_data in result['band_results'].items():
            acc_peak_idx = np.argmax(band_data['accuracy'])
            summary_data['Metric'].append('Accuracy')
            summary_data['Band'].append(band_name)
            summary_data['Peak_Value'].append(band_data['accuracy'][acc_peak_idx])
            summary_data['Peak_Time'].append(times[acc_peak_idx])
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Peak_Summary', index=False)
    
    logging.info(f"Saved dual metrics to {excel_path}")

if __name__ == "__main__":
    main()