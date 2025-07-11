#!/usr/bin/env python
"""
Group Analysis for Hemifield Voice Decoding
Aggregates individual results and creates group-level plots
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from scipy import stats
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths
HOME_DIR = '/mnt/hpc/projects/awm4/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
RESULTS_DIR = PROCESSED_DIR + 'HemifieldVoiceDecoding/'

# Parameters
N_PERMUTATIONS = 1000
CLUSTER_THRESHOLD = 2.086  # t-value for p < 0.05, df=29

def load_individual_results(subjects, stimulus='S1'):
    """Load all individual hemifield results"""
    
    individual_dir = f"{RESULTS_DIR}/{stimulus}/individual_results/"
    
    all_results = []
    successful_subjects = []
    
    for subject in subjects:
        result_file = f"{individual_dir}/sub-{subject}_hemifield_results.pkl"
        
        if os.path.exists(result_file):
            try:
                with open(result_file, 'rb') as f:
                    result = pickle.load(f)
                    all_results.append(result)
                    successful_subjects.append(subject)
                    logging.info(f"Loaded results for subject {subject}")
            except Exception as e:
                logging.error(f"Error loading subject {subject}: {str(e)}")
        else:
            logging.warning(f"Missing results for subject {subject}")
    
    logging.info(f"Successfully loaded {len(successful_subjects)} subjects for {stimulus}")
    
    return all_results, successful_subjects

def aggregate_results(all_results):
    """Aggregate individual results into group arrays"""
    
    # Get times from first subject
    times = all_results[0]['within_left']['combined']['times']
    n_subjects = len(all_results)
    n_times = len(times)
    
    # Initialize arrays
    within_left_auc = np.zeros((n_subjects, n_times))
    within_right_auc = np.zeros((n_subjects, n_times))
    cross_lr_auc = np.zeros((n_subjects, n_times))
    cross_rl_auc = np.zeros((n_subjects, n_times))
    invariance_index = np.zeros((n_subjects, n_times))
    
    # Collect data from each subject
    for i, result in enumerate(all_results):
        within_left_auc[i] = result['within_left']['combined']['auc']
        within_right_auc[i] = result['within_right']['combined']['auc']
        cross_lr_auc[i] = result['cross_hemifield']['train_left_test_right']['combined']['auc']
        cross_rl_auc[i] = result['cross_hemifield']['train_right_test_left']['combined']['auc']
        invariance_index[i] = result['location_invariance']['invariance_index']
    
    return {
        'times': times,
        'within_left_auc': within_left_auc,
        'within_right_auc': within_right_auc,
        'cross_lr_auc': cross_lr_auc,
        'cross_rl_auc': cross_rl_auc,
        'invariance_index': invariance_index,
        'n_subjects': n_subjects
    }

def perform_statistical_tests(aggregated_data):
    """Perform cluster-based permutation tests"""
    
    results = {}
    
    # Test each condition against chance (0.5)
    conditions = ['within_left_auc', 'within_right_auc', 'cross_lr_auc', 'cross_rl_auc']
    
    for condition in conditions:
        data = aggregated_data[condition] - 0.5  # Center on chance
        t_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
            data, 
            n_permutations=N_PERMUTATIONS,
            threshold=CLUSTER_THRESHOLD,
            tail=1
        )
        results[f'{condition}_vs_chance'] = (clusters, cluster_p_values)
    
    # Test within vs cross comparisons
    # Within average vs Cross average
    within_avg = (aggregated_data['within_left_auc'] + aggregated_data['within_right_auc']) / 2
    cross_avg = (aggregated_data['cross_lr_auc'] + aggregated_data['cross_rl_auc']) / 2
    
    t_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [within_avg, cross_avg],
        n_permutations=N_PERMUTATIONS,
        threshold=CLUSTER_THRESHOLD
    )
    results['within_vs_cross'] = (clusters, cluster_p_values)
    
    # Test invariance index against 0.5 (chance ratio)
    invariance_data = aggregated_data['invariance_index'] - 0.5
    t_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        invariance_data,
        n_permutations=N_PERMUTATIONS,
        threshold=CLUSTER_THRESHOLD,
        tail=1
    )
    results['invariance_vs_chance'] = (clusters, cluster_p_values)
    
    return results

def plot_group_results_comparison(data_s1, data_s2, stats_s1, stats_s2, subjects, save_dir):
    """Create comprehensive group plots comparing S1 and S2"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    times = data_s1['times']
    n_subjects = data_s1['n_subjects']
    
    # Calculate means and SEMs
    def get_mean_sem(data):
        mean = np.mean(data, axis=0)
        sem = np.std(data, axis=0) / np.sqrt(n_subjects)
        return mean, sem
    
    # PLOT 1: Within vs Cross comparison for both stimuli
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # S1 - AUC comparison
    ax = axes[0, 0]
    
    # Within-hemifield
    mean_left_s1, sem_left_s1 = get_mean_sem(data_s1['within_left_auc'])
    mean_right_s1, sem_right_s1 = get_mean_sem(data_s1['within_right_auc'])
    mean_within_s1 = (mean_left_s1 + mean_right_s1) / 2
    sem_within_s1 = np.sqrt(sem_left_s1**2 + sem_right_s1**2) / 2
    
    # Cross-hemifield
    mean_lr_s1, sem_lr_s1 = get_mean_sem(data_s1['cross_lr_auc'])
    mean_rl_s1, sem_rl_s1 = get_mean_sem(data_s1['cross_rl_auc'])
    mean_cross_s1 = (mean_lr_s1 + mean_rl_s1) / 2
    sem_cross_s1 = np.sqrt(sem_lr_s1**2 + sem_rl_s1**2) / 2
    
    ax.plot(times, mean_within_s1, 'b-', label='Within-hemifield', linewidth=2)
    ax.fill_between(times, mean_within_s1 - sem_within_s1, 
                    mean_within_s1 + sem_within_s1, alpha=0.3, color='blue')
    
    ax.plot(times, mean_cross_s1, 'r--', label='Cross-hemifield', linewidth=2)
    ax.fill_between(times, mean_cross_s1 - sem_cross_s1, 
                    mean_cross_s1 + sem_cross_s1, alpha=0.3, color='red')
    
    # Add significance markers
    clusters, p_values = stats_s1['within_vs_cross']
    for cluster, p_val in zip(clusters, p_values):
        if p_val < 0.05:
            cluster_times = times[cluster[0]]
            ax.axvspan(cluster_times[0], cluster_times[-1], alpha=0.1, color='gray')
    
    ax.axhline(0.5, color='k', linestyle=':', alpha=0.5)
    ax.set_ylabel('Voice Decoding AUC')
    ax.set_title(f'S1 Period (N={n_subjects})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 0.65)
    
    # S2 - AUC comparison
    ax = axes[0, 1]
    
    mean_within_s2 = (np.mean(data_s2['within_left_auc'], axis=0) + 
                      np.mean(data_s2['within_right_auc'], axis=0)) / 2
    sem_within_s2 = (np.std(data_s2['within_left_auc'], axis=0) / np.sqrt(n_subjects) + 
                     np.std(data_s2['within_right_auc'], axis=0) / np.sqrt(n_subjects)) / 2
    
    mean_cross_s2 = (np.mean(data_s2['cross_lr_auc'], axis=0) + 
                     np.mean(data_s2['cross_rl_auc'], axis=0)) / 2
    sem_cross_s2 = (np.std(data_s2['cross_lr_auc'], axis=0) / np.sqrt(n_subjects) + 
                    np.std(data_s2['cross_rl_auc'], axis=0) / np.sqrt(n_subjects)) / 2
    
    ax.plot(times, mean_within_s2, 'b-', label='Within-hemifield', linewidth=2)
    ax.fill_between(times, mean_within_s2 - sem_within_s2, 
                    mean_within_s2 + sem_within_s2, alpha=0.3, color='blue')
    
    ax.plot(times, mean_cross_s2, 'r--', label='Cross-hemifield', linewidth=2)
    ax.fill_between(times, mean_cross_s2 - sem_cross_s2, 
                    mean_cross_s2 + sem_cross_s2, alpha=0.3, color='red')
    
    clusters, p_values = stats_s2['within_vs_cross']
    for cluster, p_val in zip(clusters, p_values):
        if p_val < 0.05:
            cluster_times = times[cluster[0]]
            ax.axvspan(cluster_times[0], cluster_times[-1], alpha=0.1, color='gray')
    
    ax.axhline(0.5, color='k', linestyle=':', alpha=0.5)
    ax.set_title(f'S2 Period (N={n_subjects})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 0.65)
    
    # Location Invariance Index comparison
    ax = axes[1, 0]
    
    mean_inv_s1, sem_inv_s1 = get_mean_sem(data_s1['invariance_index'])
    mean_inv_s2, sem_inv_s2 = get_mean_sem(data_s2['invariance_index'])
    
    ax.plot(times, mean_inv_s1, 'purple', label='S1', linewidth=2)
    ax.fill_between(times, mean_inv_s1 - sem_inv_s1, 
                    mean_inv_s1 + sem_inv_s1, alpha=0.3, color='purple')
    
    ax.plot(times, mean_inv_s2, 'orange', label='S2', linewidth=2)
    ax.fill_between(times, mean_inv_s2 - sem_inv_s2, 
                    mean_inv_s2 + sem_inv_s2, alpha=0.3, color='orange')
    
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Perfect invariance')
    ax.axhline(0.5, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Location Invariance Index')
    ax.set_title('Voice Representation Location Invariance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.1)
    
    # Peak performance summary
    ax = axes[1, 1]
    
    # Find peak times
    peak_within_s1_idx = np.argmax(mean_within_s1)
    peak_within_s2_idx = np.argmax(mean_within_s2)
    peak_cross_s1_idx = np.argmax(mean_cross_s1)
    peak_cross_s2_idx = np.argmax(mean_cross_s2)
    
    # Create bar plot
    conditions = ['Within\nS1', 'Cross\nS1', 'Within\nS2', 'Cross\nS2']
    peak_values = [mean_within_s1[peak_within_s1_idx], 
                   mean_cross_s1[peak_cross_s1_idx],
                   mean_within_s2[peak_within_s2_idx], 
                   mean_cross_s2[peak_cross_s2_idx]]
    peak_sems = [sem_within_s1[peak_within_s1_idx], 
                 sem_cross_s1[peak_cross_s1_idx],
                 sem_within_s2[peak_within_s2_idx], 
                 sem_cross_s2[peak_cross_s2_idx]]
    
    x = np.arange(len(conditions))
    colors = ['blue', 'red', 'blue', 'red']
    
    bars = ax.bar(x, peak_values, yerr=peak_sems, color=colors, alpha=0.7, capsize=10)
    ax.axhline(0.5, color='k', linestyle='--', alpha=0.5)
    ax.set_ylabel('Peak AUC')
    ax.set_title('Peak Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0.45, 0.65)
    
    # Add peak time annotations
    for i, (bar, peak_val, peak_idx) in enumerate(zip(bars, peak_values, 
                                                      [peak_within_s1_idx, peak_cross_s1_idx, 
                                                       peak_within_s2_idx, peak_cross_s2_idx])):
        time_val = times[peak_idx]
        ax.text(bar.get_x() + bar.get_width()/2, peak_val + peak_sems[i] + 0.005,
                f'{time_val:.2f}s', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Group Hemifield Voice Decoding Results', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/group_within_vs_cross_comparison.png", dpi=300)
    plt.close()
    
    # PLOT 2: Detailed breakdown by hemifield
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # S1 - Detailed hemifield breakdown
    ax = axes[0, 0]
    
    mean_left_s1, sem_left_s1 = get_mean_sem(data_s1['within_left_auc'])
    mean_right_s1, sem_right_s1 = get_mean_sem(data_s1['within_right_auc'])
    mean_lr_s1, sem_lr_s1 = get_mean_sem(data_s1['cross_lr_auc'])
    mean_rl_s1, sem_rl_s1 = get_mean_sem(data_s1['cross_rl_auc'])
    
    ax.plot(times, mean_left_s1, 'b-', label='Within Left', linewidth=2)
    ax.plot(times, mean_right_s1, 'r-', label='Within Right', linewidth=2)
    ax.plot(times, mean_lr_s1, 'b--', label='Train L→Test R', linewidth=2, alpha=0.7)
    ax.plot(times, mean_rl_s1, 'r--', label='Train R→Test L', linewidth=2, alpha=0.7)
    
    ax.axhline(0.5, color='k', linestyle=':', alpha=0.5)
    ax.set_ylabel('Voice Decoding AUC')
    ax.set_title('S1 - Detailed Hemifield Breakdown')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 0.65)
    
    # S2 - Detailed hemifield breakdown
    ax = axes[0, 1]
    
    mean_left_s2, sem_left_s2 = get_mean_sem(data_s2['within_left_auc'])
    mean_right_s2, sem_right_s2 = get_mean_sem(data_s2['within_right_auc'])
    mean_lr_s2, sem_lr_s2 = get_mean_sem(data_s2['cross_lr_auc'])
    mean_rl_s2, sem_rl_s2 = get_mean_sem(data_s2['cross_rl_auc'])
    
    ax.plot(times, mean_left_s2, 'b-', label='Within Left', linewidth=2)
    ax.plot(times, mean_right_s2, 'r-', label='Within Right', linewidth=2)
    ax.plot(times, mean_lr_s2, 'b--', label='Train L→Test R', linewidth=2, alpha=0.7)
    ax.plot(times, mean_rl_s2, 'r--', label='Train R→Test L', linewidth=2, alpha=0.7)
    
    ax.axhline(0.5, color='k', linestyle=':', alpha=0.5)
    ax.set_title('S2 - Detailed Hemifield Breakdown')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 0.65)
    
    # Difference plots
    ax = axes[1, 0]
    
    # Within-hemifield difference (Right - Left)
    within_diff_s1 = mean_right_s1 - mean_left_s1
    within_diff_s2 = mean_right_s2 - mean_left_s2
    
    ax.plot(times, within_diff_s1, 'purple', label='S1 (R-L)', linewidth=2)
    ax.plot(times, within_diff_s2, 'orange', label='S2 (R-L)', linewidth=2)
    ax.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('AUC Difference')
    ax.set_title('Within-Hemifield Asymmetry (Right - Left)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 0.05)
    
    # Cross-decoding asymmetry
    ax = axes[1, 1]
    
    cross_asym_s1 = mean_lr_s1 - mean_rl_s1
    cross_asym_s2 = mean_lr_s2 - mean_rl_s2
    
    ax.plot(times, cross_asym_s1, 'purple', label='S1 (L→R - R→L)', linewidth=2)
    ax.plot(times, cross_asym_s2, 'orange', label='S2 (L→R - R→L)', linewidth=2)
    ax.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('AUC Difference')
    ax.set_title('Cross-Decoding Asymmetry')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 0.05)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/group_detailed_hemifield_analysis.png", dpi=300)
    plt.close()
    
    # Save summary statistics
    save_summary_statistics(data_s1, data_s2, stats_s1, stats_s2, subjects, save_dir)

def save_summary_statistics(data_s1, data_s2, stats_s1, stats_s2, subjects, save_dir):
    """Save detailed summary statistics"""
    
    times = data_s1['times']
    
    with open(f"{save_dir}/group_summary_statistics.txt", 'w') as f:
        f.write("GROUP HEMIFIELD VOICE DECODING ANALYSIS\n")
        f.write("="*60 + "\n")
        f.write(f"Number of subjects: {len(subjects)}\n")
        f.write(f"Subjects: {subjects}\n\n")
        
        # Peak performance for S1
        f.write("S1 PERIOD:\n")
        f.write("-"*30 + "\n")
        
        mean_within_s1 = (np.mean(data_s1['within_left_auc'], axis=0) + 
                          np.mean(data_s1['within_right_auc'], axis=0)) / 2
        mean_cross_s1 = (np.mean(data_s1['cross_lr_auc'], axis=0) + 
                         np.mean(data_s1['cross_rl_auc'], axis=0)) / 2
        
        peak_within_idx = np.argmax(mean_within_s1)
        peak_cross_idx = np.argmax(mean_cross_s1)
        
        f.write(f"Within-hemifield peak: {mean_within_s1[peak_within_idx]:.3f} at {times[peak_within_idx]:.2f}s\n")
        f.write(f"Cross-hemifield peak: {mean_cross_s1[peak_cross_idx]:.3f} at {times[peak_cross_idx]:.2f}s\n")
        f.write(f"Peak invariance index: {np.max(np.mean(data_s1['invariance_index'], axis=0)):.3f}\n\n")
        
        # Peak performance for S2
        f.write("S2 PERIOD:\n")
        f.write("-"*30 + "\n")
        
        mean_within_s2 = (np.mean(data_s2['within_left_auc'], axis=0) + 
                          np.mean(data_s2['within_right_auc'], axis=0)) / 2
        mean_cross_s2 = (np.mean(data_s2['cross_lr_auc'], axis=0) + 
                         np.mean(data_s2['cross_rl_auc'], axis=0)) / 2
        
        peak_within_idx = np.argmax(mean_within_s2)
        peak_cross_idx = np.argmax(mean_cross_s2)
        
        f.write(f"Within-hemifield peak: {mean_within_s2[peak_within_idx]:.3f} at {times[peak_within_idx]:.2f}s\n")
        f.write(f"Cross-hemifield peak: {mean_cross_s2[peak_cross_idx]:.3f} at {times[peak_cross_idx]:.2f}s\n")
        f.write(f"Peak invariance index: {np.max(np.mean(data_s2['invariance_index'], axis=0)):.3f}\n\n")
        
        # Statistical significance
        f.write("STATISTICAL SIGNIFICANCE:\n")
        f.write("-"*30 + "\n")
        
        # Count significant time points
        def count_sig_timepoints(stats_dict, test_name):
            clusters, p_values = stats_dict[test_name]
            sig_timepoints = 0
            for cluster, p_val in zip(clusters, p_values):
                if p_val < 0.05:
                    sig_timepoints += len(cluster[0])
            return sig_timepoints
        
        f.write("S1 significant timepoints:\n")
        f.write(f"  Within-left > chance: {count_sig_timepoints(stats_s1, 'within_left_auc_vs_chance')}\n")
        f.write(f"  Within-right > chance: {count_sig_timepoints(stats_s1, 'within_right_auc_vs_chance')}\n")
        f.write(f"  Cross L→R > chance: {count_sig_timepoints(stats_s1, 'cross_lr_auc_vs_chance')}\n")
        f.write(f"  Cross R→L > chance: {count_sig_timepoints(stats_s1, 'cross_rl_auc_vs_chance')}\n")
        
        f.write("\nS2 significant timepoints:\n")
        f.write(f"  Within-left > chance: {count_sig_timepoints(stats_s2, 'within_left_auc_vs_chance')}\n")
        f.write(f"  Within-right > chance: {count_sig_timepoints(stats_s2, 'within_right_auc_vs_chance')}\n")
        f.write(f"  Cross L→R > chance: {count_sig_timepoints(stats_s2, 'cross_lr_auc_vs_chance')}\n")
        f.write(f"  Cross R→L > chance: {count_sig_timepoints(stats_s2, 'cross_rl_auc_vs_chance')}\n")

def main():
    """Main function for group analysis"""
    
    # Define subjects (1-30)
    subjects = list(range(1, 31))
    
    # Load results for both stimuli
    logging.info("Loading individual results...")
    
    results_s1, subjects_s1 = load_individual_results(subjects, 'S1')
    results_s2, subjects_s2 = load_individual_results(subjects, 'S2')
    
    # Find common subjects (those with both S1 and S2)
    common_subjects = list(set(subjects_s1) & set(subjects_s2))
    common_subjects.sort()
    
    logging.info(f"Found {len(common_subjects)} subjects with both S1 and S2 results")
    
    # Filter results to common subjects only
    results_s1_filtered = [r for r, s in zip(results_s1, subjects_s1) if s in common_subjects]
    results_s2_filtered = [r for r, s in zip(results_s2, subjects_s2) if s in common_subjects]
    
    # Aggregate results
    logging.info("Aggregating results...")
    data_s1 = aggregate_results(results_s1_filtered)
    data_s2 = aggregate_results(results_s2_filtered)
    
    # Perform statistical tests
    logging.info("Performing statistical tests...")
    stats_s1 = perform_statistical_tests(data_s1)
    stats_s2 = perform_statistical_tests(data_s2)
    
    # Create plots
    logging.info("Creating group plots...")
    save_dir = f"{RESULTS_DIR}/group_analysis/"
    plot_group_results_comparison(data_s1, data_s2, stats_s1, stats_s2, common_subjects, save_dir)
    
    # Save aggregated data
    with open(f"{save_dir}/aggregated_data.pkl", 'wb') as f:
        pickle.dump({
            'subjects': common_subjects,
            'data_s1': data_s1,
            'data_s2': data_s2,
            'stats_s1': stats_s1,
            'stats_s2': stats_s2
        }, f)
    
    logging.info(f"Group analysis complete! Results saved to {save_dir}")

if __name__ == "__main__":
    main()