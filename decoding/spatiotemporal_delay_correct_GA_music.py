#!/usr/bin/env python
"""
Grand Average Script for Delay Period Spatiotemporal Analysis with Behavioral Correctness
Now includes musician level analysis alongside the standard grand average
Handles both "all trials" and "correct trials only" datasets from spatiotemporal_delay_correct.py

Usage: 
  python spatiotemporal_delay_correct_GA.py --analysis_type all_trials
  python spatiotemporal_delay_correct_GA.py --analysis_type correct_only  
  python spatiotemporal_delay_correct_GA.py --analysis_type comparison
"""

import os
import sys
import argparse
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Import MNE for cluster permutation tests
try:
    import mne
    from mne.stats import permutation_cluster_1samp_test
    MNE_AVAILABLE = True
except ImportError:
    print("Warning: MNE not available. Cluster permutation tests will be skipped.")
    MNE_AVAILABLE = False

# Try to import for statistical testing
try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Warning: Statsmodels not available. FDR correction will be skipped.")
    STATSMODELS_AVAILABLE = False

# Parse arguments
parser = argparse.ArgumentParser(description='Grand average delay period analysis with correctness')
parser.add_argument('--analysis_type', type=str, choices=['all_trials', 'correct_only', 'comparison'], 
                   default='all_trials', help='Type of analysis to run')
parser.add_argument('--exclude_subjects', type=str, default='', 
                   help='Comma-separated list of subjects to exclude')
parser.add_argument('--min_subjects', type=int, default=3, 
                   help='Minimum number of subjects required')
parser.add_argument('--stats', action='store_true', 
                   help='Run statistical tests (permutation tests)')
parser.add_argument('--individual_traces', action='store_true',
                   help='Plot individual subject traces')
parser.add_argument('--musician_analysis', action='store_true',
                   help='Run musician level analysis')
parser.add_argument('--verbose', action='store_true',
                   help='Print detailed information')
args = parser.parse_args()

# Set paths
HOME_DIR = '/mnt/hpc/projects/awm4/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'

# Define analysis directories based on analysis type
ANALYSIS_DIRS = {
    'all_trials': PROCESSED_DIR + 'spatiotemporal_delay_400/200Hz/',
    'correct_only': PROCESSED_DIR + 'spatiotemporal_delay_400_correct/50ms200Hz/'
}

# Create output directory
if args.analysis_type == 'comparison':
    OUTPUT_DIR = PROCESSED_DIR + 'spatiotemporal_delay_comparison_GA/'
else:
    OUTPUT_DIR = ANALYSIS_DIRS[args.analysis_type] + 'grand_average_correct/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Analysis parameters
FEATURES = ['maintained_voice', 'maintained_location']
FEATURE_COLORS = {'maintained_voice': '#e41a1c', 'maintained_location': '#377eb8'}
PING_TIME = 3.5

# Musician level colors for each feature
FEATURE_MUSICIAN_COLORS = {
    'maintained_location': {
        1: "#ffcb99",  # Light orange
        2: "#ff996a",  # Medium orange  
        3: "#dc5e23",  # Dark orange
    },
    'maintained_voice': {
        1: "#ffcccb",  # Light red
        2: "#ff6666",  # Medium red
        3: "#cc0000",  # Dark red
    }
}

def setup_logging():
    """Simple logging setup"""
    log_file = f'{OUTPUT_DIR}/ga_log.txt'
    
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            pass
    
    sys.stdout = Logger(log_file)
    print(f"Grand Average analysis started: {datetime.now()}")
    print(f"Analysis type: {args.analysis_type}")

def load_metadata():
    """Load participant metadata including musician categories"""
    print(f"Loading metadata from: {META_FILE}")
    
    if not os.path.exists(META_FILE):
        print(f"Warning: Metadata file not found at {META_FILE}")
        return {}
    
    try:
        meta_df = pd.read_excel(META_FILE)
        print(f"Available columns in metadata: {list(meta_df.columns)}")
        
        # Create a mapping of subject ID to musician category
        musician_mapping = {}
        
        # Try to identify the subject ID column
        subject_col = None
        for col in ['Subject', 'ID', 'subject', 'subject_id', 'SubjectID', 'Participant']:
            if col in meta_df.columns:
                subject_col = col
                break
        
        # Try to identify the musician column
        musician_col = None
        for col in ['Musician', 'musician', 'MusicianLevel', 'Musical_ability']:
            if col in meta_df.columns:
                musician_col = col
                break
        
        if subject_col is None or musician_col is None:
            print(f"Warning: Could not find subject ID or musician columns")
            return {}
        
        print(f"Using '{subject_col}' as subject ID column and '{musician_col}' as musician level column")
        
        # Create mapping
        for idx, row in meta_df.iterrows():
            subject_id = row[subject_col]
            musician_level = row[musician_col]
            
            if pd.notna(subject_id) and pd.notna(musician_level):
                # Convert to string and handle different formats
                subject_id_str = str(int(subject_id)).strip()
                try:
                    musician_level_int = int(float(musician_level))
                    if musician_level_int in [1, 2, 3]:
                        musician_mapping[subject_id_str] = musician_level_int
                        # Also add variations of the subject ID
                        if subject_id_str.startswith('sub-'):
                            musician_mapping[subject_id_str[4:]] = musician_level_int
                        else:
                            musician_mapping[f'sub-{subject_id_str}'] = musician_level_int
                except (ValueError, TypeError):
                    print(f"Warning: Invalid musician level for subject {subject_id}: {musician_level}")
        
        print(f"Successfully mapped {len(musician_mapping)} subject entries")
        print(f"Musician level distribution: Level 1: {sum(1 for v in musician_mapping.values() if v==1)}, "
              f"Level 2: {sum(1 for v in musician_mapping.values() if v==2)}, "
              f"Level 3: {sum(1 for v in musician_mapping.values() if v==3)}")
        
        return musician_mapping
        
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return {}

def discover_and_load_subjects(analysis_dir, analysis_label):
    """Discover subjects and load their results"""
    print(f"\n=== LOADING SUBJECT RESULTS: {analysis_label} ===")
    print(f"Source directory: {analysis_dir}")
    
    # Parse excluded subjects
    excluded = set()
    if args.exclude_subjects:
        excluded = set(map(int, args.exclude_subjects.split(',')))
        print(f"Excluding subjects: {sorted(excluded)}")
    
    # Find subject directories
    subject_dirs = glob.glob(f"{analysis_dir}/subject_*")
    loaded_data = {}
    
    for subject_dir in subject_dirs:
        subject_id = int(Path(subject_dir).name.split('_')[1])
        
        if subject_id in excluded:
            if args.verbose:
                print(f"Subject {subject_id}: Excluded by user")
            continue
        
        # Try to load this subject's results
        subject_data = {}
        success = True
        
        # Check if summary.json exists (indicates completed analysis)
        summary_file = f"{subject_dir}/summary.json"
        if not os.path.exists(summary_file):
            if args.verbose:
                print(f"Subject {subject_id}: No summary.json - analysis incomplete")
            continue
        
        # Load summary to check analysis parameters
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                correct_only_flag = summary.get('correct_only', False)
                if args.verbose:
                    print(f"Subject {subject_id}: correct_only={correct_only_flag}")
        except:
            pass
        
        for feature in FEATURES:
            try:
                # Load the files
                mean_scores = np.load(f"{subject_dir}/{feature}_mean_scores.npy")
                timepoints = np.load(f"{subject_dir}/{feature}_timepoints.npy")
                
                # Basic sanity checks
                if len(mean_scores) == len(timepoints) and len(mean_scores) > 10:
                    # Check for reasonable accuracy values
                    if (np.all(np.isfinite(mean_scores)) and 
                        np.all(mean_scores >= 0) and 
                        np.all(mean_scores <= 1.0)):
                        
                        subject_data[feature] = {
                            'mean_scores': mean_scores,
                            'timepoints': timepoints
                        }
                        
                        if args.verbose:
                            print(f"Subject {subject_id}: {feature} loaded ({len(mean_scores)} timepoints, "
                                  f"range {np.min(timepoints):.3f}-{np.max(timepoints):.3f}s)")
                    else:
                        print(f"Subject {subject_id}: {feature} has invalid accuracy values")
                        success = False
                        break
                else:
                    print(f"Subject {subject_id}: {feature} shape mismatch or too short")
                    success = False
                    break
                    
            except FileNotFoundError:
                print(f"Subject {subject_id}: Missing {feature} files")
                success = False
                break
            except Exception as e:
                print(f"Subject {subject_id}: Error loading {feature} - {e}")
                success = False
                break
        
        if success and len(subject_data) == len(FEATURES):
            loaded_data[subject_id] = subject_data
            print(f"Subject {subject_id}: Successfully loaded")
    
    print(f"\nSuccessfully loaded {len(loaded_data)} subjects: {sorted(loaded_data.keys())}")
    
    if len(loaded_data) < args.min_subjects:
        print(f"ERROR: Only {len(loaded_data)} subjects loaded, need {args.min_subjects}")
        print(f"Make sure you've run spatiotemporal_delay_correct.py for multiple subjects first!")
        return None
    
    return loaded_data

def align_timepoints(all_subjects_data):
    """Align timepoints across subjects using interpolation"""
    print(f"\n=== ALIGNING TIMEPOINTS ===")
    
    aligned_data = {}
    
    for feature in FEATURES:
        print(f"Processing {feature}...")
        
        # Collect all timepoints and scores for this feature
        all_timepoints = []
        all_scores = []
        valid_subjects = []
        
        for subject_id, subject_data in all_subjects_data.items():
            if feature in subject_data:
                timepoints = subject_data[feature]['timepoints']
                scores = subject_data[feature]['mean_scores']
                
                all_timepoints.append(timepoints)
                all_scores.append(scores)
                valid_subjects.append(subject_id)
        
        if len(all_timepoints) < args.min_subjects:
            print(f"  WARNING: Not enough subjects for {feature} ({len(all_timepoints)} < {args.min_subjects})")
            continue
        
        # Check if timepoints are already aligned
        reference_timepoints = all_timepoints[0]
        aligned = True
        
        for tp in all_timepoints[1:]:
            if not np.allclose(tp, reference_timepoints, atol=1e-6):
                aligned = False
                break
        
        if aligned:
            print(f"  Timepoints already aligned across {len(valid_subjects)} subjects")
            common_timepoints = reference_timepoints
            aligned_scores = np.array(all_scores)
        else:
            print(f"  Timepoint mismatch detected - interpolating...")
            
            # Find common timepoint range
            min_start = max(np.min(tp) for tp in all_timepoints)
            max_end = min(np.max(tp) for tp in all_timepoints)
            
            print(f"  Common range: {min_start:.3f}s to {max_end:.3f}s")
            
            # Create common timepoint grid (10ms resolution)
            common_timepoints = np.arange(min_start, max_end + 0.005, 0.01)
            
            # Interpolate all subjects to common grid
            interpolated_scores = []
            
            for i, (timepoints, scores) in enumerate(zip(all_timepoints, all_scores)):
                interp_func = interp1d(timepoints, scores, kind='linear', 
                                     bounds_error=False, fill_value='extrapolate')
                interpolated = interp_func(common_timepoints)
                interpolated_scores.append(interpolated)
                
                if args.verbose:
                    print(f"    Subject {valid_subjects[i]}: {len(timepoints)} → {len(common_timepoints)} points")
            
            aligned_scores = np.array(interpolated_scores)
        
        # Store aligned data
        aligned_data[feature] = {
            'timepoints': common_timepoints,
            'individual_scores': aligned_scores,
            'valid_subjects': valid_subjects,
            'n_subjects': len(valid_subjects)
        }
        
        print(f"  Final: {len(valid_subjects)} subjects, {len(common_timepoints)} timepoints")
    
    return aligned_data

def organize_by_musician_level(aligned_data, musician_mapping):
    """Organize aligned data by musician level"""
    print(f"\n=== ORGANIZING BY MUSICIAN LEVEL ===")
    
    if not musician_mapping:
        print("No musician mapping available")
        return None
    
    musician_data = {}
    
    for feature in FEATURES:
        if feature not in aligned_data:
            continue
        
        print(f"Processing {feature} by musician level...")
        
        data = aligned_data[feature]
        musician_data[feature] = {1: [], 2: [], 3: []}
        
        # Organize subjects by musician level
        for i, subject_id in enumerate(data['valid_subjects']):
            subject_id_str = str(subject_id)
            
            # Try different variations to match with musician mapping
            musician_level = None
            for id_variant in [subject_id_str, f'sub-{subject_id_str}', 
                             subject_id_str.zfill(2), f'sub-{subject_id_str.zfill(2)}']:
                if id_variant in musician_mapping:
                    musician_level = musician_mapping[id_variant]
                    break
            
            if musician_level is not None:
                musician_data[feature][musician_level].append({
                    'subject_id': subject_id,
                    'scores': data['individual_scores'][i],
                    'timepoints': data['timepoints']
                })
                if args.verbose:
                    print(f"  Subject {subject_id} -> Musician Level {musician_level}")
            else:
                if args.verbose:
                    print(f"  Subject {subject_id} -> No musician level found")
        
        # Convert to numpy arrays and add metadata
        for level in [1, 2, 3]:
            if musician_data[feature][level]:
                scores_array = np.array([subj['scores'] for subj in musician_data[feature][level]])
                musician_data[feature][level] = {
                    'scores': scores_array,
                    'timepoints': data['timepoints'],
                    'n_subjects': len(scores_array),
                    'subject_ids': [subj['subject_id'] for subj in musician_data[feature][level]]
                }
                print(f"  {feature} Level {level}: {len(scores_array)} subjects")
            else:
                musician_data[feature][level] = None
                print(f"  {feature} Level {level}: 0 subjects")
    
    return musician_data

def compute_grand_averages(aligned_data):
    """Compute grand averages with confidence intervals"""
    print(f"\n=== COMPUTING GRAND AVERAGES ===")
    
    results = {}
    
    for feature in FEATURES:
        if feature not in aligned_data:
            print(f"Skipping {feature} - no aligned data")
            continue
        
        print(f"Computing {feature}...")
        
        data = aligned_data[feature]
        scores_matrix = data['individual_scores']  # Shape: (n_subjects, n_timepoints)
        
        # Compute basic statistics
        grand_mean = np.mean(scores_matrix, axis=0)
        grand_std = np.std(scores_matrix, axis=0)
        grand_sem = grand_std / np.sqrt(data['n_subjects'])
        
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_indices = np.random.choice(data['n_subjects'], data['n_subjects'], replace=True)
            bootstrap_sample = scores_matrix[bootstrap_indices]
            bootstrap_means.append(np.mean(bootstrap_sample, axis=0))
        
        bootstrap_means = np.array(bootstrap_means)
        ci_lower = np.percentile(bootstrap_means, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_means, 97.5, axis=0)
        
        # Store results
        results[feature] = {
            'timepoints': data['timepoints'],
            'grand_mean': grand_mean,
            'grand_std': grand_std,
            'grand_sem': grand_sem,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'individual_scores': scores_matrix,
            'valid_subjects': data['valid_subjects'],
            'n_subjects': data['n_subjects']
        }
        
        # Summary stats
        mean_acc = np.mean(grand_mean)
        peak_acc = np.max(grand_mean)
        peak_time = data['timepoints'][np.argmax(grand_mean)]
        
        # Find ping accuracy
        ping_idx = np.argmin(np.abs(data['timepoints'] - PING_TIME))
        ping_acc = grand_mean[ping_idx]
        
        print(f"  {feature}: n={data['n_subjects']}")
        print(f"    Mean accuracy: {mean_acc:.3f}")
        print(f"    Peak accuracy: {peak_acc:.3f} @ {peak_time:.2f}s")
        print(f"    Ping accuracy: {ping_acc:.3f}")
    
    return results

def run_cluster_permutation_test(subject_array, n_permutations=1000):
    """
    Run cluster permutation test against chance (0.5)
    
    Parameters:
    - subject_array: shape (n_subjects, n_timepoints)
    - n_permutations: number of permutations for test
    
    Returns:
    - T_obs: observed test statistic
    - clusters: list of cluster masks
    - p_values: p-values for each cluster
    - H0: null distribution
    """
    try:
        # Test against chance level (0.5)
        T_obs, clusters, p_values, H0 = permutation_cluster_1samp_test(
            subject_array - 0.5,
            n_permutations=n_permutations,
            threshold=None,
            tail=1,  # One-tailed test (above chance)
            out_type='mask',
            seed=42
        )
        
        return T_obs, clusters, p_values, H0
    
    except Exception as e:
        print(f"    Error in cluster test: {str(e)}")
        return None, None, None, None

def run_cluster_test_wrapper(scores, window_centers, feature_name, n_permutations=10000):
    """Wrapper for cluster permutation test"""
    if not MNE_AVAILABLE:
        print("MNE not available - skipping cluster test")
        return None
    
    print(f"  Running cluster permutation test with {n_permutations} permutations...")
    
    # Run the cluster test
    T_obs, clusters, cluster_p_values, H0 = run_cluster_permutation_test(scores, n_permutations)
    
    if T_obs is None:
        return None
    
    # Extract cluster information
    cluster_info = []
    if len(clusters) > 0:
        for c_idx, (cluster, p_val) in enumerate(zip(clusters, cluster_p_values)):
            cluster_mask = cluster[0]
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 0:
                start_idx = cluster_indices[0]
                end_idx = cluster_indices[-1]
                start_time = window_centers[start_idx]
                end_time = window_centers[end_idx]
                cluster_info.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time': start_time,
                    'end_time': end_time,
                    'p_value': p_val,
                    'mask': cluster_mask,
                    'indices': cluster_indices,
                    'cluster_idx': c_idx
                })
                
                print(f"  Cluster {c_idx}: p={p_val:.4f}, time={start_time:.2f}s-{end_time:.2f}s")
    
    # Find significant clusters (p < 0.05)
    sig_clusters = [ci for ci in cluster_info if ci['p_value'] < 0.05]
    
    print(f"  Cluster results: {len(sig_clusters)}/{len(cluster_info)} significant clusters")
    
    return {
        'clusters': clusters,
        'cluster_p_values': cluster_p_values,
        'cluster_info': cluster_info,
        'sig_clusters': sig_clusters,
        'T_obs': T_obs
    }

def run_sign_flipping_test(scores, window_centers, feature_name, 
                          n_permutations=10000, alpha=0.05):
    """
    One-sided sign flipping permutation test with FDR correction
    """
    print(f"  Running sign flipping test with FDR correction...")
    
    n_subjects, n_timepoints = scores.shape
    
    # Observed test statistics (t-statistics at each timepoint)
    observed_diff = scores - 0.5  # Difference from chance
    observed_t_stats = np.zeros(n_timepoints)
    
    for t_idx in range(n_timepoints):
        t_stat, _ = stats.ttest_1samp(observed_diff[:, t_idx], 0)
        observed_t_stats[t_idx] = t_stat
    
    # Permutation procedure: sign flipping
    print(f"  Running {n_permutations} permutations...")
    null_t_stats = np.zeros((n_permutations, n_timepoints))
    
    for perm_idx in range(n_permutations):
        if perm_idx % 2000 == 0 and perm_idx > 0:
            print(f"    Permutation {perm_idx}/{n_permutations}")
        
        # Random sign flips for each subject
        sign_flips = np.random.choice([-1, 1], size=n_subjects)
        
        # Apply sign flips to difference scores
        permuted_diff = observed_diff * sign_flips[:, np.newaxis]
        
        # Calculate t-statistics for this permutation
        for t_idx in range(n_timepoints):
            t_stat, _ = stats.ttest_1samp(permuted_diff[:, t_idx], 0)
            null_t_stats[perm_idx, t_idx] = t_stat
    
    # Calculate uncorrected p-values (one-sided: observed > null)
    p_values_uncorrected = np.zeros(n_timepoints)
    for t_idx in range(n_timepoints):
        # Count how many permutations had t-stats >= observed
        n_greater = np.sum(null_t_stats[:, t_idx] >= observed_t_stats[t_idx])
        p_values_uncorrected[t_idx] = (n_greater + 1) / (n_permutations + 1)  # Add 1 for observed
    
    # Apply multiple comparison correction
    print(f"  Applying FDR correction for {n_timepoints} comparisons...")
    
    if STATSMODELS_AVAILABLE:
        # FDR correction using Benjamini-Hochberg
        rejected, p_values_corrected, alpha_sidak, alpha_bonf = multipletests(
            p_values_uncorrected, alpha=alpha, method='fdr_bh'
        )
        sig_mask_corrected = rejected
        correction_method = 'FDR (Benjamini-Hochberg)'
        print(f"  FDR correction applied successfully")
    else:
        # Fallback to Bonferroni correction
        bonferroni_alpha = alpha / n_timepoints
        sig_mask_corrected = p_values_uncorrected < bonferroni_alpha
        p_values_corrected = np.minimum(p_values_uncorrected * n_timepoints, 1.0)
        correction_method = 'Bonferroni'
        print(f"  Statsmodels unavailable - using Bonferroni correction")
    
    # Uncorrected significance mask for comparison
    sig_mask_uncorrected = p_values_uncorrected < alpha
    
    results = {
        'observed_t_stats': observed_t_stats,
        'p_values_uncorrected': p_values_uncorrected,
        'p_values_corrected': p_values_corrected,
        'significant_mask': sig_mask_corrected,  # Use corrected by default
        'significant_mask_uncorrected': sig_mask_uncorrected,
        'significant_mask_corrected': sig_mask_corrected,
        'null_distribution': null_t_stats,
        'n_significant': np.sum(sig_mask_corrected),
        'n_significant_uncorrected': np.sum(sig_mask_uncorrected),
        'alpha': alpha,
        'correction_method': correction_method,
        'timepoints': window_centers
    }
    
    print(f"  Sign flipping results:")
    print(f"    Uncorrected: {np.sum(sig_mask_uncorrected)}/{n_timepoints} significant timepoints")
    print(f"    {correction_method}: {np.sum(sig_mask_corrected)}/{n_timepoints} significant timepoints")
    
    return results

def run_permutation_tests(results):
    """Run both cluster and sign flipping permutation tests"""
    print(f"\n=== PERMUTATION TESTS ===")
    
    stats_results = {}
    
    for feature in FEATURES:
        if feature not in results:
            continue
        
        print(f"Testing {feature}...")
        
        data = results[feature]
        scores = data['individual_scores']  # subjects × timepoints
        timepoints = data['timepoints']
        
        print(f"  Data shape: {scores.shape[0]} subjects × {scores.shape[1]} timepoints")
        
        # 1. Cluster permutation test
        print(f"  --- CLUSTER PERMUTATION TEST ---")
        cluster_results = run_cluster_test_wrapper(scores, timepoints, feature)
        
        # 2. Sign flipping test  
        print(f"  --- SIGN FLIPPING TEST ---")
        sign_flip_results = run_sign_flipping_test(scores, timepoints, feature)
        
        stats_results[feature] = {
            'cluster': cluster_results,
            'sign_flip': sign_flip_results
        }
    
    return stats_results

def create_plots(results, stats_results=None, analysis_label=""):
    """Create plots for single analysis type"""
    print(f"\n=== CREATING PLOTS: {analysis_label} ===")
    
    # Main comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    title_suffix = f" ({analysis_label})" if analysis_label else ""
    fig.suptitle(f'Delay Period Grand Average - Working Memory Maintenance{title_suffix}', 
                 fontsize=16, fontweight='bold')
    
    for idx, feature in enumerate(FEATURES):
        if feature not in results:
            axes[idx].text(0.5, 0.5, f'No data for {feature}', ha='center', va='center', transform=axes[idx].transAxes)
            continue
        
        ax = axes[idx]
        data = results[feature]
        color = FEATURE_COLORS[feature]
        
        # Plot individual traces if requested
        if args.individual_traces:
            for i, subject_scores in enumerate(data['individual_scores']):
                alpha = 0.3 if data['n_subjects'] > 10 else 0.4
                ax.plot(data['timepoints'], subject_scores, color='lightgray', alpha=alpha, linewidth=0.7)
        
        # Plot grand average
        ax.plot(data['timepoints'], data['grand_mean'], 
               color=color, linewidth=3, label=f'Grand Average (n={data["n_subjects"]})')
        
        # Add SEM confidence intervals only (no bootstrap)
        ax.fill_between(data['timepoints'],
                       data['grand_mean'] - data['grand_sem'],
                       data['grand_mean'] + data['grand_sem'],
                       color=color, alpha=0.3)
        
        # Add permutation test results
        if stats_results and feature in stats_results:
            feature_stats = stats_results[feature]
            
            # Debug print
            if args.verbose:
                print(f"  Plotting stats for {feature}:")
                if feature_stats['cluster']:
                    print(f"    Clusters: {len(feature_stats['cluster']['sig_clusters']) if feature_stats['cluster']['sig_clusters'] else 0}")
                if feature_stats['sign_flip']:
                    print(f"    Sign flip significant: {np.sum(feature_stats['sign_flip']['significant_mask'])}")
            
            # 1. Cluster results - horizontal bars at y=0.57
            if feature_stats['cluster'] and feature_stats['cluster']['sig_clusters']:
                cluster_y = 0.57
                for cluster in feature_stats['cluster']['sig_clusters']:
                    ax.plot([cluster['start_time'], cluster['end_time']], 
                           [cluster_y, cluster_y], 
                           color=color, linewidth=6, alpha=0.8,
                           solid_capstyle='round')
                    # Add p-value text
                    mid_time = (cluster['start_time'] + cluster['end_time']) / 2
                    ax.text(mid_time, cluster_y + 0.005, f'p={cluster["p_value"]:.3f}', 
                           va='bottom', ha='center', fontsize=8, 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # 2. Sign flipping results - dots at y=0.59
            if feature_stats['sign_flip'] is not None:
                sign_results = feature_stats['sign_flip']
                sig_mask = sign_results['significant_mask']
                n_sig = np.sum(sig_mask)
                
                if args.verbose:
                    print(f"    Sign flip mask shape: {sig_mask.shape}, n_significant: {n_sig}")
                
                if n_sig > 0:
                    sig_timepoints = sign_results['timepoints'][sig_mask]
                    sig_y = np.full(len(sig_timepoints), 0.59)
                    
                    if args.verbose:
                        print(f"    Plotting {len(sig_timepoints)} sign flip points")
                    
                    ax.scatter(sig_timepoints, sig_y, color=color, s=30, alpha=0.9, 
                             marker='o', edgecolor='white', linewidth=0.8, zorder=10)
        
        # Reference lines
        ax.axvline(x=2.0, color='green', linestyle='--', alpha=0.7, linewidth=2)
        ax.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.7, linewidth=1)
        
        # Formatting
        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel('Decoding Accuracy', fontsize=14)
        title = feature.replace('maintained_', '').replace('_', ' ').title()
        ax.set_title(f'{title} Maintenance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set y-limits to accommodate statistical markers
        ax.set_ylim(0.45, 0.60)
        
        # Stats summary box
        mean_acc = np.mean(data['grand_mean'])
        peak_acc = np.max(data['grand_mean'])
        peak_time = data['timepoints'][np.argmax(data['grand_mean'])]
        
        stats_text = f"n = {data['n_subjects']}\nMean: {mean_acc:.3f}\nPeak: {peak_acc:.3f}\n@ {peak_time:.2f}s"
        
        # Add statistical results to summary
        if stats_results and feature in stats_results:
            feature_stats = stats_results[feature]
            if feature_stats['cluster'] and feature_stats['cluster']['sig_clusters']:
                n_clusters = len(feature_stats['cluster']['sig_clusters'])
                stats_text += f"\nClusters: {n_clusters}"
            
            if feature_stats['sign_flip'] and np.any(feature_stats['sign_flip']['significant_mask']):
                n_sign_flip = np.sum(feature_stats['sign_flip']['significant_mask'])
                stats_text += f"\nSign flip: {n_sign_flip}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save with analysis-specific naming
    if analysis_label:
        filename = f'{OUTPUT_DIR}/delay_grand_average_{analysis_label.lower().replace(" ", "_")}.png'
    else:
        filename = f'{OUTPUT_DIR}/delay_grand_average.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Grand average plot saved: {filename}")

def create_musician_plots(musician_data, analysis_label=""):
    """Create plots by musician level"""
    print(f"\n=== CREATING MUSICIAN LEVEL PLOTS: {analysis_label} ===")
    
    if not musician_data:
        print("No musician data available")
        return
    
    # Create musician-specific output directory
    musician_output_dir = f'{OUTPUT_DIR}/musician_level_analysis/'
    os.makedirs(musician_output_dir, exist_ok=True)
    
    # Create individual plots for each feature
    for feature in FEATURES:
        if feature not in musician_data:
            continue
        
        plt.figure(figsize=(12, 7))
        
        # Plot each musician level
        for level in [1, 2, 3]:
            if musician_data[feature][level] is not None:
                data = musician_data[feature][level]
                all_scores = data['scores']
                n_subjects = data['n_subjects']
                timepoints = data['timepoints']
                
                mean_scores = np.mean(all_scores, axis=0)
                std_error = np.std(all_scores, axis=0) / np.sqrt(n_subjects)
                
                color = FEATURE_MUSICIAN_COLORS[feature][level]
                
                plt.plot(timepoints, mean_scores,
                        label=f'Level {level} (N={n_subjects})',
                        color=color, linewidth=2)
                
                plt.fill_between(timepoints,
                                mean_scores - std_error,
                                mean_scores + std_error,
                                alpha=0.2, color=color)
        
        # Add reference lines
        plt.axvline(x=2.0, color='green', linestyle='--', alpha=0.7, linewidth=2)
        plt.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7, linewidth=2)
        plt.axhline(y=0.5, color='black', linestyle=':', alpha=0.7, linewidth=1)
        
        # Formatting
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Decoding Accuracy', fontsize=14)
        title = feature.replace('maintained_', '').replace('_', ' ').title()
        plt.title(f'{title} Maintenance by Musician Level{" (" + analysis_label + ")" if analysis_label else ""}', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(0.45, 0.60)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        if analysis_label:
            filename = f'{musician_output_dir}/{feature}_musician_levels_{analysis_label.lower().replace(" ", "_")}.png'
        else:
            filename = f'{musician_output_dir}/{feature}_musician_levels.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        
        print(f"Musician level plot for {feature} saved: {filename}")
    
    # Create comparison plot across features by musician level
    for level in [1, 2, 3]:
        plt.figure(figsize=(12, 7))
        
        level_has_data = False
        for feature in FEATURES:
            if (feature in musician_data and 
                musician_data[feature][level] is not None):
                
                data = musician_data[feature][level]
                all_scores = data['scores']
                n_subjects = data['n_subjects']
                timepoints = data['timepoints']
                
                mean_scores = np.mean(all_scores, axis=0)
                std_error = np.std(all_scores, axis=0) / np.sqrt(n_subjects)
                
                color = FEATURE_COLORS[feature]
                feature_name = feature.replace('maintained_', '').replace('_', ' ').title()
                
                plt.plot(timepoints, mean_scores,
                        label=f'{feature_name} (N={n_subjects})',
                        color=color, linewidth=2)
                
                plt.fill_between(timepoints,
                                mean_scores - std_error,
                                mean_scores + std_error,
                                alpha=0.2, color=color)
                level_has_data = True
        
        if level_has_data:
            # Add reference lines
            plt.axvline(x=2.0, color='green', linestyle='--', alpha=0.7, linewidth=2)
            plt.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7, linewidth=2)
            plt.axhline(y=0.5, color='black', linestyle=':', alpha=0.7, linewidth=1)
            
            # Formatting
            plt.xlabel('Time (s)', fontsize=14)
            plt.ylabel('Decoding Accuracy', fontsize=14)
            plt.title(f'Working Memory Maintenance - Musician Level {level}{" (" + analysis_label + ")" if analysis_label else ""}', 
                     fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.ylim(0.45, 0.60)
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            if analysis_label:
                filename = f'{musician_output_dir}/feature_comparison_level_{level}_{analysis_label.lower().replace(" ", "_")}.png'
            else:
                filename = f'{musician_output_dir}/feature_comparison_level_{level}.png'
            
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
            plt.close()
            
            print(f"Feature comparison plot for Level {level} saved: {filename}")

def create_comparison_plots(all_trials_results, correct_only_results):
    """Create comparison plots between all trials and correct trials only"""
    print(f"\n=== CREATING COMPARISON PLOTS ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Delay Period Comparison: All Trials vs Correct Trials Only', 
                 fontsize=16, fontweight='bold')
    
    for idx, feature in enumerate(FEATURES):
        # Check if feature is in both datasets
        if feature not in all_trials_results or feature not in correct_only_results:
            print(f"Skipping {feature} - missing in one dataset")
            continue
        
        color = FEATURE_COLORS[feature]
        feature_name = feature.replace('maintained_', '').replace('_', ' ').title()
        
        # Top row: Overlay comparison
        ax_overlay = axes[0, idx]
        
        # All trials
        all_data = all_trials_results[feature]
        ax_overlay.plot(all_data['timepoints'], all_data['grand_mean'], 
                       color=color, linewidth=3, alpha=0.8, label=f'All Trials (n={all_data["n_subjects"]})')
        ax_overlay.fill_between(all_data['timepoints'],
                               all_data['grand_mean'] - all_data['grand_sem'],
                               all_data['grand_mean'] + all_data['grand_sem'],
                               color=color, alpha=0.2)
        
        # Correct trials only
        correct_data = correct_only_results[feature]
        ax_overlay.plot(correct_data['timepoints'], correct_data['grand_mean'], 
                       color=color, linewidth=3, linestyle='--', alpha=0.8, 
                       label=f'Correct Only (n={correct_data["n_subjects"]})')
        ax_overlay.fill_between(correct_data['timepoints'],
                               correct_data['grand_mean'] - correct_data['grand_sem'],
                               correct_data['grand_mean'] + correct_data['grand_sem'],
                               color=color, alpha=0.2, linestyle='--')
        
        # Reference lines
        ax_overlay.axvline(x=2.0, color='green', linestyle='--', alpha=0.7)
        ax_overlay.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7)
        ax_overlay.axhline(y=0.5, color='black', linestyle=':', alpha=0.7)
        
        ax_overlay.set_xlabel('Time (s)', fontsize=12)
        ax_overlay.set_ylabel('Decoding Accuracy', fontsize=12)
        ax_overlay.set_title(f'{feature_name} Comparison', fontsize=13, fontweight='bold')
        ax_overlay.grid(True, alpha=0.3)
        ax_overlay.set_ylim(0.45, 0.60)
        ax_overlay.legend(fontsize=9)
        
        # Bottom row: Difference plot
        ax_diff = axes[1, idx]
        
        # Interpolate to common timepoints for difference calculation
        common_timepoints = all_data['timepoints']
        if not np.array_equal(all_data['timepoints'], correct_data['timepoints']):
            # Find common range
            min_start = max(np.min(all_data['timepoints']), np.min(correct_data['timepoints']))
            max_end = min(np.max(all_data['timepoints']), np.max(correct_data['timepoints']))
            common_timepoints = np.arange(min_start, max_end + 0.005, 0.01)
            
            # Interpolate both
            interp_all = interp1d(all_data['timepoints'], all_data['grand_mean'], kind='linear')
            interp_correct = interp1d(correct_data['timepoints'], correct_data['grand_mean'], kind='linear')
            
            all_interpolated = interp_all(common_timepoints)
            correct_interpolated = interp_correct(common_timepoints)
        else:
            all_interpolated = all_data['grand_mean']
            correct_interpolated = correct_data['grand_mean']
        
        # Calculate difference
        difference = correct_interpolated - all_interpolated
        
        ax_diff.plot(common_timepoints, difference, color=color, linewidth=3, 
                    label='Correct Only - All Trials')
        ax_diff.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
        ax_diff.axvline(x=2.0, color='green', linestyle='--', alpha=0.7)
        ax_diff.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7)
        
        ax_diff.set_xlabel('Time (s)', fontsize=12)
        ax_diff.set_ylabel('Accuracy Difference', fontsize=12)
        ax_diff.set_title(f'{feature_name} Difference', fontsize=13, fontweight='bold')
        ax_diff.grid(True, alpha=0.3)
        ax_diff.legend(fontsize=9)
        
        # Add stats
        mean_diff = np.mean(difference)
        max_diff = np.max(np.abs(difference))
        ax_diff.text(0.02, 0.98, f'Mean Δ: {mean_diff:.3f}\nMax |Δ|: {max_diff:.3f}', 
                    transform=ax_diff.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/delay_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/delay_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {OUTPUT_DIR}/delay_comparison.png")

def save_results(results, stats_results=None, analysis_label="", musician_data=None):
    """Save results"""
    print(f"\n=== SAVING RESULTS: {analysis_label} ===")
    
    # Save arrays for each feature
    for feature in FEATURES:
        if feature not in results:
            continue
        
        data = results[feature]
        
        if analysis_label:
            prefix = f"{OUTPUT_DIR}/{feature}_{analysis_label.lower().replace(' ', '_')}"
        else:
            prefix = f"{OUTPUT_DIR}/{feature}"
        
        # Save main arrays
        np.save(f"{prefix}_timepoints.npy", data['timepoints'])
        np.save(f"{prefix}_grand_mean.npy", data['grand_mean'])
        np.save(f"{prefix}_grand_sem.npy", data['grand_sem'])
        np.save(f"{prefix}_grand_std.npy", data['grand_std'])
        np.save(f"{prefix}_ci_lower.npy", data['ci_lower'])
        np.save(f"{prefix}_ci_upper.npy", data['ci_upper'])
        np.save(f"{prefix}_individual_scores.npy", data['individual_scores'])
        
        print(f"Saved {feature} arrays")
    
    # Save musician data if available
    if musician_data:
        musician_output_dir = f'{OUTPUT_DIR}/musician_level_analysis/'
        os.makedirs(musician_output_dir, exist_ok=True)
        
        for feature in FEATURES:
            if feature not in musician_data:
                continue
            
            for level in [1, 2, 3]:
                if musician_data[feature][level] is not None:
                    data = musician_data[feature][level]
                    if analysis_label:
                        prefix = f"{musician_output_dir}/{feature}_level_{level}_{analysis_label.lower().replace(' ', '_')}"
                    else:
                        prefix = f"{musician_output_dir}/{feature}_level_{level}"
                    
                    np.save(f"{prefix}_timepoints.npy", data['timepoints'])
                    np.save(f"{prefix}_scores.npy", data['scores'])
                    
                    # Save subject IDs as well
                    with open(f"{prefix}_subject_ids.txt", 'w') as f:
                        for subj_id in data['subject_ids']:
                            f.write(f"{subj_id}\n")
    
    # Create summary
    summary = {
        'analysis_info': {
            'timestamp': str(datetime.now()),
            'analysis_type': args.analysis_type,
            'analysis_label': analysis_label,
            'method': 'delay_period_grand_average_with_correctness',
            'source_analysis': 'spatiotemporal_delay_correct.py results',
            'window_parameters': '400ms windows, 10ms steps',
            'analysis_period': '2.0-4.5s delay period',
            'excluded_subjects': args.exclude_subjects,
            'statistical_testing_enabled': args.stats,
            'musician_analysis_enabled': args.musician_analysis,
            'individual_traces_plotted': args.individual_traces
        },
        'results': {}
    }
    
    # Add results for each feature
    for feature in FEATURES:
        if feature not in results:
            continue
        
        data = results[feature]
        mean_acc = float(np.mean(data['grand_mean']))
        peak_acc = float(np.max(data['grand_mean']))
        peak_time = float(data['timepoints'][np.argmax(data['grand_mean'])])
        
        # Find ping accuracy
        ping_idx = np.argmin(np.abs(data['timepoints'] - PING_TIME))
        ping_acc = float(data['grand_mean'][ping_idx])
        
        # Time above chance
        above_chance_mask = data['grand_mean'] > 0.5
        time_above_chance = float(np.sum(above_chance_mask) * 0.01)
        
        result_summary = {
            'subjects_included': data['valid_subjects'],
            'n_subjects': data['n_subjects'],
            'descriptive_statistics': {
                'mean_accuracy': mean_acc,
                'peak_accuracy': peak_acc,
                'peak_time': peak_time,
                'ping_accuracy': ping_acc,
                'time_above_chance_seconds': time_above_chance,
                'time_range': [float(np.min(data['timepoints'])), float(np.max(data['timepoints']))],
                'n_timepoints': len(data['timepoints'])
            }
        }
        
        # Add statistical results
        if stats_results and feature in stats_results:
            feature_stats = stats_results[feature]
            result_summary['permutation_tests'] = {}
            
            if feature_stats['cluster']:
                cluster_data = feature_stats['cluster']
                result_summary['permutation_tests']['cluster_permutation'] = {
                    'n_clusters_total': len(cluster_data['cluster_info']) if cluster_data['cluster_info'] else 0,
                    'n_clusters_significant': len(cluster_data['sig_clusters']) if cluster_data['sig_clusters'] else 0,
                    'significant_clusters': []
                }
                
                if cluster_data['sig_clusters']:
                    for cluster in cluster_data['sig_clusters']:
                        result_summary['permutation_tests']['cluster_permutation']['significant_clusters'].append({
                            'start_time': float(cluster['start_time']),
                            'end_time': float(cluster['end_time']),
                            'duration': float(cluster['end_time'] - cluster['start_time']),
                            'p_value': float(cluster['p_value'])
                        })
            
            if feature_stats['sign_flip']:
                sign_data = feature_stats['sign_flip']
                result_summary['permutation_tests']['sign_flipping'] = {
                    'n_significant_timepoints': int(sign_data['n_significant']),
                    'n_significant_timepoints_uncorrected': int(sign_data['n_significant_uncorrected']),
                    'total_timepoints': len(sign_data['timepoints']),
                    'alpha': sign_data['alpha'],
                    'correction_method': sign_data['correction_method']
                }
        
        # Add musician level results
        if musician_data and feature in musician_data:
            result_summary['musician_level_analysis'] = {}
            for level in [1, 2, 3]:
                if musician_data[feature][level] is not None:
                    level_data = musician_data[feature][level]
                    level_scores = level_data['scores']
                    level_mean = np.mean(level_scores)
                    
                    result_summary['musician_level_analysis'][f'level_{level}'] = {
                        'n_subjects': level_data['n_subjects'],
                        'subject_ids': level_data['subject_ids'],
                        'mean_accuracy': float(level_mean),
                        'peak_accuracy': float(np.max(np.mean(level_scores, axis=0))),
                        'peak_time': float(level_data['timepoints'][np.argmax(np.mean(level_scores, axis=0))])
                    }
        
        summary['results'][feature] = result_summary
    
    # Save summary
    if analysis_label:
        summary_file = f'{OUTPUT_DIR}/grand_average_summary_{analysis_label.lower().replace(" ", "_")}.json'
    else:
        summary_file = f'{OUTPUT_DIR}/grand_average_summary.json'
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved: {summary_file}")

def main():
    """Main function"""
    setup_logging()
    
    print("=== DELAY PERIOD GRAND AVERAGE ANALYSIS WITH BEHAVIORAL CORRECTNESS ===")
    print(f"Analysis type: {args.analysis_type}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Features: {FEATURES}")
    print(f"Statistical testing: {args.stats}")
    print(f"Musician analysis: {args.musician_analysis}")
    
    # Load musician mapping if requested
    musician_mapping = {}
    if args.musician_analysis:
        musician_mapping = load_metadata()
    
    if args.analysis_type == 'comparison':
        print("\n=== COMPARISON ANALYSIS ===")
        print("Loading both all_trials and correct_only datasets...")
        
        # Load both datasets
        all_trials_data = discover_and_load_subjects(ANALYSIS_DIRS['all_trials'], "All Trials")
        correct_only_data = discover_and_load_subjects(ANALYSIS_DIRS['correct_only'], "Correct Only")
        
        if all_trials_data is None or correct_only_data is None:
            print("ERROR: Could not load both datasets for comparison")
            sys.exit(1)
        
        # Process both datasets
        all_trials_aligned = align_timepoints(all_trials_data)
        correct_only_aligned = align_timepoints(correct_only_data)
        
        all_trials_results = compute_grand_averages(all_trials_aligned)
        correct_only_results = compute_grand_averages(correct_only_aligned)
        
        # Musician analysis for both if requested
        all_trials_musician = None
        correct_only_musician = None
        if args.musician_analysis and musician_mapping:
            all_trials_musician = organize_by_musician_level(all_trials_aligned, musician_mapping)
            correct_only_musician = organize_by_musician_level(correct_only_aligned, musician_mapping)
        
        # Run stats if requested
        all_trials_stats = None
        correct_only_stats = None
        if args.stats:
            all_trials_stats = run_permutation_tests(all_trials_results)
            correct_only_stats = run_permutation_tests(correct_only_results)
        
        # Create comparison plots
        create_comparison_plots(all_trials_results, correct_only_results)
        
        # Create individual plots for each dataset
        create_plots(all_trials_results, all_trials_stats, "All Trials")
        create_plots(correct_only_results, correct_only_stats, "Correct Only")
        
        # Create musician plots if available
        if args.musician_analysis:
            if all_trials_musician:
                create_musician_plots(all_trials_musician, "All Trials")
            if correct_only_musician:
                create_musician_plots(correct_only_musician, "Correct Only")
        
        # Save both results
        save_results(all_trials_results, all_trials_stats, "all_trials", all_trials_musician)
        save_results(correct_only_results, correct_only_stats, "correct_only", correct_only_musician)
        
    else:
        # Single analysis type
        analysis_dir = ANALYSIS_DIRS[args.analysis_type]
        analysis_label = "All Trials" if args.analysis_type == 'all_trials' else "Correct Only"
        
        # Load subject results
        all_subjects_data = discover_and_load_subjects(analysis_dir, analysis_label)
        
        if all_subjects_data is None:
            sys.exit(1)
        
        # Align timepoints
        aligned_data = align_timepoints(all_subjects_data)
        
        # Check if we have data
        if not aligned_data:
            print("ERROR: No feature data could be aligned across subjects")
            sys.exit(1)
        
        # Compute grand averages
        results = compute_grand_averages(aligned_data)
        
        # Organize by musician level if requested
        musician_data = None
        if args.musician_analysis and musician_mapping:
            musician_data = organize_by_musician_level(aligned_data, musician_mapping)
        
        # Run permutation tests if requested
        stats_results = None
        if args.stats:
            stats_results = run_permutation_tests(results)
        
        # Create plots
        create_plots(results, stats_results, analysis_label)
        
        # Create musician plots if available
        if args.musician_analysis and musician_data:
            create_musician_plots(musician_data, analysis_label)
        
        # Save results
        save_results(results, stats_results, args.analysis_type, musician_data)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Grand average analysis completed successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")
    if args.musician_analysis:
        print(f"Musician level analysis saved to: {OUTPUT_DIR}/musician_level_analysis/")

if __name__ == "__main__":
    main()