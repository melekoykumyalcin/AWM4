#!/usr/bin/env python
"""
Grand Average Analysis for Spatio-Temporal Decoding
Aggregates results from spatio-temporal sliding window analysis across subjects.

Features analyzed:
- maintained_voice (speaker identity in working memory)
- maintained_location (spatial location in working memory)

Window lengths analyzed:
- 100ms spatio-temporal windows
- 400ms spatio-temporal windows

Includes cluster permutation testing and musician level analysis.

Usage: python spatiotemporal_ga_analysis.py --subjects 1 2 3 4 5 --n_subjects 30
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
from datetime import datetime
import pandas as pd
from scipy import stats
from glob import glob
import mne
from mne.stats import permutation_cluster_1samp_test

# Optional tqdm import for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, desc=None, disable=False):
        """Fallback when tqdm is not available"""
        return iterable

# Parse arguments
parser = argparse.ArgumentParser(description='Grand average analysis for spatio-temporal decoding')
parser.add_argument('--subjects', nargs='+', type=int, default=None, 
                   help='Subject IDs to include (default: all subjects found)')
parser.add_argument('--n_subjects', type=int, default=30, 
                   help='Total number of subjects if using all (default: 30)')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory for GA results')
parser.add_argument('--n_permutations', type=int, default=100000, 
                   help='Number of permutations for significance tests (default: 10000)')
parser.add_argument('--cluster_alpha', type=float, default=0.05, 
                   help='Alpha level for cluster significance (default: 0.05)')
parser.add_argument('--stat_method', type=str, default='both', choices=['cluster', 'sign', 'both'],
                   help='Statistical method: cluster, sign, or both (default: both)')
args = parser.parse_args()

# Set paths - ADAPT THESE TO YOUR PATHS
HOME_DIR = '/mnt/hpc/projects/awm4/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
BASE_DIR = PROCESSED_DIR + 'spatiotemporalAnalysis/'  # Your spatio-temporal results directory
META_FILE = HOME_DIR + 'MEGNotes.xlsx'

# Output directory
if args.output_dir is None:
    if args.subjects is None:
        OUTPUT_DIR = PROCESSED_DIR + f'spatiotemporalAnalysis/grand_average_all_subjects/'
    else:
        OUTPUT_DIR = PROCESSED_DIR + f'spatiotemporalAnalysis/grand_average_{len(args.subjects)}subjects/'
else:
    OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Important time markers
CUE_TIME = 2.0
PING_TIME = 3.5
PROBE_TIME = 4.7

# Define your spatio-temporal analysis structure
SPATIOTEMPORAL_WINDOWS = [100, 400]  # ms - window lengths you analyzed
FEATURES = ['maintained_voice', 'maintained_location']
TARGET_SAMPLING_RATE = 50  # Hz - your final sampling rate

# Feature labels for plots
FEATURE_LABELS = {
    'maintained_voice': 'Voice Identity',
    'maintained_location': 'Spatial Location'
}

# Colors for different window lengths
WINDOW_COLORS = {
    100: '#e41a1c',  # Red for 100ms
    400: '#377eb8'   # Blue for 400ms
}

def find_available_subjects():
    """Find all subjects with spatio-temporal results"""
    available_subjects = []
    
    # Look for subject directories
    pattern = f"{BASE_DIR}subject_*/"
    subject_dirs = glob(pattern)
    
    for subj_dir in subject_dirs:
        # Extract subject number
        subj_num = int(subj_dir.split('subject_')[1].rstrip('/'))
        
        # Check if this subject has spatio-temporal results
        has_results = True
        for window_ms in SPATIOTEMPORAL_WINDOWS:
            for feature in FEATURES:
                # Check for required files
                mean_file = f"{subj_dir}spatiotemporal_{feature}_{window_ms}ms_mean_scores.npy"
                if not os.path.exists(mean_file):
                    has_results = False
                    break
            if not has_results:
                break
        
        if has_results:
            available_subjects.append(subj_num)
    
    return sorted(available_subjects)

def load_musician_metadata():
    """Load participant metadata including musician categories"""
    print(f"Loading musician metadata from: {META_FILE}")
    
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
            print(f"Warning: Could not find subject/musician columns")
            print(f"Available columns: {list(meta_df.columns)}")
            return musician_mapping
        
        print(f"Using '{subject_col}' as subject ID and '{musician_col}' as musician level")
        
        # Create mapping
        for idx, row in meta_df.iterrows():
            subject_id = row[subject_col]
            musician_level = row[musician_col]
            
            if pd.notna(subject_id) and pd.notna(musician_level):
                try:
                    subject_id_int = int(float(subject_id))
                    musician_level_int = int(float(musician_level))
                    if musician_level_int in [1, 2, 3]:
                        musician_mapping[subject_id_int] = musician_level_int
                except (ValueError, TypeError):
                    print(f"Warning: Invalid data for subject {subject_id}: {musician_level}")
        
        print(f"Successfully mapped {len(musician_mapping)} subjects")
        print(f"Musician distribution: Level 1: {sum(1 for v in musician_mapping.values() if v==1)}, "
              f"Level 2: {sum(1 for v in musician_mapping.values() if v==2)}, "
              f"Level 3: {sum(1 for v in musician_mapping.values() if v==3)}")
        
        return musician_mapping
        
    except Exception as e:
        print(f"Error loading musician metadata: {e}")
        return {}

def load_subject_spatiotemporal_data(subject, window_ms, feature):
    """Load spatio-temporal data for a single subject"""
    subject_dir = f"{BASE_DIR}subject_{subject}/"
    
    # File paths for spatio-temporal analysis
    mean_file = f"{subject_dir}spatiotemporal_{feature}_{window_ms}ms_mean_scores.npy"
    std_file = f"{subject_dir}spatiotemporal_{feature}_{window_ms}ms_std_scores.npy"
    time_file = f"{subject_dir}spatiotemporal_{feature}_{window_ms}ms_timepoints.npy"
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [mean_file, std_file, time_file]):
        print(f"  Missing files for subject {subject}, window {window_ms}ms, feature {feature}")
        return None, None, None
    
    try:
        # Load data
        mean_scores = np.load(mean_file)
        std_scores = np.load(std_file)
        timepoints = np.load(time_file)
        
        return mean_scores, std_scores, timepoints
    except Exception as e:
        print(f"  Error loading data for subject {subject}: {e}")
        return None, None, None

def compute_cluster_permutation_test(individual_data, feature_name, timepoints):
    """
    Compute cluster permutation test for significance above chance level
    
    Parameters:
    -----------
    individual_data : array, shape (n_subjects, n_timepoints)
        Individual subject data
    feature_name : str
        Name of the feature being tested
    timepoints : array
        Time points corresponding to data
        
    Returns:
    --------
    clusters : dict
        Dictionary containing cluster information
    """
    print(f"  Running cluster permutation test for {feature_name}...")
    
    # Subtract chance level (0.5) to test against chance
    stats_data = individual_data - 0.5
    n_subjects = len(individual_data)
    t_thresh = stats.t.ppf(0.95, n_subjects-1)  # p<0.05, one-tailed
    
    try:
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
            stats_data,
            n_permutations=args.n_permutations,
            threshold=t_thresh,
            tail=1,  # One-tailed test (above chance)
            out_type='mask'
        )
        
        # Find significant clusters
        significant_clusters = []
        for i, (cluster, p_val) in enumerate(zip(clusters, cluster_p_values)):
            if p_val < args.cluster_alpha:
                # Find start and end times of cluster
                cluster_indices = np.where(cluster)[0]
                start_time = timepoints[cluster_indices[0]]
                end_time = timepoints[cluster_indices[-1]]
                
                significant_clusters.append({
                    'cluster_id': i,
                    'p_value': p_val,
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_idx': cluster_indices[0],
                    'end_idx': cluster_indices[-1],
                    'duration': end_time - start_time,
                    'mask': cluster
                })
        
        cluster_results = {
            'T_obs': T_obs,
            'clusters': clusters,
            'cluster_p_values': cluster_p_values,
            'significant_clusters': significant_clusters,
            'n_significant': len(significant_clusters),
            'feature_name': feature_name,
            'timepoints': timepoints
        }
        
        print(f"    Found {len(significant_clusters)} significant clusters (p < {args.cluster_alpha})")
        for cluster in significant_clusters:
            print(f"      Cluster {cluster['cluster_id']}: {cluster['start_time']:.3f}s - {cluster['end_time']:.3f}s, p = {cluster['p_value']:.6f}")
        
        return cluster_results
        
    except Exception as e:
        print(f"    Error in cluster permutation test: {str(e)}")
        return None

def compute_sign_permutation_test(individual_data, feature_name, timepoints, n_permutations=10000, alpha=0.05):
    """
    Compute sign-flipping permutation test for significance above chance level
    """
    print(f"  Running sign permutation test for {feature_name}...")
    
    n_subjects, n_timepoints = individual_data.shape
    
    # Calculate empirical means (original data)
    empirical_means = np.mean(individual_data, axis=0)
    
    # Convert scores to differences from chance (0.5)
    diffs_from_chance = individual_data - 0.5
    
    # Initialize storage for null distribution
    null_distribution = np.zeros((n_permutations, n_timepoints))
    
    # Run permutations
    for perm_idx in range(n_permutations):
        # Generate random signs for each subject and timepoint
        random_signs = np.random.choice([-1, 1], size=(n_subjects, n_timepoints))
        
        # Apply random sign flips
        permuted_diffs = diffs_from_chance * random_signs
        
        # Compute mean of this permutation (add back 0.5 to get accuracy)
        null_distribution[perm_idx] = np.mean(permuted_diffs, axis=0) + 0.5
    
    # Compute p-values for each timepoint (one-tailed test for > chance)
    p_values = np.zeros(n_timepoints)
    for t in range(n_timepoints):
        p_values[t] = np.mean(null_distribution[:, t] >= empirical_means[t])
    
    # Find significant timepoints
    significant_indices = np.where(p_values <= alpha)[0]
    
    # Group consecutive significant timepoints into clusters
    significant_clusters = []
    if len(significant_indices) > 0:
        # Group consecutive timepoints
        consec_groups = []
        current_group = [significant_indices[0]]
        
        for i in range(1, len(significant_indices)):
            if significant_indices[i] == significant_indices[i-1] + 1:
                current_group.append(significant_indices[i])
            else:
                consec_groups.append(current_group)
                current_group = [significant_indices[i]]
        
        # Add the last group
        if current_group:
            consec_groups.append(current_group)
        
        # Create cluster information
        for cluster_id, group in enumerate(consec_groups):
            start_time = timepoints[group[0]]
            end_time = timepoints[group[-1]]
            min_p_value = np.min(p_values[group])
            
            significant_clusters.append({
                'cluster_id': cluster_id,
                'p_value': min_p_value,
                'start_time': start_time,
                'end_time': end_time,
                'start_idx': group[0],
                'end_idx': group[-1],
                'duration': end_time - start_time,
                'indices': group
            })
    
    sign_results = {
        'p_values': p_values,
        'empirical_means': empirical_means,
        'null_distribution': null_distribution,
        'significant_clusters': significant_clusters,
        'n_significant': len(significant_clusters),
        'feature_name': feature_name,
        'timepoints': timepoints,
        'alpha': alpha
    }
    
    print(f"    Found {len(significant_clusters)} significant time periods (p < {alpha})")
    for cluster in significant_clusters:
        print(f"      Period {cluster['cluster_id']}: {cluster['start_time']:.3f}s - {cluster['end_time']:.3f}s, p = {cluster['p_value']:.6f}")
    
    return sign_results

def add_significance_markers(ax, cluster_results, sign_results, stat_method, y_position=0.57, line_width=3):
    """Add significance markers to plot based on statistical method"""
    
    # Add cluster permutation markers
    if stat_method in ['cluster', 'both'] and cluster_results is not None:
        if 'significant_clusters' in cluster_results and cluster_results['significant_clusters']:
            cluster_y = y_position
            
            for cluster in cluster_results['significant_clusters']:
                # Add horizontal line for significant cluster
                ax.plot([cluster['start_time'], cluster['end_time']], 
                       [cluster_y, cluster_y], 
                       color='black', linewidth=line_width, alpha=0.8)
                
                # Add text annotation
                mid_time = (cluster['start_time'] + cluster['end_time']) / 2
                ax.text(mid_time, cluster_y + 0.005, 
                       f"p={cluster['p_value']:.3f}", 
                       ha='center', va='bottom', fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Add sign permutation markers
    if stat_method in ['sign', 'both'] and sign_results is not None:
        if 'significant_clusters' in sign_results and sign_results['significant_clusters']:
            # Offset y-position if both methods are used
            sign_y = y_position + 0.03 if stat_method == 'both' else y_position
            
            for cluster in sign_results['significant_clusters']:
                # Add horizontal line for significant period
                ax.plot([cluster['start_time'], cluster['end_time']], 
                       [sign_y, sign_y], 
                       color='darkred', linewidth=line_width, alpha=0.8)
                
                # Add text annotation
                mid_time = (cluster['start_time'] + cluster['end_time']) / 2
                ax.text(mid_time, sign_y + 0.005, 
                       f"p={cluster['p_value']:.3f}", 
                       ha='center', va='bottom', fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.8))

def compute_grand_average():
    """Compute grand average across subjects for spatio-temporal results"""
    print(f"Computing spatio-temporal grand average...")
    
    # Load musician mapping
    musician_mapping = load_musician_metadata()
    
    # Storage for all data
    ga_data = {}
    
    for window_ms in SPATIOTEMPORAL_WINDOWS:
        ga_data[f'{window_ms}ms'] = {}
        
        for feature in FEATURES:
            print(f"\nProcessing {feature} with {window_ms}ms windows...")
            
            # Collect data from all subjects
            all_means = []
            all_stds = []
            valid_subjects = []
            musician_levels = []
            
            for subject in args.subjects:
                mean_scores, std_scores, timepoints = load_subject_spatiotemporal_data(
                    subject, window_ms, feature
                )
                
                if mean_scores is not None:
                    all_means.append(mean_scores)
                    all_stds.append(std_scores)
                    valid_subjects.append(subject)
                    
                    # Add musician level if available
                    musician_level = musician_mapping.get(subject, None)
                    musician_levels.append(musician_level)
            
            if len(all_means) > 0:
                # Stack data
                all_means = np.array(all_means)
                all_stds = np.array(all_stds)
                
                # Compute grand average
                ga_mean = np.mean(all_means, axis=0)
                ga_sem = stats.sem(all_means, axis=0)
                
                # Within-subject standard deviation
                within_std = np.mean(all_stds, axis=0)
                
                # Compute statistical tests based on method
                cluster_results = None
                sign_results = None
                
                if args.stat_method in ['cluster', 'both']:
                    cluster_results = compute_cluster_permutation_test(
                        all_means, f"{window_ms}ms_{feature}", timepoints
                    )
                
                if args.stat_method in ['sign', 'both']:
                    sign_results = compute_sign_permutation_test(
                        all_means, f"{window_ms}ms_{feature}", timepoints, 
                        n_permutations=args.n_permutations, alpha=args.cluster_alpha
                    )
                
                # Organize data by musician level
                musician_data = {1: [], 2: [], 3: [], None: []}
                musician_subjects = {1: [], 2: [], 3: [], None: []}
                
                for idx, (subject, level) in enumerate(zip(valid_subjects, musician_levels)):
                    musician_data[level].append(all_means[idx])
                    musician_subjects[level].append(subject)
                
                # Convert to arrays and compute stats for each musician level
                musician_stats = {}
                for level in [1, 2, 3, None]:
                    if musician_data[level]:
                        level_data = np.array(musician_data[level])
                        musician_stats[level] = {
                            'data': level_data,
                            'mean': np.mean(level_data, axis=0),
                            'sem': stats.sem(level_data, axis=0),
                            'n_subjects': len(level_data),
                            'subjects': musician_subjects[level]
                        }
                
                ga_data[f'{window_ms}ms'][feature] = {
                    'mean': ga_mean,
                    'sem': ga_sem,
                    'within_std': within_std,
                    'timepoints': timepoints,
                    'n_subjects': len(valid_subjects),
                    'subjects': valid_subjects,
                    'individual_means': all_means,
                    'cluster_results': cluster_results,
                    'sign_results': sign_results,
                    'musician_levels': musician_levels,
                    'musician_stats': musician_stats,
                    'window_ms': window_ms
                }
                
                print(f"  {window_ms}ms, {feature}: {len(valid_subjects)} subjects")
                # Print musician breakdown
                for level in [1, 2, 3, None]:
                    if level in musician_stats:
                        n = musician_stats[level]['n_subjects']
                        level_name = f"Level {level}" if level is not None else "Unknown"
                        print(f"    {level_name}: {n} subjects")
    
    return ga_data

def plot_spatiotemporal_comparison(ga_data):
    """Create comprehensive comparison plot of spatio-temporal results"""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[2, 2, 1], width_ratios=[1, 1, 1])
    
    # Main time course plots
    ax_voice = fig.add_subplot(gs[0, :])
    ax_location = fig.add_subplot(gs[1, :])
    
    # Summary plots
    ax_peak_voice = fig.add_subplot(gs[2, 0])
    ax_peak_location = fig.add_subplot(gs[2, 1])
    ax_comparison = fig.add_subplot(gs[2, 2])
    
    fig.suptitle('Spatio-Temporal Decoding Analysis: Window Length Comparison', fontsize=18)
    
    # Plot voice decoding time courses
    ax_voice.set_title('Voice Identity Decoding', fontsize=16)
    for window_ms in SPATIOTEMPORAL_WINDOWS:
        window_key = f'{window_ms}ms'
        if window_key in ga_data and 'maintained_voice' in ga_data[window_key]:
            data = ga_data[window_key]['maintained_voice']
            label = f"{window_ms}ms windows (n={data['n_subjects']})"
            
            ax_voice.plot(data['timepoints'], data['mean'], 
                        label=label, color=WINDOW_COLORS[window_ms], 
                        linewidth=2.5, alpha=0.8)
            
            ax_voice.fill_between(data['timepoints'],
                                data['mean'] - data['sem'],
                                data['mean'] + data['sem'],
                                color=WINDOW_COLORS[window_ms], alpha=0.2)
            
            # Add significance markers
            add_significance_markers(ax_voice, data['cluster_results'], data['sign_results'], args.stat_method)
    
    # Plot location decoding time courses
    ax_location.set_title('Spatial Location Decoding', fontsize=16)
    for window_ms in SPATIOTEMPORAL_WINDOWS:
        window_key = f'{window_ms}ms'
        if window_key in ga_data and 'maintained_location' in ga_data[window_key]:
            data = ga_data[window_key]['maintained_location']
            label = f"{window_ms}ms windows (n={data['n_subjects']})"
            
            ax_location.plot(data['timepoints'], data['mean'], 
                           label=label, color=WINDOW_COLORS[window_ms], 
                           linewidth=2.5, alpha=0.8)
            
            ax_location.fill_between(data['timepoints'],
                                   data['mean'] - data['sem'],
                                   data['mean'] + data['sem'],
                                   color=WINDOW_COLORS[window_ms], alpha=0.2)
            
            # Add significance markers
            add_significance_markers(ax_location, data['cluster_results'], data['sign_results'], args.stat_method)
    
    # Format time course plots
    for ax in [ax_voice, ax_location]:
        ax.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Ping')
        ax.axvline(x=CUE_TIME, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Cue')
        ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, label='Chance')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Decoding Accuracy', fontsize=12)
        ax.set_ylim(0.45, 0.65)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(CUE_TIME - 0.1, PROBE_TIME + 0.1)
    
    # Peak accuracy comparison for voice
    voice_peaks = []
    voice_labels = []
    voice_colors = []
    
    for window_ms in SPATIOTEMPORAL_WINDOWS:
        window_key = f'{window_ms}ms'
        if window_key in ga_data and 'maintained_voice' in ga_data[window_key]:
            data = ga_data[window_key]['maintained_voice']
            peak = np.max(data['mean'])
            voice_peaks.append(peak)
            voice_labels.append(f'{window_ms}ms')
            voice_colors.append(WINDOW_COLORS[window_ms])
    
    x_voice = np.arange(len(voice_peaks))
    ax_peak_voice.bar(x_voice, voice_peaks, color=voice_colors, alpha=0.8)
    ax_peak_voice.set_title('Voice: Peak Accuracy', fontsize=12)
    ax_peak_voice.set_xticks(x_voice)
    ax_peak_voice.set_xticklabels(voice_labels, fontsize=10)
    ax_peak_voice.set_ylim(0.5, 0.65)
    ax_peak_voice.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
    ax_peak_voice.set_ylabel('Peak Accuracy', fontsize=10)
    ax_peak_voice.grid(True, alpha=0.3, axis='y')
    
    # Peak accuracy comparison for location
    location_peaks = []
    location_labels = []
    location_colors = []
    
    for window_ms in SPATIOTEMPORAL_WINDOWS:
        window_key = f'{window_ms}ms'
        if window_key in ga_data and 'maintained_location' in ga_data[window_key]:
            data = ga_data[window_key]['maintained_location']
            peak = np.max(data['mean'])
            location_peaks.append(peak)
            location_labels.append(f'{window_ms}ms')
            location_colors.append(WINDOW_COLORS[window_ms])
    
    x_location = np.arange(len(location_peaks))
    ax_peak_location.bar(x_location, location_peaks, color=location_colors, alpha=0.8)
    ax_peak_location.set_title('Location: Peak Accuracy', fontsize=12)
    ax_peak_location.set_xticks(x_location)
    ax_peak_location.set_xticklabels(location_labels, fontsize=10)
    ax_peak_location.set_ylim(0.5, 0.65)
    ax_peak_location.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
    ax_peak_location.set_ylabel('Peak Accuracy', fontsize=10)
    ax_peak_location.grid(True, alpha=0.3, axis='y')
    
    # Feature comparison scatter plot
    comparison_data = []
    for window_ms in SPATIOTEMPORAL_WINDOWS:
        window_key = f'{window_ms}ms'
        voice_peak = location_peak = None
        
        if (window_key in ga_data and 
            'maintained_voice' in ga_data[window_key] and 
            'maintained_location' in ga_data[window_key]):
            
            voice_peak = np.max(ga_data[window_key]['maintained_voice']['mean'])
            location_peak = np.max(ga_data[window_key]['maintained_location']['mean'])
            
            comparison_data.append({
                'window': f'{window_ms}ms',
                'voice': voice_peak,
                'location': location_peak,
                'color': WINDOW_COLORS[window_ms]
            })
    
    # Scatter plot
    for item in comparison_data:
        ax_comparison.scatter(item['voice'], item['location'], 
                            s=200, c=item['color'], alpha=0.8, edgecolors='black',
                            label=item['window'])
    
    ax_comparison.plot([0.5, 0.70], [0.5, 0.70], 'k--', alpha=0.3)
    ax_comparison.set_xlabel('Voice Peak Accuracy', fontsize=10)
    ax_comparison.set_ylabel('Location Peak Accuracy', fontsize=10)
    ax_comparison.set_title('Voice vs Location Performance', fontsize=12)
    ax_comparison.set_xlim(0.50, 0.70)
    ax_comparison.set_ylim(0.50, 0.65)
    ax_comparison.legend(fontsize=8)
    ax_comparison.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/spatiotemporal_ga_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved main comparison plot: spatiotemporal_ga_comparison.png")

def plot_individual_differences(ga_data):
    """Plot individual subject variability"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Individual Subject Variability - Spatio-Temporal Analysis', fontsize=16)
    
    for row_idx, feature in enumerate(FEATURES):
        for col_idx, window_ms in enumerate(SPATIOTEMPORAL_WINDOWS):
            ax = axes[row_idx, col_idx]
            window_key = f'{window_ms}ms'
            
            if window_key in ga_data and feature in ga_data[window_key]:
                data = ga_data[window_key][feature]
                
                # Plot individual subjects
                for i, subj_mean in enumerate(data['individual_means']):
                    ax.plot(data['timepoints'], subj_mean, 
                           color='gray', alpha=0.3, linewidth=0.5)
                
                # Plot grand average
                ax.plot(data['timepoints'], data['mean'], 
                       color=WINDOW_COLORS[window_ms], linewidth=3, 
                       label=f'Grand Average (n={data["n_subjects"]})')
                
                # Add significance markers
                add_significance_markers(ax, data['cluster_results'], data['sign_results'], args.stat_method)
                
                # Add markers
                ax.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7)
                ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
                
                # Labels
                feature_name = FEATURE_LABELS[feature]
                ax.set_title(f'{feature_name} - {window_ms}ms Windows')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Decoding Accuracy')
                ax.set_ylim(0.4, 0.65)
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/spatiotemporal_individual_differences.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved individual differences plot: spatiotemporal_individual_differences.png")

def create_summary_table(ga_data):
    """Create a comprehensive summary table"""
    summary_data = []
    
    for window_ms in SPATIOTEMPORAL_WINDOWS:
        window_key = f'{window_ms}ms'
        if window_key in ga_data:
            for feature in FEATURES:
                if feature in ga_data[window_key]:
                    data = ga_data[window_key][feature]
                    
                    # Calculate statistics
                    peak_idx = np.argmax(data['mean'])
                    peak_acc = data['mean'][peak_idx]
                    peak_time = data['timepoints'][peak_idx]
                    
                    ping_idx = np.argmin(np.abs(data['timepoints'] - PING_TIME))
                    ping_acc = data['mean'][ping_idx]
                    
                    pre_mask = (data['timepoints'] >= 2.5) & (data['timepoints'] < PING_TIME)
                    post_mask = (data['timepoints'] >= PING_TIME) & (data['timepoints'] <= 4.5)
                    pre_mean = np.mean(data['mean'][pre_mask])
                    post_mean = np.mean(data['mean'][post_mask])
                    
                    above_chance = data['mean'] > 0.5
                    time_above = np.sum(above_chance) * 0.02  # 20ms steps
                    
                    # Cluster statistics
                    n_clusters = 0
                    total_cluster_duration = 0
                    min_cluster_p = 1.0
                    n_sign_periods = 0
                    total_sign_duration = 0
                    min_sign_p = 1.0
                    
                    if data['cluster_results']:
                        n_clusters = data['cluster_results']['n_significant']
                        if n_clusters > 0:
                            total_cluster_duration = sum([c['duration'] for c in data['cluster_results']['significant_clusters']])
                            min_cluster_p = min([c['p_value'] for c in data['cluster_results']['significant_clusters']])
                    
                    if data.get('sign_results'):
                        n_sign_periods = data['sign_results']['n_significant']
                        if n_sign_periods > 0:
                            total_sign_duration = sum([c['duration'] for c in data['sign_results']['significant_clusters']])
                            min_sign_p = min([c['p_value'] for c in data['sign_results']['significant_clusters']])
                    
                    # Musician breakdown
                    musician_breakdown = []
                    for level in [1, 2, 3, None]:
                        if level in data['musician_stats']:
                            n = data['musician_stats'][level]['n_subjects']
                            level_name = f"L{level}" if level is not None else "Unk"
                            musician_breakdown.append(f"{level_name}:{n}")
                    
                    # Expected features calculation
                    timepoints_per_channel = int(window_ms * TARGET_SAMPLING_RATE / 1000)
                    n_channels = 102  # Typical number of MEG magnetometers
                    expected_features = n_channels * timepoints_per_channel
                    
                    summary_data.append({
                        'Window Length (ms)': window_ms,
                        'Feature': FEATURE_LABELS[feature],
                        'Expected Features': expected_features,
                        'Timepoints per Channel': timepoints_per_channel,
                        'N Subjects': data['n_subjects'],
                        'Musician Breakdown': ', '.join(musician_breakdown),
                        'Peak Accuracy': f"{peak_acc:.3f}",
                        'Peak Time (s)': f"{peak_time:.2f}",
                        'Ping Accuracy': f"{ping_acc:.3f}",
                        'Pre-ping Mean': f"{pre_mean:.3f}",
                        'Post-ping Mean': f"{post_mean:.3f}",
                        'Δ(Post-Pre)': f"{post_mean - pre_mean:+.3f}",
                        'Time > Chance (s)': f"{time_above:.2f}",
                        'N Significant Clusters': n_clusters,
                        'Total Cluster Duration (s)': f"{total_cluster_duration:.3f}",
                        'Min Cluster p-value': f"{min_cluster_p:.6f}" if min_cluster_p < 1.0 else "n.s.",
                        'N Sign Periods': n_sign_periods,
                        'Total Sign Duration (s)': f"{total_sign_duration:.3f}",
                        'Min Sign p-value': f"{min_sign_p:.6f}" if min_sign_p < 1.0 else "n.s."
                    })
    
    df = pd.DataFrame(summary_data)
    return df

def save_ga_data(ga_data):
    """Save grand average data and statistics"""
    # Save raw GA data
    np.save(f'{OUTPUT_DIR}/spatiotemporal_ga_data.npy', ga_data, allow_pickle=True)
    
    # Create and save summary table
    summary_df = create_summary_table(ga_data)
    summary_df.to_csv(f'{OUTPUT_DIR}/spatiotemporal_ga_summary.csv', index=False)
    
    print("\nSpatio-Temporal Grand Average Summary:")
    print(summary_df.to_string(index=False))
    
    # Create JSON summary
    summary_json = {
        'analysis_type': 'spatio_temporal_sliding_window',
        'n_subjects_requested': len(args.subjects),
        'subjects_requested': args.subjects,
        'processing_date': str(datetime.now()),
        'window_lengths_ms': SPATIOTEMPORAL_WINDOWS,
        'features': FEATURES,
        'target_sampling_rate_hz': TARGET_SAMPLING_RATE,
        'step_size_ms': 20,  # 20ms steps at 50Hz
        'statistical_parameters': {
            'method': args.stat_method,
            'n_permutations': args.n_permutations,
            'alpha_level': args.cluster_alpha
        },
        'results': {}
    }
    
    # Add detailed results
    for window_ms in SPATIOTEMPORAL_WINDOWS:
        window_key = f'{window_ms}ms'
        if window_key in ga_data:
            summary_json['results'][window_key] = {}
            
            for feature in FEATURES:
                if feature in ga_data[window_key]:
                    data = ga_data[window_key][feature]
                    
                    feature_results = {
                        'n_subjects': data['n_subjects'],
                        'subject_ids': data['subjects'],
                        'mean_accuracy': float(np.mean(data['mean'])),
                        'peak_accuracy': float(np.max(data['mean'])),
                        'peak_time': float(data['timepoints'][np.argmax(data['mean'])]),
                        'window_length_ms': window_ms,
                        'expected_features': int(window_ms * TARGET_SAMPLING_RATE / 1000) * 102
                    }
                    
                    # Add statistical results
                    if data['cluster_results']:
                        feature_results['cluster_results'] = {
                            'n_significant_clusters': data['cluster_results']['n_significant'],
                            'significant_clusters': data['cluster_results']['significant_clusters']
                        }
                    
                    if data.get('sign_results'):
                        feature_results['sign_results'] = {
                            'n_significant_periods': data['sign_results']['n_significant'],
                            'significant_periods': data['sign_results']['significant_clusters']
                        }
                    
                    # Add musician level breakdown
                    feature_results['musician_level_breakdown'] = {}
                    for level in [1, 2, 3, None]:
                        if level in data['musician_stats']:
                            level_data = data['musician_stats'][level]
                            level_key = f"level_{level}" if level is not None else "unknown"
                            feature_results['musician_level_breakdown'][level_key] = {
                                'n_subjects': level_data['n_subjects'],
                                'peak_accuracy': float(np.max(level_data['mean'])),
                                'mean_accuracy': float(np.mean(level_data['mean']))
                            }
                    
                    summary_json['results'][window_key][feature] = feature_results
    
    with open(f'{OUTPUT_DIR}/spatiotemporal_ga_summary.json', 'w') as f:
        json.dump(summary_json, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}")

def main():
    """Main analysis pipeline"""
    print(f"Spatio-Temporal Grand Average Analysis")
    print(f"Statistical method: {args.stat_method}")
    print(f"Permutation parameters: {args.n_permutations} permutations, α = {args.cluster_alpha}")
    
    # Find available subjects if not specified
    if args.subjects is None:
        available_subjects = find_available_subjects()
        if not available_subjects:
            print("No subjects with spatio-temporal results found!")
            sys.exit(1)
        args.subjects = available_subjects
        print(f"Found {len(available_subjects)} subjects with spatio-temporal results")
    else:
        print(f"Using {len(args.subjects)} specified subjects")
    
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Compute grand average
    ga_data = compute_grand_average()
    
    # Check how many subjects actually had data
    total_valid = 0
    for window_ms in SPATIOTEMPORAL_WINDOWS:
        window_key = f'{window_ms}ms'
        if window_key in ga_data:
            for feature in FEATURES:
                if feature in ga_data[window_key]:
                    n_subj = ga_data[window_key][feature]['n_subjects']
                    total_valid = max(total_valid, n_subj)
    
    print(f"\nMaximum subjects with valid data: {total_valid}/{len(args.subjects)}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_spatiotemporal_comparison(ga_data)
    plot_individual_differences(ga_data)
    
    # Save results
    print("\nSaving results...")
    save_ga_data(ga_data)
    
    print("\nSpatio-Temporal Grand Average Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()