#!/usr/bin/env python
"""
Grand Average Analysis for Comprehensive Decoding (Three Approaches)
Aggregates results from:
1. Extreme Contrast (Sp1 vs Sp4, L1 vs L4)
2. Feature-Pure (ignoring orthogonal dimension)
3. Condition-Specific (maintaining all 16 conditions)

Includes cluster permutation testing and musician level analysis.

Usage: python ga_comprehensive_decoding.py
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

# Parse arguments
parser = argparse.ArgumentParser(description='Grand average analysis for comprehensive decoding')
parser.add_argument('--subjects', nargs='+', type=int, default=None, 
                   help='Subject IDs to include (default: all 30 subjects)')
parser.add_argument('--n_subjects', type=int, default=30, 
                   help='Total number of subjects if using all (default: 30)')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory for GA results')
parser.add_argument('--n_permutations', type=int, default=100000, 
                   help='Number of permutations for cluster tests (default: 100000)')
parser.add_argument('--cluster_alpha', type=float, default=0.05, 
                   help='Alpha level for cluster significance (default: 0.05)')
args = parser.parse_args()

# Set subjects - use all 30 by default
if args.subjects is None:
    args.subjects = list(range(1, args.n_subjects + 1))
    print(f"Using all {args.n_subjects} subjects")

# Set paths
HOME_DIR = '/mnt/hpc/projects/awm4/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
BASE_DIR = PROCESSED_DIR + 'extremesDecoding/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'

# Output directory
if args.output_dir is None:
    OUTPUT_DIR = PROCESSED_DIR + f'extremesDecoding/grand_average_{len(args.subjects)}subjects/'
else:
    OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Important time markers
CUE_TIME = 2.0
PING_TIME = 3.5
PROBE_TIME = 4.7
CLUSTER_Y = 0.57  # Y position for cluster significance markers

# Define approach structure
APPROACHES = {
    'extreme_contrast': {
        'features': ['voice_extreme', 'location_extreme'],
        'labels': {
            'voice_extreme': 'Sp1 vs Sp4',
            'location_extreme': 'L1 vs L4'
        },
        'color': '#e41a1c'
    },
    'feature_pure': {
        'features': ['voice_pure', 'location_pure'],
        'labels': {
            'voice_pure': 'Voice (location-invariant)',
            'location_pure': 'Location (voice-invariant)'
        },
        'color': '#377eb8'
    },
    'condition_specific': {
        'features': ['voice_specific', 'location_specific'],
        'labels': {
            'voice_specific': 'Voice (16 conditions)',
            'location_specific': 'Location (16 conditions)'
        },
        'color': '#4daf4a'
    }
}

# Feature types for organization
FEATURE_TYPES = {
    'voice': ['voice_extreme', 'voice_pure', 'voice_specific'],
    'location': ['location_extreme', 'location_pure', 'location_specific']
}

# Musician level colors for each approach
MUSICIAN_COLORS = {
    'extreme_contrast': {
        1: '#ff4d4d',  # Strong light red
        2: '#ff1a1a',  # Clear medium red
        3: '#800000',  # Dark red
    },
    'feature_pure': {
        1: '#4da6ff',  # Strong light blue
        2: '#1a75ff',  # Clear medium blue
        3: '#003366',  # Dark blue
    },
    'condition_specific': {
        1: '#4dff88',  # Strong light green
        2: '#1aff5c',  # Clear medium green
        3: '#006622',  # Dark green
    }
}



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

def load_subject_data(subject, approach, feature):
    """Load data for a single subject"""
    subject_dir = f"{BASE_DIR}subject_{subject}/"
    
    # File paths
    mean_file = f"{subject_dir}full_delay_{approach}_{feature}_mean_scores.npy"
    std_file = f"{subject_dir}full_delay_{approach}_{feature}_std_scores.npy"
    time_file = f"{subject_dir}full_delay_{approach}_{feature}_timepoints.npy"
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [mean_file, std_file, time_file]):
        return None, None, None
    
    # Load data
    mean_scores = np.load(mean_file)
    std_scores = np.load(std_file)
    timepoints = np.load(time_file)
    
    return mean_scores, std_scores, timepoints

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
    print(f"Running cluster permutation test for {feature_name}...")
    
    # Subtract chance level (0.5) to test against chance
    stats_data = individual_data - 0.5
    n_subjects = len(individual_data)
    t_thresh = stats.t.ppf(0.95, n_subjects-1)  # p<0.05, one-tailed

    
    # Run cluster permutation test
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
        
        print(f"  Found {len(significant_clusters)} significant clusters (p < {args.cluster_alpha})")
        for cluster in significant_clusters:
            print(f"    Cluster {cluster['cluster_id']}: {cluster['start_time']:.3f}s - {cluster['end_time']:.3f}s, p = {cluster['p_value']:.6f}")
        
        return cluster_results
        
    except Exception as e:
        print(f"  Error in cluster permutation test: {str(e)}")
        return None

def add_cluster_markers(ax, cluster_results, y_position=CLUSTER_Y, line_width=4):
    """Add cluster significance markers to plot"""
    if cluster_results is None:
        return
    
    for cluster in cluster_results['significant_clusters']:
        # Add horizontal line for significant cluster
        ax.plot([cluster['start_time'], cluster['end_time']], 
               [y_position, y_position], 
               color='black', linewidth=line_width, alpha=0.8)
        
        # Add text annotation
        mid_time = (cluster['start_time'] + cluster['end_time']) / 2
        ax.text(mid_time, y_position + 0.01, 
               f"p={cluster['p_value']:.3f}", 
               ha='center', va='bottom', fontsize=8, 
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

def compute_grand_average():
    """Compute grand average across subjects for all approaches"""
    print(f"Computing grand average for {len(args.subjects)} subjects...")
    
    # Load musician mapping
    musician_mapping = load_musician_metadata()
    
    # Storage for all data
    ga_data = {}
    
    for approach_name, approach_info in APPROACHES.items():
        ga_data[approach_name] = {}
        
        for feature in approach_info['features']:
            # Collect data from all subjects
            all_means = []
            all_stds = []
            valid_subjects = []
            musician_levels = []
            
            for subject in args.subjects:
                mean_scores, std_scores, timepoints = load_subject_data(
                    subject, approach_name, feature
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
                
                # Compute cluster permutation test
                cluster_results = compute_cluster_permutation_test(
                    all_means, f"{approach_name}_{feature}", timepoints
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
                
                ga_data[approach_name][feature] = {
                    'mean': ga_mean,
                    'sem': ga_sem,
                    'within_std': within_std,
                    'timepoints': timepoints,
                    'n_subjects': len(valid_subjects),
                    'subjects': valid_subjects,
                    'individual_means': all_means,
                    'cluster_results': cluster_results,
                    'musician_levels': musician_levels,
                    'musician_stats': musician_stats
                }
                
                print(f"  {approach_name}, {feature}: {len(valid_subjects)} subjects")
                # Print musician breakdown
                for level in [1, 2, 3, None]:
                    if level in musician_stats:
                        n = musician_stats[level]['n_subjects']
                        level_name = f"Level {level}" if level is not None else "Unknown"
                        print(f"    {level_name}: {n} subjects")
    
    return ga_data

def plot_approach_comparison(ga_data):
    """Create comprehensive comparison plot of all three approaches"""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[2, 2, 1], width_ratios=[1, 1, 1])
    
    # Main time course plots
    ax_voice = fig.add_subplot(gs[0, :])
    ax_location = fig.add_subplot(gs[1, :])
    
    # Summary plots
    ax_peak_voice = fig.add_subplot(gs[2, 0])
    ax_peak_location = fig.add_subplot(gs[2, 1])
    ax_comparison = fig.add_subplot(gs[2, 2])
    
    fig.suptitle('Comprehensive Decoding Analysis: Three Approaches Comparison', fontsize=18)
    
    # Plot voice decoding time courses
    ax_voice.set_title('Voice Identity Decoding', fontsize=16)
    for approach_name, approach_info in APPROACHES.items():
        for feature in approach_info['features']:
            if 'voice' in feature and feature in ga_data[approach_name]:
                data = ga_data[approach_name][feature]
                label = f"{approach_info['labels'][feature]} (n={data['n_subjects']})"
                
                ax_voice.plot(data['timepoints'], data['mean'], 
                            label=label, color=approach_info['color'], 
                            linewidth=2.5, alpha=0.8)
                
                ax_voice.fill_between(data['timepoints'],
                                    data['mean'] - data['sem'],
                                    data['mean'] + data['sem'],
                                    color=approach_info['color'], alpha=0.2)
                
                # Add cluster markers
                add_cluster_markers(ax_voice, data['cluster_results'])
    
    # Plot location decoding time courses
    ax_location.set_title('Location Decoding', fontsize=16)
    for approach_name, approach_info in APPROACHES.items():
        for feature in approach_info['features']:
            if 'location' in feature and feature in ga_data[approach_name]:
                data = ga_data[approach_name][feature]
                label = f"{approach_info['labels'][feature]} (n={data['n_subjects']})"
                
                ax_location.plot(data['timepoints'], data['mean'], 
                               label=label, color=approach_info['color'], 
                               linewidth=2.5, alpha=0.8)
                
                ax_location.fill_between(data['timepoints'],
                                       data['mean'] - data['sem'],
                                       data['mean'] + data['sem'],
                                       color=approach_info['color'], alpha=0.2)
                
                # Add cluster markers
                add_cluster_markers(ax_location, data['cluster_results'])
    
    # Format time course plots
    for ax in [ax_voice, ax_location]:
        ax.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Decoding Accuracy', fontsize=12)
        ax.set_ylim(0.45, 0.60)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(CUE_TIME - 0.1, PROBE_TIME + 0.1)
    
    # Add cluster significance legend
    ax_voice.plot([], [], color='black', linewidth=4, label='Significant clusters', alpha=0.8)
    ax_voice.legend(loc='best', fontsize=10)
    
    # Peak accuracy comparison for voice
    voice_peaks = []
    voice_labels = []
    voice_colors = []
    
    for approach_name in ['extreme_contrast', 'feature_pure', 'condition_specific']:
        for feature in APPROACHES[approach_name]['features']:
            if 'voice' in feature and feature in ga_data[approach_name]:
                data = ga_data[approach_name][feature]
                peak = np.max(data['mean'])
                voice_peaks.append(peak)
                voice_labels.append(approach_name.replace('_', '\n'))
                voice_colors.append(APPROACHES[approach_name]['color'])
    
    x_voice = np.arange(len(voice_peaks))
    ax_peak_voice.bar(x_voice, voice_peaks, color=voice_colors, alpha=0.8)
    ax_peak_voice.set_title('Voice: Peak Accuracy', fontsize=12)
    ax_peak_voice.set_xticks(x_voice)
    ax_peak_voice.set_xticklabels(voice_labels, fontsize=10)
    ax_peak_voice.set_ylim(0.5, 0.60)
    ax_peak_voice.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
    ax_peak_voice.set_ylabel('Peak Accuracy', fontsize=10)
    ax_peak_voice.grid(True, alpha=0.3, axis='y')
    
    # Peak accuracy comparison for location
    location_peaks = []
    location_labels = []
    location_colors = []
    
    for approach_name in ['extreme_contrast', 'feature_pure', 'condition_specific']:
        for feature in APPROACHES[approach_name]['features']:
            if 'location' in feature and feature in ga_data[approach_name]:
                data = ga_data[approach_name][feature]
                peak = np.max(data['mean'])
                location_peaks.append(peak)
                location_labels.append(approach_name.replace('_', '\n'))
                location_colors.append(APPROACHES[approach_name]['color'])
    
    x_location = np.arange(len(location_peaks))
    ax_peak_location.bar(x_location, location_peaks, color=location_colors, alpha=0.8)
    ax_peak_location.set_title('Location: Peak Accuracy', fontsize=12)
    ax_peak_location.set_xticks(x_location)
    ax_peak_location.set_xticklabels(location_labels, fontsize=10)
    ax_peak_location.set_ylim(0.5, 0.60)
    ax_peak_location.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
    ax_peak_location.set_ylabel('Peak Accuracy', fontsize=10)
    ax_peak_location.grid(True, alpha=0.3, axis='y')
    
    # Direct comparison plot
    comparison_data = []
    for approach_name in ['extreme_contrast', 'feature_pure', 'condition_specific']:
        voice_peak = location_peak = None
        
        for feature in APPROACHES[approach_name]['features']:
            if feature in ga_data[approach_name]:
                data = ga_data[approach_name][feature]
                peak = np.max(data['mean'])
                if 'voice' in feature:
                    voice_peak = peak
                else:
                    location_peak = peak
        
        if voice_peak and location_peak:
            comparison_data.append({
                'approach': approach_name,
                'voice': voice_peak,
                'location': location_peak,
                'color': APPROACHES[approach_name]['color']
            })
    
    # Scatter plot
    for item in comparison_data:
        ax_comparison.scatter(item['voice'], item['location'], 
                            s=200, c=item['color'], alpha=0.8, edgecolors='black',
                            label=item['approach'].replace('_', ' ').title())
    
    ax_comparison.plot([0.5, 0.75], [0.5, 0.75], 'k--', alpha=0.3)
    ax_comparison.set_xlabel('Voice Peak Accuracy', fontsize=10)
    ax_comparison.set_ylabel('Location Peak Accuracy', fontsize=10)
    ax_comparison.set_title('Voice vs Location Performance', fontsize=12)
    ax_comparison.set_xlim(0.55, 0.72)
    ax_comparison.set_ylim(0.55, 0.60)
    ax_comparison.legend(fontsize=8)
    ax_comparison.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/ga_approach_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_individual_differences_with_musicians(ga_data):
    """Plot individual subject variability colored by musician level"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Individual Subject Variability by Musician Level', fontsize=16)
    
    approach_names = list(APPROACHES.keys())
    
    for row_idx, feature_type in enumerate(['voice', 'location']):
        for col_idx, approach_name in enumerate(approach_names):
            ax = axes[row_idx, col_idx]
            
            # Find the relevant feature
            relevant_feature = None
            for feature in APPROACHES[approach_name]['features']:
                if feature_type in feature:
                    relevant_feature = feature
                    break
            
            if relevant_feature and relevant_feature in ga_data[approach_name]:
                data = ga_data[approach_name][relevant_feature]
                
                # Plot individual subjects colored by musician level
                for i, (subj_mean, musician_level) in enumerate(zip(data['individual_means'], data['musician_levels'])):
                    if musician_level is not None:
                        color = MUSICIAN_COLORS[approach_name][musician_level]
                        alpha = 0.6
                        linewidth = 1.0
                    else:
                        color = 'gray'
                        alpha = 0.3
                        linewidth = 0.5
                    
                    ax.plot(data['timepoints'], subj_mean, 
                           color=color, alpha=alpha, linewidth=linewidth)
                
                # Plot grand average
                ax.plot(data['timepoints'], data['mean'], 
                       color='black', linewidth=3, label='Grand Average')
                
                # Add cluster markers
                add_cluster_markers(ax, data['cluster_results'])
                
                # Add markers
                ax.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7)
                ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
                
                # Labels
                ax.set_title(f'{feature_type.title()} - {approach_name.replace("_", " ").title()}')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Decoding Accuracy')
                ax.set_ylim(0.4, 0.6)
                ax.grid(True, alpha=0.3)
                
                # Add legend for musician levels
                for level in [1, 2, 3]:
                    ax.plot([], [], color=MUSICIAN_COLORS[approach_name][level], 
                           linewidth=2, label=f'Musician Level {level}', alpha=0.8)
                ax.plot([], [], color='gray', linewidth=1, label='Unknown Level', alpha=0.5)
                ax.legend(loc='best', fontsize=8)
                
                # Add text with n subjects by musician level
                text_lines = [f'Total n = {data["n_subjects"]}']
                for level in [1, 2, 3]:
                    if level in data['musician_stats']:
                        n = data['musician_stats'][level]['n_subjects']
                        text_lines.append(f'Level {level}: {n}')
                if None in data['musician_stats']:
                    n = data['musician_stats'][None]['n_subjects']
                    text_lines.append(f'Unknown: {n}')
                
                ax.text(0.02, 0.98, '\n'.join(text_lines), 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/ga_individual_differences_by_musicians.png', dpi=300, bbox_inches='tight')
    plt.close()

def compute_musician_level_clusters(ga_data):
    """Compute cluster permutation tests for each musician level separately"""
    print("\nComputing cluster permutation tests for each musician level...")
    
    for approach_name, approach_info in APPROACHES.items():
        for feature in approach_info['features']:
            if feature in ga_data[approach_name]:
                data = ga_data[approach_name][feature]
                
                # Compute clusters for each musician level
                musician_cluster_results = {}
                
                for level in [1, 2, 3]:
                    if level in data['musician_stats'] and data['musician_stats'][level]['n_subjects'] >= 5:
                        level_data = data['musician_stats'][level]['data']
                        
                        # Run cluster permutation test for this musician level
                        cluster_results = compute_cluster_permutation_test(
                            level_data, 
                            f"{approach_name}_{feature}_musician_level_{level}", 
                            data['timepoints']
                        )
                        
                        musician_cluster_results[level] = cluster_results
                
                # Store musician cluster results
                ga_data[approach_name][feature]['musician_cluster_results'] = musician_cluster_results
    
    return ga_data

def add_musician_cluster_markers(ax, musician_cluster_results, level, y_base=0.57, level_offset=0.02):
    """Add cluster significance markers for specific musician level"""
    if musician_cluster_results is None or level not in musician_cluster_results:
        return
    
    cluster_results = musician_cluster_results[level]
    if cluster_results is None:
        return
    
    # Offset y-position for different levels
    y_position = y_base + (level - 1) * level_offset
    
    for cluster in cluster_results['significant_clusters']:
        # Add horizontal line for significant cluster
        ax.plot([cluster['start_time'], cluster['end_time']], 
               [y_position, y_position], 
               color='black', linewidth=3, alpha=0.8)
        
        # Add small text annotation with level info
        mid_time = (cluster['start_time'] + cluster['end_time']) / 2
        ax.text(mid_time, y_position + 0.005, 
               f"L{level}", 
               ha='center', va='bottom', fontsize=6, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))

def plot_musician_level_timecourses_with_clusters(ga_data):
    """Create detailed timecourse plots by musician level with individual cluster tests"""
    
    # Create musician analysis subdirectory
    musician_dir = f'{OUTPUT_DIR}/musician_analysis/'
    os.makedirs(musician_dir, exist_ok=True)
    
    for approach_name, approach_info in APPROACHES.items():
        for feature in approach_info['features']:
            if feature in ga_data[approach_name]:
                data = ga_data[approach_name][feature]
                
                # Skip if no musician data
                if not any(level in data['musician_stats'] for level in [1, 2, 3]):
                    continue
                
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Plot each musician level
                for level in [1, 2, 3]:
                    if level in data['musician_stats']:
                        level_data = data['musician_stats'][level]
                        
                        ax.plot(data['timepoints'], level_data['mean'],
                               label=f'Musician Level {level} (n={level_data["n_subjects"]})',
                               color=MUSICIAN_COLORS[approach_name][level],
                               linewidth=3.0)
                        
                        ax.fill_between(data['timepoints'],
                                       level_data['mean'] - level_data['sem'],
                                       level_data['mean'] + level_data['sem'],
                                       color=MUSICIAN_COLORS[approach_name][level],
                                       alpha=0.2)
                        
                        # Add individual cluster markers for this musician level
                        if 'musician_cluster_results' in data:
                            add_musician_cluster_markers(ax, data['musician_cluster_results'], level)
                
                # Add overall cluster markers from grand average (at bottom)
                add_cluster_markers(ax, data['cluster_results'], y_position=0.51, line_width=2)
                
                # Add markers
                ax.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7, linewidth=2)
                ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
                
                # Format plot
                feature_type = 'Voice' if 'voice' in feature else 'Location'
                approach_title = approach_name.replace('_', ' ').title()
                ax.set_title(f'{feature_type} Decoding by Musician Level\n{approach_title} Approach (with Individual Cluster Tests)')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Decoding Accuracy')
                ax.set_ylim(0.45, 0.65)  # Increased upper limit for cluster markers
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(CUE_TIME - 0.1, PROBE_TIME + 0.1)
                
                # Add legend for cluster markers
                ax.plot([], [], color='black', linewidth=3, alpha=0.8, label='Individual level clusters')
                ax.plot([], [], color='black', linewidth=2, alpha=0.8, label='Overall group clusters')
                ax.legend(loc='upper right')
                
                plt.tight_layout()
                plt.savefig(f'{musician_dir}/{approach_name}_{feature}_musician_levels_with_clusters.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Saved musician level plot with clusters: {approach_name}_{feature}")

def plot_individual_musician_level_plots(ga_data):
    """Create separate plots for each musician level within each approach"""
    
    musician_dir = f'{OUTPUT_DIR}/musician_analysis/individual_levels/'
    os.makedirs(musician_dir, exist_ok=True)
    
    for approach_name, approach_info in APPROACHES.items():
        # Create subplot for this approach showing all features by musician level
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{approach_name.replace("_", " ").title()} - Decoding by Musician Level', fontsize=16)
        
        # Features for this approach
        voice_feature = next((f for f in approach_info['features'] if 'voice' in f), None)
        location_feature = next((f for f in approach_info['features'] if 'location' in f), None)
        
        for level in [1, 2, 3]:
            col_idx = level - 1
            
            # Voice plot
            if voice_feature and voice_feature in ga_data[approach_name]:
                ax = axes[0, col_idx]
                data = ga_data[approach_name][voice_feature]
                
                if level in data['musician_stats']:
                    level_data = data['musician_stats'][level]
                    
                    # Plot individual subjects for this level
                    for subj_data in level_data['data']:
                        ax.plot(data['timepoints'], subj_data, 
                               color=MUSICIAN_COLORS[approach_name][level], 
                               alpha=0.3, linewidth=0.5)
                    
                    # Plot group average for this level
                    ax.plot(data['timepoints'], level_data['mean'],
                           color=MUSICIAN_COLORS[approach_name][level],
                           linewidth=3.0, label=f'Mean (n={level_data["n_subjects"]})')
                    
                    ax.fill_between(data['timepoints'],
                                   level_data['mean'] - level_data['sem'],
                                   level_data['mean'] + level_data['sem'],
                                   color=MUSICIAN_COLORS[approach_name][level],
                                   alpha=0.3)
                    
                    # Add cluster markers for this level
                    if 'musician_cluster_results' in data:
                        add_musician_cluster_markers(ax, data['musician_cluster_results'], level)
                
                ax.set_title(f'Voice - Musician Level {level}')
                ax.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7)
                ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
                ax.set_ylabel('Decoding Accuracy')
                ax.set_ylim(0.4, 0.65)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Location plot
            if location_feature and location_feature in ga_data[approach_name]:
                ax = axes[1, col_idx]
                data = ga_data[approach_name][location_feature]
                
                if level in data['musician_stats']:
                    level_data = data['musician_stats'][level]
                    
                    # Plot individual subjects for this level
                    for subj_data in level_data['data']:
                        ax.plot(data['timepoints'], subj_data, 
                               color=MUSICIAN_COLORS[approach_name][level], 
                               alpha=0.3, linewidth=0.5)
                    
                    # Plot group average for this level
                    ax.plot(data['timepoints'], level_data['mean'],
                           color=MUSICIAN_COLORS[approach_name][level],
                           linewidth=3.0, label=f'Mean (n={level_data["n_subjects"]})')
                    
                    ax.fill_between(data['timepoints'],
                                   level_data['mean'] - level_data['sem'],
                                   level_data['mean'] + level_data['sem'],
                                   color=MUSICIAN_COLORS[approach_name][level],
                                   alpha=0.3)
                    
                    # Add cluster markers for this level
                    if 'musician_cluster_results' in data:
                        add_musician_cluster_markers(ax, data['musician_cluster_results'], level)
                
                ax.set_title(f'Location - Musician Level {level}')
                ax.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7)
                ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Decoding Accuracy')
                ax.set_ylim(0.4, 0.65)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{musician_dir}/{approach_name}_all_levels_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved detailed musician level plot: {approach_name}")

def create_musician_cluster_summary_table(ga_data):
    """Create summary table of cluster results by musician level"""
    
    musician_dir = f'{OUTPUT_DIR}/musician_analysis/'
    
    cluster_summary = []
    
    for approach_name, approach_info in APPROACHES.items():
        for feature in approach_info['features']:
            if feature in ga_data[approach_name]:
                data = ga_data[approach_name][feature]
                feature_type = 'Voice' if 'voice' in feature else 'Location'
                
                # Overall group clusters
                overall_clusters = 0
                overall_duration = 0
                if data['cluster_results']:
                    overall_clusters = data['cluster_results']['n_significant']
                    overall_duration = sum([c['duration'] for c in data['cluster_results']['significant_clusters']])
                
                # Individual musician level clusters
                for level in [1, 2, 3]:
                    level_clusters = 0
                    level_duration = 0
                    level_n = 0
                    
                    if level in data['musician_stats']:
                        level_n = data['musician_stats'][level]['n_subjects']
                        
                        if ('musician_cluster_results' in data and 
                            level in data['musician_cluster_results'] and
                            data['musician_cluster_results'][level] is not None):
                            
                            level_cluster_results = data['musician_cluster_results'][level]
                            level_clusters = level_cluster_results['n_significant']
                            level_duration = sum([c['duration'] for c in level_cluster_results['significant_clusters']])
                    
                    cluster_summary.append({
                        'Approach': approach_name.replace('_', ' ').title(),
                        'Feature': feature_type,
                        'Musician_Level': level,
                        'N_Subjects': level_n,
                        'N_Clusters': level_clusters,
                        'Total_Cluster_Duration': f"{level_duration:.3f}",
                        'Overall_Group_Clusters': overall_clusters,
                        'Overall_Group_Duration': f"{overall_duration:.3f}"
                    })
    
    df = pd.DataFrame(cluster_summary)
    df.to_csv(f'{musician_dir}/musician_cluster_summary.csv', index=False)
    
    print("\nMusician Cluster Summary:")
    print("="*80)
    print(df.to_string(index=False))
    
    return df

def plot_musician_level_summary(ga_data):
    """Create summary plots comparing musician levels across approaches"""
    
    musician_dir = f'{OUTPUT_DIR}/musician_analysis/'
    os.makedirs(musician_dir, exist_ok=True)
    
    # Create comparison plots for voice and location separately
    for feature_type in ['voice', 'location']:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{feature_type.title()} Decoding by Musician Level Across Approaches', fontsize=16)
        
        for col_idx, approach_name in enumerate(['extreme_contrast', 'feature_pure', 'condition_specific']):
            ax = axes[col_idx]
            
            # Find relevant feature
            relevant_feature = None
            for feature in APPROACHES[approach_name]['features']:
                if feature_type in feature:
                    relevant_feature = feature
                    break
            
            if relevant_feature and relevant_feature in ga_data[approach_name]:
                data = ga_data[approach_name][relevant_feature]
                
                # Plot each musician level
                for level in [1, 2, 3]:
                    if level in data['musician_stats']:
                        level_data = data['musician_stats'][level]
                        
                        ax.plot(data['timepoints'], level_data['mean'],
                               label=f'Level {level} (n={level_data["n_subjects"]})',
                               color=MUSICIAN_COLORS[approach_name][level],
                               linewidth=2.5)
                        
                        ax.fill_between(data['timepoints'],
                                       level_data['mean'] - level_data['sem'],
                                       level_data['mean'] + level_data['sem'],
                                       color=MUSICIAN_COLORS[approach_name][level],
                                       alpha=0.2)
                
                # Add cluster markers
                add_cluster_markers(ax, data['cluster_results'])
                
                # Format subplot
                ax.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7)
                ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
                ax.set_title(approach_name.replace('_', ' ').title())
                ax.set_xlabel('Time (s)')
                if col_idx == 0:
                    ax.set_ylabel('Decoding Accuracy')
                ax.set_ylim(0.45, 0.60)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(CUE_TIME - 0.1, PROBE_TIME + 0.1)
        
        plt.tight_layout()
        plt.savefig(f'{musician_dir}/{feature_type}_comparison_by_musician_level.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {feature_type} comparison by musician level")

def plot_musician_level_statistics(ga_data):
    """Create statistical comparison plots for musician levels"""
    
    musician_dir = f'{OUTPUT_DIR}/musician_analysis/'
    os.makedirs(musician_dir, exist_ok=True)
    
    # Collect peak accuracy data by musician level
    peak_data = []
    
    for approach_name, approach_info in APPROACHES.items():
        for feature in approach_info['features']:
            if feature in ga_data[approach_name]:
                data = ga_data[approach_name][feature]
                feature_type = 'Voice' if 'voice' in feature else 'Location'
                
                for level in [1, 2, 3]:
                    if level in data['musician_stats']:
                        level_data = data['musician_stats'][level]
                        peak_acc = np.max(level_data['mean'])
                        
                        peak_data.append({
                            'Approach': approach_name.replace('_', ' ').title(),
                            'Feature': feature_type,
                            'Musician_Level': level,
                            'Peak_Accuracy': peak_acc,
                            'N_Subjects': level_data['n_subjects']
                        })
    
    if peak_data:
        df = pd.DataFrame(peak_data)
        
        # Create bar plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Peak Decoding Accuracy by Musician Level', fontsize=16)
        
        for row_idx, feature_type in enumerate(['Voice', 'Location']):
            for col_idx, approach_name in enumerate(['Extreme Contrast', 'Feature Pure', 'Condition Specific']):
                ax = axes[row_idx, col_idx]
                
                # Filter data
                subset = df[(df['Feature'] == feature_type) & (df['Approach'] == approach_name)]
                
                if not subset.empty:
                    # Get approach name for colors
                    approach_key = approach_name.lower().replace(' ', '_')
                    colors = [MUSICIAN_COLORS[approach_key][level] for level in subset['Musician_Level']]
                    
                    bars = ax.bar(subset['Musician_Level'], subset['Peak_Accuracy'], 
                                 color=colors, alpha=0.8, edgecolor='black')
                    
                    # Add N labels on bars
                    for bar, n in zip(bars, subset['N_Subjects']):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                               f'N={n}', ha='center', va='bottom', fontsize=10)
                
                ax.set_title(f'{feature_type} - {approach_name}')
                ax.set_xlabel('Musician Level')
                if col_idx == 0:
                    ax.set_ylabel('Peak Accuracy')
                ax.set_ylim(0.5, 0.60)
                ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_xticks([1, 2, 3])
        
        plt.tight_layout()
        plt.savefig(f'{musician_dir}/peak_accuracy_by_musician_level.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save summary table
        summary_table = df.groupby(['Approach', 'Feature', 'Musician_Level']).agg({
            'Peak_Accuracy': 'mean',
            'N_Subjects': 'first'
        }).round(3)
        
        summary_table.to_csv(f'{musician_dir}/musician_level_summary_stats.csv')
        
        print("Saved musician level statistical comparisons")

def plot_detailed_approach_analysis(ga_data):
    """Create detailed analysis plots for each approach"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Detailed Analysis by Approach', fontsize=16)
    
    for idx, (approach_name, approach_info) in enumerate(APPROACHES.items()):
        ax_time = axes[idx, 0]
        ax_stats = axes[idx, 1]
        
        # Time course plot
        ax_time.set_title(f'{approach_name.replace("_", " ").title()}', fontsize=14)
        
        for feature in approach_info['features']:
            if feature in ga_data[approach_name]:
                data = ga_data[approach_name][feature]
                feature_type = 'Voice' if 'voice' in feature else 'Location'
                
                ax_time.plot(data['timepoints'], data['mean'], 
                           label=f"{feature_type} (n={data['n_subjects']})", 
                           linewidth=2.5)
                
                ax_time.fill_between(data['timepoints'],
                                   data['mean'] - data['sem'],
                                   data['mean'] + data['sem'],
                                   alpha=0.2)
                
                # Add cluster markers
                add_cluster_markers(ax_time, data['cluster_results'])
        
        ax_time.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7)
        ax_time.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
        ax_time.set_xlabel('Time (s)')
        ax_time.set_ylabel('Decoding Accuracy')
        ax_time.set_ylim(0.45, 0.60)
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
        
        # Statistics plot
        stats_data = []
        for feature in approach_info['features']:
            if feature in ga_data[approach_name]:
                data = ga_data[approach_name][feature]
                
                # Calculate statistics
                peak_acc = np.max(data['mean'])
                peak_time = data['timepoints'][np.argmax(data['mean'])]
                ping_idx = np.argmin(np.abs(data['timepoints'] - PING_TIME))
                ping_acc = data['mean'][ping_idx]
                
                # Pre/post ping
                pre_mask = (data['timepoints'] >= 2.5) & (data['timepoints'] < PING_TIME)
                post_mask = (data['timepoints'] >= PING_TIME) & (data['timepoints'] <= 4.5)
                pre_mean = np.mean(data['mean'][pre_mask])
                post_mean = np.mean(data['mean'][post_mask])
                
                # Cluster statistics
                n_clusters = 0
                total_cluster_duration = 0
                if data['cluster_results']:
                    n_clusters = data['cluster_results']['n_significant']
                    total_cluster_duration = sum([c['duration'] for c in data['cluster_results']['significant_clusters']])
                
                stats_data.append({
                    'Feature': 'Voice' if 'voice' in feature else 'Location',
                    'Peak': peak_acc,
                    'Peak Time': peak_time,
                    'Ping Acc': ping_acc,
                    'Pre-ping': pre_mean,
                    'Post-ping': post_mean,
                    'Δ(Post-Pre)': post_mean - pre_mean,
                    'N Clusters': n_clusters,
                    'Cluster Duration': total_cluster_duration
                })
        
        # Create bar plot
        if stats_data:
            df = pd.DataFrame(stats_data)
            x = np.arange(len(df))
            width = 0.15
            
            metrics = ['Peak', 'Ping Acc', 'Pre-ping', 'Post-ping']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, metric in enumerate(metrics):
                offset = (i - 1.5) * width
                ax_stats.bar(x + offset, df[metric], width, 
                           label=metric, color=colors[i], alpha=0.8)
            
            ax_stats.set_xlabel('Feature')
            ax_stats.set_ylabel('Accuracy')
            ax_stats.set_title(f'{approach_name.replace("_", " ").title()} Statistics')
            ax_stats.set_xticks(x)
            ax_stats.set_xticklabels(df['Feature'])
            ax_stats.legend()
            ax_stats.set_ylim(0.5, 0.60)
            ax_stats.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
            ax_stats.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/ga_detailed_approach_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cluster_summary(ga_data):
    """Create a summary plot of all significant clusters"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Significant Clusters Summary', fontsize=16)
    
    # Collect all cluster data
    cluster_summary = []
    
    for approach_name, approach_info in APPROACHES.items():
        for feature in approach_info['features']:
            if feature in ga_data[approach_name]:
                data = ga_data[approach_name][feature]
                
                if data['cluster_results'] and data['cluster_results']['n_significant'] > 0:
                    for cluster in data['cluster_results']['significant_clusters']:
                        cluster_summary.append({
                            'approach': approach_name,
                            'feature': feature,
                            'feature_type': 'Voice' if 'voice' in feature else 'Location',
                            'start_time': cluster['start_time'],
                            'end_time': cluster['end_time'],
                            'duration': cluster['duration'],
                            'p_value': cluster['p_value'],
                            'color': approach_info['color']
                        })
    
    # Plot cluster timing for voice
    voice_clusters = [c for c in cluster_summary if c['feature_type'] == 'Voice']
    if voice_clusters:
        ax = axes[0]
        ax.set_title('Voice Decoding - Significant Clusters', fontsize=14)
        
        # Track unique approaches for legend
        plotted_approaches = set()
        
        for i, cluster in enumerate(voice_clusters):
            approach_label = cluster['approach'].replace('_', ' ').title()
            
            # Only add label for first occurrence of each approach
            label = approach_label if approach_label not in plotted_approaches else ""
            if label:
                plotted_approaches.add(approach_label)
            
            ax.barh(i, cluster['duration'], 
                   left=cluster['start_time'], 
                   color=cluster['color'], alpha=0.7,
                   label=label)
            
            # Add p-value text
            mid_time = cluster['start_time'] + cluster['duration']/2
            ax.text(mid_time, i, f"p={cluster['p_value']:.3f}", 
                   ha='center', va='center', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cluster')
        ax.set_xlim(CUE_TIME, PROBE_TIME)
        ax.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7)
        ax.grid(True, alpha=0.3)
        if plotted_approaches:
            ax.legend()
    
    # Plot cluster timing for location
    location_clusters = [c for c in cluster_summary if c['feature_type'] == 'Location']
    if location_clusters:
        ax = axes[1]
        ax.set_title('Location Decoding - Significant Clusters', fontsize=14)
        
        # Track unique approaches for legend
        plotted_approaches = set()
        
        for i, cluster in enumerate(location_clusters):
            approach_label = cluster['approach'].replace('_', ' ').title()
            
            # Only add label for first occurrence of each approach
            label = approach_label if approach_label not in plotted_approaches else ""
            if label:
                plotted_approaches.add(approach_label)
            
            ax.barh(i, cluster['duration'], 
                   left=cluster['start_time'], 
                   color=cluster['color'], alpha=0.7,
                   label=label)
            
            # Add p-value text
            mid_time = cluster['start_time'] + cluster['duration']/2
            ax.text(mid_time, i, f"p={cluster['p_value']:.3f}", 
                   ha='center', va='center', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cluster')
        ax.set_xlim(CUE_TIME, PROBE_TIME)
        ax.axvline(x=PING_TIME, color='red', linestyle='--', alpha=0.7)
        ax.grid(True, alpha=0.3)
        if plotted_approaches:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/ga_cluster_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(ga_data):
    """Create a comprehensive summary table including cluster results"""
    summary_data = []
    
    for approach_name, approach_info in APPROACHES.items():
        for feature in approach_info['features']:
            if feature in ga_data[approach_name]:
                data = ga_data[approach_name][feature]
                
                # Calculate all statistics
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
                time_above = np.sum(above_chance) * 0.01
                
                # Cluster statistics
                n_clusters = 0
                total_cluster_duration = 0
                min_cluster_p = 1.0
                if data['cluster_results']:
                    n_clusters = data['cluster_results']['n_significant']
                    if n_clusters > 0:
                        total_cluster_duration = sum([c['duration'] for c in data['cluster_results']['significant_clusters']])
                        min_cluster_p = min([c['p_value'] for c in data['cluster_results']['significant_clusters']])
                
                # Musician level statistics
                musician_breakdown = []
                for level in [1, 2, 3, None]:
                    if level in data['musician_stats']:
                        n = data['musician_stats'][level]['n_subjects']
                        level_name = f"L{level}" if level is not None else "Unk"
                        musician_breakdown.append(f"{level_name}:{n}")
                
                summary_data.append({
                    'Approach': approach_name.replace('_', ' ').title(),
                    'Feature': 'Voice' if 'voice' in feature else 'Location',
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
                    'Min Cluster p-value': f"{min_cluster_p:.6f}" if min_cluster_p < 1.0 else "n.s."
                })
    
    df = pd.DataFrame(summary_data)
    return df

def save_ga_data(ga_data):
    """Save grand average data and statistics including cluster results"""
    # Save raw GA data
    np.save(f'{OUTPUT_DIR}/ga_comprehensive_data.npy', ga_data, allow_pickle=True)
    
    # Create and save summary table
    summary_df = create_summary_table(ga_data)
    summary_df.to_csv(f'{OUTPUT_DIR}/ga_comprehensive_summary_table.csv', index=False)
    
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))
    
    # Save cluster results separately
    cluster_results = {}
    for approach_name, approach_info in APPROACHES.items():
        cluster_results[approach_name] = {}
        for feature in approach_info['features']:
            if feature in ga_data[approach_name]:
                cluster_results[approach_name][feature] = ga_data[approach_name][feature]['cluster_results']
    
    np.save(f'{OUTPUT_DIR}/ga_cluster_results.npy', cluster_results, allow_pickle=True)
    
    # Save musician level data separately
    musician_results = {}
    for approach_name, approach_info in APPROACHES.items():
        musician_results[approach_name] = {}
        for feature in approach_info['features']:
            if feature in ga_data[approach_name]:
                musician_results[approach_name][feature] = {
                    'musician_levels': ga_data[approach_name][feature]['musician_levels'],
                    'musician_stats': ga_data[approach_name][feature]['musician_stats'],
                    'musician_cluster_results': ga_data[approach_name][feature].get('musician_cluster_results', {})
                }
    
    np.save(f'{OUTPUT_DIR}/ga_musician_results.npy', musician_results, allow_pickle=True)
    
    # Create JSON summary
    summary_json = {
        'n_subjects_requested': len(args.subjects),
        'subjects_requested': args.subjects,
        'processing_date': str(datetime.now()),
        'approaches': list(APPROACHES.keys()),
        'cluster_parameters': {
            'n_permutations': args.n_permutations,
            'alpha_level': args.cluster_alpha,
            'test_type': 'one_sample_t_test_vs_chance'
        },
        'statistics': {}
    }
    
    # Add detailed statistics
    for approach_name in APPROACHES.keys():
        summary_json['statistics'][approach_name] = {}
        
        for feature in APPROACHES[approach_name]['features']:
            if feature in ga_data[approach_name]:
                data = ga_data[approach_name][feature]
                
                # Basic statistics
                stats_dict = {
                    'n_subjects': data['n_subjects'],
                    'subject_ids': data['subjects'],
                    'mean_accuracy': float(np.mean(data['mean'])),
                    'peak_accuracy': float(np.max(data['mean'])),
                    'peak_time': float(data['timepoints'][np.argmax(data['mean'])]),
                    'cluster_results': {},
                    'musician_level_breakdown': {}
                }
                
                # Cluster statistics
                if data['cluster_results']:
                    stats_dict['cluster_results'] = {
                        'n_significant_clusters': data['cluster_results']['n_significant'],
                        'significant_clusters': data['cluster_results']['significant_clusters']
                    }
                
                # Musician level statistics
                for level in [1, 2, 3, None]:
                    if level in data['musician_stats']:
                        level_data = data['musician_stats'][level]
                        level_key = f"level_{level}" if level is not None else "unknown"
                        level_stats = {
                            'n_subjects': level_data['n_subjects'],
                            'peak_accuracy': float(np.max(level_data['mean'])),
                            'mean_accuracy': float(np.mean(level_data['mean']))
                        }
                        
                        # Add cluster results for this level if available
                        if ('musician_cluster_results' in data and 
                            level in data.get('musician_cluster_results', {}) and
                            data['musician_cluster_results'][level] is not None):
                            
                            level_clusters = data['musician_cluster_results'][level]
                            level_stats['cluster_results'] = {
                                'n_significant_clusters': level_clusters['n_significant'],
                                'significant_clusters': level_clusters['significant_clusters']
                            }
                        
                        stats_dict['musician_level_breakdown'][level_key] = level_stats
                
                summary_json['statistics'][approach_name][feature] = stats_dict
    
    with open(f'{OUTPUT_DIR}/ga_comprehensive_summary.json', 'w') as f:
        json.dump(summary_json, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}")

def main():
    """Main analysis pipeline"""
    print(f"Grand Average Analysis for Comprehensive Decoding with Musician Levels")
    print(f"Total subjects requested: {len(args.subjects)}")
    print(f"Cluster permutation parameters: {args.n_permutations} permutations, α = {args.cluster_alpha}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Compute grand average
    ga_data = compute_grand_average()
    
    # Compute musician-level cluster permutation tests
    ga_data = compute_musician_level_clusters(ga_data)
    
    # Check how many subjects actually had data
    total_valid = 0
    for approach in APPROACHES.keys():
        for feature in APPROACHES[approach]['features']:
            if approach in ga_data and feature in ga_data[approach]:
                n_subj = ga_data[approach][feature]['n_subjects']
                total_valid = max(total_valid, n_subj)
    
    print(f"\nMaximum subjects with valid data: {total_valid}/{len(args.subjects)}")
    
    # Generate plots
    print("\nGenerating main plots...")
    plot_approach_comparison(ga_data)
    plot_detailed_approach_analysis(ga_data)
    plot_individual_differences_with_musicians(ga_data)
    plot_cluster_summary(ga_data)
    
    # Generate musician-specific plots with individual cluster tests
    print("\nGenerating musician-level analysis plots with cluster tests...")
    plot_musician_level_timecourses_with_clusters(ga_data)
    plot_individual_musician_level_plots(ga_data)
    plot_musician_level_summary(ga_data)
    plot_musician_level_statistics(ga_data)
    
    # Create musician cluster summary
    print("\nCreating musician cluster summary...")
    create_musician_cluster_summary_table(ga_data)
    
    # Save results
    print("\nSaving results...")
    save_ga_data(ga_data)
    
    print("\nAnalysis complete!")
    print(f"Check the following directories for results:")
    print(f"- Main plots: {OUTPUT_DIR}")
    print(f"- Musician analysis: {OUTPUT_DIR}/musician_analysis/")
    print(f"- Individual level details: {OUTPUT_DIR}/musician_analysis/individual_levels/")

if __name__ == "__main__":
    main()