#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Correlation Analysis for Delay-Period MEG Decoding

This script performs:
1. Loading existing decoding results from the sliding window analysis
2. Correlating decoding performance with behavioral measures
3. Enhancing single-subject analysis with detailed visualizations 
4. Generating group-level summaries and comparisons

Run this script directly after the original sliding window decoding is complete.
"""

import os
import glob
import locale
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pickle
import scipy
from sklearn.metrics import roc_curve, auc
import warnings
import sys
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import mne

warnings.filterwarnings("ignore")

# Set non-interactive backend for server/headless running
matplotlib.use('Agg')

# Define constants
HOME_DIR = '/media/headmodel/Elements/AWM4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
DATA_PATH = HOME_DIR + '/AWM4_data/raw/'
CORRECTED_DATA = HOME_DIR + '/AWM4_data/raw/correctTriggers/'
BEHAVIOR_DIR = HOME_DIR + '/AWM4_data/behavioral'

# Analysis constants
CRITICAL_TIMEPOINTS = [3.5, 4.5]  # Critical time points to mark on plots
RESAMPLE_FREQ = 100  # Hz
WINDOW_LENGTH_SEC = 0.1  # seconds (100ms)
WINDOW_STEP_SEC = 0.01  # seconds (10ms)

# Pre-post ping analysis constants
PRE_PING_START = 2.0
PRE_PING_END = 3.5
POST_PING_START = 3.5
POST_PING_END = 4.7

# Feature definitions
FEATURES = {
    'maintained_voice_identity': {
        'name': 'Maintained Voice Identity',
        'color': '#1c686b',  # Darker teal
        'description': 'Voice identity information maintained in working memory'
    },
    'maintained_location': {
        'name': 'Maintained Location',
        'color': '#cb6a3e',  # Darker orange
        'description': 'Location information maintained in working memory'
    }
}

# Common directory paths
def get_feature_dir(feature_name):
    """Get directory for a given feature"""
    return f"{PROCESSED_DIR}/timepoints/delay_sliding_window/{feature_name}"

def get_corr_dir(feature_name):
    """Get correlation directory for a given feature"""
    return f"{PROCESSED_DIR}/timepoints/delay_sliding_window/behavior_correlations/{feature_name}"

def get_single_subject_dir(feature_name):
    """Get single subject directory for a given feature"""
    return f"{PROCESSED_DIR}/timepoints/delay_sliding_window/single_subject/{feature_name}"

def get_music_dir(feature_name):
    """Get musical experience directory for a given feature"""
    return f"{PROCESSED_DIR}/timepoints/delay_sliding_window/musical_experience/{feature_name}"

def get_summary_dir():
    """Get summary directory"""
    return f"{PROCESSED_DIR}/timepoints/delay_sliding_window/summary"

def get_subject_results_path(feature_name, subject):
    """Get path to subject's decoding results"""
    return f"{get_feature_dir(feature_name)}/sub{subject}_results.pkl"

# Utility functions
def add_significance_stars(p_value):
    """Add significance stars based on p-value"""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""

def get_pre_post_ping_masks(window_centers):
    """Get boolean masks for pre and post ping periods"""
    pre_ping_mask = (window_centers >= PRE_PING_START) & (window_centers < PRE_PING_END)
    post_ping_mask = (window_centers >= POST_PING_START) & (window_centers <= POST_PING_END)
    return pre_ping_mask, post_ping_mask

def setup_directories():
    """Create all necessary directories for analysis"""
    # Main directory
    os.makedirs(f"{PROCESSED_DIR}/timepoints/delay_sliding_window", exist_ok=True)
    
    # Create directories for each feature
    for feature in FEATURES.keys():
        # Main feature directory
        os.makedirs(get_feature_dir(feature), exist_ok=True)
        
        # Correlation directory
        os.makedirs(get_corr_dir(feature), exist_ok=True)
        
        # Single subject directory
        os.makedirs(get_single_subject_dir(feature), exist_ok=True)
        
        # Musical experience directory
        os.makedirs(get_music_dir(feature), exist_ok=True)
        
        # MATLAB-style correlation directory
        os.makedirs(f"{PROCESSED_DIR}/timepoints/delay_sliding_window/matlab_correlations/{feature}", exist_ok=True)
    
    # Summary directory
    os.makedirs(get_summary_dir(), exist_ok=True)
    
    # Musical experience directory
    os.makedirs(f"{PROCESSED_DIR}/timepoints/delay_sliding_window/musical_experience", exist_ok=True)
    
    # Behavioral analysis directory
    os.makedirs(f"{PROCESSED_DIR}/behavioral_analysis", exist_ok=True)
    
    print("Created all necessary directories")

def load_subject_metadata(subjects):
    """
    Load subject metadata including musical experience from the MEGNotes.xlsx file.
    """
    print(f"\nLoading subject metadata from {META_FILE}")
    
    # Load metadata file
    metadata_df = pd.read_excel(META_FILE)
    print(f"Loaded metadata with {len(metadata_df)} entries")
    
    # Extract unique subjects and their musician level
    unique_rows = []
    for subject in subjects:
        # Get all rows for this subject
        subject_rows = metadata_df[metadata_df['Subject'] == subject]
        
        if len(subject_rows) > 0:
            # Get the first row (all 8 should have the same Musician value)
            first_row = subject_rows.iloc[0]
            
            # Add to our list of unique rows
            if 'Musician' in first_row:
                unique_rows.append({
                    'Subject': subject,
                    'Musician': first_row['Musician'],
                    'SubjectCode': first_row['SubjectCode'] if 'SubjectCode' in first_row else None
                })
    
    # Create a new DataFrame from the list of rows
    unique_subjects_df = pd.DataFrame(unique_rows)
    
    print(f"Found {len(unique_subjects_df)} unique subjects with metadata")
    
    # Check if Musician column exists
    if 'Musician' in unique_subjects_df.columns:
        print("Found 'Musician' information in metadata")
        # Count subjects in each musical experience category
        musician_counts = unique_subjects_df['Musician'].value_counts().sort_index()
        print("Musical experience levels distribution:")
        for level, count in musician_counts.items():
            if level == 1:
                description = "Little to no musical experience"
            elif level == 2:
                description = "Moderate musical experience"
            elif level == 3:
                description = "Experienced (>10 years)"
            else:
                description = "Unknown level"
                
            print(f"  Level {level} ({description}): {count} subjects")
    else:
        print("'Musician' column not found in metadata")
    
    return unique_subjects_df


def build_subject_code_map(metadata_df):
    """
    Build a mapping from subject IDs to subject codes.
    """
    subject_code_map = {}
    for _, row in metadata_df.iterrows():
        if 'Subject' in row and 'SubjectCode' in row:
            subject_id = row['Subject']
            subject_code = row['SubjectCode']
            if isinstance(subject_code, str) and subject_code.startswith('S'):
                subject_code = 'VP' + subject_code[1:]
            if subject_id not in subject_code_map:
                subject_code_map[subject_id] = subject_code
    
    print(f"Created mapping for {len(subject_code_map)} subjects")
    return subject_code_map

def calculate_dprime(data, match_column):
    """
    Calculate d-prime for a given match condition.
    
    Parameters:
    data - DataFrame with behavioral data
    match_column - Column name for match condition ('SpeakerMatch' or 'LocMatch')
    
    Returns:
    dprime - Calculated d-prime value
    """
    # Calculate hits, misses, false alarms, correct rejections
    hits = len(data[(data[match_column] == 1) & (data['Answer'] == 1)])
    misses = len(data[(data[match_column] == 1) & (data['Answer'] == 0)])
    fas = len(data[(data[match_column] == 0) & (data['Answer'] == 1)])
    crs = len(data[(data[match_column] == 0) & (data['Answer'] == 0)])
    
    # Adjust to avoid infinite values (add 0.5 to empty cells)
    total_yes = len(data[data[match_column] == 1])
    total_no = len(data[data[match_column] == 0])
    
    hits = max(0.5, min(hits, total_yes - 0.5))
    fas = max(0.5, min(fas, total_no - 0.5))
    
    # Calculate hit rate and false alarm rate
    hit_rate = hits / (hits + misses)
    fa_rate = fas / (fas + crs)
    
    # Calculate d-prime
    dprime = scipy.stats.norm.ppf(hit_rate) - scipy.stats.norm.ppf(fa_rate)
    
    return dprime

def analyze_behavioral_data_detailed(subjects):
    """
    Analyze behavioral data following the MATLAB approach, calculating accuracy and d-prime
    for different conditions (speaker match/non-match, location match/non-match).
    
    Parameters:
    subjects - List of subject IDs
    
    Returns:
    behavior_df - DataFrame with detailed behavioral measures for each subject
    """
    print(f"\nPerforming detailed behavioral analysis following MATLAB approach...")
    
    behavior_dir = f"{HOME_DIR}/AWM4_data/behavioral"
    
    # Column names based on specified format
    columns = ['Block', 'Trial', 'BlockTrial', 'SpeakerS1', 'LocS1', 'SpeakerS2', 
              'LocS2', 'TargetPos', 'SpeakerMatch', 'ProbeDistance', 'LocMatch', 
              'ProbeLoc', 'Answer', 'Correct', 'RT', 'Trialduration', 'Syllable']
    
    # Initialize arrays to store behavioral metrics
    all_results = []
    
    # Load metadata for subject codes
    metadata_df = pd.read_excel(META_FILE)
    subject_code_map = build_subject_code_map(metadata_df)
    
    # Process each subject
    for subject in subjects:
        # Get subject code if available, otherwise use default naming
        if subject in subject_code_map:
            subject_code = subject_code_map[subject]
        else:
            subject_code = f"VP{subject:02d}"
        
        # Find the subject's behavioral data file
        file_patterns = [
            f"{behavior_dir}/Output_AWM4_Exp1_MEG_{subject_code}.txt"
        ]
        
        file_found = False
        for pattern in file_patterns:
            matching_files = glob.glob(pattern)
            if matching_files:
                file_path = matching_files[0]
                file_found = True
                break
                
        if not file_found:
            print(f"No behavioral data file found for subject {subject}")
            continue
        
        print(f"Analyzing behavioral data for subject {subject} using file: {os.path.basename(file_path)}")
        
        # Read the data file - keeping this exactly as requested
        data = pd.read_csv(file_path, sep=' ', skiprows=1, names=columns)
        
        # Initialize results dictionary for this subject
        subject_results = {'Subject': subject}
        
        # Calculate accuracies for each condition (SpeakerMatch x LocMatch)
        for speaker_match in [0, 1]:  # Non-match, Match
            for loc_match in [0, 1]:   # Non-match, Match
                # Filter trials for this condition
                condition_mask = (data['SpeakerMatch'] == speaker_match) & (data['LocMatch'] == loc_match)
                condition_trials = data[condition_mask]
                
                if len(condition_trials) > 0:
                    # Calculate accuracy (percentage correct)
                    accuracy = condition_trials['Correct'].mean() * 100
                    subject_results[f'Acc_Spk{speaker_match}_Loc{loc_match}'] = accuracy
                else:
                    subject_results[f'Acc_Spk{speaker_match}_Loc{loc_match}'] = np.nan
        
        # Overall accuracy
        subject_results['Accuracy'] = data['Correct'].mean() * 100
        
        # Calculate d-prime for Voice (Speaker) identity
        voice_dprime = calculate_dprime(data, 'SpeakerMatch')
        subject_results['VoiceDprime'] = voice_dprime
        
        # Location d-prime
        location_dprime = calculate_dprime(data, 'LocMatch')
        subject_results['LocationDprime'] = location_dprime
        
        # Overall d-prime (average of voice and location)
        subject_results['OverallDprime'] = (voice_dprime + location_dprime) / 2
        
        # First half vs second half accuracy
        mid_point = len(data) // 2
        first_half = data.iloc[:mid_point]
        second_half = data.iloc[mid_point:]
        
        subject_results['FirstHalfAcc'] = first_half['Correct'].mean() * 100
        subject_results['SecondHalfAcc'] = second_half['Correct'].mean() * 100
        
        # Add average RT
        subject_results['RT'] = data['RT'].mean()
        
        # Add to results
        all_results.append(subject_results)
    
    if not all_results:
        print("No behavioral data was successfully analyzed")
        return None
    
    # Create DataFrame from results
    behavior_df = pd.DataFrame(all_results)
    
    # Save detailed results
    behavior_df.to_excel(f"{PROCESSED_DIR}/behavioral_analysis/detailed_behavioral_metrics.xlsx")
    
    print(f"Successfully analyzed behavioral data for {len(behavior_df)} subjects")
    print(f"Calculated metrics: {list(behavior_df.columns)[1:]}")
    
    return behavior_df

def load_subject_decoding_data(subject, feature_name):
    """
    Load decoding data for a specific subject and feature.
    
    Parameters:
    subject - Subject ID
    feature_name - Name of the feature
    
    Returns:
    subject_data - Dictionary with subject's decoding data or None if not found
    """
    subject_results_path = get_subject_results_path(feature_name, subject)
    
    if os.path.exists(subject_results_path):
        with open(subject_results_path, 'rb') as f:
            return pickle.load(f)
    else:
        return None


def calculate_decoding_metrics(mean_scores, window_centers):
    """
    Calculate summary metrics from decoding scores.
    
    Parameters:
    mean_scores - Array of mean decoding scores
    window_centers - Array of window centers (time points)
    
    Returns:
    metrics - Dictionary with decoding metrics
    """
    # Get pre/post ping masks
    pre_ping_mask, post_ping_mask = get_pre_post_ping_masks(window_centers)
    
    # Calculate metrics
    metrics = {
        'mean_accuracy': np.mean(mean_scores),
        'peak_accuracy': np.max(mean_scores),
        'pre_ping': np.mean(mean_scores[pre_ping_mask]),
        'post_ping': np.mean(mean_scores[post_ping_mask]),
        'ping_difference': np.mean(mean_scores[post_ping_mask]) - np.mean(mean_scores[pre_ping_mask])
    }
    
    # Time point of peak accuracy
    peak_idx = np.argmax(mean_scores)
    metrics['peak_time'] = window_centers[peak_idx]
    
    return metrics


def analyze_behavior_decoding_correlation(subjects, feature_name, behavior_df=None):
    """
    Analyze correlation between decoding performance and behavioral metrics
    
    Parameters:
    subjects - List of subject IDs
    feature_name - Name of the feature to analyze
    behavior_df - Optional pre-loaded behavioral DataFrame
    
    Returns:
    correlation_results - Dictionary with correlation results
    analysis_df - DataFrame with combined behavioral and decoding data
    """
    print(f"\nAnalyzing correlations for {FEATURES[feature_name]['name']}...")
    
    # Load behavioral data if not provided
    if behavior_df is None:
        behavior_df = analyze_behavioral_data_detailed(subjects)
    
    if behavior_df is None:
        print("No behavioral data available for analysis")
        return None, None
        
    # Create path for correlation results
    corr_path = get_corr_dir(feature_name)
    
    # Load decoding scores for each subject
    decoding_scores = {}
    
    for subject in subjects:
        subject_data = load_subject_decoding_data(subject, feature_name)
        
        if subject_data is not None:
            # Calculate summary metrics
            mean_scores = subject_data['mean_scores']
            window_centers = subject_data['window_centers']
            
            # Store metrics for this subject
            decoding_scores[subject] = calculate_decoding_metrics(mean_scores, window_centers)
        else:
            print(f"No decoding results found for subject {subject}")
    
    if not decoding_scores:
        print("No decoding data available for correlation")
        return None, None
    
    # Create DataFrame from decoding scores
    decoding_df = pd.DataFrame.from_dict(decoding_scores, orient='index')
    decoding_df['Subject'] = decoding_df.index
    
    # Merge with behavioral data
    analysis_df = pd.merge(decoding_df, behavior_df, on='Subject', how='inner')
    
    if len(analysis_df) == 0:
        print("No matching subjects found between decoding and behavioral data")
        return None, None
    
    print(f"Found {len(analysis_df)} subjects with both decoding and behavioral data")
    
    # Save merged data for reference
    analysis_df.to_excel(f"{corr_path}/combined_data.xlsx")
    
    # Perform correlations
    behavioral_metrics = ['Accuracy', 'RT', 'VoiceDprime', 'LocationDprime', 'OverallDprime']
    if 'Musician' in analysis_df.columns:
        behavioral_metrics.append('Musician')
        
    decoding_metrics = ['mean_accuracy', 'peak_accuracy', 'pre_ping', 'post_ping', 'ping_difference']
    
    # Initialize results dictionary
    correlation_results = {}
    
    # Calculate and visualize correlations
    for b_metric in behavioral_metrics:
        if b_metric in analysis_df.columns:
            correlation_results[b_metric] = {}
            
            for d_metric in decoding_metrics:
                # Skip if one of the columns doesn't exist
                if d_metric not in analysis_df.columns:
                    continue
                    
                # Drop rows with NaN values
                valid_data = analysis_df[[b_metric, d_metric, 'Subject']].dropna()
                
                if len(valid_data) < 3:
                    print(f"Not enough data for {b_metric} vs {d_metric}")
                    continue
                
                # Calculate correlation
                r, p = scipy.stats.pearsonr(valid_data[b_metric], valid_data[d_metric])
                correlation_results[b_metric][d_metric] = {'r': r, 'p': p}
                
                # Create scatter plot
                create_correlation_plot(valid_data, b_metric, d_metric, r, p, corr_path)
                
                print(f"  {b_metric} vs {d_metric}: r={r:.3f}, p={p:.3f}")
    
    return correlation_results, analysis_df


def create_correlation_plot(data, x_var, y_var, r, p, save_path):
    """
    Create and save correlation scatter plot.
    
    Parameters:
    data - DataFrame with data to plot
    x_var - Name of x variable
    y_var - Name of y variable
    r - Correlation coefficient
    p - p-value
    save_path - Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(data[x_var], data[y_var], alpha=0.7)
    
    # Add regression line
    x = data[x_var]
    y = data[y_var]
    z = np.polyfit(x, y, 1)
    p_fit = np.poly1d(z)
    plt.plot(x, p_fit(x), "r--", alpha=0.7)
    
    # Annotate points with subject IDs
    for _, row in data.iterrows():
        plt.annotate(str(int(row['Subject'])), 
                  (row[x_var], row[y_var]), 
                  fontsize=9)
    
    # Add correlation statistics
    plt.title(f"{x_var} vs {y_var}")
    plt.xlabel(x_var)
    plt.ylabel(y_var.replace('_', ' ').title())
    
    stars = add_significance_stars(p)
    
    plt.annotate(f"r = {r:.2f}, p = {p:.3f}{stars}", 
               xy=(0.05, 0.95), xycoords='axes fraction', 
               fontsize=12, ha='left', va='top',
               bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{x_var}_vs_{y_var}.png")
    plt.close()


def analyze_musical_experience(subjects, feature_name, metadata_df=None):
    """
    Analyze how musical experience relates to decoding performance
    
    Parameters:
    subjects - List of subject IDs
    feature_name - Name of the feature to analyze
    metadata_df - Optional pre-loaded metadata DataFrame
    
    Returns:
    analysis_results - Dictionary with analysis results
    """
    print(f"\n==== Analyzing musical experience effects on {FEATURES[feature_name]['name']} ====")
    
    # Create output directory
    music_path = get_music_dir(feature_name)
    
    # Load metadata if not provided
    if metadata_df is None:
        metadata_df = load_subject_metadata(subjects)
    
    if metadata_df is None or 'Musician' not in metadata_df.columns:
        print("Cannot analyze musical experience: missing metadata")
        return None
    
    # Load decoding results for each subject
    decoding_measures = {}
    
    for subject in subjects:
        subject_data = load_subject_decoding_data(subject, feature_name)
        
        if subject_data is not None:
            # Calculate metrics
            mean_scores = subject_data['mean_scores']
            window_centers = subject_data['window_centers']
            
            # Store metrics and original data
            metrics = calculate_decoding_metrics(mean_scores, window_centers)
            decoding_measures[subject] = metrics
            
            # Also store raw time course for later
            decoding_measures[subject]['time_course'] = mean_scores
            decoding_measures[subject]['window_centers'] = window_centers
        else:
            print(f"No decoding results found for subject {subject}")
    
    if not decoding_measures:
        print("No decoding measures available for analysis")
        return None
    
    # Create DataFrame from decoding measures
    decoding_df = pd.DataFrame.from_dict(decoding_measures, orient='index')
    
    # Combine with metadata
    analysis_df = pd.merge(
        decoding_df.drop(['time_course', 'window_centers'], axis=1, errors='ignore'),
        metadata_df[['Subject', 'Musician']], 
        left_index=True, 
        right_on='Subject',
        how='inner'
    )
    
    if len(analysis_df) == 0:
        print("No subjects with both decoding data and musical experience information")
        return None
    
    print(f"Analysis includes {len(analysis_df)} subjects with musical experience data")
    
    # Convert Musician to categorical for better analysis
    analysis_df['Musician'] = analysis_df['Musician'].astype('category')
    
    # Save combined data
    analysis_df.to_excel(f"{music_path}/musical_experience_decoding.xlsx")
    
    # Analyze correlations between musical experience and decoding metrics
    correlation_results = analyze_music_correlations(analysis_df)
    
    # Group subjects by musical experience
    grouped_results = analysis_df.groupby('Musician')
    group_means = grouped_results.mean()
    group_sems = grouped_results.sem()
    
    # Create comparison plots
    create_music_comparison_plot(group_means, group_sems, grouped_results, feature_name, music_path)
    
    # Create time course plot by musical experience
    create_music_timecourse_plot_with_clusters(decoding_measures, metadata_df, feature_name, music_path)
    
    # Return results
    return {
        'correlation': correlation_results,
        'group_means': group_means,
        'combined_data': analysis_df
    }


def analyze_music_correlations(analysis_df):
    """
    Analyze correlations between musical experience and decoding metrics.
    
    Parameters:
    analysis_df - DataFrame with analysis data
    
    Returns:
    correlation_results - Dictionary with correlation results
    """
    print("\nCorrelating musical experience with decoding metrics:")
    decoding_metrics = ['mean_accuracy', 'peak_accuracy', 'pre_ping', 'post_ping', 'ping_difference']
    
    correlation_results = {}
    for metric in decoding_metrics:
        if metric in analysis_df.columns:
            # Use Spearman correlation since musical experience is ordinal
            r, p = scipy.stats.spearmanr(analysis_df['Musician'], analysis_df[metric])
            correlation_results[metric] = {'r': r, 'p': p}
            
            print(f"  {metric}: r={r:.3f}, p={p:.3f}")
    
    return correlation_results


def create_music_comparison_plot(group_means, group_sems, grouped_results, feature_name, save_path):
    """
    Create bar chart comparing pre and post ping by musical experience.
    
    Parameters:
    group_means - DataFrame with group means
    group_sems - DataFrame with group SEMs
    grouped_results - GroupBy object for getting group counts
    feature_name - Name of the feature
    save_path - Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Set width of bars
    barWidth = 0.3
    
    # Set positions of bars on X axis
    r1 = np.arange(len(group_means.index))
    r2 = [x + barWidth for x in r1]
    
    # Create bars
    plt.bar(r1, group_means['pre_ping'], width=barWidth, edgecolor='white', 
           yerr=group_sems['pre_ping'], capsize=7, label='Pre-Ping', color='blue', alpha=0.7)
    plt.bar(r2, group_means['post_ping'], width=barWidth, edgecolor='white',
           yerr=group_sems['post_ping'], capsize=7, label='Post-Ping', color='orange', alpha=0.7)
    
    # Add chance level
    plt.axhline(y=0.5, color='black', linestyle='--', label='Chance')
    
    # Add labels
    plt.xlabel('Musical Experience Level')
    plt.ylabel('Decoding Accuracy')
    plt.title(f'Pre vs Post Ping Decoding by Musical Experience\n({FEATURES[feature_name]["name"]})')
    
    # Adjust x-axis
    plt.xticks([r + barWidth/2 for r in range(len(group_means.index))], 
              [f'Level {int(i)}' for i in group_means.index])
    
    # Add legend
    plt.legend()
    
    # Add counts
    for i, level in enumerate(group_means.index):
        count = len(grouped_results.get_group(level))
        plt.text(i, 0.3, f'n={count}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/pre_post_ping_by_experience.png")
    plt.close()

CLUSTER_OUTPUT_PATH = f"{PROCESSED_DIR}/timepoints/delay_sliding_window/cluster_stats"

def run_cluster_permutation_test(scores, window_centers, feature_name):
    """
    Run a cluster permutation test on the decoding scores.
    
    Args:
        scores (numpy.ndarray): Array of decoding scores (subjects × timepoints)
        window_centers (numpy.ndarray): Array of time points
        feature_name (str): Name of the feature
        
    Returns:
        tuple: (clusters, cluster_p_values, significant_points, significant_times)
    """
    # Create results path
    results_path = f"{CLUSTER_OUTPUT_PATH}/{feature_name}"
    os.makedirs(results_path, exist_ok=True)
    
    # Convert scores to stats (difference from chance)
    stats_mean = scores - 0.5
    
    # Run cluster permutation test
    try:
        print(f"Running cluster permutation test for {feature_name}...")
        T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
            stats_mean, 
            n_permutations=100000,
            threshold=None,
            tail=1,  # One-tailed test (looking for accuracy > chance)
            out_type='mask'
        )
        
        # If no clusters were found
        if len(clusters) == 0:
            print(f"No clusters found for {feature_name}")
            return [], [], [], []
        
        # Convert clusters to start/stop indices
        clusters_converted = np.array([[cl.start, cl.stop] for cl in [c[0] for c in clusters]])
        
        # Stack clusters and p-values
        results = np.column_stack((clusters_converted, cluster_p_values))
        
        # Save to file
        output_file = os.path.join(results_path, f'clusters_{feature_name}.txt')
        np.savetxt(output_file, results, fmt='%d %d %f', header="start stop p_value", comments='')
        
        # Find significant clusters (p < 0.05)
        sigPs = np.argwhere(cluster_p_values < 0.05)[:, 0]
        
        # Extract significant time points
        significant_points = []
        significant_times = []
        
        for p_idx in sigPs:
            cluster_mask = clusters[p_idx][0]
            sig_indices = np.where(cluster_mask)[0]
            significant_points.extend(sig_indices)
            
            # Extract corresponding time values
            sig_times = window_centers[sig_indices]
            significant_times.extend(sig_times)
            
            # Print cluster information
            print(f"Significant cluster found: p={cluster_p_values[p_idx]:.4f}, "
                  f"time range: {window_centers[sig_indices[0]]:.2f}s - {window_centers[sig_indices[-1]]:.2f}s")
        
        return clusters, cluster_p_values, significant_points, significant_times
    
    except Exception as e:
        print(f"Error in cluster permutation test: {e}")
        import traceback
        traceback.print_exc()
        return [], [], [], []

def create_detailed_significance_plot(time_courses_by_level, window_centers, significant_times_by_level, feature_name, save_path):
    """
    Create a detailed plot focusing on significant time periods.
    
    Parameters:
    time_courses_by_level - Dictionary of time courses by level
    window_centers - Array of time points
    significant_times_by_level - Dictionary of significant times by level
    feature_name - Name of the feature
    save_path - Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot average time course for each level
    for level, time_courses in time_courses_by_level.items():
        if time_courses:  # Skip empty levels
            time_courses_array = np.array(time_courses)
            mean_time_course = np.mean(time_courses_array, axis=0)
            sem_time_course = np.std(time_courses_array, axis=0) / np.sqrt(len(time_courses))
            
            if level == 1:
                label = "Level 1 (Little/No Experience)"
                color = '#1f77b4'
            elif level == 2:
                label = "Level 2 (Moderate Experience)"
                color = '#ff7f0e'
            else:
                label = "Level 3 (>10 Years Experience)"
                color = '#2ca02c'
            
            # Plot the time course
            plt.plot(window_centers, mean_time_course, '-', 
                   label=f"{label} (n={len(time_courses)})",
                   color=color, linewidth=2)
            
            # Get significant time indices
            significant_times = significant_times_by_level.get(level, [])
            if significant_times:
                # Find indices of significant times
                sig_indices = [np.where(window_centers >= t)[0][0] for t in significant_times
                             if t <= window_centers[-1]]
                
                # Extract significant points from the time course
                sig_scores = mean_time_course[sig_indices]
                
                # Plot significant points with markers
                plt.scatter(window_centers[sig_indices], sig_scores, 
                          color=color, s=50, marker='o', zorder=10)
    
    # Add ping timepoint
    plt.axvline(x=3.5, color='red', linestyle='--', alpha=0.7, label='Ping (3.5s)')
    
    # Add chance level
    plt.axhline(y=0.5, color='black', linestyle='--', label='Chance')
    
    # Add labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('Decoding Accuracy')
    plt.title(f'Significant Time Points in {FEATURES[feature_name]["name"]} Decoding\nBy Musical Experience Level')
    plt.legend(loc='best')
    
    plt.figtext(0.5, 0.01, 
              "Markers indicate time points with significant decoding (p<0.05, cluster-corrected)", 
              ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/significant_timepoints_by_experience.png")
    plt.close()

def create_music_timecourse_plot_with_clusters(decoding_measures, metadata_df, feature_name, save_path):
    """
    Create time course plot by musical experience level with cluster-based permutation testing.
    
    Parameters:
    decoding_measures - Dictionary with decoding measures for each subject
    metadata_df - DataFrame with subject metadata
    feature_name - Name of the feature
    save_path - Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Use the first subject's time points
    window_centers = None
    for subject in decoding_measures:
        if 'window_centers' in decoding_measures[subject]:
            window_centers = decoding_measures[subject]['window_centers']
            break
    
    if window_centers is not None:
        # Group time courses by musical experience level
        time_courses_by_level = {1: [], 2: [], 3: []}
        
        for subject in decoding_measures:
            if 'time_course' in decoding_measures[subject]:
                # Get musical experience level for this subject
                subject_data = metadata_df[metadata_df['Subject'] == subject]
                if not subject_data.empty and 'Musician' in subject_data.columns:
                    level = subject_data['Musician'].values[0]
                    if level in time_courses_by_level:
                        time_courses_by_level[level].append(decoding_measures[subject]['time_course'])
        
        # Create cluster output directory
        cluster_path = os.path.join(save_path, "cluster_results")
        os.makedirs(cluster_path, exist_ok=True)
        
        # Dictionary to store significant time points for each level
        significant_times_by_level = {}
        
        # Plot average time course for each level
        for level, time_courses in time_courses_by_level.items():
            if time_courses:  # Skip empty levels
                time_courses_array = np.array(time_courses)
                mean_time_course = np.mean(time_courses_array, axis=0)
                sem_time_course = np.std(time_courses_array, axis=0) / np.sqrt(len(time_courses))
                
                if level == 1:
                    label = "Level 1 (Little/No Experience)"
                    color = '#1f77b4'
                elif level == 2:
                    label = "Level 2 (Moderate Experience)"
                    color = '#ff7f0e'
                else:
                    label = "Level 3 (>10 Years Experience)"
                    color = '#2ca02c'
                
                # Run cluster permutation test for this level
                print(f"\nRunning cluster permutation test for {feature_name}, Level {level}...")
                _, _, _, significant_times = run_cluster_permutation_test(
                    time_courses_array, window_centers, f"{feature_name}_level{level}"
                )
                significant_times_by_level[level] = significant_times
                
                # Plot the time course
                plt.plot(window_centers, mean_time_course, '-', 
                       label=f"{label} (n={len(time_courses)})",
                       color=color, linewidth=2)
                plt.fill_between(window_centers, 
                               mean_time_course - sem_time_course,
                               mean_time_course + sem_time_course,
                               alpha=0.2, color=color)
                
                # Highlight significant time points if any
                if significant_times:
                    # Convert list of times to time ranges
                    sig_ranges = []
                    current_range = [significant_times[0], significant_times[0]]
                    
                    for t in significant_times[1:]:
                        # Check if time point is contiguous with current range
                        if abs(t - current_range[1]) <= (window_centers[1] - window_centers[0]) * 1.1:
                            current_range[1] = t
                        else:
                            sig_ranges.append(current_range)
                            current_range = [t, t]
                    
                    sig_ranges.append(current_range)  # Add the last range
                    
                    # Highlight significant time ranges
                    for range_start, range_end in sig_ranges:
                        plt.axvspan(range_start, range_end, alpha=0.3, color=color)
                    
                    print(f"Highlighted {len(sig_ranges)} significant time ranges for Level {level}")
        
        # Add ping timepoint and highlight pre/post ping periods
        plt.axvline(x=3.5, color='red', linestyle='--', alpha=0.7, label='Ping (3.5s)')
        
        # Add chance level
        plt.axhline(y=0.5, color='black', linestyle='--', label='Chance')
        
        # Add labels and title
        plt.xlabel('Time (s)')
        plt.ylabel('Decoding Accuracy')
        plt.title(f'Time Course by Musical Experience Level\n({FEATURES[feature_name]["name"]})')
        plt.legend(loc='best')
        
        # Add annotation about significant time periods
        has_significance = any(significant_times_by_level.values())
        if has_significance:
            plt.figtext(0.5, 0.01, 
                      "Highlighted regions show time points with significant decoding (p<0.05, cluster-corrected)", 
                      ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/time_course_by_experience_with_clusters.png")
        plt.close()
        
        # Create a more detailed plot showing only the significant regions
        if has_significance:
            create_detailed_significance_plot(
                time_courses_by_level, window_centers, significant_times_by_level, 
                feature_name, save_path
            )


def main():
    """Main execution function"""
    print("\n" + "="*50)
    print("DELAY-PERIOD MEG DECODING ANALYSIS")
    print("="*50 + "\n")
    
    # Set locale for number formatting
    try:
        locale.setlocale(locale.LC_ALL, "en_US.utf8")
    except:
        try:
            locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
        except:
            print("Warning: Could not set locale. Continue anyway.")
    
    # Create necessary output directories
    setup_directories()
    
    # Load metadata to get all subjects
    try:
        metaInfo = pd.read_excel(META_FILE)
        all_subjects = list(np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject']))
        
        # Allow specifying subjects from command line
        if len(sys.argv) > 1:
            try:
                # If arguments are provided, use them as subject IDs
                subjects_to_process = [int(arg) for arg in sys.argv[1:]]
                print(f"Using command-line specified subjects: {subjects_to_process}")
            except ValueError:
                print("Invalid subject IDs provided. Using all subjects from metadata.")
                subjects_to_process = all_subjects.copy()
        else:
            subjects_to_process = all_subjects.copy()
            
        print(f"\nRunning analysis for {len(subjects_to_process)} subjects: {subjects_to_process}")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        print("Falling back to default subject list (1-30)")
        subjects_to_process = list(range(1, 31))
    
    # Load metadata and behavioral data once
    metadata_df = load_subject_metadata(subjects_to_process)
    behavior_df = analyze_behavioral_data_detailed(subjects_to_process)
    
    # Run analysis for each feature
    for feature_name in FEATURES.keys():
        print(f"\n==== Processing {FEATURES[feature_name]['name']} ====")
        
        # Collect all subject data for this feature
        all_subject_ids = []
        for subject in subjects_to_process:
            subject_data = load_subject_decoding_data(subject, feature_name)
            
            if subject_data is not None:
                all_subject_ids.append(subject)
                
        if not all_subject_ids:
            print(f"No decoding results found for {feature_name}. Skipping.")
            continue
        
        # Perform behavior-decoding correlation analysis
        print(f"\n==== Correlating {FEATURES[feature_name]['name']} with behavior ====")
        analyze_behavior_decoding_correlation(all_subject_ids, feature_name, behavior_df)
        
        # Analyze musical experience
        print(f"\n==== Analyzing musical experience for {FEATURES[feature_name]['name']} ====")
        analyze_musical_experience(all_subject_ids, feature_name, metadata_df)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()