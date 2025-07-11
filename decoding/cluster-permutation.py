#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create a comparison plot for delay phase results using cluster permutation test.
This script combines techniques from comparisonPlot.py but uses the data structure
and paths from delaySlidingWindow.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

# Constants and Configuration from delaySlidingWindow.py
HOME_DIR = '/media/headmodel/Elements/AWM4/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'

# Delay Period Configuration
DELAY_CONFIG = {
    'tmin': 0.0,
    'tmax': 1.0,
}

# Critical time points to mark on plots
CRITICAL_TIMEPOINTS = [3.5, 4.5]  # Gray dashed lines at these times

# Feature Definitions for Maintained Information
FEATURES = {
    'voice': {
        'name': 'Maintained Voice Identity',
        'color': '#1c686b',  # Darker teal
        'description': 'Voice identity information maintained in working memory'
    },
    'location': {
        'name': 'Maintained Location',
        'color': '#cb6a3e',  # Darker orange
        'description': 'Location information maintained in working memory'
    }
}

# Path to the delay sliding window results
DELAY3PATH = f"{PROCESSED_DIR}/delayPseudo"

# Output path for the cluster permutation results
CLUSTER_OUTPUT_PATH = f"{DELAY3PATH}/cluster_permutation"
os.makedirs(CLUSTER_OUTPUT_PATH, exist_ok=True)


def load_delay_data(feature_name):
    """
    Load the delay phase decoding data for a specific feature.
    
    Args:
        feature_name (str): Name of the feature to load data for.
        
    Returns:
        tuple: (scores_array, window_centers, subject_ids)
    """
    results_path = f"{DELAY3PATH}"
    results_file = f"{results_path}/S1_avg12_voice_results.xlsx"
    
    try:
        results_df = pd.read_excel(results_file, index_col=0)
        
        # Extract subject IDs
        subject_ids = list(map(int, results_df.index))
        
        # Extract time points (column names)
        time_labels = results_df.columns
        window_centers = np.array([float(t.split('_')[1].replace('s', '')) for t in time_labels])
        
        # Extract scores
        scores_array = results_df.values
        
        print(f"Loaded data for {feature_name}: {scores_array.shape[0]} subjects, {scores_array.shape[1]} timepoints")
        return scores_array, window_centers, subject_ids
    
    except FileNotFoundError:
        print(f"Warning: Results file not found for {feature_name} at {results_file}")
        return None, None, None
    except Exception as e:
        print(f"Error loading data for {feature_name}: {str(e)}")
        return None, None, None


def run_cluster_permutation_test(scores, window_centers, feature_name):
    """
    Run a cluster permutation test on the decoding scores.
    
    Args:
        scores (numpy.ndarray): Array of decoding scores (subjects × timepoints)
        window_centers (numpy.ndarray): Array of time points
        feature_name (str): Name of the feature
        
    Returns:
        tuple: (clusters, cluster_p_values, significant_cluster_info)
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
            n_permutations=1000000,  
            threshold=None,
            tail=1,
            out_type='mask'
        )
        
        # If no clusters were found
        if len(clusters) == 0:
            print(f"No clusters found for {feature_name}")
            return [], [], []
        
        # Extract cluster information
        cluster_info = []
        
        for c_idx, (cluster, p_val) in enumerate(zip(clusters, cluster_p_values)):
            # Get the mask array and find where it's True
            cluster_mask = cluster[0]  # This is a boolean mask array
            cluster_indices = np.where(cluster_mask)[0]
            
            # Extract start and stop indices and times
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
                    'indices': cluster_indices
                })
                
                # Save cluster information to file
                np.savetxt(
                    os.path.join(results_path, f'cluster_{c_idx}_{feature_name}.txt'),
                    np.column_stack((window_centers[cluster_indices], np.ones_like(cluster_indices))),
                    fmt='%.3f %d',
                    header=f"Time Significant (p={p_val:.5f})",
                    comments=''
                )
        
        # Find significant clusters (p < 0.05)
        sig_clusters = [ci for ci in cluster_info if ci['p_value'] < 0.05]
        
        # Print significant cluster information
        for ci in sig_clusters:
            print(f"Significant cluster found: p={ci['p_value']:.4f}, "
                  f"time range: {ci['start_time']:.2f}s - {ci['end_time']:.2f}s")
        
        return clusters, cluster_p_values, cluster_info
    
    except Exception as e:
        print(f"Error in cluster permutation test for {feature_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], [], []


def create_delay_comparison_plot():
    """
    Create a comparison plot for the delay phase results with cluster permutation testing.
    """
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Dictionary to store results for each feature
    results_dict = {}
    
    # Process each feature
    for feature_name in FEATURES.keys():
        # Load data
        scores, window_centers, subject_ids = load_delay_data(feature_name)
        
        if scores is None or window_centers is None:
            continue
            
        # Run cluster permutation test
        clusters, cluster_p_values, cluster_info = run_cluster_permutation_test(
            scores, window_centers, feature_name)
        
        # Calculate mean and standard error
        mean_accuracy = np.mean(scores, axis=0)
        std_error = np.std(scores, axis=0) / np.sqrt(scores.shape[0])
        
        # Store results
        results_dict[feature_name] = {
            'mean_accuracy': mean_accuracy,
            'std_error': std_error,
            'window_centers': window_centers,
            'clusters': clusters,
            'cluster_p_values': cluster_p_values,
            'cluster_info': cluster_info,
            'n_subjects': scores.shape[0]
        }
    
    # Plot each feature
    for feature_name, data in results_dict.items():
        color = FEATURES[feature_name]['color']
        timepoints = data['window_centers']
        mean_accuracy = data['mean_accuracy']
        std_error = data['std_error']
        cluster_info = data['cluster_info']
        
        # Plot mean accuracy
        plt.plot(timepoints, mean_accuracy, 
                color=color, 
                linewidth=2,
                label=f"{FEATURES[feature_name]['name']} (N={data['n_subjects']})")
        
        # Add confidence interval
        plt.fill_between(timepoints, 
                         mean_accuracy - std_error,
                         mean_accuracy + std_error,
                         alpha=0.2,
                         color=color)
        
        # Add significant time periods as horizontal lines for clusters with p < 0.05
        sig_clusters = [ci for ci in cluster_info if ci['p_value'] < 0.05]
        
        # Height offset based on feature (to separate lines for different features)
        height_offset = 0.67 if feature_name == 'maintained_voice_identity' else 0.68
        
        for ci in sig_clusters:
            start_time = ci['start_time']
            end_time = ci['end_time']
            
            plt.plot([start_time, end_time], 
                    [height_offset, height_offset], 
                    color=color, 
                    linewidth=3, 
                    solid_capstyle='round')
    
    # Add critical timepoints as vertical lines
    for tp in CRITICAL_TIMEPOINTS:
        plt.axvline(x=tp, color='gray', linestyle='--', alpha=0.7)
    
    # Add chance level
    plt.axhline(y=0.5, color='black', linestyle='--', label='Chance (50%)')
    
    # Set axis labels and title
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Decoding Accuracy', fontsize=14)
    plt.title('Comparison of Working Memory Content Decoding during Delay Period', fontsize=16)
    
    # Set axis limits
    plt.xlim(DELAY_CONFIG['tmin'], DELAY_CONFIG['tmax'])
    plt.ylim(0.45, 0.70)
    
    # Add legend
    plt.legend(loc='upper right', fontsize=12)
    
    # Add grid
    plt.grid(True, linestyle=':', alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{CLUSTER_OUTPUT_PATH}/delay_comparison_cluster_permutation.png', dpi=300)
    plt.savefig(f'{CLUSTER_OUTPUT_PATH}/delay_comparison_cluster_permutation.pdf')
    
    print(f"\nComparison plot saved to {CLUSTER_OUTPUT_PATH}/delay_comparison_cluster_permutation.png")
    plt.close()


def main():
    """Main function to create comparison plot with cluster permutation tests"""
    print("\n=== Creating Delay Phase Comparison Plot with Cluster Permutation Tests ===")
    
    # Create combined comparison plot
    print("\nCreating comparison plot...")
    create_delay_comparison_plot()
    
    print("\nPlot created successfully!")


if __name__ == "__main__":
    main()