#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Significance testing for delay period decoding results using sign-flipping permutation test.
This approach follows the method described in the paper:
1. Subtract chance level (0.5) from individual accuracies
2. Randomly invert the sign of these differences
3. Compute the group mean for each permutation
4. Calculate p-values by comparing empirical means to the null distribution
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm

# Constants and Configuration
HOME_DIR = '/media/headmodel/Elements/AWM4/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'

# Delay Period Configuration
DELAY_CONFIG = {
    'tmin': 2.0,
    'tmax': 4.7,
}

# Critical time points to mark on plots
CRITICAL_TIMEPOINTS = [3.5, 4.5]  # Gray dashed lines at these times

# Feature Definitions for Maintained Information
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

# Path to the delay sliding window results
DELAY3PATH = f"{PROCESSED_DIR}/timepoints/delay_binned(3)"

# Output path for the permutation test results
PERMTEST_OUTPUT_PATH = f"{DELAY3PATH}/sign_permutation"
os.makedirs(PERMTEST_OUTPUT_PATH, exist_ok=True)


def load_delay_data(feature_name):
    """
    Load the delay phase decoding data for a specific feature.
    
    Args:
        feature_name (str): Name of the feature to load data for.
        
    Returns:
        tuple: (scores_array, window_centers, subject_ids)
    """
    results_path = f"{DELAY3PATH}/{feature_name}"
    results_file = f"{results_path}/decoding_results.xlsx"
    
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


def run_sign_flipping_permutation_test(scores, window_centers, feature_name, n_permutations=10000, alpha=0.05):
    """
    Run a sign-flipping permutation test on the decoding scores.
    
    Args:
        scores (numpy.ndarray): Array of decoding scores (subjects × timepoints)
        window_centers (numpy.ndarray): Array of time points
        feature_name (str): Name of the feature
        n_permutations (int): Number of permutations to perform
        alpha (float): Significance threshold
        
    Returns:
        tuple: (p_values, significant_timepoints, significant_timepoint_indices)
    """
    # Create results path
    results_path = f"{PERMTEST_OUTPUT_PATH}/{feature_name}"
    os.makedirs(results_path, exist_ok=True)
    
    # Get dimensions
    n_subjects, n_timepoints = scores.shape
    
    # Calculate empirical means (original data)
    empirical_means = np.mean(scores, axis=0)
    
    # Convert scores to differences from chance (0.5)
    diffs_from_chance = scores - 0.5
    
    # Initialize storage for permutation distribution
    null_distribution = np.zeros((n_permutations, n_timepoints))
    
    # Run permutations
    print(f"Running {n_permutations} sign-flipping permutations for {feature_name}...")
    for perm_idx in tqdm(range(n_permutations)):
        # Generate random signs for each subject and timepoint
        random_signs = np.random.choice([-1, 1], size=(n_subjects, n_timepoints))
        
        # Apply random sign flips
        permuted_diffs = diffs_from_chance * random_signs
        
        # Compute mean of this permutation
        null_distribution[perm_idx] = np.mean(permuted_diffs, axis=0)
    
    # Compute p-values for each timepoint
    p_values = np.zeros(n_timepoints)
    
    for t in range(n_timepoints):
        # Count permutation means >= empirical mean - 0.5 (to compare with chance)
        p_values[t] = np.mean(null_distribution[:, t] >= (empirical_means[t] - 0.5))
    
    # Find significant timepoints
    significant_timepoint_indices = np.where(p_values <= alpha)[0]
    significant_timepoints = window_centers[significant_timepoint_indices]
    
    # Print results
    print(f"\nResults for {feature_name}:")
    print(f"Significant timepoints (p ≤ {alpha}):")
    
    if len(significant_timepoints) > 0:
        # Group consecutive timepoints
        consec_groups = []
        current_group = [significant_timepoint_indices[0]]
        
        for i in range(1, len(significant_timepoint_indices)):
            if significant_timepoint_indices[i] == significant_timepoint_indices[i-1] + 1:
                current_group.append(significant_timepoint_indices[i])
            else:
                consec_groups.append(current_group)
                current_group = [significant_timepoint_indices[i]]
        
        # Add the last group
        if current_group:
            consec_groups.append(current_group)
        
        # Print all significant timepoints in grouped format
        for group in consec_groups:
            start_time = window_centers[group[0]]
            end_time = window_centers[group[-1]]
            p_value_range = f"{p_values[group[0]]:.4f}-{p_values[group[-1]]:.4f}"
            
            print(f"  Time range: {start_time:.2f}s - {end_time:.2f}s (p-values: {p_value_range})")
    else:
        print("  No significant timepoints found")
    
    # Save results to file
    results = np.column_stack((window_centers, empirical_means, p_values))
    header = "Time Accuracy p_value"
    np.savetxt(f"{results_path}/permutation_results.txt", results, header=header, fmt='%.4f')
    
    # Save full null distribution for potential later analysis
    with open(f"{results_path}/null_distribution.npy", 'wb') as f:
        np.save(f, null_distribution)
    
    return p_values, significant_timepoints, significant_timepoint_indices


def create_permutation_plots():
    """
    Create plots of the decoding results with permutation test significance.
    """
    # Set up the figure for the combined plot
    plt.figure(figsize=(12, 8))
    
    # Dictionary to store results for each feature
    results_dict = {}
    
    # Process each feature
    for feature_name in FEATURES.keys():
        # Load data
        scores, window_centers, subject_ids = load_delay_data(feature_name)
        
        if scores is None or window_centers is None:
            continue
            
        # Run sign-flipping permutation test
        p_values, sig_timepoints, sig_indices = run_sign_flipping_permutation_test(
            scores, window_centers, feature_name)
        
        # Calculate mean and standard error
        mean_accuracy = np.mean(scores, axis=0)
        std_error = np.std(scores, axis=0) / np.sqrt(scores.shape[0])
        
        # Store results
        results_dict[feature_name] = {
            'mean_accuracy': mean_accuracy,
            'std_error': std_error,
            'window_centers': window_centers,
            'p_values': p_values,
            'significant_timepoints': sig_timepoints,
            'significant_indices': sig_indices,
            'n_subjects': scores.shape[0]
        }
        
        # Create individual plot for this feature
        plt.figure(figsize=(10, 6))
        
        # Plot mean accuracy
        plt.plot(window_centers, mean_accuracy, 
                color=FEATURES[feature_name]['color'], 
                linewidth=2)
        
        # Add confidence interval
        plt.fill_between(window_centers, 
                        mean_accuracy - std_error,
                        mean_accuracy + std_error,
                        alpha=0.2,
                        color=FEATURES[feature_name]['color'])
        
        # Add significance markers
        if len(sig_timepoints) > 0:
            # Group consecutive significant timepoints
            consec_groups = []
            current_group = [sig_indices[0]]
            
            for i in range(1, len(sig_indices)):
                if sig_indices[i] == sig_indices[i-1] + 1:
                    current_group.append(sig_indices[i])
                else:
                    consec_groups.append(current_group)
                    current_group = [sig_indices[i]]
            
            # Add the last group
            if current_group:
                consec_groups.append(current_group)
            
            # Plot horizontal lines for significant time ranges
            for group in consec_groups:
                start_time = window_centers[group[0]]
                end_time = window_centers[group[-1]]
                
                plt.plot([start_time, end_time], [0.62, 0.62], 
                        color=FEATURES[feature_name]['color'], 
                        linewidth=3, 
                        solid_capstyle='round')
                
                # Add p-value text
                min_p = np.min(p_values[group])
                plt.text((start_time + end_time) / 2, 0.63, 
                        f"p<{min_p:.3f}", 
                        ha='center', va='bottom', 
                        color=FEATURES[feature_name]['color'])
        
        # Add critical timepoints as vertical lines
        for tp in CRITICAL_TIMEPOINTS:
            plt.axvline(x=tp, color='gray', linestyle='--', alpha=0.7)
        
        # Add chance level
        plt.axhline(y=0.5, color='black', linestyle='--')
        
        # Set axis labels and title
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Decoding Accuracy', fontsize=14)
        plt.title(f'{FEATURES[feature_name]["name"]} Decoding during Delay Period (N={scores.shape[0]})', 
                fontsize=16)
        
        # Set axis limits
        plt.xlim(DELAY_CONFIG['tmin'], DELAY_CONFIG['tmax'])
        plt.ylim(0.45, 0.75)
        
        # Add grid
        plt.grid(True, linestyle=':', alpha=0.3)
        
        # Save the figure
        plt.tight_layout()
        
        results_path = f"{PERMTEST_OUTPUT_PATH}/{feature_name}"
        plt.savefig(f'{results_path}/{feature_name}_permutation_test.png', dpi=300)
        plt.savefig(f'{results_path}/{feature_name}_permutation_test.pdf')
        
        print(f"Individual plot saved to {results_path}/{feature_name}_permutation_test.png")
        plt.close()
    
    # Create the combined plot
    plt.figure(figsize=(12, 8))
    
    # Plot each feature
    for feature_name, data in results_dict.items():
        color = FEATURES[feature_name]['color']
        timepoints = data['window_centers']
        mean_accuracy = data['mean_accuracy']
        std_error = data['std_error']
        sig_indices = data['significant_indices']
        
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
        
        # Add significant time periods as horizontal lines
        if len(sig_indices) > 0:
            # Group consecutive significant timepoints
            consec_groups = []
            current_group = [sig_indices[0]]
            
            for i in range(1, len(sig_indices)):
                if sig_indices[i] == sig_indices[i-1] + 1:
                    current_group.append(sig_indices[i])
                else:
                    consec_groups.append(current_group)
                    current_group = [sig_indices[i]]
            
            # Add the last group
            if current_group:
                consec_groups.append(current_group)
            
            # Height offset based on feature (to separate lines for different features)
            height_offset = 0.60 if feature_name == 'maintained_voice_identity' else 0.62
            
            for group in consec_groups:
                start_time = timepoints[group[0]]
                end_time = timepoints[group[-1]]
                
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
    plt.ylim(0.45, 0.75)
    
    # Add legend
    plt.legend(loc='upper right', fontsize=12)
    
    # Add grid
    plt.grid(True, linestyle=':', alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{PERMTEST_OUTPUT_PATH}/delay_comparison_permutation_test.png', dpi=300)
    plt.savefig(f'{PERMTEST_OUTPUT_PATH}/delay_comparison_permutation_test.pdf')
    
    print(f"\nComparison plot saved to {PERMTEST_OUTPUT_PATH}/delay_comparison_permutation_test.png")
    plt.close()


def main():
    """Main function to run sign-flipping permutation tests"""
    print("\n=== Running Sign-Flipping Permutation Tests for Delay Phase Decoding ===")
    
    # Create the plots with permutation test results
    create_permutation_plots()
    
    print("\nAll analyses completed successfully!")


if __name__ == "__main__":
    main()