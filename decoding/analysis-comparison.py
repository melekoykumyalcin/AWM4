#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create a comprehensive figure with 3 subplots comparing results from:
- Averaged analysis 
- Binned analysis
- Overlapping bins with PCA

With S2 data consistently plotted between 1.0-2.0 seconds across all analysis methods.
N values are removed from the plots as requested.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines

# Set paths
HOME_DIR = '/media/headmodel/Elements/AWM4/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
BASE_PATH = f"{PROCESSED_DIR}/timepoints/PhD/"

# Path to each analysis type
AVERAGED_PATH = f"{BASE_PATH}/averaged"
BINS_PATH = f"{BASE_PATH}/bins"
OVERLAP_PCA_PATH = f"{BASE_PATH}/bins/overlapPCA"

# Output path for the combined plot
OUTPUT_PATH = f"{BASE_PATH}/combined_comparisons"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Feature definitions with colors (copied from original script)
FEATURES = {
    'voice_identity': {
        'name': 'voice identity',
        'color': {
            'S1': '#1c686b',  # Darker teal
            'S2': '#64b1b5'   # Lighter teal
        }
    },
    'location': {
        'name': 'location',
        'color': {
            'S1': '#cb6a3e',  # Darker orange
            'S2': '#e8a87c'   # Lighter orange
        }
    }
}

# Stimulus definitions with fixed timepoints
STIMULI = {
    'S1': {'name': 'S1', 'timerange': (0.0, 1.0)}, 
    'S2': {'name': 'S2', 'timerange': (1.0, 2.0)}
}

# Analysis types with their paths and titles
ANALYSIS_TYPES = {
    'averaged': {
        'path': AVERAGED_PATH,
        'title': 'Averaged Analysis',
        'filename_pattern': 'all_features_stimuli_comparison.png'
    },
    'bins': {
        'path': BINS_PATH,
        'title': 'Binned Analysis',
        'filename_pattern': 'all_features_stimuli_comparison_bins.png'
    },
    'overlap_pca': {
        'path': OVERLAP_PCA_PATH,
        'title': 'Overlapping Bins with PCA',
        'filename_pattern': 'all_features_stimuli_comparison_overlapping_bins.png'
    }
}

def load_decoding_results(analysis_type, feature_name, stimulus_name):
    """
    Load saved decoding results for a specific analysis type, feature, and stimulus
    
    Parameters:
    analysis_type - Type of analysis ('averaged', 'bins', 'overlap_pca')
    feature_name - Name of feature ('voice_identity' or 'location')
    stimulus_name - Name of stimulus ('S1' or 'S2')
    
    Returns:
    (scores, time_points, significant_points) - Tuple with results data
    """
    base_path = ANALYSIS_TYPES[analysis_type]['path']
    results_path = f"{base_path}/{feature_name}({stimulus_name})"
    
    # Define file patterns based on analysis type
    if analysis_type == 'overlap_pca':
        scores_file = f"{results_path}/all_{feature_name}_{stimulus_name}_byBin_overlapping_SVM.xlsx"
        time_points_file = f"{results_path}/bin_centers_overlapping.csv"
        p_values_file = f"{results_path}/p_values.csv"
        time_col = "bin_centers"
    elif analysis_type == 'bins':
        scores_file = f"{results_path}/all_{feature_name}_{stimulus_name}_byBin_SVM.xlsx"
        time_points_file = f"{results_path}/bin_centers.csv"
        p_values_file = f"{results_path}/p_values.csv"
        time_col = "bin_centers"
    else:  # averaged
        scores_file = f"{results_path}/all_{feature_name}_{stimulus_name}_byTimepoint_SVM.xlsx"
        p_values_file = f"{results_path}/p_values.csv"
    
    # Load scores
    try:
        scores_df = pd.read_excel(scores_file, index_col=0)
        scores = scores_df.values
        
        # For averaged analysis, generate timepoints based on stimulus
        if analysis_type == 'averaged':
            start_time, end_time = STIMULI[stimulus_name]['timerange']
            time_points = np.linspace(start_time, end_time, scores.shape[1])
        else:
            # Load time points from file for other analysis types
            try:
                original_time_points = pd.read_csv(time_points_file)[time_col].values
                
                # Check if we need to adjust S2 timepoints
                if stimulus_name == 'S2':
                    start_time, end_time = STIMULI['S2']['timerange']
                    
                    # Check if time points need shifting to 1.0-2.0 range
                    if original_time_points[0] < 0.5:  # Assuming S2 should start around 1.0
                        # These timepoints are likely in the 0-1 range instead of 1-2
                        time_points = original_time_points + 1.0
                        print(f"Shifting S2 timepoints by +1.0 for {analysis_type} - {feature_name}")
                    else:
                        time_points = original_time_points
                else:
                    time_points = original_time_points
                    
            except FileNotFoundError:
                print(f"Warning: Time points file not found for {analysis_type} - {feature_name} ({stimulus_name})")
                # Generate synthetic timepoints if file not found
                start_time, end_time = STIMULI[stimulus_name]['timerange']
                time_points = np.linspace(start_time, end_time, scores.shape[1])
    except FileNotFoundError:
        print(f"Warning: Scores file not found for {analysis_type} - {feature_name} ({stimulus_name})")
        return None, None, None
    
    # Load significance data
    try:
        p_values = pd.read_csv(p_values_file)["p_values"].values
        
        # Handle case where p_values length doesn't match time_points
        if len(p_values) != len(time_points):
            print(f"Warning: P-values length ({len(p_values)}) doesn't match timepoints length ({len(time_points)})")
            if len(p_values) < len(time_points):
                # Pad with non-significant values
                p_values = np.pad(p_values, (0, len(time_points) - len(p_values)), 
                                 constant_values=1.0)
            else:
                # Truncate
                p_values = p_values[:len(time_points)]
        
        # Calculate significant points
        significant_points = []
        for idx, p in enumerate(p_values):
            if p <= 0.01:
                significant_points.append((idx, '**'))
            elif p <= 0.05:
                significant_points.append((idx, '*'))
    except FileNotFoundError:
        print(f"Warning: P-values file not found for {analysis_type} - {feature_name} ({stimulus_name})")
        significant_points = None
    
    return scores, time_points, significant_points

def create_comprehensive_plot():
    """Create a comprehensive figure with 3 subplots for different analysis methods"""
    
    # Create figure with specific dimensions for consistent spacing
    fig = plt.figure(figsize=(20, 8))
    
    # Set up subplots with equal width
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.15, 0.85], wspace=0.2, hspace=0.05)
    
    # Define the order of subplots (excluding sliding window)
    subplot_order = ['averaged', 'bins', 'overlap_pca']
    
    # Create a single shared legend in the top row
    legend_ax = fig.add_subplot(gs[0, :])
    legend_ax.axis('off')
    
    # Prepare legend handles
    legend_handles = []
    
    # Add legend items for each feature/stimulus combination
    for feature_name in FEATURES.keys():
        for stimulus_name in STIMULI.keys():
            color = FEATURES[feature_name]['color'][stimulus_name]
            label = f"{feature_name.title()} ({stimulus_name})"
            legend_handles.append(mlines.Line2D([], [], color=color, linewidth=2, label=label))
    
    # Add reference lines to legend
    legend_handles.append(mlines.Line2D([], [], color='black', linestyle='--', label='Chance (50%)'))
    legend_handles.append(mlines.Line2D([], [], color='gray', linestyle='--', label='S1/S2 boundary'))
    
    # Place the legend at the top
    legend_ax.legend(handles=legend_handles, loc='center', ncol=len(legend_handles), 
                    frameon=False, fontsize=12)
    
    # Process each analysis type
    for i, analysis_type in enumerate(subplot_order):
        # Add subplot in the bottom row
        ax = fig.add_subplot(gs[1, i])
        
        # Generate plot from data
        print(f"Generating plot for {analysis_type}...")
        results_dict = {}
        significance_dict = {}
        
        # Load results for all feature-stimulus combinations
        for feature_name in FEATURES.keys():
            for stimulus_name in STIMULI.keys():
                scores, time_points, sig_points = load_decoding_results(
                    analysis_type, feature_name, stimulus_name)
                
                if scores is not None and time_points is not None:
                    results_dict[(feature_name, stimulus_name)] = (scores, time_points)
                    if sig_points is not None:
                        significance_dict[(feature_name, stimulus_name)] = sig_points
        
        if not results_dict:
            ax.text(0.5, 0.5, f"No data found for {analysis_type}", 
                    ha='center', va='center', fontsize=14)
            ax.set_title(ANALYSIS_TYPES[analysis_type]['title'], fontsize=16)
            ax.axis('off')
            continue
        
        # Plot each feature/stimulus combination
        significance_heights = {}
        current_height = 0.62
        height_step = 0.02
        
        for (feature, stimulus) in results_dict.keys():
            significance_heights[(feature, stimulus)] = current_height
            current_height += height_step
        
        # Plot each feature/stimulus data
        for (feature, stimulus), (data, time_points) in results_dict.items():
            mean_accuracy = np.mean(data, axis=0)
            std_error = np.std(data, axis=0) / np.sqrt(data.shape[0])
            
            color = FEATURES[feature]['color'][stimulus]
            
            # Plot the mean accuracy line
            ax.plot(time_points, mean_accuracy, 
                    color=color,
                    linewidth=2)
            
            # Add confidence interval
            ax.fill_between(time_points, 
                            mean_accuracy - std_error,
                            mean_accuracy + std_error,
                            alpha=0.2,
                            color=color)
            
            # Plot significance markers if provided
            if significance_dict and (feature, stimulus) in significance_dict:
                sig_points = significance_dict[(feature, stimulus)]
                if sig_points:
                    # Extract significant indices
                    sig_indices = [idx for idx, _ in sig_points]
                    
                    if sig_indices:
                        # Get the height for this feature/stimulus
                        sig_height = significance_heights[(feature, stimulus)]
                        
                        # Group consecutive significant points
                        segments = []
                        current_segment = []
                        
                        for j in range(len(sig_indices)):
                            if j == 0 or sig_indices[j] > sig_indices[j-1] + 1:
                                if current_segment:
                                    segments.append(current_segment)
                                current_segment = [sig_indices[j]]
                            else:
                                current_segment.append(sig_indices[j])
                        
                        if current_segment:
                            segments.append(current_segment)
                        
                        # Plot each segment as a horizontal line
                        for segment in segments:
                            # Get actual timepoints for these indices
                            try:
                                x_start = time_points[segment[0]]
                                x_end = time_points[segment[-1]]
                                
                                ax.plot([x_start, x_end], [sig_height, sig_height], 
                                        color=color, linewidth=2, solid_capstyle='round')
                                
                                # Add asterisks at the center of the segment
                                x_center = (x_start + x_end) / 2
                                marker = next((m for idx, m in sig_points if idx == segment[len(segment)//2]), '*')
                                ax.text(x_center, sig_height + 0.01, marker, color=color, 
                                        ha='center', va='bottom', fontsize=14)
                            except IndexError:
                                print(f"Warning: Skipping segment due to index error: {segment}")
                                continue
        
        # Add reference lines
        ax.axhline(y=0.5, color='black', linestyle='--')
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
        
        # Set consistent axis limits for all plots
        ax.set_xlim(0, 2)
        ax.set_ylim(0.45, 0.7)
        
        # Add labels
        ax.set_xlabel('Time (s)', fontsize=14)
        if i == 0:  # Only add y-label on the first plot
            ax.set_ylabel('Decoding Accuracy', fontsize=14)
        
        ax.set_title(ANALYSIS_TYPES[analysis_type]['title'], fontsize=16)
        
        # Ensure consistent tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add grid for better readability
        ax.grid(True, linestyle=':', alpha=0.5)
    
    # Add overall title
    fig.suptitle('Comparison of Analysis Methods for Feature-Stimulus Decoding', 
                fontsize=20, y=0.98)
    
    # Adjust layout to ensure proper spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save plot
    plt.savefig(f'{OUTPUT_PATH}/comprehensive_comparison_final.pdf')
    plt.savefig(f'{OUTPUT_PATH}/comprehensive_comparison_final.png', dpi=300)
    print(f"Saved comprehensive comparison plot to {OUTPUT_PATH}/comprehensive_comparison_final.png")
    plt.close()

def main():
    """Main function to create all comparison plots"""
    print("Creating final comprehensive comparison plot...")
    create_comprehensive_plot()
    print("Comprehensive comparison plot created successfully!")

if __name__ == "__main__":
    main()