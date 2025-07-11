#!/usr/bin/env python
"""
Delay Period Sliding Window Analysis - Group Analysis
For HPC cluster - to be run after all individual subjects are processed
Usage: python delay_sliding_group.py
"""

import os
import locale
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
plt.ioff()
plt.rcParams['figure.figsize'] = [10, 8]

import pandas as pd
import pickle
from glob import glob

# Constants and Configuration
HOME_DIR = '/mnt/hpc/projects/awm4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'

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

def compute_statistics(scores):
    """Compute statistical significance using permutation testing"""
    p_values = []
    significant_points = []
    
    for timepoint in range(scores.shape[1]):
        actual = scores[:, timepoint] - 0.5
        actual_mean = np.mean(actual)
        
        # Create permutation distribution
        permuted = [[np.abs(value), -np.abs(value)] for value in actual]
        
        # Permutation test
        population = []
        for _ in range(100000):
            sample = [np.random.choice(p, 1)[0] for p in permuted]
            population.append(np.mean(sample))
        
        # Calculate p-value
        p = np.sum(np.array(population) >= actual_mean) / 100000
        p_values.append(p)
        
        # Store significance
        if p <= 0.01:
            significant_points.append((timepoint, '**'))
        elif p <= 0.05:
            significant_points.append((timepoint, '*'))
    
    return p_values, significant_points

def load_individual_results():
    """
    Load all individual subject results from pickle files
    
    Returns:
        dict: Dictionary with feature names as keys, containing scores and metadata
    """
    results_dict = {}
    
    # Process each feature
    for feature_name in FEATURES.keys():
        results_path = f"{PROCESSED_DIR}/timepoints/delay_sliding_highres/{feature_name}"
        
        # Find all individual result files
        result_files = glob(f"{results_path}/sub*_results.pkl")
        
        if not result_files:
            print(f"No individual results found for {feature_name}")
            continue
            
        # Initialize storage
        all_scores = []
        all_subjects = []
        window_centers = None
        
        # Load each subject's results
        for result_file in sorted(result_files):
            try:
                with open(result_file, 'rb') as f:
                    subject_result = pickle.load(f)
                
                # Extract subject number from filename
                subject = int(os.path.basename(result_file).split('sub')[1].split('_')[0])
                
                all_scores.append(subject_result['mean_scores'])
                all_subjects.append(subject)
                
                # Store window centers (same for all subjects)
                if window_centers is None:
                    window_centers = subject_result['window_centers']
                    
            except Exception as e:
                print(f"Error loading {result_file}: {str(e)}")
                continue
        
        # Store in results dictionary
        if all_scores:
            results_dict[feature_name] = {
                'scores': np.array(all_scores),
                'subjects': all_subjects,
                'window_centers': window_centers
            }
            print(f"Loaded {len(all_subjects)} subjects for {feature_name}")
    
    return results_dict

def create_group_plots(results_dict):
    """
    Create group-level plots and save results
    
    Args:
        results_dict: Dictionary containing all subject results
    """
    # Process each feature
    for feature_name, feature_data in results_dict.items():
        all_scores = feature_data['scores']
        valid_subjects = feature_data['subjects']
        window_centers = feature_data['window_centers']
        
        results_path = f"{PROCESSED_DIR}/timepoints/delay_sliding_highres/{feature_name}"
        
        # Save results to Excel
        time_labels = [f'Time_{t:.3f}s' for t in window_centers]
        results_df = pd.DataFrame(all_scores, index=valid_subjects, columns=time_labels)
        results_df.to_excel(f'{results_path}/decoding_results.xlsx')
        print(f"Saved Excel results for {feature_name}: shape {all_scores.shape}")
        
        # Calculate mean across subjects and timepoints
        mean_across_time = results_df.mean(axis=1)
        mean_across_time.to_excel(f'{results_path}/{feature_name}_mean_across_time.xlsx')
        
        # Compute significance
        try:
            p_values, significant_points = compute_statistics(all_scores)
            pd.DataFrame(p_values, columns=["p_values"]).to_csv(f'{results_path}/p_values.csv', index=False)
        except Exception as e:
            print(f"Error computing statistics for {feature_name}: {e}")
            significant_points = []
        
        # Compute statistics for plotting
        mean_scores = np.mean(all_scores, axis=0)
        std_error = np.std(all_scores, axis=0) / np.sqrt(len(valid_subjects))
        
        # Create group-level plot
        plt.figure(figsize=(12, 7))
        plt.plot(window_centers, mean_scores, 
                label=f'Mean Accuracy (N={len(valid_subjects)})', 
                color=FEATURES[feature_name]['color'],
                linewidth=2)
                
        plt.fill_between(window_centers, 
                        mean_scores - std_error, 
                        mean_scores + std_error, 
                        alpha=0.2,
                        color=FEATURES[feature_name]['color'])
                        
        # Add critical timepoints as vertical lines
        for tp in CRITICAL_TIMEPOINTS:
            plt.axvline(x=tp, color='gray', linestyle='--', alpha=0.7)
        
        # Add significance markers if applicable
        if significant_points:
            sig_times = [window_centers[tp] for tp, _ in significant_points if tp < len(window_centers)]
            if sig_times:
                plt.plot(sig_times, [0.65] * len(sig_times), 
                        marker='*', linestyle='', color=FEATURES[feature_name]['color'],
                        markersize=10)
            
        plt.title(f'Group-Level {FEATURES[feature_name]["name"]} Decoding during Delay Period\n(100ms window, 10ms steps)')
        plt.xlabel('Time (s)')
        plt.ylabel('Decoding Accuracy')
        plt.axhline(y=0.5, color='black', linestyle='--', label='Chance')
        plt.ylim(0.45, 0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{results_path}/group_decoding_result.png', dpi=300)
        plt.savefig(f'{results_path}/group_decoding_result.pdf')
        plt.close()
        
        print(f"Created group plot for {feature_name}")
        
        # Save group statistics
        group_stats = {
            'mean_scores': mean_scores,
            'std_error': std_error,
            'n_subjects': len(valid_subjects),
            'subjects': valid_subjects,
            'window_centers': window_centers,
            'p_values': p_values if 'p_values' in locals() else None,
            'significant_points': significant_points
        }
        
        with open(f'{results_path}/group_statistics.pkl', 'wb') as f:
            pickle.dump(group_stats, f)

def create_comparison_plot(results_dict):
    """
    Create comparison plot for all features
    
    Args:
        results_dict: Dictionary containing all subject results
    """
    if len(results_dict) < 2:
        print("Not enough features for comparison plot")
        return
        
    comparison_path = f"{PROCESSED_DIR}/timepoints/delay_sliding_highres/comparison"
    os.makedirs(comparison_path, exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    
    # Store data for combined statistics
    comparison_data = {}
    
    for feature_name, feature_data in results_dict.items():
        all_scores = feature_data['scores']
        window_centers = feature_data['window_centers']
        
        mean_scores = np.mean(all_scores, axis=0)
        std_error = np.std(all_scores, axis=0) / np.sqrt(all_scores.shape[0])
        
        plt.plot(window_centers, mean_scores, 
                label=f'{FEATURES[feature_name]["name"]} (N={all_scores.shape[0]})',
                color=FEATURES[feature_name]['color'],
                linewidth=2)
                
        plt.fill_between(window_centers, 
                        mean_scores - std_error, 
                        mean_scores + std_error, 
                        alpha=0.2,
                        color=FEATURES[feature_name]['color'])
        
        # Store for comparison statistics
        comparison_data[feature_name] = {
            'mean_scores': mean_scores,
            'std_error': std_error,
            'window_centers': window_centers,
            'n_subjects': all_scores.shape[0]
        }
    
    # Add critical timepoints as vertical lines
    for tp in CRITICAL_TIMEPOINTS:
        plt.axvline(x=tp, color='gray', linestyle='--', alpha=0.7)
        
    plt.axhline(y=0.5, color='black', linestyle='--', label='Chance')
    plt.xlabel('Time (s)')
    plt.ylabel('Decoding Accuracy')
    plt.title('Comparison of Maintained Information Decoding during Delay Period\n(100ms window, 10ms steps)')
    plt.ylim(0.45, 0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{comparison_path}/feature_comparison.png', dpi=300)
    plt.savefig(f'{comparison_path}/feature_comparison.pdf')
    plt.close()
    
    # Save comparison data
    with open(f'{comparison_path}/comparison_data.pkl', 'wb') as f:
        pickle.dump(comparison_data, f)
    
    print("Created feature comparison plot")

def generate_summary_report(results_dict):
    """
    Generate a summary report of the group analysis
    
    Args:
        results_dict: Dictionary containing all subject results
    """
    report_path = f"{PROCESSED_DIR}/timepoints/delay_sliding_highres/group_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("DELAY PERIOD SLIDING WINDOW ANALYSIS - GROUP REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
        f.write(f"Home Directory: {HOME_DIR}\n")
        f.write(f"Window Size: 100ms, Step Size: 10ms\n\n")
        
        for feature_name, feature_data in results_dict.items():
            f.write(f"\n{FEATURES[feature_name]['name']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of subjects: {len(feature_data['subjects'])}\n")
            f.write(f"Subject IDs: {sorted(feature_data['subjects'])}\n")
            
            # Calculate peak accuracy
            mean_scores = np.mean(feature_data['scores'], axis=0)
            peak_idx = np.argmax(mean_scores)
            peak_time = feature_data['window_centers'][peak_idx]
            peak_accuracy = mean_scores[peak_idx]
            
            f.write(f"Peak accuracy: {peak_accuracy:.3f} at {peak_time:.3f}s\n")
            f.write(f"Mean accuracy across time: {np.mean(mean_scores):.3f}\n")
            
            # Count significant timepoints
            try:
                _, significant_points = compute_statistics(feature_data['scores'])
                n_sig = len(significant_points)
                f.write(f"Significant timepoints: {n_sig} out of {len(mean_scores)}\n")
            except:
                f.write("Significance testing failed\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Analysis completed successfully\n")
    
    print(f"Summary report saved to: {report_path}")

def check_missing_subjects():
    """
    Check which subjects might be missing from the analysis
    """
    # Load metadata to get expected subjects
    metaInfo = pd.read_excel(META_FILE)
    all_expected_subjects = set(np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject']))
    
    # Check what subjects we have results for
    processed_subjects = set()
    for feature_name in FEATURES.keys():
        results_path = f"{PROCESSED_DIR}/timepoints/delay_sliding_highres/{feature_name}"
        result_files = glob(f"{results_path}/sub*_results.pkl")
        
        for result_file in result_files:
            subject = int(os.path.basename(result_file).split('sub')[1].split('_')[0])
            processed_subjects.add(subject)
    
    missing_subjects = all_expected_subjects - processed_subjects
    
    if missing_subjects:
        print(f"\nWARNING: The following subjects are missing: {sorted(missing_subjects)}")
        print("These subjects may still be processing or may have failed.")
    else:
        print(f"\nAll {len(all_expected_subjects)} expected subjects have been processed!")
    
    return processed_subjects, missing_subjects

# Main execution
if __name__ == "__main__":
    print("DELAY PERIOD SLIDING WINDOW ANALYSIS - GROUP ANALYSIS")
    print("=" * 60)
    
    # Set locale
    try:
        locale.setlocale(locale.LC_ALL, "en_US.utf8")
    except:
        pass
    
    # Check for missing subjects
    print("\nChecking for processed subjects...")
    processed_subjects, missing_subjects = check_missing_subjects()
    print(f"Found results for {len(processed_subjects)} subjects")
    
    # Load all individual results
    print("\nLoading individual subject results...")
    results_dict = load_individual_results()
    
    if not results_dict:
        print("ERROR: No individual results found! Make sure individual subjects have been processed.")
        exit(1)
    
    # Create group plots
    print("\nCreating group-level plots...")
    create_group_plots(results_dict)
    
    # Create comparison plot
    print("\nCreating feature comparison plot...")
    create_comparison_plot(results_dict)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(results_dict)
    
    print("\n" + "=" * 60)
    print("GROUP ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {PROCESSED_DIR}/timepoints/delay_sliding_highres/")
    
    if missing_subjects:
        print(f"\nNote: {len(missing_subjects)} subjects were missing from the analysis")
        print("You may want to check the processing logs for these subjects.")