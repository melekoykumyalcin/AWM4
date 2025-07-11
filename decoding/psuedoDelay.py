#!/usr/bin/env python
"""
Script to collect and grand average delay period results from all subjects
Usage: python hpc_delay_grand_average.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import json
from datetime import datetime

# Set paths
HOME_DIR = '/mnt/hpc/projects/awm4/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
RESULTS_DIR = PROCESSED_DIR + 'delayPseudo/'
OUTPUT_DIR = PROCESSED_DIR + 'delayPseudo/grand_average/'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotting parameters
plt.rcParams['figure.figsize'] = [14, 10]
plt.rcParams['font.size'] = 12
colors = {'avg4': '#1f77b4', 'avg8': '#ff7f0e', 'avg12': '#2ca02c'}
feature_colors = {'maintained_voice': '#1c686b', 'maintained_location': '#cb6a3e'}

# Critical timepoints for delay period
CRITICAL_TIMEPOINTS = [3.5, 4.5]

def collect_subject_results():
    """Collect results from all processed subjects"""
    print("Collecting delay period results from all subjects...")
    
    # Find all subject directories
    subject_dirs = glob(f"{RESULTS_DIR}/subject_*/")
    subjects = []
    
    all_results = {
        'avg4': {'maintained_voice': {'mean': [], 'std': []}, 'maintained_location': {'mean': [], 'std': []}},
        'avg8': {'maintained_voice': {'mean': [], 'std': []}, 'maintained_location': {'mean': [], 'std': []}},
        'avg12': {'maintained_voice': {'mean': [], 'std': []}, 'maintained_location': {'mean': [], 'std': []}}
    }
    
    summaries = []
    
    for subject_dir in sorted(subject_dirs):
        # Extract subject ID
        subject_id = int(subject_dir.split('subject_')[-1].rstrip('/'))
        
        # Check if summary exists
        summary_file = f"{subject_dir}/summary.json"
        if not os.path.exists(summary_file):
            print(f"  Skipping subject {subject_id} - no summary file")
            continue
        
        # Load summary
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        summary['subject_id'] = subject_id
        summaries.append(summary)
        
        subjects.append(subject_id)
        
        # Load results for each scheme
        for scheme in ['avg4', 'avg8', 'avg12']:
            for feature in ['maintained_voice', 'maintained_location']:
                # Load mean and std
                mean_file = f"{subject_dir}/{scheme}_{feature}_mean.npy"
                std_file = f"{subject_dir}/{scheme}_{feature}_std.npy"
                
                if os.path.exists(mean_file) and os.path.exists(std_file):
                    all_results[scheme][feature]['mean'].append(np.load(mean_file))
                    all_results[scheme][feature]['std'].append(np.load(std_file))
                else:
                    print(f"  Missing data for subject {subject_id}, {scheme}, {feature}")
    
    print(f"Collected results from {len(subjects)} subjects: {subjects}")
    
    # Load timepoints (should be same for all subjects)
    if subject_dirs:
        first_subject_dir = sorted(subject_dirs)[0]
        timepoints = np.load(f"{first_subject_dir}/avg4_maintained_voice_timepoints.npy")
    else:
        timepoints = np.linspace(2.0, 4.7, 270)  # Default for delay period
    
    return all_results, subjects, timepoints, summaries

def compute_grand_average(all_results):
    """Compute grand average across subjects"""
    grand_avg = {}
    
    for scheme in all_results:
        grand_avg[scheme] = {}
        for feature in ['maintained_voice', 'maintained_location']:
            if all_results[scheme][feature]['mean']:
                # Stack all subjects
                all_means = np.stack(all_results[scheme][feature]['mean'])
                
                # Grand average
                grand_mean = np.mean(all_means, axis=0)
                grand_std = np.std(all_means, axis=0)
                grand_sem = grand_std / np.sqrt(len(all_means))
                
                grand_avg[scheme][feature] = {
                    'mean': grand_mean,
                    'std': grand_std,
                    'sem': grand_sem,
                    'n_subjects': len(all_means)
                }
            else:
                grand_avg[scheme][feature] = None
    
    return grand_avg

def plot_grand_average_comparison(grand_avg, timepoints, subjects):
    """Create grand average comparison plots for delay period"""
    print("Creating delay period grand average plots...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Maintained Voice comparison across schemes
    ax = axes[0, 0]
    for scheme in ['avg4', 'avg8', 'avg12']:
        if grand_avg[scheme]['maintained_voice'] is not None:
            data = grand_avg[scheme]['maintained_voice']
            ax.plot(timepoints, data['mean'], 
                   label=f'{scheme} (n={data["n_subjects"]})',
                   color=colors[scheme], linewidth=2)
            ax.fill_between(timepoints, 
                           data['mean'] - data['sem'],
                           data['mean'] + data['sem'],
                           alpha=0.2, color=colors[scheme])
    
    # Add critical timepoints
    for tp in CRITICAL_TIMEPOINTS:
        ax.axvline(x=tp, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Decoding Accuracy')
    ax.set_title(f'Maintained Voice Identity - Grand Average (N={len(subjects)})')
    ax.legend()
    ax.set_ylim([0.45, 0.65])
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Maintained Location comparison across schemes
    ax = axes[0, 1]
    for scheme in ['avg4', 'avg8', 'avg12']:
        if grand_avg[scheme]['maintained_location'] is not None:
            data = grand_avg[scheme]['maintained_location']
            ax.plot(timepoints, data['mean'], 
                   label=f'{scheme} (n={data["n_subjects"]})',
                   color=colors[scheme], linewidth=2)
            ax.fill_between(timepoints, 
                           data['mean'] - data['sem'],
                           data['mean'] + data['sem'],
                           alpha=0.2, color=colors[scheme])
    
    # Add critical timepoints
    for tp in CRITICAL_TIMEPOINTS:
        ax.axvline(x=tp, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Decoding Accuracy')
    ax.set_title(f'Maintained Location - Grand Average (N={len(subjects)})')
    ax.legend()
    ax.set_ylim([0.45, 0.65])
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Direct comparison of features for best scheme
    ax = axes[1, 0]
    best_scheme = 'avg8'  # Default, or determine programmatically
    
    for feature in ['maintained_voice', 'maintained_location']:
        if grand_avg[best_scheme][feature] is not None:
            data = grand_avg[best_scheme][feature]
            ax.plot(timepoints, data['mean'], 
                   label=f'{feature.replace("maintained_", "").title()}',
                   color=feature_colors[feature], linewidth=2)
            ax.fill_between(timepoints, 
                           data['mean'] - data['sem'],
                           data['mean'] + data['sem'],
                           alpha=0.2, color=feature_colors[feature])
    
    # Add critical timepoints
    for tp in CRITICAL_TIMEPOINTS:
        ax.axvline(x=tp, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Decoding Accuracy')
    ax.set_title(f'{best_scheme.upper()} - Feature Comparison (Delay Period)')
    ax.legend()
    ax.set_ylim([0.45, 0.65])
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary table
    summary_text = f"Delay Period Grand Average Summary (N={len(subjects)})\n"
    summary_text += "="*45 + "\n\n"
    
    for scheme in ['avg4', 'avg8', 'avg12']:
        summary_text += f"{scheme.upper()}:\n"
        for feature in ['maintained_voice', 'maintained_location']:
            if grand_avg[scheme][feature] is not None:
                data = grand_avg[scheme][feature]
                peak_acc = np.max(data['mean'])
                peak_time = timepoints[np.argmax(data['mean'])]
                mean_acc = np.mean(data['mean'])
                
                feature_name = feature.replace('maintained_', '').title()
                summary_text += f"  {feature_name}:\n"
                summary_text += f"    Peak: {peak_acc:.3f} at {peak_time:.3f}s\n"
                summary_text += f"    Mean: {mean_acc:.3f}\n"
        summary_text += "\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Delay Period Analysis - Grand Average Results', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/delay_grand_average_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/delay_grand_average_comparison.pdf', bbox_inches='tight')
    plt.close()

def plot_time_course_comparison(grand_avg, timepoints):
    """Create detailed time course plots"""
    print("Creating detailed time course plots...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    for idx, feature in enumerate(['maintained_voice', 'maintained_location']):
        ax = axes[idx]
        
        # Plot all schemes
        for scheme in ['avg4', 'avg8', 'avg12']:
            if grand_avg[scheme][feature] is not None:
                data = grand_avg[scheme][feature]
                ax.plot(timepoints, data['mean'], 
                       label=f'{scheme}',
                       color=colors[scheme], linewidth=2)
                
                # Add confidence interval
                ax.fill_between(timepoints, 
                               data['mean'] - data['sem'] * 1.96,
                               data['mean'] + data['sem'] * 1.96,
                               alpha=0.15, color=colors[scheme])
        
        # Add critical timepoints with labels
        for tp in CRITICAL_TIMEPOINTS:
            ax.axvline(x=tp, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            ax.text(tp, ax.get_ylim()[1] - 0.01, f'{tp}s', 
                   ha='center', va='top', fontsize=9, color='gray')
        
        # Add chance level
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Chance')
        
        # Formatting
        ax.set_ylabel('Decoding Accuracy')
        ax.set_title(f'{feature.replace("maintained_", "").title()} Decoding During Delay Period')
        ax.legend(loc='best')
        ax.set_ylim([0.47, 0.58])
        ax.grid(True, alpha=0.3)
        
        # Add shaded regions for different delay phases
        ax.axvspan(2.0, 3.5, alpha=0.1, color='blue', label='Early delay')
        ax.axvspan(3.5, 4.5, alpha=0.1, color='green', label='Late delay')
        ax.axvspan(4.5, 4.7, alpha=0.1, color='red', label='Pre-test')
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('Delay Period Time Course Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/delay_time_course_detailed.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/delay_time_course_detailed.pdf', bbox_inches='tight')
    plt.close()

def create_summary_statistics(all_results, grand_avg, summaries):
    """Create summary statistics and save to files"""
    print("Creating summary statistics...")
    
    # Create DataFrame for overall statistics
    stats_rows = []
    
    for scheme in ['avg4', 'avg8', 'avg12']:
        for feature in ['maintained_voice', 'maintained_location']:
            if grand_avg[scheme][feature] is not None:
                data = grand_avg[scheme][feature]
                
                row = {
                    'scheme': scheme,
                    'feature': feature,
                    'n_subjects': data['n_subjects'],
                    'mean_accuracy': np.mean(data['mean']),
                    'peak_accuracy': np.max(data['mean']),
                    'peak_time': 2.0 + np.argmax(data['mean']) * 0.01,  # Approximate
                    'std_across_time': np.mean(data['std']),
                    'sem_across_time': np.mean(data['sem'])
                }
                stats_rows.append(row)
    
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(f'{OUTPUT_DIR}/delay_grand_average_statistics.csv', index=False)
    
    # Create subject-wise summary
    subject_rows = []
    for summary in summaries:
        subject_id = summary['subject_id']
        for scheme in summary['results_summary']:
            for feature in ['maintained_voice', 'maintained_location']:
                row = {
                    'subject': subject_id,
                    'scheme': scheme,
                    'feature': feature,
                    'mean_accuracy': summary['results_summary'][scheme][feature]['mean_accuracy'],
                    'max_accuracy': summary['results_summary'][scheme][feature]['max_accuracy']
                }
                subject_rows.append(row)
    
    subject_df = pd.DataFrame(subject_rows)
    subject_df.to_csv(f'{OUTPUT_DIR}/delay_subject_wise_summary.csv', index=False)
    
    # Save grand average arrays
    for scheme in grand_avg:
        for feature in ['maintained_voice', 'maintained_location']:
            if grand_avg[scheme][feature] is not None:
                np.save(f'{OUTPUT_DIR}/grand_avg_{scheme}_{feature}_mean.npy',
                       grand_avg[scheme][feature]['mean'])
                np.save(f'{OUTPUT_DIR}/grand_avg_{scheme}_{feature}_sem.npy',
                       grand_avg[scheme][feature]['sem'])

def main():
    """Main function"""
    print(f"Delay period grand averaging analysis started at: {datetime.now()}")
    
    # Collect results
    all_results, subjects, timepoints, summaries = collect_subject_results()
    
    if not subjects:
        print("No subjects found. Exiting.")
        return
    
    # Compute grand average
    grand_avg = compute_grand_average(all_results)
    
    # Create plots
    plot_grand_average_comparison(grand_avg, timepoints, subjects)
    plot_time_course_comparison(grand_avg, timepoints)
    
    # Create summary statistics
    create_summary_statistics(all_results, grand_avg, summaries)
    
    # Save timepoints
    np.save(f'{OUTPUT_DIR}/timepoints.npy', timepoints)
    
    # Create final report
    report = f"""
Delay Period Grand Average Analysis Report
==========================================
Generated: {datetime.now()}

Subjects Included: {len(subjects)}
Subject IDs: {subjects}

Time Period: 2.0 - 4.7 seconds (delay period)
Analysis Method: Proper nested CV with pseudo-trials created within folds

Summary of Results:
------------------
"""
    
    for scheme in ['avg4', 'avg8', 'avg12']:
        report += f"\n{scheme.upper()}:\n"
        for feature in ['maintained_voice', 'maintained_location']:
            if grand_avg[scheme][feature] is not None:
                data = grand_avg[scheme][feature]
                report += f"  {feature.replace('maintained_', '').title()}: "
                report += f"Peak={np.max(data['mean']):.3f}, "
                report += f"Mean={np.mean(data['mean']):.3f}\n"
    
    with open(f'{OUTPUT_DIR}/delay_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nAnalysis completed at: {datetime.now()}")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()