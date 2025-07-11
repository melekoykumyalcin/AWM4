#!/usr/bin/env python
"""
Updated pseudoGA.py using analysis-comparison color scheme
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
RESULTS_DIR = PROCESSED_DIR + 'pseudo/'
OUTPUT_DIR = PROCESSED_DIR + 'pseudo/grand_average/'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotting parameters
plt.rcParams['figure.figsize'] = [14, 10]
plt.rcParams['font.size'] = 12

# Color scheme from analysis-comparison
FEATURES = {
    'voice': {
        'name': 'voice identity',
        'color': {
            'S1': '#1c686b',
            'S2': '#64b1b5'
        }
    },
    'location': {
        'name': 'location',
        'color': {
            'S1': '#cb6a3e',
            'S2': '#e8a87c'
        }
    }
}

SCHEME_PALETTE = {
    'avg4': '#1f77b4',
    'avg8': '#ff7f0e',
    'avg12': '#2ca02c'
}

def plot_grand_average_comparison(grand_avg, timepoints, subjects):
    print("Creating grand average plots...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Voice across schemes
    ax = axes[0, 0]
    for scheme in SCHEME_PALETTE:
        if grand_avg[scheme]['voice'] is not None:
            data = grand_avg[scheme]['voice']
            ax.plot(timepoints, data['mean'], label=f'{scheme}', color=SCHEME_PALETTE[scheme], linewidth=2)
            ax.fill_between(timepoints, data['mean'] - data['sem'], data['mean'] + data['sem'],
                            color=SCHEME_PALETTE[scheme], alpha=0.3)
    ax.axhline(0.5, linestyle='--', color='black', alpha=0.5)
    ax.set_title('Voice Decoding')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Decoding Accuracy')
    ax.set_ylim(0.45, 0.9)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    # Plot 2: Location across schemes
    ax = axes[0, 1]
    for scheme in SCHEME_PALETTE:
        if grand_avg[scheme]['location'] is not None:
            data = grand_avg[scheme]['location']
            ax.plot(timepoints, data['mean'], label=f'{scheme}', color=SCHEME_PALETTE[scheme], linewidth=2)
            ax.fill_between(timepoints, data['mean'] - data['sem'], data['mean'] + data['sem'],
                            color=SCHEME_PALETTE[scheme], alpha=0.3)
    ax.axhline(0.5, linestyle='--', color='black', alpha=0.5)
    ax.set_title('Location Decoding')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Decoding Accuracy')
    ax.set_ylim(0.45, 0.9)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    # Plot 3: Feature comparison for best scheme
    ax = axes[1, 0]
    best_scheme = 'avg8'
    for feature in FEATURES:
        if grand_avg[best_scheme][feature] is not None:
            data = grand_avg[best_scheme][feature]
            ax.plot(timepoints, data['mean'], label=FEATURES[feature]['name'].title(),
                    color=FEATURES[feature]['color']['S1'], linewidth=2)
            ax.fill_between(timepoints, data['mean'] - data['sem'], data['mean'] + data['sem'],
                            color=FEATURES[feature]['color']['S1'], alpha=0.3)
    ax.axhline(0.5, linestyle='--', color='black', alpha=0.5)
    ax.set_title(f'{best_scheme.upper()} Feature Comparison')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Decoding Accuracy')
    ax.set_ylim(0.45, 0.9)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')
    summary = "Summary:\n\n"
    for scheme in SCHEME_PALETTE:
        summary += f"{scheme.upper()}\n"
        for feature in FEATURES:
            if grand_avg[scheme][feature] is not None:
                mean_val = np.mean(grand_avg[scheme][feature]['mean'])
                peak_val = np.max(grand_avg[scheme][feature]['mean'])
                peak_time = timepoints[np.argmax(grand_avg[scheme][feature]['mean'])]
                summary += f"  {feature.title()}: Mean={mean_val:.3f}, Peak={peak_val:.3f} @ {peak_time:.2f}s\n"
        summary += "\n"
    ax.text(0.05, 0.95, summary, va='top', fontsize=11, family='monospace')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/grand_average_comparison.png', dpi=300)
    plt.savefig(f'{OUTPUT_DIR}/grand_average_comparison.pdf')
    plt.close()

def plot_individual_scheme_details(all_results, grand_avg, timepoints, subjects):
    print("Creating individual scheme detail plots...")
    for scheme in SCHEME_PALETTE:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"{scheme.upper()} Detailed Analysis", fontsize=16)

        for i, feature in enumerate(FEATURES):
            # Left: subject lines + grand average
            ax = axes[i, 0]
            for subj in all_results[scheme][feature]['mean']:
                ax.plot(timepoints, subj, color='gray', alpha=0.3)
            avg = grand_avg[scheme][feature]
            ax.plot(timepoints, avg['mean'], color=FEATURES[feature]['color']['S1'], linewidth=3)
            ax.fill_between(timepoints, avg['mean'] - avg['sem'], avg['mean'] + avg['sem'],
                            color=FEATURES[feature]['color']['S1'], alpha=0.3)
            ax.set_title(f"{FEATURES[feature]['name'].title()} - Grand Avg")
            ax.set_ylim(0.35, 0.9)
            ax.axhline(0.5, linestyle='--', color='black', alpha=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Accuracy')
            ax.grid(True, linestyle=':', alpha=0.5)

            # Right: heatmap
            ax = axes[i, 1]
            subj_data = np.array(all_results[scheme][feature]['mean'])
            sort_idx = np.argsort(np.mean(subj_data, axis=1))[::-1]
            sorted_data = subj_data[sort_idx]
            im = ax.imshow(sorted_data, aspect='auto', cmap='RdBu_r', vmin=0.4, vmax=0.6)
            ax.set_title(f"{FEATURES[feature]['name'].title()} - Heatmap")
            ax.set_xlabel('Time')
            ax.set_ylabel('Subjects')
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/{scheme}_detail.png', dpi=300)
        plt.savefig(f'{OUTPUT_DIR}/{scheme}_detail.pdf')
        plt.close()

def plot_statistical_comparison(all_results, timepoints):
    print("Creating statistical comparison plots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, feature in enumerate(FEATURES):
        ax = axes[idx]
        scheme_data = {}
        for scheme in SCHEME_PALETTE:
            if all_results[scheme][feature]['mean']:
                scheme_data[scheme] = [np.max(x) for x in all_results[scheme][feature]['mean']]
        if scheme_data:
            data = []
            labels = []
            for scheme, vals in scheme_data.items():
                data.extend(vals)
                labels.extend([scheme] * len(vals))
            df = pd.DataFrame({'Scheme': labels, 'Peak Accuracy': data})
            sns.boxplot(data=df, x='Scheme', y='Peak Accuracy', palette=SCHEME_PALETTE, ax=ax)
            sns.stripplot(data=df, x='Scheme', y='Peak Accuracy', color='black', alpha=0.4, size=3, ax=ax)
            ax.axhline(0.5, linestyle='--', color='black', alpha=0.5)
            ax.set_title(FEATURES[feature]['name'].title())
            ax.set_ylim(0.45, 0.9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/statistical_comparison.png', dpi=300)
    plt.savefig(f'{OUTPUT_DIR}/statistical_comparison.pdf')
    plt.close()


def collect_subject_results():
    """Collect results from all processed subjects"""
    print("Collecting results from all subjects...")
    
    # Find all subject directories
    subject_dirs = glob(f"{RESULTS_DIR}/subject_*/")
    subjects = []
    
    all_results = {
        'avg4': {'voice': {'mean': [], 'std': []}, 'location': {'mean': [], 'std': []}},
        'avg8': {'voice': {'mean': [], 'std': []}, 'location': {'mean': [], 'std': []}},
        'avg12': {'voice': {'mean': [], 'std': []}, 'location': {'mean': [], 'std': []}}
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
            for feature in ['voice', 'location']:
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
        timepoints = np.load(f"{first_subject_dir}/avg4_voice_timepoints.npy")
    else:
        timepoints = np.linspace(0.1, 0.5, 40)
    
    return all_results, subjects, timepoints, summaries

def compute_grand_average(all_results):
    """Compute grand average across subjects"""
    grand_avg = {}
    
    for scheme in all_results:
        grand_avg[scheme] = {}
        for feature in ['voice', 'location']:
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

def create_summary_statistics(all_results, grand_avg, summaries):
    """Create summary statistics and save to files"""
    print("Creating summary statistics...")
    
    # Create DataFrame for overall statistics
    stats_rows = []
    
    for scheme in ['avg4', 'avg8', 'avg12']:
        for feature in ['voice', 'location']:
            if grand_avg[scheme][feature] is not None:
                data = grand_avg[scheme][feature]
                
                row = {
                    'scheme': scheme,
                    'feature': feature,
                    'n_subjects': data['n_subjects'],
                    'mean_accuracy': np.mean(data['mean']),
                    'peak_accuracy': np.max(data['mean']),
                    'peak_time': 0.1 + np.argmax(data['mean']) * 0.01,  # Approximate
                    'std_across_time': np.mean(data['std']),
                    'sem_across_time': np.mean(data['sem'])
                }
                stats_rows.append(row)
    
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(f'{OUTPUT_DIR}/grand_average_statistics.csv', index=False)
    
    # Create subject-wise summary
    subject_rows = []
    for summary in summaries:
        subject_id = summary['subject_id']
        for scheme in summary['results_summary']:
            for feature in ['voice', 'location']:
                row = {
                    'subject': subject_id,
                    'scheme': scheme,
                    'feature': feature,
                    'mean_accuracy': summary['results_summary'][scheme][feature]['mean_accuracy'],
                    'max_accuracy': summary['results_summary'][scheme][feature]['max_accuracy']
                }
                subject_rows.append(row)
    
    subject_df = pd.DataFrame(subject_rows)
    subject_df.to_csv(f'{OUTPUT_DIR}/subject_wise_summary.csv', index=False)
    
    # Save grand average arrays
    for scheme in grand_avg:
        for feature in ['voice', 'location']:
            if grand_avg[scheme][feature] is not None:
                np.save(f'{OUTPUT_DIR}/grand_avg_{scheme}_{feature}_mean.npy',
                       grand_avg[scheme][feature]['mean'])
                np.save(f'{OUTPUT_DIR}/grand_avg_{scheme}_{feature}_sem.npy',
                       grand_avg[scheme][feature]['sem'])
def main():
    """Main function"""
    print(f"Grand averaging analysis started at: {datetime.now()}")
    
    # Collect results
    all_results, subjects, timepoints, summaries = collect_subject_results()
    
    if not subjects:
        print("No subjects found. Exiting.")
        return
    
    # Compute grand average
    grand_avg = compute_grand_average(all_results)
    
    # Create plots
    # Create plots
    plot_grand_average_comparison(grand_avg, timepoints, subjects)
    plot_individual_scheme_details(all_results, grand_avg, timepoints, subjects)
    plot_statistical_comparison(all_results, timepoints)
    
    # Create summary statistics
    create_summary_statistics(all_results, grand_avg, summaries)
    
    # Save timepoints
    np.save(f'{OUTPUT_DIR}/timepoints.npy', timepoints)
    
    # Create final report
    report = f"""

Grand Average Analysis Report
=============================
Generated: {datetime.now()}

Subjects Included: {len(subjects)}
Subject IDs: {subjects}

Summary of Results:
------------------
"""
    
    for scheme in ['avg4', 'avg8', 'avg12']:
        report += f"\n{scheme.upper()}:\n"
        for feature in ['voice', 'location']:
            if grand_avg[scheme][feature] is not None:
                data = grand_avg[scheme][feature]
                report += f"  {feature.title()}: "
                report += f"Peak={np.max(data['mean']):.3f}, "
                report += f"Mean={np.mean(data['mean']):.3f}\n"
    
    with open(f'{OUTPUT_DIR}/analysis_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nAnalysis completed at: {datetime.now()}")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()


