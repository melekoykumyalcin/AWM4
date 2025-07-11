#!/usr/bin/env python
"""
Cluster permutation test for encoding period results only.
Simplified version for running just encoding analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from glob import glob
from datetime import datetime

# Set paths
HOME_DIR = '/mnt/hpc/projects/awm4/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
delay_results_dir = PROCESSED_DIR + 'delayPseudo/'
OUTPUT_DIR = delay_results_dir + 'cluster_permutation/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
SCHEME = 'avg12'  # Change this to analyze different schemes
N_PERMUTATIONS = 1000000  # Reduce for faster testing

# Features and colors
FEATURES = {
    'voice': {'name': 'Voice Identity', 'color': '#1c686b'},
    'location': {'name': 'Location', 'color': '#cb6a3e'}
}


def collect_encoding_data():
    """Collect all subject data for encoding period."""
    print(f"Collecting encoding data for {SCHEME}")
    
    subject_dirs = glob(f"{delay_results_dir}/subject_*/")
    results = {feat: {'data': [], 'timepoints': None} for feat in FEATURES}
    
    for subject_dir in sorted(subject_dirs):
        subject_id = int(subject_dir.split('subject_')[-1].rstrip('/'))
        
        # Check if all features exist
        all_exist = all(
            os.path.exists(f"{subject_dir}/{SCHEME}_{feat}_mean.npy") 
            for feat in FEATURES
        )
        
        if not all_exist:
            continue
            
        # Load data
        for feature in FEATURES:
            mean_data = np.load(f"{subject_dir}/{SCHEME}_{feature}_mean.npy")
            results[feature]['data'].append(mean_data)
            
            if results[feature]['timepoints'] is None:
                tp_file = f"{subject_dir}/{SCHEME}_{feature}_timepoints.npy"
                if os.path.exists(tp_file):
                    results[feature]['timepoints'] = np.load(tp_file)
    
    # Convert to arrays
    for feature in FEATURES:
        if results[feature]['data']:
            results[feature]['data'] = np.stack(results[feature]['data'])
            print(f"  {feature}: {results[feature]['data'].shape[0]} subjects")
    
    return results


def run_encoding_clusters():
    """Run cluster permutation tests and create plots."""
    
    # Collect data
    data = collect_encoding_data()
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Process each feature
    cluster_y_base = 0.68
    
    for idx, (feature, feature_info) in enumerate(FEATURES.items()):
        if not data[feature]['data'].size:
            continue
            
        print(f"\nProcessing {feature}...")
        
        # Get data
        scores = data[feature]['data']
        timepoints = data[feature]['timepoints']
        
        # Calculate statistics
        mean_acc = np.mean(scores, axis=0)
        sem = np.std(scores, axis=0) / np.sqrt(scores.shape[0])
        
        # Run cluster test
        print(f"  Running cluster permutation test...")
        T_obs, clusters, p_values, _ = mne.stats.permutation_cluster_1samp_test(
            scores - 0.5,
            n_permutations=N_PERMUTATIONS,
            threshold=None,
            tail=1,
            out_type='mask',
            seed=42
        )
        
        # Plot results
        color = feature_info['color']
        ax.plot(timepoints, mean_acc, color=color, linewidth=2,
                label=f"{feature_info['name']} (N={scores.shape[0]})")
        ax.fill_between(timepoints, mean_acc - sem, mean_acc + sem,
                       alpha=0.2, color=color)
        
        # Add significant clusters
        cluster_y = cluster_y_base - idx * 0.02
        
        for c_idx, (cluster_mask, p_val) in enumerate(zip(clusters, p_values)):
            if p_val < 0.05:
                cluster_times = timepoints[cluster_mask]
                if len(cluster_times) > 0:
                    ax.plot([cluster_times[0], cluster_times[-1]], 
                           [cluster_y, cluster_y],
                           color=color, linewidth=4, solid_capstyle='round')
                    
                    # Save cluster info
                    output_file = f"{OUTPUT_DIR}/{SCHEME}_{feature}_cluster_{c_idx}.txt"
                    with open(output_file, 'w') as f:
                        f.write(f"Significant cluster for {feature}\n")
                        f.write(f"p-value: {p_val:.5f}\n")
                        f.write(f"Time range: {cluster_times[0]:.3f} - {cluster_times[-1]:.3f}s\n")
                    
                    print(f"  Found significant cluster: p={p_val:.4f}, "
                          f"time={cluster_times[0]:.2f}-{cluster_times[-1]:.2f}s")
    
    # Finalize plot
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Decoding Accuracy', fontsize=14)
    ax.set_title(f'Encoding Period - Cluster Permutation Results ({SCHEME})', fontsize=16)
    ax.set_xlim(2.0, 4.7)
    ax.set_ylim(0.45, 0.65)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{SCHEME}_encoding_clusters.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/{SCHEME}_encoding_clusters.pdf")
    plt.close()
    
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    print(f"Starting encoding cluster permutation analysis at {datetime.now()}")
    run_encoding_clusters()
    print(f"Completed at {datetime.now()}")