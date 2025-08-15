#!/usr/bin/env python
"""
Feature-Level Representational Similarity Analysis (RSA) Script
Tests whether task-irrelevant features are maintained in working memory
Based on template-based leave-one-out approach (Günseli & Aly, 2020)

Feature-Level RSA Script - MATCHED TEMPLATES VERSION
Voice RSA: Tests speaker identity while holding location constant  
Location RSA: Tests location identity while holding speaker constant

Usage: python feature_level_rsa.py --subject 23
"""

import os
import sys
import argparse
import numpy as np
import mne
mne.set_log_level('WARNING')
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp
import json
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run feature-level RSA analysis')
parser.add_argument('--subject', type=int, required=True, help='Subject ID')
parser.add_argument('--n_iterations', type=int, default=100, help='Number of iterations for statistical robustness')
args = parser.parse_args()

# Set paths for HPC (same as original script)
HOME_DIR = '/mnt/hpc/projects/awm4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
DATA_PATH = HOME_DIR + '/AWM4_data/raw/'
CORRECTED_DATA = HOME_DIR + '/AWM4_data/raw/correctTriggers/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'

# Create output directory
OUTPUT_DIR = PROCESSED_DIR + f'featureRSA_matched/subject_{args.subject}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Log file path
LOG_FILE_PATH = f'{OUTPUT_DIR}/rsa_processing_log.txt'

def write_log(message):
    """Write message to log file"""
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(message + '\n')
    print(message)

# Initialize log
write_log(f"Feature-level RSA analysis started at: {datetime.now()}")
write_log(f"Subject: {args.subject}")

# Analysis parameters
RESAMPLE_FREQ = 100  # Hz to match original script
TIME_WINDOWS = {
    'pre_ping': {'tmin': 2.0, 'tmax': 3.5, 'name': 'Pre-ping (2.0-3.5s)'},
    'post_ping': {'tmin': 3.5, 'tmax': 4.5, 'name': 'Post-ping (3.5-4.5s)'}
}

# Event dictionary (same as original)
EVENT_DICT = {
    'S1/Sp1/L1': 111, 'S1/Sp1/L2': 112, 'S1/Sp1/L3': 113, 'S1/Sp1/L4': 114,
    'S1/Sp2/L1': 121, 'S1/Sp2/L2': 122, 'S1/Sp2/L3': 123, 'S1/Sp2/L4': 124,
    'S1/Sp3/L1': 131, 'S1/Sp3/L2': 132, 'S1/Sp3/L3': 133, 'S1/Sp3/L4': 134,
    'S1/Sp4/L1': 141, 'S1/Sp4/L2': 142, 'S1/Sp4/L3': 143, 'S1/Sp4/L4': 144,
    'S2/Sp1/L1': 211, 'S2/Sp1/L2': 212, 'S2/Sp1/L3': 213, 'S2/Sp1/L4': 214,
    'S2/Sp2/L1': 221, 'S2/Sp2/L2': 222, 'S2/Sp2/L3': 223, 'S2/Sp2/L4': 224,
    'S2/Sp3/L1': 231, 'S2/Sp3/L2': 232, 'S2/Sp3/L3': 233, 'S2/Sp3/L4': 234,
    'S2/Sp4/L1': 241, 'S2/Sp4/L2': 242, 'S2/Sp4/L3': 243, 'S2/Sp4/L4': 244,
    'Cue_S1': 101, 'Cue_S2': 201
}

def extract_maintained_information(subject, metaInfo):
    """Extract which information is maintained in working memory based on retro-cues"""
    write_log(f"\nExtracting maintained information for subject {subject}...")
    
    try:
        # Get file information
        actInd = (metaInfo.Subject==subject) & (metaInfo.Valid==1)
        
        # Determine if subject is in early subjects
        early_subject = subject in np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])[:7]
        
        if early_subject:
            actFiles = pd.Series([f.split('.')[0] + '_correct_triggers.fif' for f in metaInfo['MEG_Name']])[actInd]
        else:
            actFiles = metaInfo['MEG_Name'][actInd]
        
        # Load and concatenate events
        all_events = None
        reference_dev_head_t_ref = None
        
        for ff in range(actFiles.count()):
            if early_subject:
                fname = CORRECTED_DATA + actFiles.iloc[ff]
                raw = mne.io.read_raw_fif(fname, preload=True)
            else:
                fname = DATA_PATH + actFiles.iloc[ff]
                raw = mne.io.read_raw_ctf(fname, 'truncate', True)
                
            if ff == 0:
                reference_dev_head_t_ref = raw.info["dev_head_t"]
            else:
                raw.info['dev_head_t'] = reference_dev_head_t_ref
                
            events = mne.find_events(raw, 'UPPT001', shortest_event=1)
            if ff != 0:
                events = events[events[:, 1] == 0, :]
                
            if ff == 0:
                all_events = events
            else:
                all_events = np.concatenate((all_events, events), axis=0)
            del raw
        
        # Find retro-cue events
        S1_cue_indices = np.where(all_events[:,2] == 101)[0]
        S2_cue_indices = np.where(all_events[:,2] == 201)[0]
        
        write_log(f"Found {len(S1_cue_indices)} S1 cues and {len(S2_cue_indices)} S2 cues")
        
        # Create memorized array
        memorized = np.zeros(len(all_events[:,2]))
        
        # For S1 cues, maintained stimulus is 4 positions before
        for i in S1_cue_indices:
            if i >= 4:
                memorized[i - 4] = all_events[i - 4, 2]
        
        # For S2 cues, maintained stimulus is 2 positions before
        for i in S2_cue_indices:
            if i >= 2:
                memorized[i - 2] = all_events[i - 2, 2]
        
        # Keep only non-zero values
        memorized_values = memorized[memorized != 0]
        
        write_log(f"Extracted {len(memorized_values)} maintained trials")
        
        return memorized_values, all_events
        
    except Exception as e:
        write_log(f"Error extracting maintained information: {str(e)}")
        return None, None

def load_subject_data(subject, meta_info):
    """Load and preprocess data for RSA analysis"""
    write_log(f"\nLoading data for subject {subject}...")
    
    try:
        # First extract maintained information
        memorized, all_events = extract_maintained_information(subject, meta_info)
        
        if memorized is None:
            write_log("Could not extract maintained information")
            return None, None
        
        # Load cleaned epochs
        clean_trials = mne.read_epochs(
            f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif",
            preload=True,
            verbose='ERROR'
        )
        
        # Special case handling 
        if subject == 23:
            drop_idx = 64 * 7
            if drop_idx < len(clean_trials):
                memorized = np.delete(memorized, drop_idx)

        if subject == 28:
            drop_idx = 63
            if drop_idx < len(clean_trials):
                clean_trials.drop(drop_idx)
          
        # Check for jump artifacts
        jname = f"{PROCESSED_DIR}/ICAs/Jumps{subject}.npy"
        if os.path.isfile(jname):
            jump_inds = np.load(jname)
            if len(jump_inds) > 0:
                jump_inds = np.array(jump_inds, dtype=int)
                valid_jump_inds = jump_inds[jump_inds < len(clean_trials)]
                
                if len(valid_jump_inds) > 0:
                    clean_trials.drop(valid_jump_inds, reason='jump')
                    memorized = np.delete(memorized, valid_jump_inds)
        
        # Update event codes to reflect maintained information
        clean_trials.events[:, 2] = memorized[:len(clean_trials.events)]
        clean_trials.event_id = EVENT_DICT
        
        # Select magnetometers and resample
        mag_epochs = clean_trials.copy().pick_types(meg='mag')
        mag_epochs = mag_epochs.resample(RESAMPLE_FREQ, npad='auto')
        
        # Get data and maintained events
        maintained_events = mag_epochs.events[:, 2]
        
        write_log(f"Data loaded successfully. Total trials: {len(maintained_events)}")
        
        # Log distribution
        speakers = {
            'Sp1+2': sum(1 for v in maintained_events if (v//10)%10 in [1, 2]),
            'Sp3+4': sum(1 for v in maintained_events if (v//10)%10 in [3, 4])
        }
        locations = {
            'L1+2': sum(1 for v in maintained_events if v%10 in [1, 2]),
            'L3+4': sum(1 for v in maintained_events if v%10 in [3, 4])
        }
        write_log(f"Speaker distribution: {speakers}")
        write_log(f"Location distribution: {locations}")
        
        return mag_epochs, maintained_events
        
    except Exception as e:
        write_log(f"Error loading data: {str(e)}")
        import traceback
        write_log(traceback.format_exc())
        return None, None

def feature_level_rsa(epochs_data, maintained_events, feature_type, time_window_name, time_window_params):
    """
    Perform feature-level RSA using location/speaker-matched templates
    
    Parameters:
    -----------
    epochs_data : ndarray, shape (n_trials, n_channels, n_times)
        Epoch data for specific time window
    maintained_events : array
        Event codes indicating which feature was maintained
    feature_type : str
        'voice' or 'location'
    time_window_name : str
        Name of time window for logging
    time_window_params : dict
        Time window parameters
    
    Returns:
    --------
    same_correlations : array
        Correlations between left-out trial and same-feature template
    different_correlations : array  
        Correlations between left-out trial and different-feature template
    difference_scores : array
        Fisher-transformed same-different correlation differences
    """
    
    write_log(f"\n  Running {feature_type} RSA (MATCHED TEMPLATES) for {time_window_name}")
    
    # Flatten data to trial x (channel * timepoint) matrix
    n_trials, n_channels, n_times = epochs_data.shape
    flattened_data = epochs_data.reshape(n_trials, n_channels * n_times)
    write_log(f"  Flattened data shape: {flattened_data.shape}")
    
    # Log trial distribution first
    if feature_type == 'voice':
        group1_all = [i for i, event in enumerate(maintained_events) if (event//10)%10 in [1, 2]]
        group2_all = [i for i, event in enumerate(maintained_events) if (event//10)%10 in [3, 4]]
        write_log(f"  Total Sp1+2 trials: {len(group1_all)}")
        write_log(f"  Total Sp3+4 trials: {len(group2_all)}")
    else:  # location
        group1_all = [i for i, event in enumerate(maintained_events) if event%10 in [1, 2]]
        group2_all = [i for i, event in enumerate(maintained_events) if event%10 in [3, 4]]
        write_log(f"  Total L1+2 trials: {len(group1_all)}")
        write_log(f"  Total L3+4 trials: {len(group2_all)}")
    
    # Storage for correlations
    same_correlations = []
    different_correlations = []
    skipped_trials = 0
    processed_trials = 0
    
    # Leave-one-out procedure with matched templates
    all_trial_indices = np.arange(len(maintained_events))
    
    for i, left_out_idx in enumerate(all_trial_indices):
        left_out_event = maintained_events[left_out_idx]
        left_out_trial = flattened_data[left_out_idx]
        
        # Create remaining trials mask
        remaining_indices = np.concatenate([all_trial_indices[:i], all_trial_indices[i+1:]])
        remaining_events = maintained_events[remaining_indices]
        
        if feature_type == 'voice':
            # VOICE RSA: Match location, test speaker
            left_out_location = left_out_event % 10
            left_out_speaker_group = (left_out_event//10)%10 in [1, 2]  # True=Sp1+2, False=Sp3+4
            
            # Find remaining trials from SAME location only
            same_location_mask = remaining_events % 10 == left_out_location
            same_location_indices = remaining_indices[same_location_mask]
            same_location_events = remaining_events[same_location_mask]
            
            # Within same location, group by speaker
            if left_out_speaker_group:  # Testing Sp1+2 trial
                same_speaker_indices = same_location_indices[(same_location_events//10)%10 <= 2]  # Sp1+2 from same location
                diff_speaker_indices = same_location_indices[(same_location_events//10)%10 >= 3]  # Sp3+4 from same location
            else:  # Testing Sp3+4 trial
                same_speaker_indices = same_location_indices[(same_location_events//10)%10 >= 3]  # Sp3+4 from same location
                diff_speaker_indices = same_location_indices[(same_location_events//10)%10 <= 2]  # Sp1+2 from same location
            
        else:  # location RSA
            # LOCATION RSA: Match speaker, test location
            left_out_speaker = (left_out_event//10)%10
            left_out_location_group = left_out_event%10 in [1, 2]  # True=L1+2, False=L3+4
            
            # Find remaining trials from SAME speaker only
            same_speaker_mask = (remaining_events//10)%10 == left_out_speaker
            same_speaker_indices = remaining_indices[same_speaker_mask]
            same_speaker_events = remaining_events[same_speaker_mask]
            
            # Within same speaker, group by location
            if left_out_location_group:  # Testing L1+2 trial
                same_location_indices = same_speaker_indices[same_speaker_events%10 <= 2]  # L1+2 from same speaker
                diff_location_indices = same_speaker_indices[same_speaker_events%10 >= 3]  # L3+4 from same speaker
            else:  # Testing L3+4 trial
                same_location_indices = same_speaker_indices[same_speaker_events%10 >= 3]  # L3+4 from same speaker
                diff_location_indices = same_speaker_indices[same_speaker_events%10 <= 2]  # L1+2 from same speaker
            
            # Rename for consistency
            same_speaker_indices = same_location_indices
            diff_speaker_indices = diff_location_indices
        
        # Check minimum template sizes (need at least 2 trials per template)
        min_template_size = 2
        if len(same_speaker_indices) < min_template_size or len(diff_speaker_indices) < min_template_size:
            skipped_trials += 1
            continue
        
        # Create templates
        same_template = np.mean(flattened_data[same_speaker_indices], axis=0)
        different_template = np.mean(flattened_data[diff_speaker_indices], axis=0)
        
        # Compute Spearman correlations
        same_corr, _ = spearmanr(left_out_trial, same_template)
        different_corr, _ = spearmanr(left_out_trial, different_template)
        
        # Handle NaN correlations
        if np.isnan(same_corr):
            same_corr = 0
        if np.isnan(different_corr):
            different_corr = 0
        
        same_correlations.append(same_corr)
        different_correlations.append(different_corr)
        processed_trials += 1
    
    # Log processing statistics
    write_log(f"  Processed trials: {processed_trials}")
    write_log(f"  Skipped trials (insufficient templates): {skipped_trials}")
    write_log(f"  Processing rate: {processed_trials/(processed_trials+skipped_trials)*100:.1f}%")
    
    if processed_trials == 0:
        write_log(f"  No trials could be processed!")
        return None, None, None
    
    # Convert to arrays
    same_correlations = np.array(same_correlations)
    different_correlations = np.array(different_correlations)
    
    # Fisher transform correlations to ensure normality
    same_correlations_fisher = np.arctanh(np.clip(same_correlations, -0.999, 0.999))
    different_correlations_fisher = np.arctanh(np.clip(different_correlations, -0.999, 0.999))
    
    # Compute difference scores (same - different)
    difference_scores = same_correlations_fisher - different_correlations_fisher
    
    write_log(f"  Mean same correlation: {np.mean(same_correlations):.4f}")
    write_log(f"  Mean different correlation: {np.mean(different_correlations):.4f}")
    write_log(f"  Mean difference score: {np.mean(difference_scores):.4f}")
    
    return same_correlations, different_correlations, difference_scores

def run_rsa_iterations(epochs, maintained_events, n_iterations):
    """Run RSA analysis multiple times for statistical robustness"""
    
    write_log(f"\nRunning RSA analysis with {n_iterations} iterations...")
    
    # Storage for results across iterations
    results = {
        'voice': {'pre_ping': [], 'post_ping': []},
        'location': {'pre_ping': [], 'post_ping': []}
    }
    
    for iteration in range(n_iterations):
        if iteration % 20 == 0:
            write_log(f"  Iteration {iteration + 1}/{n_iterations}")
        
        # Set random seed for this iteration
        np.random.seed(iteration)
        
        # Process each time window
        for window_name, window_params in TIME_WINDOWS.items():
            
            # Crop epochs to time window
            window_epochs = epochs.copy().crop(tmin=window_params['tmin'], 
                                              tmax=window_params['tmax'])
            epochs_data = window_epochs.get_data()
            
            # Test both features
            for feature_type in ['voice', 'location']:
                
                # Run RSA
                same_corrs, diff_corrs, diff_scores = feature_level_rsa(
                    epochs_data, maintained_events, feature_type, 
                    window_params['name'], window_params
                )
                
                if diff_scores is not None:
                    # Store mean difference score for this iteration
                    results[feature_type][window_name].append(np.mean(diff_scores))
                else:
                    # Store NaN if analysis failed
                    results[feature_type][window_name].append(np.nan)
    
    return results

def statistical_analysis(results):
    """Perform statistical analysis on RSA results"""
    
    write_log(f"\n=== Statistical Analysis ===")
    
    stats_results = {}
    
    for feature_type in ['voice', 'location']:
        stats_results[feature_type] = {}
        
        write_log(f"\n{feature_type.upper()} RSA:")
        
        for window_name in ['pre_ping', 'post_ping']:
            window_data = TIME_WINDOWS[window_name]
            scores = np.array(results[feature_type][window_name])
            
            # Remove NaN values
            valid_scores = scores[~np.isnan(scores)]
            
            if len(valid_scores) == 0:
                write_log(f"  {window_data['name']}: No valid data")
                continue
            
            # One-sample t-test against zero
            t_stat, p_value = ttest_1samp(valid_scores, 0)
            
            # Effect size (Cohen's d)
            effect_size = np.mean(valid_scores) / np.std(valid_scores, ddof=1)
            
            # Confidence interval (95%)
            se = np.std(valid_scores, ddof=1) / np.sqrt(len(valid_scores))
            ci_lower = np.mean(valid_scores) - 1.96 * se
            ci_upper = np.mean(valid_scores) + 1.96 * se
            
            # Store results
            stats_results[feature_type][window_name] = {
                'mean_difference': float(np.mean(valid_scores)),
                'std_difference': float(np.std(valid_scores, ddof=1)),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'effect_size': float(effect_size),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'n_iterations': int(len(valid_scores))
            }
            
            # Log results
            write_log(f"  {window_data['name']}:")
            write_log(f"    Mean difference: {np.mean(valid_scores):.4f}")
            write_log(f"    t({len(valid_scores)-1}) = {t_stat:.3f}, p = {p_value:.4f}")
            write_log(f"    Effect size (d): {effect_size:.3f}")
            write_log(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # Significance indicator
            if p_value < 0.001:
                sig_str = "***"
            elif p_value < 0.01:
                sig_str = "**"
            elif p_value < 0.05:
                sig_str = "*"
            else:
                sig_str = "n.s."
            write_log(f"    Significance: {sig_str}")
    
    return stats_results

def create_summary_plot(results, stats_results):
    """Create summary plot of RSA results"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Subject {args.subject} - Feature-Level RSA Results', fontsize=14)
    
    # Colors for different features
    colors = {'voice': '#2E86AB', 'location': '#A23B72'}
    
    for idx, feature_type in enumerate(['voice', 'location']):
        ax = axes[idx]
        
        window_names = ['pre_ping', 'post_ping']
        x_pos = np.arange(len(window_names))
        
        means = []
        cis = []
        
        for window_name in window_names:
            if window_name in stats_results[feature_type]:
                stats = stats_results[feature_type][window_name]
                means.append(stats['mean_difference'])
                ci_width = stats['ci_upper'] - stats['mean_difference']
                cis.append(ci_width)
            else:
                means.append(0)
                cis.append(0)
        
        # Bar plot
        bars = ax.bar(x_pos, means, yerr=cis, capsize=5, 
                     color=colors[feature_type], alpha=0.7, 
                     edgecolor='black', linewidth=1)
        
        # Add significance stars
        for i, window_name in enumerate(window_names):
            if window_name in stats_results[feature_type]:
                stats = stats_results[feature_type][window_name]
                p_val = stats['p_value']
                
                if p_val < 0.001:
                    sig_str = "***"
                elif p_val < 0.01:
                    sig_str = "**"
                elif p_val < 0.05:
                    sig_str = "*"
                else:
                    sig_str = ""
                
                if sig_str:
                    y_pos = means[i] + cis[i] + 0.01
                    ax.text(i, y_pos, sig_str, ha='center', va='bottom', 
                           fontsize=12, fontweight='bold')
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Pre-ping\n(2.0-3.5s)', 'Post-ping\n(3.5-4.5s)'])
        ax.set_ylabel('Same-Different Correlation\n(Fisher-transformed)')
        ax.set_title(f'{feature_type.title()} RSA')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits
        max_y = max([abs(m) + c for m, c in zip(means, cis)]) if any(means) else 0.1
        ax.set_ylim(-max_y * 1.2, max_y * 1.2)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/rsa_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    write_log(f"\nSummary plot saved to {OUTPUT_DIR}/rsa_summary.png")

def main():
    """Main processing function"""
    
    # Load metadata
    meta_info = pd.read_excel(META_FILE)
    
    # Load subject data
    epochs, maintained_events = load_subject_data(args.subject, meta_info)
    
    if epochs is None:
        write_log("Failed to load data. Exiting.")
        sys.exit(1)
    
    # Run RSA analysis
    rsa_results = run_rsa_iterations(epochs, maintained_events, args.n_iterations)
    
    # Statistical analysis
    stats_results = statistical_analysis(rsa_results)
    
    # Create summary plot
    create_summary_plot(rsa_results, stats_results)
    
    # Save detailed results
    detailed_results = {
        'subject': args.subject,
        'n_iterations': args.n_iterations,
        'time_windows': TIME_WINDOWS,
        'processing_time': str(datetime.now()),
        'method': 'feature_level_rsa_leave_one_out',
        'raw_results': rsa_results,
        'statistical_results': stats_results
    }
    
    # Save raw iteration data
    for feature_type in ['voice', 'location']:
        for window_name in ['pre_ping', 'post_ping']:
            scores = np.array(rsa_results[feature_type][window_name])
            np.save(f'{OUTPUT_DIR}/{feature_type}_{window_name}_difference_scores.npy', scores)
    
    # Save summary
    with open(f'{OUTPUT_DIR}/rsa_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Final summary
    write_log(f"\n=== FINAL SUMMARY ===")
    write_log(f"Subject {args.subject} RSA analysis completed")
    write_log(f"Results saved to: {OUTPUT_DIR}")
    
    # Key findings
    for feature_type in ['voice', 'location']:
        write_log(f"\n{feature_type.upper()} RSA:")
        for window_name in ['pre_ping', 'post_ping']:
            if window_name in stats_results[feature_type]:
                stats = stats_results[feature_type][window_name]
                sig = "SIGNIFICANT" if stats['p_value'] < 0.05 else "NOT SIGNIFICANT"
                write_log(f"  {TIME_WINDOWS[window_name]['name']}: {sig} (p = {stats['p_value']:.4f})")
    
    write_log(f"\nProcessing completed at: {datetime.now()}")
    
    print(f"Subject {args.subject} RSA analysis completed successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()