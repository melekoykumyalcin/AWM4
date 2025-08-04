#!/usr/bin/env python
"""
Group-Level Analysis Script for Mahalanobis Decoding and RSA Results
Collects individual subject results and creates group statistics

Usage: python group_analysis.py --subjects 1 2 3 4 5 6 7 8 9 --output_dir /path/to/group_results
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
import json
from datetime import datetime
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser(description='Run group-level analysis')
parser.add_argument('--subjects', nargs='+', type=int, required=True, help='List of subject IDs')
parser.add_argument('--base_dir', type=str, default='/mnt/hpc/projects/awm4/AWM4_data/processed/mahalanobis_rsa_analysis/', 
                    help='Base directory with individual results')
parser.add_argument('--output_dir', type=str, default='/mnt/hpc/projects/awm4/AWM4_data/processed/mahalanobis_rsa_analysis/group_results/', 
                    help='Output directory for group results')
parser.add_argument('--tmin', type=float, default=2.0, help='Start time')
parser.add_argument('--tmax', type=float, default=4.5, help='End time')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Important time markers
PING_TIME = 3.5
CUE_TIME = 2.0

# Initialize log
LOG_FILE = os.path.join(args.output_dir, 'group_analysis_log.txt')

def write_log(message):
    """Write to log file"""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{datetime.now()}: {message}\n")
    print(message)

write_log(f"Starting group analysis for subjects: {args.subjects}")

class GroupAnalyzer:
    """Class to handle group-level analyses"""
    
    def __init__(self, subjects, base_dir):
        self.subjects = subjects
        self.base_dir = base_dir
        self.valid_subjects = []
        self.group_data = {
            'speaker_decoding': {},
            'location_decoding': {},
            'speaker_rsa': {},
            'location_rsa': {}
        }
        
    def load_subject_data(self, subject):
        """Load all data for a single subject"""
        subject_dir = os.path.join(self.base_dir, f'subject_{subject}')
        
        if not os.path.exists(subject_dir):
            write_log(f"Warning: No data found for subject {subject}")
            return None
            
        try:
            data = {}
            
            # Load decoding results
            data['speaker_accuracies'] = np.load(os.path.join(subject_dir, 'speaker_decoding_accuracies.npy'))
            data['speaker_timepoints'] = np.load(os.path.join(subject_dir, 'speaker_decoding_timepoints.npy'))
            data['location_accuracies'] = np.load(os.path.join(subject_dir, 'location_decoding_accuracies.npy'))
            data['location_timepoints'] = np.load(os.path.join(subject_dir, 'location_decoding_timepoints.npy'))
            
            # Load RSA results
            data['speaker_rsa'] = np.load(os.path.join(subject_dir, 'speaker_rsa_results.npy'), 
                                         allow_pickle=True).item()
            data['location_rsa'] = np.load(os.path.join(subject_dir, 'location_rsa_results.npy'), 
                                          allow_pickle=True).item()
            
            write_log(f"Successfully loaded data for subject {subject}")
            return data
            
        except Exception as e:
            write_log(f"Error loading data for subject {subject}: {str(e)}")
            return None
    
    def align_timepoints(self, all_timepoints):
        """Create common timepoints and interpolate data"""
        # Find common time range
        min_time = max([tp[0] for tp in all_timepoints])
        max_time = min([tp[-1] for tp in all_timepoints])
        
        # Create common timepoints (10ms resolution)
        common_timepoints = np.arange(min_time, max_time, 0.01)
        
        return common_timepoints
    
    def collect_all_data(self):
        """Collect data from all subjects"""
        write_log("\nCollecting data from all subjects...")
        
        # Storage for aligned data
        speaker_accs_aligned = []
        location_accs_aligned = []
        speaker_timepoints_all = []
        location_timepoints_all = []
        
        # RSA data storage
        speaker_rsa_betas = {
            'speaker_harmonic': [],
            'speaker_categorical': []
        }
        location_rsa_betas = {
            'location_magnitude': [],
            'location_hemispheric': []
        }
        rsa_r_squared = {
            'speaker': [],
            'location': []
        }
        
        # Load each subject
        for subject in self.subjects:
            data = self.load_subject_data(subject)
            
            if data is None:
                continue
                
            self.valid_subjects.append(subject)
            
            # Collect timepoints
            speaker_timepoints_all.append(data['speaker_timepoints'])
            location_timepoints_all.append(data['location_timepoints'])
            
            # Store raw data for alignment
            speaker_accs_aligned.append((data['speaker_timepoints'], data['speaker_accuracies']))
            location_accs_aligned.append((data['location_timepoints'], data['location_accuracies']))
            
            # Collect RSA data
            if 'model_fits' in data['speaker_rsa']:
                for model_name in speaker_rsa_betas.keys():
                    if model_name in data['speaker_rsa']['model_fits']:
                        betas = [fit['beta'] for fit in data['speaker_rsa']['model_fits'][model_name]]
                        speaker_rsa_betas[model_name].append((data['speaker_rsa']['timepoints'], betas))
                
                if 'r_squared' in data['speaker_rsa']:
                    rsa_r_squared['speaker'].append((data['speaker_rsa']['timepoints'], 
                                                   data['speaker_rsa']['r_squared']))
            
            if 'model_fits' in data['location_rsa']:
                for model_name in location_rsa_betas.keys():
                    if model_name in data['location_rsa']['model_fits']:
                        betas = [fit['beta'] for fit in data['location_rsa']['model_fits'][model_name]]
                        location_rsa_betas[model_name].append((data['location_rsa']['timepoints'], betas))
                
                if 'r_squared' in data['location_rsa']:
                    rsa_r_squared['location'].append((data['location_rsa']['timepoints'], 
                                                    data['location_rsa']['r_squared']))
        
        write_log(f"Successfully loaded {len(self.valid_subjects)} subjects: {self.valid_subjects}")
        
        # Align all data to common timepoints
        if len(self.valid_subjects) > 0:
            # Align decoding data
            common_time_speaker = self.align_timepoints(speaker_timepoints_all)
            common_time_location = self.align_timepoints(location_timepoints_all)
            
            # Interpolate speaker decoding
            aligned_speaker_accs = []
            for timepoints, accuracies in speaker_accs_aligned:
                f = interp1d(timepoints, accuracies, kind='linear', fill_value='extrapolate')
                aligned_speaker_accs.append(f(common_time_speaker))
            
            # Interpolate location decoding
            aligned_location_accs = []
            for timepoints, accuracies in location_accs_aligned:
                f = interp1d(timepoints, accuracies, kind='linear', fill_value='extrapolate')
                aligned_location_accs.append(f(common_time_location))
            
            # Store aligned data
            self.group_data['speaker_decoding'] = {
                'timepoints': common_time_speaker,
                'accuracies': np.array(aligned_speaker_accs),  # Shape: (n_subjects, n_timepoints)
                'mean': np.mean(aligned_speaker_accs, axis=0),
                'sem': stats.sem(aligned_speaker_accs, axis=0),
                'std': np.std(aligned_speaker_accs, axis=0)
            }
            
            self.group_data['location_decoding'] = {
                'timepoints': common_time_location,
                'accuracies': np.array(aligned_location_accs),
                'mean': np.mean(aligned_location_accs, axis=0),
                'sem': stats.sem(aligned_location_accs, axis=0),
                'std': np.std(aligned_location_accs, axis=0)
            }
            
            # Align RSA data
            for model_name, subject_data in speaker_rsa_betas.items():
                if len(subject_data) > 0:
                    aligned_betas = []
                    for timepoints, betas in subject_data:
                        f = interp1d(timepoints, betas, kind='linear', fill_value='extrapolate')
                        aligned_betas.append(f(common_time_speaker))
                    
                    self.group_data['speaker_rsa'][model_name] = {
                        'betas': np.array(aligned_betas),
                        'mean': np.mean(aligned_betas, axis=0),
                        'sem': stats.sem(aligned_betas, axis=0)
                    }
            
            # Similar for location RSA
            for model_name, subject_data in location_rsa_betas.items():
                if len(subject_data) > 0:
                    aligned_betas = []
                    for timepoints, betas in subject_data:
                        f = interp1d(timepoints, betas, kind='linear', fill_value='extrapolate')
                        aligned_betas.append(f(common_time_location))
                    
                    self.group_data['location_rsa'][model_name] = {
                        'betas': np.array(aligned_betas),
                        'mean': np.mean(aligned_betas, axis=0),
                        'sem': stats.sem(aligned_betas, axis=0)
                    }
            
            # R-squared
            for feature in ['speaker', 'location']:
                if len(rsa_r_squared[feature]) > 0:
                    aligned_r2 = []
                    common_time = common_time_speaker if feature == 'speaker' else common_time_location
                    for timepoints, r2 in rsa_r_squared[feature]:
                        f = interp1d(timepoints, r2, kind='linear', fill_value='extrapolate')
                        aligned_r2.append(f(common_time))
                    
                    self.group_data[f'{feature}_rsa']['r_squared'] = {
                        'values': np.array(aligned_r2),
                        'mean': np.mean(aligned_r2, axis=0),
                        'sem': stats.sem(aligned_r2, axis=0)
                    }
    
    def statistical_tests(self):
        """Perform statistical tests on group data"""
        write_log("\nPerforming statistical tests...")
        
        results = {}
        
        # Test decoding against chance (0.25)
        for feature in ['speaker', 'location']:
            if f'{feature}_decoding' in self.group_data and 'accuracies' in self.group_data[f'{feature}_decoding']:
                accuracies = self.group_data[f'{feature}_decoding']['accuracies']
                timepoints = self.group_data[f'{feature}_decoding']['timepoints']
                
                # One-sample t-test against chance at each timepoint
                t_stats = []
                p_values = []
                
                for t in range(len(timepoints)):
                    t_stat, p_val = stats.ttest_1samp(accuracies[:, t], 0.25)
                    t_stats.append(t_stat)
                    p_values.append(p_val)
                
                # FDR correction
                from statsmodels.stats.multitest import multipletests
                _, p_fdr, _, _ = multipletests(p_values, method='fdr_bh')
                
                results[f'{feature}_decoding'] = {
                    't_stats': np.array(t_stats),
                    'p_values': np.array(p_values),
                    'p_fdr': p_fdr,
                    'significant_fdr': p_fdr < 0.05
                }
        
        # Peak analysis
        for feature in ['speaker', 'location']:
            if f'{feature}_decoding' in self.group_data:
                mean_acc = self.group_data[f'{feature}_decoding']['mean']
                peak_idx = np.argmax(mean_acc)
                peak_time = self.group_data[f'{feature}_decoding']['timepoints'][peak_idx]
                peak_acc = mean_acc[peak_idx]
                
                # Test peak accuracy across subjects
                peak_accs_subjects = self.group_data[f'{feature}_decoding']['accuracies'][:, peak_idx]
                t_stat, p_val = stats.ttest_1samp(peak_accs_subjects, 0.25)
                
                results[f'{feature}_peak'] = {
                    'time': peak_time,
                    'accuracy': peak_acc,
                    'accuracy_sem': stats.sem(peak_accs_subjects),
                    't_stat': t_stat,
                    'p_value': p_val,
                    'effect_size': (peak_acc - 0.25) / np.std(peak_accs_subjects)
                }
        
        self.stats_results = results
        return results
    
    def plot_group_results(self):
        """Create comprehensive group-level plots"""
        write_log("\nCreating group visualizations...")
        
        # Set up the figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.25)
        
        # 1. Decoding time courses with statistics
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        
        # Speaker decoding
        if 'speaker_decoding' in self.group_data and 'mean' in self.group_data['speaker_decoding']:
            time = self.group_data['speaker_decoding']['timepoints']
            mean = self.group_data['speaker_decoding']['mean']
            sem = self.group_data['speaker_decoding']['sem']
            
            # Plot mean ± SEM
            ax1.plot(time, mean, 'b-', linewidth=2, label='Group mean')
            ax1.fill_between(time, mean - sem, mean + sem, alpha=0.3, color='b')
            
            # Add individual subjects
            for i, subj_acc in enumerate(self.group_data['speaker_decoding']['accuracies']):
                ax1.plot(time, subj_acc, 'b-', alpha=0.1, linewidth=0.5)
            
            # Add significance markers
            if hasattr(self, 'stats_results') and 'speaker_decoding' in self.stats_results:
                sig_times = time[self.stats_results['speaker_decoding']['significant_fdr']]
                ax1.scatter(sig_times, np.ones_like(sig_times) * 0.15, marker='*', 
                           color='red', s=20, label='p<0.05 (FDR)')
            
            ax1.axhline(0.25, color='k', linestyle='--', alpha=0.5, label='Chance')
            ax1.axvline(PING_TIME, color='r', linestyle='--', alpha=0.7, label='Ping')
            ax1.set_ylabel('Decoding Accuracy')
            ax1.set_title(f'Speaker Identity Decoding (N={len(self.valid_subjects)})')
            ax1.set_ylim([0.15, 0.45])
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Location decoding
        if 'location_decoding' in self.group_data and 'mean' in self.group_data['location_decoding']:
            time = self.group_data['location_decoding']['timepoints']
            mean = self.group_data['location_decoding']['mean']
            sem = self.group_data['location_decoding']['sem']
            
            ax2.plot(time, mean, 'r-', linewidth=2, label='Group mean')
            ax2.fill_between(time, mean - sem, mean + sem, alpha=0.3, color='r')
            
            # Add individual subjects
            for i, subj_acc in enumerate(self.group_data['location_decoding']['accuracies']):
                ax2.plot(time, subj_acc, 'r-', alpha=0.1, linewidth=0.5)
            
            # Add significance markers
            if hasattr(self, 'stats_results') and 'location_decoding' in self.stats_results:
                sig_times = time[self.stats_results['location_decoding']['significant_fdr']]
                ax2.scatter(sig_times, np.ones_like(sig_times) * 0.15, marker='*', 
                           color='red', s=20, label='p<0.05 (FDR)')
            
            ax2.axhline(0.25, color='k', linestyle='--', alpha=0.5, label='Chance')
            ax2.axvline(PING_TIME, color='r', linestyle='--', alpha=0.7, label='Ping')
            ax2.set_ylabel('Decoding Accuracy')
            ax2.set_xlabel('Time (s)')
            ax2.set_title(f'Location Decoding (N={len(self.valid_subjects)})')
            ax2.set_ylim([0.15, 0.45])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 2. RSA Model fits
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[2, 1])
        ax5 = fig.add_subplot(gs[2, 2])
        
        # Speaker RSA
        if 'speaker_harmonic' in self.group_data['speaker_rsa']:
            time = self.group_data['speaker_decoding']['timepoints']
            
            for model_name, color in [('speaker_harmonic', 'green'), 
                                     ('speaker_categorical', 'purple')]:
                if model_name in self.group_data['speaker_rsa']:
                    mean = self.group_data['speaker_rsa'][model_name]['mean']
                    sem = self.group_data['speaker_rsa'][model_name]['sem']
                    
                    label = model_name.replace('speaker_', '').capitalize()
                    ax3.plot(time, mean, color=color, linewidth=2, label=label)
                    ax3.fill_between(time, mean - sem, mean + sem, alpha=0.3, color=color)
            
            ax3.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax3.axvline(PING_TIME, color='r', linestyle='--', alpha=0.7)
            ax3.set_ylabel('Beta Weight')
            ax3.set_xlabel('Time (s)')
            ax3.set_title('Speaker Model Fits')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Location RSA
        if 'location_magnitude' in self.group_data['location_rsa']:
            time = self.group_data['location_decoding']['timepoints']
            
            for model_name, color in [('location_magnitude', 'cyan'), 
                                     ('location_hemispheric', 'orange')]:
                if model_name in self.group_data['location_rsa']:
                    mean = self.group_data['location_rsa'][model_name]['mean']
                    sem = self.group_data['location_rsa'][model_name]['sem']
                    
                    label = model_name.replace('location_', '').capitalize()
                    ax4.plot(time, mean, color=color, linewidth=2, label=label)
                    ax4.fill_between(time, mean - sem, mean + sem, alpha=0.3, color=color)
            
            ax4.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax4.axvline(PING_TIME, color='r', linestyle='--', alpha=0.7)
            ax4.set_ylabel('Beta Weight')
            ax4.set_xlabel('Time (s)')
            ax4.set_title('Location Model Fits')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # R-squared comparison
        if 'r_squared' in self.group_data['speaker_rsa'] and 'r_squared' in self.group_data['location_rsa']:
            time_sp = self.group_data['speaker_decoding']['timepoints']
            time_loc = self.group_data['location_decoding']['timepoints']
            
            mean_sp = self.group_data['speaker_rsa']['r_squared']['mean']
            sem_sp = self.group_data['speaker_rsa']['r_squared']['sem']
            
            mean_loc = self.group_data['location_rsa']['r_squared']['mean']
            sem_loc = self.group_data['location_rsa']['r_squared']['sem']
            
            ax5.plot(time_sp, mean_sp, 'b-', linewidth=2, label='Speaker')
            ax5.fill_between(time_sp, mean_sp - sem_sp, mean_sp + sem_sp, alpha=0.3, color='b')
            
            ax5.plot(time_loc, mean_loc, 'r-', linewidth=2, label='Location')
            ax5.fill_between(time_loc, mean_loc - sem_loc, mean_loc + sem_loc, alpha=0.3, color='r')
            
            ax5.axvline(PING_TIME, color='r', linestyle='--', alpha=0.7)
            ax5.set_ylabel('R²')
            ax5.set_xlabel('Time (s)')
            ax5.set_title('Model Variance Explained')
            ax5.set_ylim([0, 0.3])
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 3. Individual subject summary
        ax6 = fig.add_subplot(gs[3, :])
        
        # Create summary data for each subject
        subject_summary = []
        for i, subject in enumerate(self.valid_subjects):
            if i < len(self.group_data['speaker_decoding']['accuracies']):
                speaker_acc = self.group_data['speaker_decoding']['accuracies'][i]
                location_acc = self.group_data['location_decoding']['accuracies'][i]
                
                summary = {
                    'Subject': subject,
                    'Speaker Mean': np.mean(speaker_acc),
                    'Speaker Peak': np.max(speaker_acc),
                    'Location Mean': np.mean(location_acc),
                    'Location Peak': np.max(location_acc)
                }
                subject_summary.append(summary)
        
        if subject_summary:
            df_summary = pd.DataFrame(subject_summary)
            
            # Plot individual subject performance
            x = np.arange(len(df_summary))
            width = 0.35
            
            ax6.bar(x - width/2, df_summary['Speaker Peak'], width, label='Speaker Peak', color='blue', alpha=0.7)
            ax6.bar(x + width/2, df_summary['Location Peak'], width, label='Location Peak', color='red', alpha=0.7)
            
            ax6.axhline(0.25, color='k', linestyle='--', alpha=0.5, label='Chance')
            ax6.set_ylabel('Peak Decoding Accuracy')
            ax6.set_xlabel('Subject')
            ax6.set_title('Individual Subject Peak Performance')
            ax6.set_xticks(x)
            ax6.set_xticklabels(df_summary['Subject'])
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Group-Level Results (N={len(self.valid_subjects)} subjects)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'group_results_comprehensive.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional plot: Time-frequency style plot for all subjects
        self.plot_subject_heatmaps()
    
    def plot_subject_heatmaps(self):
        """Create heatmaps showing all subjects across time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Speaker decoding heatmap
        if 'speaker_decoding' in self.group_data and 'accuracies' in self.group_data['speaker_decoding']:
            data = self.group_data['speaker_decoding']['accuracies']
            time = self.group_data['speaker_decoding']['timepoints']
            
            im1 = ax1.imshow(data, aspect='auto', cmap='RdBu_r', vmin=0.15, vmax=0.35,
                            extent=[time[0], time[-1], len(self.valid_subjects)-0.5, -0.5])
            
            ax1.axvline(PING_TIME, color='green', linestyle='--', linewidth=2)
            ax1.set_ylabel('Subject')
            ax1.set_title('Speaker Decoding Accuracy Across Subjects')
            ax1.set_yticks(range(len(self.valid_subjects)))
            ax1.set_yticklabels(self.valid_subjects)
            
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('Accuracy')
        
        # Location decoding heatmap
        if 'location_decoding' in self.group_data and 'accuracies' in self.group_data['location_decoding']:
            data = self.group_data['location_decoding']['accuracies']
            time = self.group_data['location_decoding']['timepoints']
            
            im2 = ax2.imshow(data, aspect='auto', cmap='RdBu_r', vmin=0.15, vmax=0.35,
                            extent=[time[0], time[-1], len(self.valid_subjects)-0.5, -0.5])
            
            ax2.axvline(PING_TIME, color='green', linestyle='--', linewidth=2)
            ax2.set_ylabel('Subject')
            ax2.set_xlabel('Time (s)')
            ax2.set_title('Location Decoding Accuracy Across Subjects')
            ax2.set_yticks(range(len(self.valid_subjects)))
            ax2.set_yticklabels(self.valid_subjects)
            
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'subject_heatmaps.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_group_data(self):
        """Save all group data"""
        write_log("\nSaving group data...")
        
        # Save numpy arrays
        for feature in ['speaker', 'location']:
            if f'{feature}_decoding' in self.group_data:
                np.save(os.path.join(args.output_dir, f'group_{feature}_accuracies.npy'),
                       self.group_data[f'{feature}_decoding']['accuracies'])
                np.save(os.path.join(args.output_dir, f'group_{feature}_timepoints.npy'),
                       self.group_data[f'{feature}_decoding']['timepoints'])
                np.save(os.path.join(args.output_dir, f'group_{feature}_mean.npy'),
                       self.group_data[f'{feature}_decoding']['mean'])
                np.save(os.path.join(args.output_dir, f'group_{feature}_sem.npy'),
                       self.group_data[f'{feature}_decoding']['sem'])
        
        # Save summary statistics
        summary = {
            'n_subjects': len(self.valid_subjects),
            'valid_subjects': self.valid_subjects,
            'analysis_date': str(datetime.now()),
            'results': {}
        }
        
        # Add peak statistics
        if hasattr(self, 'stats_results'):
            for feature in ['speaker', 'location']:
                if f'{feature}_peak' in self.stats_results:
                    peak_data = self.stats_results[f'{feature}_peak']
                    summary['results'][f'{feature}_peak'] = {
                        'time': float(peak_data['time']),
                        'accuracy': float(peak_data['accuracy']),
                        'accuracy_sem': float(peak_data['accuracy_sem']),
                        'p_value': float(peak_data['p_value']),
                        'effect_size': float(peak_data['effect_size'])
                    }
        
        # Add time windows of significance
        for feature in ['speaker', 'location']:
            if hasattr(self, 'stats_results') and f'{feature}_decoding' in self.stats_results:
                sig_mask = self.stats_results[f'{feature}_decoding']['significant_fdr']
                if np.any(sig_mask):
                    time = self.group_data[f'{feature}_decoding']['timepoints']
                    sig_start = time[sig_mask][0]
                    sig_end = time[sig_mask][-1]
                    summary['results'][f'{feature}_significant_window'] = {
                        'start': float(sig_start),
                        'end': float(sig_end),
                        'duration': float(sig_end - sig_start)
                    }
        
        with open(os.path.join(args.output_dir, 'group_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        write_log("Group data saved successfully")

def main():
    """Main execution function"""
    
    # Initialize analyzer
    analyzer = GroupAnalyzer(args.subjects, args.base_dir)
    
    # Collect all data
    analyzer.collect_all_data()
    
    if len(analyzer.valid_subjects) == 0:
        write_log("No valid subjects found. Exiting.")
        return
    
    # Run statistical tests
    analyzer.statistical_tests()
    
    # Create visualizations
    analyzer.plot_group_results()
    
    # Save results
    analyzer.save_group_data()
    
    # Print summary
    write_log("\n" + "="*60)
    write_log("GROUP ANALYSIS SUMMARY")
    write_log("="*60)
    write_log(f"Analyzed {len(analyzer.valid_subjects)} subjects: {analyzer.valid_subjects}")
    
    if hasattr(analyzer, 'stats_results'):
        for feature in ['speaker', 'location']:
            if f'{feature}_peak' in analyzer.stats_results:
                peak = analyzer.stats_results[f'{feature}_peak']
                write_log(f"\n{feature.capitalize()} Decoding:")
                write_log(f"  Peak accuracy: {peak['accuracy']:.3f} ± {peak['accuracy_sem']:.3f}")
                write_log(f"  Peak time: {peak['time']:.3f}s")
                write_log(f"  p-value: {peak['p_value']:.6f}")
                write_log(f"  Effect size (Cohen's d): {peak['effect_size']:.3f}")
    
    write_log(f"\nResults saved to: {args.output_dir}")
    write_log(f"Analysis completed at: {datetime.now()}")

if __name__ == "__main__":
    main()