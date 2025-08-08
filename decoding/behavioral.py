#!/usr/bin/env python
"""
AWM4 Behavioral Analysis by Musician Level
Replicates MATLAB analysis but separately for each musician level (1, 2, 3)

This script performs the same 2x2 factorial analysis (Speaker Match/Nonmatch × Location Match/Nonmatch)
that the MATLAB script does, but groups subjects by their musician level and analyzes each group separately.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================
HOME_DIR = '/mnt/hpc/projects/awm4/'
META_FILE = '/mnt/hpc/projects/awm4/MEGNotes.xlsx'
BEHAVIORAL_DIR = '/mnt/hpc/projects/awm4/AWM4_data/behavioral/'
OUTPUT_DIR = BEHAVIORAL_DIR + 'behavioral_analysis_musician_levels/'
# ============================================================================

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Musician level colors (matching your MEG analysis style)
MUSICIAN_COLORS = {
    1: '#ff6b6b',  # Light red - Level 1 (lowest musical training)
    2: '#4ecdc4',  # Teal - Level 2 (intermediate)
    3: '#45b7d1',  # Blue - Level 3 (highest musical training)
}

def load_musician_metadata():
    """Load VP -> musician level mapping via SubjectCode"""
    print(f"Loading musician metadata from: {META_FILE}")
    
    try:
        meta_df = pd.read_excel(META_FILE)
        print(f"Metadata shape: {meta_df.shape}")
        
        # Get unique Subject-SubjectCode-Musician combinations
        unique_combinations = meta_df[['Subject', 'SubjectCode', 'Musician']].drop_duplicates()
        print(f"Unique combinations: {len(unique_combinations)}")
        
        # Create VP -> Musician mapping
        vp_to_musician = {}
        
        for _, row in unique_combinations.iterrows():
            subject_code = row['SubjectCode']
            musician_level = row['Musician']
            
            if pd.notna(subject_code) and pd.notna(musician_level):
                if isinstance(subject_code, str) and subject_code.startswith('S'):
                    vp_number = int(subject_code[1:])  # S51 -> 51
                    musician_level_int = int(float(musician_level))
                    
                    if musician_level_int in [1, 2, 3]:
                        vp_to_musician[vp_number] = musician_level_int
        
        print(f"Mapped {len(vp_to_musician)} VPs to musician levels")
        level_counts = {level: sum(1 for v in vp_to_musician.values() if v == level) for level in [1, 2, 3]}
        print(f"Distribution: {level_counts}")
        
        return vp_to_musician
        
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return {}

def load_behavioral_file(file_path):
    """Load and parse a single behavioral file (replicating MATLAB importdata)"""
    
    # Try different separators
    separators = ['\t', ' ', '  ', ',']
    
    for sep in separators:
        try:
            df = pd.read_csv(file_path, sep=sep, header=None)
            
            # Check if we got reasonable number of columns (should be ~16)
            if df.shape[1] >= 14:
                # Clean data like MATLAB: remove rows where first column is 0
                df = df[df.iloc[:, 0] != 0]
                # Remove rows with NaN
                df = df.dropna()
                
                if len(df) > 0:
                    return df.values  # Return as numpy array like MATLAB
                    
        except:
            continue
    
    return None

def extract_accuracy_rt_by_conditions(data_array):
    """
    Extract accuracy and RT for 2x2 conditions (Speaker Match/Nonmatch × Location Match/Nonmatch)
    Replicating the MATLAB analysis exactly
    
    Column indices (0-based, converted from MATLAB 1-based):
    8: SpeakerMatch (9 in MATLAB)
    10: LocMatch (11 in MATLAB) 
    13: Correct (14 in MATLAB)
    14: RT (15 in MATLAB)
    """
    
    if data_array is None or data_array.shape[1] < 15:
        return None
    
    # Initialize results dictionary
    results = {
        'accuracy': np.full((2, 2), np.nan),  # [SpeakerMatch, LocMatch]
        'rt': np.full((2, 2), np.nan),
        'n_trials': np.full((2, 2), 0, dtype=int)
    }
    
    # Extract relevant columns
    speaker_match = data_array[:, 8].astype(int)    # SpeakerMatch: 1=match, 0=nonmatch
    loc_match = data_array[:, 10].astype(int)       # LocMatch: 1=match, 0=nonmatch  
    correct = data_array[:, 13].astype(int)         # Correct: 1=correct, 0=incorrect
    rt = data_array[:, 14]                          # RT in seconds
    
    # Calculate accuracy and RT for each 2x2 condition
    for speaker_mnm in [1, 0]:  # 1=match, 0=nonmatch
        for loc_mnm in [1, 0]:  # 1=match, 0=nonmatch
            
            # Find trials for this condition
            condition_trials = (speaker_match == speaker_mnm) & (loc_match == loc_mnm)
            
            if np.sum(condition_trials) > 0:
                # Accuracy: mean of correct responses * 100 (like MATLAB)
                accuracy = np.mean(correct[condition_trials]) * 100
                results['accuracy'][speaker_mnm, loc_mnm] = accuracy
                
                # RT: mean RT for correct responses only (like MATLAB)
                correct_trials = condition_trials & (correct == 1)
                if np.sum(correct_trials) > 0:
                    rt_mean = np.mean(rt[correct_trials])
                    results['rt'][speaker_mnm, loc_mnm] = rt_mean
                
                results['n_trials'][speaker_mnm, loc_mnm] = np.sum(condition_trials)
    
    return results

def load_all_behavioral_data(musician_mapping):
    """Load all behavioral files and group by musician level"""
    
    print(f"Loading behavioral data from: {BEHAVIORAL_DIR}")
    
    # Find VP files
    vp_files = glob.glob(os.path.join(BEHAVIORAL_DIR, 'Output_AWM4_Exp1_MEG_VP*.txt'))
    print(f"Found {len(vp_files)} VP files")
    
    # Group data by musician level
    musician_data = {1: {}, 2: {}, 3: {}}
    
    for file_path in vp_files:
        # Extract VP number
        filename = os.path.basename(file_path)
        import re
        vp_match = re.search(r'VP(\d+)\.txt', filename)
        
        if vp_match:
            vp_number = int(vp_match.group(1))
            
            # Check if we have musician level for this VP
            if vp_number in musician_mapping:
                musician_level = musician_mapping[vp_number]
                
                # Load behavioral data
                data_array = load_behavioral_file(file_path)
                
                if data_array is not None:
                    # Extract 2x2 condition results
                    conditions_results = extract_accuracy_rt_by_conditions(data_array)
                    
                    if conditions_results is not None:
                        musician_data[musician_level][vp_number] = conditions_results
                        print(f"  ✓ VP{vp_number} -> Musician Level {musician_level} ({len(data_array)} trials)")
                    else:
                        print(f"  ❌ VP{vp_number} -> Could not extract conditions")
                else:
                    print(f"  ❌ VP{vp_number} -> Could not load data")
            else:
                print(f"  ? VP{vp_number} -> No musician level mapping")
    
    # Summary
    print(f"\nData loaded by musician level:")
    for level in [1, 2, 3]:
        n_subjects = len(musician_data[level])
        print(f"  Level {level}: {n_subjects} subjects")
    
    return musician_data

def compute_group_statistics_for_level(level_data, level_name):
    """Compute group statistics for one musician level (replicating MATLAB analysis)"""
    
    if not level_data:
        print(f"No data for {level_name}")
        return None
    
    n_subjects = len(level_data)
    print(f"\n" + "="*50)
    print(f"MUSICIAN LEVEL {level_name} ANALYSIS (N={n_subjects})")
    print("="*50)
    
    # Initialize arrays to store data (like MATLAB)
    accuracy_array = np.full((n_subjects, 2, 2), np.nan)  # [subject, speaker_match, loc_match]
    rt_array = np.full((n_subjects, 2, 2), np.nan)
    
    # Fill arrays with subject data
    for i, (vp_num, vp_data) in enumerate(level_data.items()):
        accuracy_array[i, :, :] = vp_data['accuracy']
        rt_array[i, :, :] = vp_data['rt']
    
    # Compute means and stats for each condition
    print(f"\nACCURACY RESULTS:")
    print(f"Condition                          | Mean±SD    | SEM   | t-stat | p-value")
    print(f"-----------------------------------|------------|-------|--------|--------")
    
    condition_names = [
        "Speaker Match, Location Match    ",
        "Speaker Match, Location Nonmatch ",
        "Speaker Nonmatch, Location Match ",
        "Speaker Nonmatch, Location Nonmatch"
    ]
    
    accuracy_results = {}
    rt_results = {}
    
    for i, (speaker_idx, loc_idx) in enumerate([(1,1), (1,0), (0,1), (0,0)]):
        condition_name = condition_names[i]
        
        # Accuracy analysis
        acc_data = accuracy_array[:, speaker_idx, loc_idx]
        valid_acc = acc_data[~np.isnan(acc_data)]
        
        if len(valid_acc) > 1:
            acc_mean = np.mean(valid_acc)
            acc_std = np.std(valid_acc, ddof=1)
            acc_sem = acc_std / np.sqrt(len(valid_acc))
            
            # t-test against chance (50%)
            t_stat, p_val = stats.ttest_1samp(valid_acc, 50)
            
            accuracy_results[f'speaker_{speaker_idx}_loc_{loc_idx}'] = {
                'data': valid_acc,
                'mean': acc_mean,
                'std': acc_std,
                'sem': acc_sem,
                't_stat': t_stat,
                'p_value': p_val,
                'n': len(valid_acc)
            }
            
            print(f"{condition_name}| {acc_mean:5.1f}±{acc_std:4.1f} | {acc_sem:5.1f} | {t_stat:6.2f} | {p_val:6.3f}")
        
        # RT analysis  
        rt_data = rt_array[:, speaker_idx, loc_idx]
        valid_rt = rt_data[~np.isnan(rt_data)]
        
        if len(valid_rt) > 1:
            rt_mean = np.mean(valid_rt)
            rt_std = np.std(valid_rt, ddof=1)
            rt_sem = rt_std / np.sqrt(len(valid_rt))
            
            rt_results[f'speaker_{speaker_idx}_loc_{loc_idx}'] = {
                'data': valid_rt,
                'mean': rt_mean,
                'std': rt_std,
                'sem': rt_sem,
                'n': len(valid_rt)
            }
    
    # Perform 2x2 ANOVA (like MATLAB)
    print(f"\n2×2 ANOVA RESULTS (Speaker Match/Nonmatch × Location Match/Nonmatch):")
    print(f"Factor                    | F-stat | p-value | η²")
    print(f"--------------------------|--------|---------|--------")
    
    anova_results = {}
    
    try:
        # Reshape data for ANOVA
        acc_for_anova = []
        speaker_factor = []
        location_factor = []
        subject_factor = []
        
        for subj_idx in range(n_subjects):
            for speaker_idx in [0, 1]:
                for loc_idx in [0, 1]:
                    acc_val = accuracy_array[subj_idx, speaker_idx, loc_idx]
                    if not np.isnan(acc_val):
                        acc_for_anova.append(acc_val)
                        speaker_factor.append(speaker_idx)
                        location_factor.append(loc_idx)
                        subject_factor.append(subj_idx)
        
        if len(acc_for_anova) > 10:  # Need reasonable sample size
            # Create DataFrame for ANOVA
            anova_df = pd.DataFrame({
                'accuracy': acc_for_anova,
                'speaker': speaker_factor,
                'location': location_factor,
                'subject': subject_factor
            })
            
            # Simple ANOVA approximation (not full repeated measures, but gives idea)
            from scipy.stats import f_oneway
            
            # Main effect of speaker
            speaker_0 = anova_df[anova_df['speaker'] == 0]['accuracy']
            speaker_1 = anova_df[anova_df['speaker'] == 1]['accuracy']
            f_speaker, p_speaker = f_oneway(speaker_0, speaker_1)
            
            # Main effect of location  
            location_0 = anova_df[anova_df['location'] == 0]['accuracy']
            location_1 = anova_df[anova_df['location'] == 1]['accuracy']
            f_location, p_location = f_oneway(location_0, location_1)
            
            print(f"Speaker main effect       | {f_speaker:6.2f} | {p_speaker:7.3f} | -")
            print(f"Location main effect      | {f_location:6.2f} | {p_location:7.3f} | -")
            
            anova_results = {
                'speaker_f': f_speaker,
                'speaker_p': p_speaker,
                'location_f': f_location,
                'location_p': p_location
            }
    
    except Exception as e:
        print(f"ANOVA failed: {e}")
    
    # Paired t-tests (like MATLAB)
    print(f"\nPAIRED T-TESTS:")
    print(f"Comparison                                    | t-stat | p-value")
    print(f"----------------------------------------------|--------|--------")
    
    # Speaker match: Location match vs nonmatch
    if ('speaker_1_loc_1' in accuracy_results and 'speaker_1_loc_0' in accuracy_results):
        data1 = accuracy_results['speaker_1_loc_1']['data']
        data2 = accuracy_results['speaker_1_loc_0']['data']
        if len(data1) > 1 and len(data2) > 1:
            t_stat, p_val = stats.ttest_rel(data1, data2)
            print(f"Speaker Match: Location Match vs Nonmatch    | {t_stat:6.2f} | {p_val:7.3f}")
    
    # Speaker nonmatch: Location match vs nonmatch
    if ('speaker_0_loc_1' in accuracy_results and 'speaker_0_loc_0' in accuracy_results):
        data1 = accuracy_results['speaker_0_loc_1']['data']
        data2 = accuracy_results['speaker_0_loc_0']['data']
        if len(data1) > 1 and len(data2) > 1:
            t_stat, p_val = stats.ttest_rel(data1, data2)
            print(f"Speaker Nonmatch: Location Match vs Nonmatch | {t_stat:6.2f} | {p_val:7.3f}")
    
    return {
        'level': level_name,
        'n_subjects': n_subjects,
        'accuracy_results': accuracy_results,
        'rt_results': rt_results,
        'anova_results': anova_results,
        'accuracy_array': accuracy_array,
        'rt_array': rt_array
    }

def create_musician_level_plots(all_level_stats):
    """Create plots showing results for each musician level"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('AWM4 Behavioral Results by Musician Level\n(Speaker Match/Nonmatch × Location Match/Nonmatch)', 
                 fontsize=16, fontweight='bold')
    
    # Plot accuracy for each musician level
    for i, level in enumerate([1, 2, 3]):
        if level in all_level_stats:
            ax = axes[0, i]
            stats_data = all_level_stats[level]
            
            # Extract means and SEMs for 2x2 plot
            acc_results = stats_data['accuracy_results']
            
            means = []
            sems = []
            labels = []
            colors = []
            
            conditions = [
                ('speaker_1_loc_1', 'Speaker Match\nLocation Match', '#2E8B57'),
                ('speaker_1_loc_0', 'Speaker Match\nLocation Nonmatch', '#4682B4'),
                ('speaker_0_loc_1', 'Speaker Nonmatch\nLocation Match', '#CD853F'),
                ('speaker_0_loc_0', 'Speaker Nonmatch\nLocation Nonmatch', '#DC143C')
            ]
            
            for condition_key, label, color in conditions:
                if condition_key in acc_results:
                    means.append(acc_results[condition_key]['mean'])
                    sems.append(acc_results[condition_key]['sem'])
                    labels.append(label)
                    colors.append(color)
            
            if means:
                x_pos = np.arange(len(means))
                bars = ax.bar(x_pos, means, yerr=sems, capsize=5, 
                             color=colors, alpha=0.7, edgecolor='black')
                
                # Add individual subject dots
                for j, (condition_key, _, _) in enumerate(conditions):
                    if condition_key in acc_results:
                        data = acc_results[condition_key]['data']
                        if len(data) > 0:
                            jitter = np.random.normal(0, 0.05, len(data))
                            ax.scatter([j] * len(data) + jitter, data, 
                                     color='black', alpha=0.6, s=20, zorder=3)
                
                ax.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Chance')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel('Accuracy (%)')
                ax.set_title(f'Level {level} (N={stats_data["n_subjects"]})')
                ax.set_ylim(30, 100)
                ax.grid(True, alpha=0.3)
    
    # Plot RT for each musician level
    for i, level in enumerate([1, 2, 3]):
        if level in all_level_stats:
            ax = axes[1, i]
            stats_data = all_level_stats[level]
            
            rt_results = stats_data['rt_results']
            
            means = []
            sems = []
            
            for condition_key, _, color in conditions:
                if condition_key in rt_results:
                    means.append(rt_results[condition_key]['mean'])
                    sems.append(rt_results[condition_key]['sem'])
                else:
                    means.append(0)
                    sems.append(0)
            
            if any(m > 0 for m in means):
                x_pos = np.arange(len(means))
                bars = ax.bar(x_pos, means, yerr=sems, capsize=5,
                             color=colors, alpha=0.7, edgecolor='black')
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel('Reaction Time (s)')
                ax.set_title(f'Level {level} RT')
                ax.set_ylim(0, 1.0)
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/musician_level_behavioral_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/musician_level_behavioral_analysis.pdf', bbox_inches='tight')
    plt.show()

def save_results_to_csv(all_level_stats):
    """Save detailed results to CSV files"""
    
    # Summary table
    summary_data = []
    
    for level, stats_data in all_level_stats.items():
        acc_results = stats_data['accuracy_results']
        
        for condition_key, condition_stats in acc_results.items():
            summary_data.append({
                'Musician_Level': level,
                'Condition': condition_key,
                'N_Subjects': condition_stats['n'],
                'Mean_Accuracy': condition_stats['mean'],
                'SD_Accuracy': condition_stats['std'],
                'SEM_Accuracy': condition_stats['sem'],
                'T_Statistic': condition_stats['t_stat'],
                'P_Value': condition_stats['p_value']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{OUTPUT_DIR}/behavioral_analysis_summary.csv', index=False)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"- Summary: behavioral_analysis_summary.csv")
    print(f"- Plots: musician_level_behavioral_analysis.png/pdf")

def main():
    """Main analysis pipeline"""
    print("AWM4 BEHAVIORAL ANALYSIS BY MUSICIAN LEVEL")
    print("="*50)
    print("Replicating MATLAB analysis separately for each musician level")
    
    # Load musician mapping
    musician_mapping = load_musician_metadata()
    if not musician_mapping:
        print("❌ Could not load musician mapping")
        return
    
    # Load all behavioral data grouped by musician level
    musician_data = load_all_behavioral_data(musician_mapping)
    
    # Check if we have data for each level
    levels_with_data = [level for level in [1, 2, 3] if musician_data[level]]
    if not levels_with_data:
        print("❌ No behavioral data could be loaded")
        return
    
    print(f"\nProceeding with analysis for levels: {levels_with_data}")
    
    # Analyze each musician level separately
    all_level_stats = {}
    
    for level in levels_with_data:
        level_stats = compute_group_statistics_for_level(musician_data[level], level)
        if level_stats:
            all_level_stats[level] = level_stats
    
    if all_level_stats:
        # Create plots
        create_musician_level_plots(all_level_stats)
        
        # Save results
        save_results_to_csv(all_level_stats)
        
        print(f"\n✅ Analysis complete for {len(all_level_stats)} musician levels!")
        
        # Print summary
        print(f"\n📊 SUMMARY:")
        for level, stats in all_level_stats.items():
            print(f"  Musician Level {level}: {stats['n_subjects']} subjects analyzed")
    
    else:
        print("❌ No valid data for analysis")

if __name__ == "__main__":
    main()