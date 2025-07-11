#!/usr/bin/env python
"""
Analyze trial counts across all subjects to determine optimal pseudotrial parameters
"""

import os
import numpy as np
import pandas as pd

def analyze_group_trial_counts():
    """
    Analyze trial counts across all subjects to set optimal group parameters
    """
    HOME_DIR = '/mnt/hpc/projects/awm4/'
    META_FILE = HOME_DIR + 'MEGNotes.xlsx'
    PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
    
    print("Loading metadata...")
    try:
        metaInfo = pd.read_excel(META_FILE)
        all_subjects = list(np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject']))
        print(f"Found {len(all_subjects)} subjects in final sample: {all_subjects}")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None, None
    
    condition_counts = {}  # condition -> [counts across subjects]
    subject_summaries = {}
    failed_subjects = []
    
    all_conditions = [111, 112, 113, 114, 121, 122, 123, 124,
                     131, 132, 133, 134, 141, 142, 143, 144]
    
    for subject in all_subjects:
        print(f"Analyzing subject {subject}...")
        
        # Determine events file path
        if subject == 28:
            events_file = f"{PROCESSED_DIR}/aligned_events_fixed/sub-{subject}_events.npy"
        else:
            events_file = f"{PROCESSED_DIR}/aligned_events_corrected/sub-{subject}_events.npy"
        
        # Check if file exists
        if not os.path.exists(events_file):
            print(f"  Warning: Events file not found for subject {subject}")
            failed_subjects.append(subject)
            continue
        
        try:
            # Load events
            events = np.load(events_file)
            print(f"  Loaded {len(events)} events")
            
            # Count each condition
            subject_counts = {}
            for condition in all_conditions:
                count = np.sum(events == condition)
                subject_counts[condition] = count
                
                if condition not in condition_counts:
                    condition_counts[condition] = []
                condition_counts[condition].append(count)
            
            # Calculate binary class counts
            voice_sp12 = sum(subject_counts[c] for c in [111,112,113,114,121,122,123,124])
            voice_sp34 = sum(subject_counts[c] for c in [131,132,133,134,141,142,143,144])
            loc_12 = sum(subject_counts[c] for c in [111,112,121,122,131,132,141,142])
            loc_34 = sum(subject_counts[c] for c in [113,114,123,124,133,134,143,144])
            
            subject_summaries[subject] = {
                'condition_counts': subject_counts,
                'voice_sp12': voice_sp12,
                'voice_sp34': voice_sp34,
                'loc_12': loc_12,
                'loc_34': loc_34,
                'total_trials': sum(subject_counts.values())
            }
            
            print(f"  Total trials: {subject_summaries[subject]['total_trials']}")
            print(f"  Voice Sp1+2: {voice_sp12}, Voice Sp3+4: {voice_sp34}")
            print(f"  Location 1+2: {loc_12}, Location 3+4: {loc_34}")
            
        except Exception as e:
            print(f"  Error loading subject {subject}: {e}")
            failed_subjects.append(subject)
            continue
    
    if failed_subjects:
        print(f"\nFailed to load {len(failed_subjects)} subjects: {failed_subjects}")
    
    successful_subjects = len(subject_summaries)
    print(f"\nSuccessfully analyzed {successful_subjects} subjects")
    
    return condition_counts, subject_summaries

def determine_optimal_parameters(condition_counts, subject_summaries):
    """
    Determine group-level parameters based on trial count distributions
    """
    
    if not subject_summaries:
        print("No subjects to analyze!")
        return None, None, None
    
    # Analyze condition-level statistics
    condition_stats = {}
    for condition, counts in condition_counts.items():
        condition_stats[condition] = {
            'mean': np.mean(counts),
            'std': np.std(counts),
            'min': np.min(counts),
            'max': np.max(counts),
            'median': np.median(counts),
            'n_subjects': len(counts)
        }
    
    # Analyze binary class statistics
    voice_sp12_counts = [s['voice_sp12'] for s in subject_summaries.values()]
    voice_sp34_counts = [s['voice_sp34'] for s in subject_summaries.values()]
    loc_12_counts = [s['loc_12'] for s in subject_summaries.values()]
    loc_34_counts = [s['loc_34'] for s in subject_summaries.values()]
    
    print("\n" + "="*60)
    print("GROUP-LEVEL TRIAL COUNT ANALYSIS")
    print("="*60)
    print(f"Number of subjects analyzed: {len(subject_summaries)}")
    
    print(f"\nSubject IDs: {sorted(subject_summaries.keys())}")
    
    print("\nBINARY CLASSIFICATION COUNTS:")
    print(f"Voice Sp1+2: mean={np.mean(voice_sp12_counts):.1f}, std={np.std(voice_sp12_counts):.1f}, min={np.min(voice_sp12_counts)}, max={np.max(voice_sp12_counts)}")
    print(f"Voice Sp3+4: mean={np.mean(voice_sp34_counts):.1f}, std={np.std(voice_sp34_counts):.1f}, min={np.min(voice_sp34_counts)}, max={np.max(voice_sp34_counts)}")
    print(f"Location 1+2: mean={np.mean(loc_12_counts):.1f}, std={np.std(loc_12_counts):.1f}, min={np.min(loc_12_counts)}, max={np.max(loc_12_counts)}")
    print(f"Location 3+4: mean={np.mean(loc_34_counts):.1f}, std={np.std(loc_34_counts):.1f}, min={np.min(loc_34_counts)}, max={np.max(loc_34_counts)}")
    
    print("\nCONDITION-LEVEL STATISTICS:")
    print("Condition | Mean  | Std  | Min | Max | Median")
    print("-" * 50)
    for cond in sorted(condition_stats.keys()):
        stats = condition_stats[cond]
        print(f"{cond:8} | {stats['mean']:5.1f} | {stats['std']:4.1f} | {stats['min']:3} | {stats['max']:3} | {stats['median']:6.1f}")
    
    # Find bottlenecks
    min_condition_trials = {cond: stats['min'] for cond, stats in condition_stats.items()}
    worst_condition = min(min_condition_trials.keys(), key=lambda x: min_condition_trials[x])
    worst_condition_count = min_condition_trials[worst_condition]
    
    min_binary_class_size = min(
        np.min(voice_sp12_counts), np.min(voice_sp34_counts),
        np.min(loc_12_counts), np.min(loc_34_counts)
    )
    
    print(f"\nBOTTLENECKS:")
    print(f"Worst individual condition: {worst_condition} with {worst_condition_count} trials")
    print(f"Smallest binary class size: {min_binary_class_size} trials")
    
    # Determine optimal parameters based on worst-case scenario
    # Conservative approach: use 20-25% of worst-case condition for averaging
    trials_per_pseudotrial = max(3, worst_condition_count // 4)  # 25% of worst-case condition
    
    # Number of pseudotrials per condition: conservative to avoid overusing trials
    max_possible_pseudotrials = worst_condition_count // trials_per_pseudotrial
    pseudotrials_per_condition = max(2, min(4, max_possible_pseudotrials))  # Cap at 4 for balance
    
    recommended_params = {
        'trials_per_pseudotrial': trials_per_pseudotrial,
        'pseudotrials_per_condition': pseudotrials_per_condition,
        'min_binary_class_size': min_binary_class_size,
        'min_condition_size': worst_condition_count,
        'worst_condition': worst_condition
    }
    
    # Calculate expected pseudotrials per binary class
    # 8 conditions per binary class (e.g., for voice: conditions 111-114, 121-124 vs 131-134, 141-144)
    expected_pseudotrials_per_class = 8 * pseudotrials_per_condition
    
    recommended_params['expected_pseudotrials_per_voice_class'] = expected_pseudotrials_per_class
    recommended_params['expected_pseudotrials_per_location_class'] = expected_pseudotrials_per_class
    
    # Calculate percentage of trials used
    trials_used_per_condition = trials_per_pseudotrial * pseudotrials_per_condition
    percentage_used = (trials_used_per_condition / worst_condition_count) * 100
    
    print(f"\n" + "="*60)
    print("RECOMMENDED GROUP PARAMETERS")
    print("="*60)
    print(f"Trials per pseudotrial: {trials_per_pseudotrial}")
    print(f"Pseudotrials per condition: {pseudotrials_per_condition}")
    print(f"Expected pseudotrials per voice class: {expected_pseudotrials_per_class}")
    print(f"Expected pseudotrials per location class: {expected_pseudotrials_per_class}")
    print(f"Trials used per condition: {trials_used_per_condition} ({percentage_used:.1f}% of worst case)")
    print(f"Based on worst-case condition {worst_condition} with {worst_condition_count} trials")
    
    # Show what this means for each subject
    print(f"\nIMPACT ON INDIVIDUAL SUBJECTS:")
    print("Subject | Total | Used | Percentage")
    print("-" * 40)
    for subject_id in sorted(subject_summaries.keys()):
        total_trials = subject_summaries[subject_id]['total_trials']
        used_trials = 16 * trials_used_per_condition  # 16 conditions total
        percentage = (used_trials / total_trials) * 100
        print(f"{subject_id:7} | {total_trials:5} | {used_trials:4} | {percentage:9.1f}%")
    
    return recommended_params, condition_stats, subject_summaries

def save_results_to_file(recommended_params, condition_stats, subject_summaries):
    """Save detailed results to a file for reference"""
    
    HOME_DIR = '/mnt/hpc/projects/awm4/'
    output_file = f"{HOME_DIR}/trial_count_analysis_results.txt"
    
    with open(output_file, 'w') as f:
        f.write("TRIAL COUNT ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Analysis date: {pd.Timestamp.now()}\n")
        f.write(f"Number of subjects: {len(subject_summaries)}\n")
        f.write(f"Subject IDs: {sorted(subject_summaries.keys())}\n\n")
        
        f.write("RECOMMENDED PARAMETERS:\n")
        for key, value in recommended_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("CONDITION STATISTICS:\n")
        f.write("Condition | Mean  | Std  | Min | Max | Median\n")
        f.write("-" * 50 + "\n")
        for cond in sorted(condition_stats.keys()):
            stats = condition_stats[cond]
            f.write(f"{cond:8} | {stats['mean']:5.1f} | {stats['std']:4.1f} | {stats['min']:3} | {stats['max']:3} | {stats['median']:6.1f}\n")
        
        f.write("\nSUBJECT SUMMARIES:\n")
        for subject_id, summary in sorted(subject_summaries.items()):
            f.write(f"\nSubject {subject_id}:\n")
            f.write(f"  Total trials: {summary['total_trials']}\n")
            f.write(f"  Voice Sp1+2: {summary['voice_sp12']}\n")
            f.write(f"  Voice Sp3+4: {summary['voice_sp34']}\n")
            f.write(f"  Location 1+2: {summary['loc_12']}\n")
            f.write(f"  Location 3+4: {summary['loc_34']}\n")
    
    print(f"\nDetailed results saved to: {output_file}")

# Main execution
if __name__ == "__main__":
    print("Starting trial count analysis...")
    
    # Analyze trial counts
    condition_counts, subject_summaries = analyze_group_trial_counts()
    
    if condition_counts is None or not subject_summaries:
        print("Failed to analyze trial counts. Please check file paths and data.")
        exit(1)
    
    # Determine optimal parameters
    recommended_params, condition_stats, subject_summaries = determine_optimal_parameters(
        condition_counts, subject_summaries
    )
    
    if recommended_params is None:
        print("Failed to determine optimal parameters.")
        exit(1)
    
    # Save results
    save_results_to_file(recommended_params, condition_stats, subject_summaries)
    
    print("\nAnalysis complete!")
    print("You can now copy the output above and share it for parameter optimization.")