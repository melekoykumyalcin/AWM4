#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defining the correct trials to use later in the decoding
Handles block restarts and extracts correct trial indices
"""
import os
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Set paths
HOME_DIR = '/media/headmodel/Elements/AWM4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
BEHAVIORAL_PATH = HOME_DIR + 'AWM4_data/behavioral/'
OUTPUT_PATH = HOME_DIR + 'AWM4_data/behavioral/correctResponses/'
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(HOME_DIR)

# Load meta information
metaInfo = pd.read_excel(META_FILE)

# Debug: Check what behavioral files actually exist
print("=== DEBUGGING FILE PATHS ===")
behavioral_files = [f for f in os.listdir(BEHAVIORAL_PATH) if f.startswith('Output_AWM4_Exp1_') and f.endswith('.txt')]
print(f"Found {len(behavioral_files)} behavioral files:")
for f in sorted(behavioral_files[:10]):  # Show first 10
    print(f"  {f}")
if len(behavioral_files) > 10:
    print(f"  ... and {len(behavioral_files) - 10} more")

# Extract subject codes from actual filenames
actual_subjects = []
for f in behavioral_files:
    # Extract VP number from filename like 'Output_AWM4_Exp1_MEG_VP23.txt'
    if 'VP' in f:
        subject_part = f.split('_')[-1].replace('.txt', '')  # Get 'VP23' part
        actual_subjects.append(subject_part)
    
actual_subjects = sorted(list(set(actual_subjects)))
print(f"\nActual subjects from filenames: {actual_subjects}")

# Compare with MetaInfo extraction
meta_subjects = np.unique(metaInfo['SubjectCode'].str.extract(r'S(\d+)').astype(int))
meta_subjects = ['VP' + str(sub) for sub in meta_subjects]
print(f"Subjects from MetaInfo: {sorted(meta_subjects)}")

# Use the actual subjects found in files
Subs = actual_subjects
print(f"\nUsing {len(Subs)} subjects with existing behavioral files")

# Output logic reference:
# % 1:'Block' 2:'Trial' 3:'BlockTrial' 4:'SpeakerS1' 5:'LocS1' 6: 'SpeakerS2'
# % 7:'LocS2' 8:'TargetPos' 9:'SpeakerMatch' 10:'ProbeDistance'
# % 11:'LocMatch' 12:'ProbeLoc' 13:Answer (1:Match, 0:Nonmatch, 2:too slow)
# % 14:Correct(1)/Incorrect(0) 15:RT 16:Trialduration new 16: Syllable

def inspect_and_clean_behavioral_data(Subs):
    """
    Load and clean behavioral data for all subjects:
    - Handle block restarts (keep post-restart sequences)
    - Extract indices of correct trials
    - Report data quality metrics
    """
    all_behavioral_data = {}
    inspection_summary = {}
    
    print("=== BEHAVIORAL DATA INSPECTION ===")
    print(f"Processing {len(Subs)} subjects...")
    
    for subject in Subs:
        print(f"\n--- Inspecting {subject} ---")
        
        try:
            file_path = os.path.join(BEHAVIORAL_PATH, f'Output_AWM4_Exp1_MEG_{subject}.txt')
            if not os.path.exists(file_path):
                print(f"  WARNING: File not found for {subject}")
                continue
                
            # Read space-separated file, skip header line
            data = pd.read_csv(file_path, sep=' ', skiprows=1, header=None)
            
            # Debug: Check file structure
            print(f"    File shape: {data.shape} (rows × columns)")
            print(f"    Column count: {data.shape[1]}")
            if data.shape[1] < 14:
                print(f"    ERROR: Expected at least 14 columns, found {data.shape[1]}")
                print(f"    Sample row: {data.iloc[0, :min(10, data.shape[1])].values}")
                continue
            
            # Extract key columns (0-indexed)
            blocks = data.iloc[:, 0].values      # Block
            trials = data.iloc[:, 1].values      # Trial within block 
            cumulative = data.iloc[:, 2].values  # Cumulative trial counter
            correct = data.iloc[:, 13].values    # Correct(1)/Incorrect(0)/TooSlow(2)
            
            print(f"  Raw data: {len(data)} rows")
            
            # Clean data by removing pre-restart sequences
            cleaned_indices = []
            restart_info = {}
            
            for block_num in np.unique(blocks):
                block_mask = blocks == block_num
                block_trials = trials[block_mask]
                block_indices = np.where(block_mask)[0]
                
                # Find restart points: where trial number decreases
                restart_points = []
                for i in range(1, len(block_trials)):
                    if block_trials[i] < block_trials[i-1]:
                        restart_points.append(i)
                
                if restart_points:
                    # Keep only after the LAST restart
                    last_restart = restart_points[-1]
                    keep_indices = block_indices[last_restart:]
                    restart_info[block_num] = {
                        'had_restart': True,
                        'n_restarts': len(restart_points),
                        'trials_before_restart': last_restart,
                        'trials_kept': len(keep_indices)
                    }
                    print(f"    Block {block_num}: restart detected, keeping {len(keep_indices)} trials (dropped {last_restart})")
                else:
                    # No restart, keep all
                    keep_indices = block_indices
                    restart_info[block_num] = {
                        'had_restart': False, 
                        'trials_kept': len(keep_indices)
                    }
                
                cleaned_indices.extend(keep_indices)
            
            # Apply cleaning
            cleaned_indices = np.array(cleaned_indices)
            clean_blocks = blocks[cleaned_indices]
            clean_trials = trials[cleaned_indices]
            clean_cumulative = cumulative[cleaned_indices]
            clean_correct = correct[cleaned_indices]
            
            # Verify cleaned block structure
            unique_blocks = np.unique(clean_blocks)
            block_structure = {}
            
            for block_num in unique_blocks:
                block_mask = clean_blocks == block_num
                block_trial_numbers = clean_trials[block_mask]
                block_cumulative = clean_cumulative[block_mask]
                
                block_structure[block_num] = {
                    'n_trials': len(block_trial_numbers),
                    'starts_with_1': bool(block_trial_numbers[0] == 1),
                    'max_trial': int(np.max(block_trial_numbers)),
                    'is_sequential': bool(np.array_equal(block_trial_numbers, np.arange(1, len(block_trial_numbers) + 1))),
                    'cumulative_start': int(block_cumulative[0]),
                    'cumulative_end': int(block_cumulative[-1])
                }
            
            # Extract correctness indices (relative to cleaned data)
            correct_indices = np.where(clean_correct == 1)[0]
            incorrect_indices = np.where(clean_correct == 0)[0] 
            too_slow_indices = np.where(clean_correct == 2)[0]
            
            # Count totals
            n_total = len(clean_correct)
            n_correct = len(correct_indices)
            n_incorrect = len(incorrect_indices)
            n_too_slow = len(too_slow_indices)
            
            # Store cleaned data
            all_behavioral_data[subject] = {
                'blocks': clean_blocks,
                'trials': clean_trials,
                'cumulative': clean_cumulative,
                'correct': clean_correct,
                'correct_indices': correct_indices,
                'incorrect_indices': incorrect_indices,
                'too_slow_indices': too_slow_indices,
                'original_indices': cleaned_indices  # Original row indices from raw data
            }
            
            inspection_summary[subject] = {
                'raw_rows': len(data),
                'cleaned_rows': n_total,
                'rows_dropped': len(data) - n_total,
                'n_blocks': len(unique_blocks),
                'restart_info': restart_info,
                'block_structure': block_structure,
                'correct_trials': n_correct,
                'incorrect_trials': n_incorrect,
                'too_slow_trials': n_too_slow,
                'accuracy_excluding_slow': n_correct / (n_correct + n_incorrect) if (n_correct + n_incorrect) > 0 else 0,
                'overall_accuracy': n_correct / n_total if n_total > 0 else 0
            }
            
            print(f"    Cleaned: {n_total} trials ({len(unique_blocks)} blocks)")
            print(f"    Correct: {n_correct}, Incorrect: {n_incorrect}, Too slow: {n_too_slow}")
            print(f"    Accuracy (excl. too slow): {inspection_summary[subject]['accuracy_excluding_slow']:.3f}")
            print(f"    Overall accuracy: {inspection_summary[subject]['overall_accuracy']:.3f}")
            
            # Check for problematic blocks
            problematic_blocks = []
            for block_num, info in block_structure.items():
                issues = []
                if not info['starts_with_1']:
                    issues.append('not starting with 1')
                if not info['is_sequential']:
                    issues.append('not sequential')
                if info['n_trials'] > 64:
                    issues.append(f'too many trials ({info["n_trials"]})')
                if info['n_trials'] < 20:  # Suspiciously few trials
                    issues.append(f'too few trials ({info["n_trials"]})')
                
                if issues:
                    problematic_blocks.append(f"Block {block_num}: {', '.join(issues)}")
            
            if problematic_blocks:
                print(f"    WARNING: {'; '.join(problematic_blocks)}")
            
        except Exception as e:
            print(f"  ERROR processing {subject}: {str(e)}")
            continue
    
    return all_behavioral_data, inspection_summary

def save_correct_trial_indices(all_behavioral_data):
    """
    Save correct trial indices for each subject
    Format compatible with MEG decoding pipeline
    """
    print(f"\n=== SAVING CORRECT TRIAL INDICES ===")
    
    for subject, data in all_behavioral_data.items():
        # Save correct trial indices
        correct_indices_file = os.path.join(OUTPUT_PATH, f'correct_indices_{subject}.npy')
        np.save(correct_indices_file, data['correct_indices'])
        
        # Save all trial correctness labels (for full array)
        correctness_labels_file = os.path.join(OUTPUT_PATH, f'correctness_labels_{subject}.npy')
        np.save(correctness_labels_file, data['correct'])
        
        # Save mapping information for debugging
        mapping_file = os.path.join(OUTPUT_PATH, f'trial_mapping_{subject}.npz')
        np.savez(mapping_file,
                blocks=data['blocks'],
                trials=data['trials'],
                cumulative=data['cumulative'],
                correct=data['correct'],
                correct_indices=data['correct_indices'],
                original_indices=data['original_indices'])
        
        print(f"  {subject}: Saved {len(data['correct_indices'])} correct trial indices")

def create_summary_report(inspection_summary):
    """Create summary statistics and plots"""
    print(f"\n=== SUMMARY STATISTICS ===")
    
    # Collect statistics
    all_stats = []
    for subject, stats in inspection_summary.items():
        all_stats.append({
            'Subject': subject,
            'Raw_Rows': stats['raw_rows'],
            'Cleaned_Rows': stats['cleaned_rows'],
            'Rows_Dropped': stats['rows_dropped'],
            'N_Blocks': stats['n_blocks'],
            'Correct_Trials': stats['correct_trials'],
            'Incorrect_Trials': stats['incorrect_trials'],
            'Too_Slow_Trials': stats['too_slow_trials'],
            'Accuracy_Excl_Slow': stats['accuracy_excluding_slow'],
            'Overall_Accuracy': stats['overall_accuracy'],
            'Had_Restarts': any(info.get('had_restart', False) for info in stats['restart_info'].values())
        })
    
    df_stats = pd.DataFrame(all_stats)
    
    # Print summary
    print(f"Total subjects processed: {len(df_stats)}")
    print(f"Subjects with restarts: {df_stats['Had_Restarts'].sum()}")
    print(f"Mean trials per subject: {df_stats['Cleaned_Rows'].mean():.1f} ± {df_stats['Cleaned_Rows'].std():.1f}")
    print(f"Mean accuracy (excl. slow): {df_stats['Accuracy_Excl_Slow'].mean():.3f} ± {df_stats['Accuracy_Excl_Slow'].std():.3f}")
    print(f"Mean overall accuracy: {df_stats['Overall_Accuracy'].mean():.3f} ± {df_stats['Overall_Accuracy'].std():.3f}")
    
    # Save summary
    summary_file = os.path.join(OUTPUT_PATH, 'behavioral_summary.csv')
    df_stats.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Behavioral Data Summary', fontsize=16)
    
    # Trial counts
    axes[0,0].hist(df_stats['Cleaned_Rows'], bins=20, alpha=0.7, color='skyblue')
    axes[0,0].set_xlabel('Number of Trials')
    axes[0,0].set_ylabel('Number of Subjects')
    axes[0,0].set_title('Distribution of Trial Counts')
    axes[0,0].axvline(df_stats['Cleaned_Rows'].mean(), color='red', linestyle='--', label='Mean')
    axes[0,0].legend()
    
    # Accuracy distribution
    axes[0,1].hist(df_stats['Accuracy_Excl_Slow'], bins=20, alpha=0.7, color='lightgreen')
    axes[0,1].set_xlabel('Accuracy (excluding too slow)')
    axes[0,1].set_ylabel('Number of Subjects')
    axes[0,1].set_title('Accuracy Distribution')
    axes[0,1].axvline(df_stats['Accuracy_Excl_Slow'].mean(), color='red', linestyle='--', label='Mean')
    axes[0,1].legend()
    
    # Response types
    response_types = df_stats[['Correct_Trials', 'Incorrect_Trials', 'Too_Slow_Trials']].mean()
    axes[1,0].bar(range(len(response_types)), response_types.values, 
                  color=['green', 'red', 'orange'], alpha=0.7)
    axes[1,0].set_xticks(range(len(response_types)))
    axes[1,0].set_xticklabels(['Correct', 'Incorrect', 'Too Slow'])
    axes[1,0].set_ylabel('Mean Count per Subject')
    axes[1,0].set_title('Response Type Distribution')
    
    # Subjects with issues
    restart_subjects = df_stats['Had_Restarts'].sum()
    normal_subjects = len(df_stats) - restart_subjects
    axes[1,1].pie([normal_subjects, restart_subjects], 
                  labels=[f'Normal ({normal_subjects})', f'Had Restarts ({restart_subjects})'],
                  colors=['lightblue', 'orange'], autopct='%1.1f%%')
    axes[1,1].set_title('Subjects with Block Restarts')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(OUTPUT_PATH, 'behavioral_summary.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved to: {plot_file}")
    
    return df_stats

def main():
    """Main processing function"""
    print(f"Processing started at: {datetime.now()}")
    print(f"Found {len(Subs)} subjects: {Subs}")
    
    # Load and clean all behavioral data
    all_behavioral_data, inspection_summary = inspect_and_clean_behavioral_data(Subs)
    
    if not all_behavioral_data:
        print("No behavioral data could be loaded!")
        return
    
    # Save correct trial indices
    save_correct_trial_indices(all_behavioral_data)
    
    # Create summary report
    summary_df = create_summary_report(inspection_summary)
    
    # Save detailed inspection results
    inspection_file = os.path.join(OUTPUT_PATH, 'detailed_inspection.json')
    
    # Convert numpy types to native Python for JSON serialization
    json_summary = {}
    for subject, stats in inspection_summary.items():
        json_summary[subject] = {
            'raw_rows': int(stats['raw_rows']),
            'cleaned_rows': int(stats['cleaned_rows']),
            'rows_dropped': int(stats['rows_dropped']),
            'n_blocks': int(stats['n_blocks']),
            'correct_trials': int(stats['correct_trials']),
            'incorrect_trials': int(stats['incorrect_trials']),
            'too_slow_trials': int(stats['too_slow_trials']),
            'accuracy_excluding_slow': float(stats['accuracy_excluding_slow']),
            'overall_accuracy': float(stats['overall_accuracy']),
            'restart_info': stats['restart_info'],
            'block_structure': {str(k): v for k, v in stats['block_structure'].items()}
        }
    
    # with open(inspection_file, 'w') as f:
    #     json.dump(json_summary, f, indent=2)
    
    print(f"\nDetailed inspection saved to: {inspection_file}")
    
    # Print final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Successfully processed: {len(all_behavioral_data)} subjects")
    print(f"Total cleaned trials: {summary_df['Cleaned_Rows'].sum()}")
    print(f"Total correct trials: {summary_df['Correct_Trials'].sum()}")
    print(f"Files saved to: {OUTPUT_PATH}")
    
    print(f"\nProcessing completed at: {datetime.now()}")
    
    # Instructions for MEG integration
    print(f"\n=== MEG INTEGRATION NOTES ===")
    print("To use these correct trial indices in your MEG decoding:")
    print("1. Load: correct_indices = np.load(f'correct_indices_{subject}.npy')")
    print("2. Apply after all MEG preprocessing drops (jumps, artifacts, etc.)")
    print("3. The indices correspond to trials in your cleaned behavioral data")
    print("4. You'll need to align these with your final MEG trial set")

if __name__ == "__main__":
    main()