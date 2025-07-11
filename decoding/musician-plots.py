#!/usr/bin/env python3
"""
Decoding Timecourse Analysis by Musician Level
Plots neural decoding accuracy over time for different musician levels
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Constants and Configuration
HOME_DIR = '/media/headmodel/Elements/AWM4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'

# Decoding Parameters
RESAMPLE_FREQ = 100  # Hz
WINDOW_LENGTH_SEC = 0.1  # 100ms windows 
WINDOW_STEP_SEC = 0.01  # 10ms steps 

# Delay Period Configuration
DELAY_CONFIG = {
    'tmin': 2.0,
    'tmax': 4.7,
    'timepoints': np.linspace(2.0, 4.7, int((4.7-2.0)*RESAMPLE_FREQ))
}

# Critical time points to mark on plots
CRITICAL_TIMEPOINTS = [3.5, 4.5]  # Gray dashed lines at these times

# Feature Definitions for Maintained Information
FEATURES = {
    'maintained_voice_identity': {
        'name': 'Maintained Voice Identity',
        'color': '#1c686b',  # Darker teal
        'description': 'Voice identity information maintained in working memory',
        'file_path': f'{PROCESSED_DIR}/timepoints/delay_sliding_highres/maintained_voice_identity/decoding_results.xlsx'
    },
    'maintained_location': {
        'name': 'Maintained Location',
        'color': '#cb6a3e',  # Darker orange
        'description': 'Location information maintained in working memory',
        'file_path': f'{PROCESSED_DIR}/timepoints/delay_sliding_highres/maintained_location/decoding_results.xlsx'
    }
}

# Feature-specific musician level colors
FEATURE_MUSICIAN_COLORS = {
    'maintained_location': {
        1: "#ffcb99",  # Light orange
        2: "#ff996a",  # Medium orange
        3: "#dc5e23",  # Dark orange
    },
    'maintained_voice_identity': {
        1: "#81c7ca",  # Light teal
        2: "#009da4",  # Medium teal
        3: "#007482",  # Dark teal
    }
}

def load_metadata():
    """Load participant metadata including musician categories"""
    print(f"Loading metadata from: {META_FILE}")
    
    if not os.path.exists(META_FILE):
        print(f"Error: Metadata file not found at {META_FILE}")
        return {}
    
    meta_df = pd.read_excel(META_FILE)
    
    # First, let's check what columns are available
    print(f"Available columns in metadata: {list(meta_df.columns)}")
    
    # Create a mapping of subject ID to musician category
    musician_mapping = {}
    
    # Try to identify the subject ID column
    subject_col = None
    for col in ['Subject', 'ID', 'subject', 'subject_id', 'SubjectID', 'Participant']:
        if col in meta_df.columns:
            subject_col = col
            break
    
    # Try to identify the musician column
    musician_col = None
    for col in ['Musician', 'musician', 'MusicianLevel', 'Musical_ability']:
        if col in meta_df.columns:
            musician_col = col
            break
    
    if subject_col is None or musician_col is None:
        print(f"Warning: Could not find subject ID column (tried: {['Subject', 'ID', 'subject', 'subject_id', 'SubjectID', 'Participant']})")
        print(f"Warning: Could not find musician column (tried: {['Musician', 'musician', 'MusicianLevel', 'Musical_ability']})")
        print("Please check the column names in your metadata file")
        print("Available columns:", list(meta_df.columns))
        return musician_mapping
    
    print(f"Using '{subject_col}' as subject ID column and '{musician_col}' as musician level column")
    
    # Create mapping
    for idx, row in meta_df.iterrows():
        subject_id = row[subject_col]
        musician_level = row[musician_col]
        
        if pd.notna(subject_id) and pd.notna(musician_level):
            # Convert to string and handle different formats
            subject_id_str = str(subject_id).strip()
            try:
                musician_level_int = int(float(musician_level))
                if musician_level_int in [1, 2, 3]:
                    musician_mapping[subject_id_str] = musician_level_int
                    # Also add variations of the subject ID
                    if subject_id_str.startswith('sub-'):
                        musician_mapping[subject_id_str[4:]] = musician_level_int
                    else:
                        musician_mapping[f'sub-{subject_id_str}'] = musician_level_int
            except (ValueError, TypeError):
                print(f"Warning: Invalid musician level for subject {subject_id}: {musician_level}")
    
    print(f"Successfully mapped {len(musician_mapping)} subject entries")
    print(f"Musician level distribution: Level 1: {sum(1 for v in musician_mapping.values() if v==1)}, "
          f"Level 2: {sum(1 for v in musician_mapping.values() if v==2)}, "
          f"Level 3: {sum(1 for v in musician_mapping.values() if v==3)}")
    
    return musician_mapping

def load_decoding_data(file_path, feature_name):
    """Load decoding data from Excel file"""
    print(f"Loading {feature_name} data from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}")
        return None
    
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:\n{df.head()}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def process_decoding_by_musician_level(decoding_df, musician_mapping, feature_name):
    """Process decoding data by musician level"""
    print(f"\nProcessing {feature_name} by musician level...")
    
    if decoding_df is None:
        return None
    
    # Initialize results dictionary
    results_by_level = {1: [], 2: [], 3: []}
    
    print(f"Data structure analysis:")
    print(f"  - Shape: {decoding_df.shape}")
    print(f"  - Columns: {list(decoding_df.columns[:5])}...{list(decoding_df.columns[-2:])}")
    print(f"  - First few rows:\n{decoding_df.head(3)}")
    
    # Check if this is the timecourse format we expect
    if decoding_df.shape[1] > 10:  # Many columns indicate timecourse data
        print("Detected timecourse data format!")
        
        # Extract time information from column headers
        time_columns = [col for col in decoding_df.columns if isinstance(col, str) and 'Time_' in col]
        
        if time_columns:
            print(f"Found {len(time_columns)} time columns")
            print(f"Time range: {time_columns[0]} to {time_columns[-1]}")
            
            # Extract actual timepoints from headers (e.g., "Time_2.050s" -> 2.050)
            timepoints = []
            for col in time_columns:
                try:
                    time_str = col.replace('Time_', '').replace('s', '')
                    timepoints.append(float(time_str))
                except:
                    print(f"Warning: Could not parse time from {col}")
            
            print(f"Parsed timepoints: {timepoints[:5]}...{timepoints[-5:]} ({len(timepoints)} total)")
            
            # Get subject column (should be first column or one without 'Time_')
            subject_col = None
            for col in decoding_df.columns:
                if not (isinstance(col, str) and 'Time_' in col):
                    subject_col = col
                    break
            
            if subject_col is None:
                print("Error: Could not find subject ID column")
                return None
                
            print(f"Using column '{subject_col}' for subject IDs")
            
            # Process each subject
            for _, row in decoding_df.iterrows():
                subject_id = str(int(row[subject_col])) if pd.notna(row[subject_col]) else None
                
                if subject_id is None:
                    continue
                
                # Try different variations of subject ID to match with musician mapping
                musician_level = None
                for id_variant in [subject_id, f'sub-{subject_id}', subject_id.zfill(2), f'sub-{subject_id.zfill(2)}']:
                    if id_variant in musician_mapping:
                        musician_level = musician_mapping[id_variant]
                        break
                
                if musician_level is not None:
                    # Extract timecourse data
                    timecourse = []
                    for time_col in time_columns:
                        value = row[time_col]
                        if pd.notna(value):
                            timecourse.append(value)
                        else:
                            timecourse.append(0.5)  # Default to chance if missing
                    
                    if timecourse:
                        results_by_level[musician_level].append(timecourse)
                        print(f"Subject {subject_id} -> Musician Level {musician_level} (timecourse length: {len(timecourse)})")
                else:
                    print(f"Warning: No musician level found for subject {subject_id}")
                    print(f"  Tried variants: {[subject_id, f'sub-{subject_id}', subject_id.zfill(2), f'sub-{subject_id.zfill(2)}']}")
            
            # Store timepoints for later use
            if hasattr(process_decoding_by_musician_level, 'timepoints'):
                process_decoding_by_musician_level.timepoints = timepoints
            else:
                setattr(process_decoding_by_musician_level, 'timepoints', timepoints)
                
        else:
            print("Error: No time columns found in headers")
            return None
            
    else:
        print("Error: Expected timecourse data with many columns, but found only few columns")
        print("This appears to be summary data, not timecourse data")
        return None
    
    # Convert to the format matching your original code
    processed_results = {}
    for level in [1, 2, 3]:
        if results_by_level[level]:
            # Convert to numpy array - this matches your all_scores format
            all_scores = np.array(results_by_level[level])
            processed_results[level] = {
                'scores': all_scores,
                'n_subjects': len(all_scores)
            }
            print(f"Level {level}: {len(all_scores)} subjects")
        else:
            print(f"Level {level}: No subjects found")
    
    return processed_results

def create_summary_plots(results_dict, feature_name):
    """Create summary plots for summary data (bar plots and statistical comparisons)"""
    
    if not results_dict:
        print(f"No data to plot for {feature_name}")
        return
    
    # Create output directory
    output_path = f"{PROCESSED_DIR}/timepoints/delay_sliding_highres/musician_level_analysis"
    os.makedirs(output_path, exist_ok=True)
    
    # Extract summary statistics
    summary_stats = {}
    all_values = []
    
    for level in sorted(results_dict.keys()):
        # Get the first timepoint since all timepoints are the same (flat line)
        values = [scores[0] for scores in results_dict[level]['scores']]
        summary_stats[level] = {
            'values': values,
            'mean': np.mean(values),
            'std': np.std(values),
            'sem': np.std(values) / np.sqrt(len(values)),
            'n': len(values)
        }
        
        # For plotting individual points
        for val in values:
            all_values.append({'Musician_Level': level, 'Decoding_Accuracy': val})
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    
    levels = sorted(summary_stats.keys())
    means = [summary_stats[level]['mean'] for level in levels]
    sems = [summary_stats[level]['sem'] for level in levels]
    colors = [FEATURE_MUSICIAN_COLORS[feature_name][level]
 for level in levels]
    
    bars = plt.bar([f'Level {level}' for level in levels], means, 
                   yerr=sems, capsize=5, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=1)
    
    # Add individual data points
    for i, level in enumerate(levels):
        values = summary_stats[level]['values']
        x_positions = np.random.normal(i, 0.04, len(values))
        plt.scatter(x_positions, values, color='black', alpha=0.6, s=20, zorder=3)
    
    # Add statistical information
    for i, level in enumerate(levels):
        n = summary_stats[level]['n']
        mean = summary_stats[level]['mean']
        plt.text(i, mean + sems[i] + 0.01, f'N={n}', ha='center', va='bottom', fontweight='bold')
    
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Chance')
    plt.ylabel('Decoding Accuracy')
    plt.xlabel('Musician Level')
    plt.title(f'{FEATURES[feature_name]["name"]}\nDecoding Accuracy by Musician Level')
    plt.ylim(0.4, 0.65)
    plt.legend()
    plt.tight_layout()
    
    # Save bar plot
    plt.savefig(f'{output_path}/{feature_name}_summary_barplot.png')
    plt.savefig(f'{output_path}/{feature_name}_summary_barplot.pdf')
    plt.close()
    
    # Print summary statistics
    print(f"\nSummary Statistics for {feature_name}:")
    print("="*50)
    for level in levels:
        stats = summary_stats[level]
        print(f"Musician Level {level}:")
        print(f"  N = {stats['n']}")
        print(f"  Mean = {stats['mean']:.3f}")
        print(f"  SEM = {stats['sem']:.3f}")
        print(f"  SD = {stats['std']:.3f}")
        print()
    
    print(f"Summary bar plot for {feature_name} saved to {output_path}")
    
    return summary_stats

def create_individual_plots(results_dict, feature_name):
    """Create individual plot for each feature by musician level"""
    
    if not results_dict:
        print(f"No data to plot for {feature_name}")
        return
    
    # Create output directory matching your style
    output_path = f"{PROCESSED_DIR}/timepoints/delay_sliding_highres/musician_level_analysis"
    os.makedirs(output_path, exist_ok=True)
    
    # Get actual timepoints from the processing function
    if hasattr(process_decoding_by_musician_level, 'timepoints'):
        window_centers = np.array(process_decoding_by_musician_level.timepoints)
        print(f"Using actual timepoints: {window_centers[0]:.3f}s to {window_centers[-1]:.3f}s ({len(window_centers)} points)")
    else:
        # Fallback to default time axis
        first_level_data = list(results_dict.values())[0]
        n_timepoints = first_level_data['scores'].shape[1]
        window_centers = np.linspace(DELAY_CONFIG['tmin'], DELAY_CONFIG['tmax'], n_timepoints)
        print(f"Using default timepoints: {window_centers[0]:.3f}s to {window_centers[-1]:.3f}s ({len(window_centers)} points)")
    
    plt.figure(figsize=(12, 7))
    
    for level in sorted(results_dict.keys()):
        all_scores = results_dict[level]['scores']
        n_subjects = results_dict[level]['n_subjects']
        
        mean_scores = np.mean(all_scores, axis=0)
        std_error = np.std(all_scores, axis=0) / np.sqrt(all_scores.shape[0])
        
        plt.plot(window_centers, mean_scores,
                label=f'(N={n_subjects})',
                color=FEATURE_MUSICIAN_COLORS[feature_name][level]
,
                linewidth=2)
        
        plt.fill_between(window_centers,
                        mean_scores - std_error,
                        mean_scores + std_error,
                        alpha=0.2,
                        color=FEATURE_MUSICIAN_COLORS[feature_name][level]
)
    
    # Add critical timepoints as vertical lines
    for tp in CRITICAL_TIMEPOINTS:
        plt.axvline(x=tp, color='gray', linestyle='--', alpha=0.7)
    
    plt.axhline(y=0.5, color='black', linestyle='--', label='Chance')
    plt.xlabel('Time (s)')
    plt.ylabel('Decoding Accuracy')
    plt.title(f'{FEATURES[feature_name]["name"]} by Musician Level\nDecoding during Delay Period')
    
    # Set y-axis limits to match your style
    plt.ylim(0.4, 0.70)
    plt.legend()
    plt.tight_layout()
    
    # Save both PNG and PDF
    plt.savefig(f'{output_path}/{feature_name}_by_musician_level.png')
    plt.savefig(f'{output_path}/{feature_name}_by_musician_level.pdf')
    plt.close()
    
    print(f"Timecourse plot for {feature_name} saved to {output_path}")

def create_comparison_plots(all_results_dict):
    """Create comparison plots across features and musician levels"""
    
    if len(all_results_dict) < 2:
        print("Need at least 2 features for comparison plots")
        return
    
    # Create comparison directory
    comparison_path = f"{PROCESSED_DIR}/timepoints/delay_sliding_highres/musician_level_comparison"
    os.makedirs(comparison_path, exist_ok=True)
    
    # Get actual timepoints from the processing function
    if hasattr(process_decoding_by_musician_level, 'timepoints'):
        window_centers = np.array(process_decoding_by_musician_level.timepoints)
        print(f"Using actual timepoints for comparison plots: {window_centers[0]:.3f}s to {window_centers[-1]:.3f}s")
    else:
        # Fallback to default time axis
        first_feature = list(all_results_dict.values())[0]
        first_level = list(first_feature.keys())[0]
        n_timepoints = first_feature[first_level]['scores'].shape[1]
        window_centers = np.linspace(DELAY_CONFIG['tmin'], DELAY_CONFIG['tmax'], n_timepoints)
        print(f"Using default timepoints for comparison plots: {window_centers[0]:.3f}s to {window_centers[-1]:.3f}s")
    
    # Create separate plots for each musician level
    for level in [1, 2, 3]:
        plt.figure(figsize=(12, 7))
        
        features_with_level = []
        for feature_name, results_dict in all_results_dict.items():
            if level in results_dict:
                features_with_level.append(feature_name)
                all_scores = results_dict[level]['scores']
                n_subjects = results_dict[level]['n_subjects']
                
                mean_scores = np.mean(all_scores, axis=0)
                std_error = np.std(all_scores, axis=0) / np.sqrt(all_scores.shape[0])
                
                plt.plot(window_centers, mean_scores,
                        label=f'{FEATURES[feature_name]["name"]} (N={n_subjects})',
                        color=FEATURES[feature_name]['color'],
                        linewidth=2)
                
                plt.fill_between(window_centers,
                                mean_scores - std_error,
                                mean_scores + std_error,
                                alpha=0.2,
                                color=FEATURES[feature_name]['color'])
        
        if features_with_level:
            # Add critical timepoints as vertical lines
            for tp in CRITICAL_TIMEPOINTS:
                plt.axvline(x=tp, color='gray', linestyle='--', alpha=0.7)
            
            plt.axhline(y=0.5, color='black', linestyle='--', label='Chance')
            plt.xlabel('Time (s)')
            plt.ylabel('Decoding Accuracy')
            plt.title(f'Comparison of Maintained Information Decoding\nMusician Level {level} during Delay Period')
            
            # Set y-axis limits to match your style
            plt.ylim(0.4, 0.65)
            plt.legend()
            plt.tight_layout()
            
            # Save both PNG and PDF
            plt.savefig(f'{comparison_path}/feature_comparison_level_{level}.png')
            plt.savefig(f'{comparison_path}/feature_comparison_level_{level}.pdf')
            plt.close()
            
            print(f"Comparison plot for Musician Level {level} saved")
    
    # Also create a combined plot showing all features and levels
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each feature
    n_features = len(all_results_dict)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 5*n_features), sharex=True)
    
    if n_features == 1:
        axes = [axes]
    
    for idx, (feature_name, results_dict) in enumerate(all_results_dict.items()):
        ax = axes[idx]
        
        for level in sorted(results_dict.keys()):
            all_scores = results_dict[level]['scores']
            n_subjects = results_dict[level]['n_subjects']
            
            mean_scores = np.mean(all_scores, axis=0)
            std_error = np.std(all_scores, axis=0) / np.sqrt(all_scores.shape[0])
            
            ax.plot(window_centers, mean_scores,
                   label=f'Musician Level {level} (N={n_subjects})',
                   color=FEATURE_MUSICIAN_COLORS[feature_name][level]
,
                   linewidth=2)
            
            ax.fill_between(window_centers,
                           mean_scores - std_error,
                           mean_scores + std_error,
                           alpha=0.2,
                           color=FEATURE_MUSICIAN_COLORS[feature_name][level]
)
        
        # Add critical timepoints
        for tp in CRITICAL_TIMEPOINTS:
            ax.axvline(x=tp, color='gray', linestyle='--', alpha=0.7)
        
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        ax.set_ylabel('Decoding Accuracy')
        ax.set_title(f'{FEATURES[feature_name]["name"]}')
        ax.set_ylim(0.4, 0.65)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('Decoding Timecourse by Musician Level - All Features', fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save combined plot
    plt.savefig(f'{comparison_path}/all_features_all_levels_combined.png')
    plt.savefig(f'{comparison_path}/all_features_all_levels_combined.pdf')
    plt.close()
    
    print("\nTimecourse comparison plots created successfully!")

def main():
    """Main analysis function"""
    print("=== Decoding Timecourse Analysis by Musician Level ===\n")
    
    # Load musician mapping
    musician_mapping = load_metadata()
    if not musician_mapping:
        print("Error: Could not load musician mapping. Exiting.")
        return
    
    # Dictionary to store all results for comparison plots
    all_results_dict = {}
    
    # Process each feature
    for feature_key, feature_info in FEATURES.items():
        print(f"\n{'='*50}")
        print(f"Processing {feature_info['name']}")
        print(f"{'='*50}")
        
        # Load decoding data
        decoding_df = load_decoding_data(feature_info['file_path'], feature_key)
        
        if decoding_df is not None:
            # Process by musician level
            results = process_decoding_by_musician_level(decoding_df, musician_mapping, feature_key)
            
            if results:
                # Store results for comparison
                all_results_dict[feature_key] = results
                
                # Create individual plot
                create_individual_plots(results, feature_key)
                
                # Save processed data
                data_save_path = f"{PROCESSED_DIR}/timepoints/delay_sliding_highres/musician_level_analysis/{feature_key}_processed_data.csv"
                
                # Convert results to DataFrame for saving
                all_data = []
                
                # Get actual timepoints if available
                if hasattr(process_decoding_by_musician_level, 'timepoints'):
                    window_centers = np.array(process_decoding_by_musician_level.timepoints)
                else:
                    # Fallback to default time axis
                    first_level = list(results.keys())[0]
                    n_timepoints = results[first_level]['scores'].shape[1]
                    window_centers = np.linspace(DELAY_CONFIG['tmin'], DELAY_CONFIG['tmax'], n_timepoints)
                
                for level in sorted(results.keys()):
                    all_scores = results[level]['scores']
                    mean_scores = np.mean(all_scores, axis=0)
                    std_error = np.std(all_scores, axis=0) / np.sqrt(all_scores.shape[0])
                    
                    for i, (time, mean_acc, sem_acc) in enumerate(zip(window_centers, mean_scores, std_error)):
                        all_data.append({
                            'time': time,
                            'musician_level': level,
                            'decoding_accuracy': mean_acc,
                            'std_error': sem_acc,
                            'n_subjects': results[level]['n_subjects']
                        })
                
                df_save = pd.DataFrame(all_data)
                df_save.to_csv(data_save_path, index=False)
                print(f"Processed data saved to: {data_save_path}")
        
        else:
            print(f"Skipping {feature_key} due to data loading issues")
    
    # Create comparison plots if we have multiple features
    if len(all_results_dict) > 0:
        create_comparison_plots(all_results_dict)
    
    print(f"\nAnalysis complete! Check the following directories for results:")
    print(f"- Individual plots: {PROCESSED_DIR}/timepoints/delay_sliding_highres/musician_level_analysis/")
    print(f"- Comparison plots: {PROCESSED_DIR}/timepoints/delay_sliding_highres/musician_level_comparison/")

if __name__ == "__main__":
    main()