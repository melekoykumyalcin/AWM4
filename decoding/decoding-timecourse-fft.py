import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from pathlib import Path
import seaborn as sns

# Paths
HOME_DIR = '/media/headmodel/Elements/AWM4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'

# Parameters
SAMPLING_RATE = 100  # Hz (10ms steps)

# ========== ANALYSIS TYPE SELECTION ==========
# Set this to either 'encoding' or 'delay'
ANALYSIS_TYPE = 'encoding'  # Change this to 'encoding' or 'delay'
# ==============================================

def load_metadata():
    """Load participant metadata including musician categories"""
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

def load_timecourse_data(feature_name, stimulus_name=None):
    """Load timecourse data for all subjects"""
    
    if ANALYSIS_TYPE == 'encoding':
        # For encoding analysis
        results_path = f"{PROCESSED_DIR}/timepoints/PhD/averaged/{feature_name}({stimulus_name})"
        file_path = f'{results_path}/all_{feature_name}_{stimulus_name}_byTimepoint_SVM.xlsx'
    elif ANALYSIS_TYPE == 'delay':
        # For delay analysis
        results_path = f"{PROCESSED_DIR}/timepoints/delay_sliding_highres/"
        file_path = f'{results_path}/{feature_name}/decoding_results.xlsx'
    else:
        raise ValueError("ANALYSIS_TYPE must be either 'encoding' or 'delay'")
    
    try:
        # Load data with subjects as rows and timepoints as columns
        data_df = pd.read_excel(file_path, index_col=0)
        if stimulus_name:
            print(f"Loaded {feature_name} {stimulus_name}: {data_df.shape[0]} subjects, {data_df.shape[1]} timepoints")
        else:
            print(f"Loaded {feature_name}: {data_df.shape[0]} subjects, {data_df.shape[1]} timepoints")
        return data_df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def compute_fft_for_subject(timecourse):
    """Compute FFT for a single subject's timecourse"""
    # Remove any NaN values
    timecourse = np.nan_to_num(timecourse)
    
    # Compute FFT
    fft_values = fft(timecourse)
    frequencies = fftfreq(len(timecourse), d=1/SAMPLING_RATE)
    
    # Only keep positive frequencies
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]
    fft_power = np.abs(fft_values[positive_freq_idx])**2
    
    return frequencies, fft_power

def diagnose_subject_matching(data_df, musician_mapping):
    """Diagnostic function to help identify subject ID matching issues"""
    print("\n=== Subject ID Matching Diagnostic ===")
    print(f"Data subjects (first 5): {list(data_df.index[:5])}")
    print(f"Metadata subjects (first 5): {list(musician_mapping.keys())[:5]}")
    
    # Check for common patterns
    data_subjects = set(str(s) for s in data_df.index)
    meta_subjects = set(musician_mapping.keys())
    
    # Direct matches
    direct_matches = data_subjects & meta_subjects
    print(f"\nDirect matches found: {len(direct_matches)}")
    
    # Check if we need to add/remove prefixes
    unmatched_data = data_subjects - meta_subjects
    if unmatched_data:
        print(f"\nUnmatched subjects in data (first 5): {list(unmatched_data)[:5]}")
        
        # Check if adding 'sub-' helps
        matches_with_prefix = sum(1 for s in unmatched_data if f'sub-{s}' in meta_subjects)
        if matches_with_prefix > 0:
            print(f"  -> {matches_with_prefix} could match by adding 'sub-' prefix")
        
        # Check if removing 'sub-' helps
        matches_without_prefix = sum(1 for s in unmatched_data if s.startswith('sub-') and s[4:] in meta_subjects)
        if matches_without_prefix > 0:
            print(f"  -> {matches_without_prefix} could match by removing 'sub-' prefix")
    
    return direct_matches

def compute_all_ffts(data_df, musician_mapping, run_diagnostic=True):
    """Compute FFTs for all subjects (no grouping by musician level)"""
    if run_diagnostic:
        diagnose_subject_matching(data_df, musician_mapping)
    
    all_ffts = []
    valid_subjects = []
    
    for subject in data_df.index:
        # Try to find the subject in the musician mapping
        subject_str = str(subject)
        found_match = False
        
        # Try different formats of subject ID
        if subject_str in musician_mapping:
            found_match = True
        elif subject_str.startswith('sub-'):
            # Try without 'sub-' prefix
            subject_id = subject_str.replace('sub-', '')
            if subject_id in musician_mapping:
                found_match = True
        else:
            # Try with 'sub-' prefix
            subject_id = f'sub-{subject_str}'
            if subject_id in musician_mapping:
                found_match = True
        
        if found_match:
            timecourse = data_df.loc[subject].values
            frequencies, fft_power = compute_fft_for_subject(timecourse)
            all_ffts.append(fft_power)
            valid_subjects.append(subject)
        else:
            print(f"Warning: Subject {subject} not found in musician mapping")
    
    print(f"  Total valid subjects: {len(valid_subjects)}")
    
    return frequencies, all_ffts, valid_subjects

def plot_fft_comparison(frequencies, voice_ffts, location_ffts, condition_name):
    """Plot FFT comparison between relevant (voice) and irrelevant (location) features"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(f'FFT Analysis of Decoding Timecourses - {ANALYSIS_TYPE.title()} - {condition_name}', fontsize=16)
    
    # Define colors
    voice_color = '#1c686b'
    location_color = '#cb6a3e'
    
    # Limit frequency range for better visualization
    freq_limit = 12  # Hz
    freq_mask = frequencies <= freq_limit
    
    # Calculate means and SEMs
    if voice_ffts:
        voice_mean = np.mean(voice_ffts, axis=0)
        voice_sem = np.std(voice_ffts, axis=0) / np.sqrt(len(voice_ffts))
        
        # Plot voice identity (relevant)
        ax.plot(frequencies[freq_mask], voice_mean[freq_mask], 
               color=voice_color, linewidth=3, label='Voice Identity (Relevant)')
        ax.fill_between(frequencies[freq_mask], 
                      voice_mean[freq_mask] - voice_sem[freq_mask],
                      voice_mean[freq_mask] + voice_sem[freq_mask],
                      color=voice_color, alpha=0.3)
    
    if location_ffts:
        location_mean = np.mean(location_ffts, axis=0)
        location_sem = np.std(location_ffts, axis=0) / np.sqrt(len(location_ffts))
        
        # Plot location (irrelevant)
        ax.plot(frequencies[freq_mask], location_mean[freq_mask], 
               color=location_color, linewidth=3, label='Location (Irrelevant)')
        ax.fill_between(frequencies[freq_mask], 
                      location_mean[freq_mask] - location_sem[freq_mask],
                      location_mean[freq_mask] + location_sem[freq_mask],
                      color=location_color, alpha=0.3)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel('Power', fontsize=14)
    ax.set_title('All Subjects Combined', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add vertical lines for key frequency bands
    ax.axvspan(4, 8, alpha=0.1, color='gray', label='Theta')
    ax.axvspan(8, 12, alpha=0.1, color='blue', label='Alpha')
    
    # Add text labels for frequency bands
    ax.text(6, ax.get_ylim()[1]*0.95, 'Theta', ha='center', fontsize=10, alpha=0.7)
    ax.text(10, ax.get_ylim()[1]*0.95, 'Alpha', ha='center', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_difference_spectrum(frequencies, voice_ffts, location_ffts, condition_name):
    """Plot the difference between relevant and irrelevant feature spectra"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f'Difference Spectrum (Voice - Location) - {ANALYSIS_TYPE.title()} - {condition_name}', fontsize=16)
    
    freq_limit = 30  # Hz
    freq_mask = frequencies <= freq_limit
    
    if voice_ffts and location_ffts:
        voice_mean = np.mean(voice_ffts, axis=0)
        location_mean = np.mean(location_ffts, axis=0)
        difference = voice_mean - location_mean
        
        # Calculate SEM for difference
        n_subjects = min(len(voice_ffts), len(location_ffts))
        voice_std = np.std(voice_ffts, axis=0)
        location_std = np.std(location_ffts, axis=0)
        # Approximate SEM for difference
        difference_sem = np.sqrt(voice_std**2 + location_std**2) / np.sqrt(n_subjects)
        
        ax.plot(frequencies[freq_mask], difference[freq_mask], 
               color='#2c3e50', linewidth=3, label='Voice - Location')
        ax.fill_between(frequencies[freq_mask], 
                      difference[freq_mask] - difference_sem[freq_mask],
                      difference[freq_mask] + difference_sem[freq_mask],
                      color='#2c3e50', alpha=0.3)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel('Power Difference', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add vertical lines for key frequency bands
    ax.axvspan(4, 8, alpha=0.1, color='gray')
    ax.axvspan(8, 12, alpha=0.1, color='blue')
    ax.axvspan(13, 30, alpha=0.1, color='green')
    
    plt.tight_layout()
    return fig

def save_fft_results(frequencies, all_ffts, feature_name, condition_name, valid_subjects):
    """Save FFT results to Excel files"""
    results_path = f"{PROCESSED_DIR}/timepoints/PhD/averaged/FFT_analysis_{ANALYSIS_TYPE}_allsubjects"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    
    if all_ffts:
        # Create DataFrame with subjects as rows and frequencies as columns
        freq_labels = [f"{freq:.2f}Hz" for freq in frequencies]
        fft_df = pd.DataFrame(
            all_ffts,
            index=valid_subjects,
            columns=freq_labels
        )
        
        # Save to Excel
        filename = f"{results_path}/FFT_{feature_name}_{condition_name}_allSubjects.xlsx"
        fft_df.to_excel(filename)
        print(f"  Saved: {filename}")
        
        # Also save summary statistics
        summary_df = pd.DataFrame({
            'Frequency_Hz': frequencies,
            'Mean_Power': np.mean(all_ffts, axis=0),
            'SEM_Power': np.std(all_ffts, axis=0) / np.sqrt(len(all_ffts)),
            'N_Subjects': len(all_ffts)
        })
        summary_filename = f"{results_path}/FFT_{feature_name}_{condition_name}_summary.xlsx"
        summary_df.to_excel(summary_filename, index=False)
        print(f"  Saved summary: {summary_filename}")

def analyze_peak_frequencies(results):
    """Analyze peak frequencies in specific bands"""
    conditions = list(results.keys())
    
    # Create a summary table
    peak_summary = []
    
    for condition in conditions:
        frequencies = results[condition]['frequencies']
        
        # Define frequency bands
        bands = {
            'Theta': (4, 8),
            'Alpha': (8, 12),
            'Beta': (13, 30)
        }
        
        for feature_name, ffts in [('Voice Identity', results[condition]['voice_ffts']), 
                                   ('Location', results[condition]['location_ffts'])]:
            if ffts:
                mean_fft = np.mean(ffts, axis=0)
                
                for band_name, (low, high) in bands.items():
                    band_mask = (frequencies >= low) & (frequencies <= high)
                    if np.any(band_mask):
                        band_freqs = frequencies[band_mask]
                        band_power = mean_fft[band_mask]
                        
                        # Find peak
                        peak_idx = np.argmax(band_power)
                        peak_freq = band_freqs[peak_idx]
                        peak_power = band_power[peak_idx]
                        
                        # Calculate average power in band
                        avg_power = np.mean(band_power)
                        
                        peak_summary.append({
                            'Condition': condition,
                            'Feature': feature_name,
                            'Band': band_name,
                            'Peak_Frequency_Hz': peak_freq,
                            'Peak_Power': peak_power,
                            'Average_Band_Power': avg_power,
                            'N_Subjects': len(ffts)
                        })
    
    # Create and save summary table
    summary_df = pd.DataFrame(peak_summary)
    results_path = f"{PROCESSED_DIR}/timepoints/PhD/averaged/FFT_analysis_{ANALYSIS_TYPE}_allsubjects"
    summary_df.to_excel(f"{results_path}/peak_frequency_summary.xlsx", index=False)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for plotting
    x_labels = []
    voice_peaks = []
    location_peaks = []
    
    for band in ['Theta', 'Alpha', 'Beta']:
        for condition in conditions:
            label = f"{condition}\n{band}"
            x_labels.append(label)
            
            # Get peak frequencies
            voice_data = summary_df[(summary_df['Condition'] == condition) & 
                                   (summary_df['Feature'] == 'Voice Identity') & 
                                   (summary_df['Band'] == band)]
            location_data = summary_df[(summary_df['Condition'] == condition) & 
                                      (summary_df['Feature'] == 'Location') & 
                                      (summary_df['Band'] == band)]
            
            if not voice_data.empty:
                voice_peaks.append(voice_data['Peak_Frequency_Hz'].values[0])
            else:
                voice_peaks.append(np.nan)
                
            if not location_data.empty:
                location_peaks.append(location_data['Peak_Frequency_Hz'].values[0])
            else:
                location_peaks.append(np.nan)
    
    # Create bar plot
    x = np.arange(len(x_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, voice_peaks, width, label='Voice Identity', color='#1c686b', alpha=0.8)
    bars2 = ax.bar(x + width/2, location_peaks, width, label='Location', color='#cb6a3e', alpha=0.8)
    
    ax.set_ylabel('Peak Frequency (Hz)', fontsize=12)
    ax.set_title(f'Peak Frequencies by Band - {ANALYSIS_TYPE.title()} Analysis', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{results_path}/peak_frequencies_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nPeak Frequency Summary:")
    print(summary_df.to_string(index=False))

def main():
    """Main analysis function"""
    print(f"=== Starting {ANALYSIS_TYPE.upper()} Analysis (All Subjects Combined) ===")
    print("Loading metadata...")
    musician_mapping = load_metadata()
    print(f"Found {len(musician_mapping)} subjects with musician categories")
    
    # Dictionary to store results
    results = {}
    
    # Determine feature names and conditions based on analysis type
    if ANALYSIS_TYPE == 'encoding':
        voice_feature = 'voice_identity'
        location_feature = 'location'
        conditions = ['S1', 'S2']
    elif ANALYSIS_TYPE == 'delay':
        voice_feature = 'maintained_voice_identity'
        location_feature = 'maintained_location'
        conditions = ['Delay']  # Only one condition for delay
    else:
        raise ValueError("ANALYSIS_TYPE must be either 'encoding' or 'delay'")
    
    # Process conditions
    for condition in conditions:
        print(f"\nProcessing {condition}...")
        
        # Load voice identity (relevant) data
        print("Loading voice identity data...")
        if ANALYSIS_TYPE == 'encoding':
            voice_data = load_timecourse_data(voice_feature, condition)
        else:  # delay
            voice_data = load_timecourse_data(voice_feature)
        
        # Load location (irrelevant) data
        print("Loading location data...")
        if ANALYSIS_TYPE == 'encoding':
            location_data = load_timecourse_data(location_feature, condition)
        else:  # delay
            location_data = load_timecourse_data(location_feature)
        
        if voice_data is not None and location_data is not None:
            # Compute FFTs for all subjects
            print("Computing FFTs for voice identity...")
            frequencies, voice_ffts, voice_subjects = compute_all_ffts(voice_data, musician_mapping, run_diagnostic=True)
            
            print("Computing FFTs for location...")
            _, location_ffts, location_subjects = compute_all_ffts(location_data, musician_mapping, run_diagnostic=False)
            
            # Store results
            results[condition] = {
                'frequencies': frequencies,
                'voice_ffts': voice_ffts,
                'location_ffts': location_ffts,
                'voice_subjects': voice_subjects,
                'location_subjects': location_subjects
            }
            
            # Save FFT results
            print("Saving FFT results...")
            save_fft_results(frequencies, voice_ffts, voice_feature, condition, voice_subjects)
            save_fft_results(frequencies, location_ffts, location_feature, condition, location_subjects)
            
            # Create plots
            print("Creating plots...")
            fig1 = plot_fft_comparison(frequencies, voice_ffts, location_ffts, condition)
            plt.savefig(f"{PROCESSED_DIR}/timepoints/PhD/averaged/FFT_comparison_{ANALYSIS_TYPE}_{condition}_allsubjects.png", 
                       dpi=300, bbox_inches='tight')
            
            fig2 = plot_difference_spectrum(frequencies, voice_ffts, location_ffts, condition)
            plt.savefig(f"{PROCESSED_DIR}/timepoints/PhD/averaged/FFT_difference_{ANALYSIS_TYPE}_{condition}_allsubjects.png", 
                       dpi=300, bbox_inches='tight')
            
            plt.show()
    
    # Analyze peak frequencies
    if len(results) > 0:
        analyze_peak_frequencies(results)
    
    print(f"\n=== {ANALYSIS_TYPE.upper()} Analysis Complete (All Subjects) ===")
    return results

# Run the analysis
if __name__ == "__main__":
    results = main()