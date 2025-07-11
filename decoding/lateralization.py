import numpy as np
import os
import mne
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy import stats
from tqdm import tqdm
import seaborn as sns
import logging

# Your existing paths and parameters
HOME_DIR = '/mnt/hpc/projects/awm4/'
PROCESSED_DIR = HOME_DIR + 'AWM4_data/processed'
tfr_dir = f"{PROCESSED_DIR}/AllSensorsTFR/data"
ALPHA_DIR = f"{PROCESSED_DIR}/AlphaLateralization"
os.makedirs(ALPHA_DIR, exist_ok=True)

# Alpha frequency band
ALPHA_BAND = (8, 12)

# Time windows for analysis
BASELINE_WINDOW = (-0.4, -0.25)
STIMULUS_WINDOWS = {
    'stimulus1': (0.0, 1),
    'stimulus2': (1.0, 2), 
    'ping': (3.5, 4.0)
}

# Key sensor regions for alpha lateralization
regions = {
    'central_left': [
        'MLC11-3609', 'MLC12-3609', 'MLC13-3609', 'MLC14-3609', 'MLC15-3609', 'MLC16-3609', 'MLC17-3609',
        'MLC21-3609', 'MLC22-3609', 'MLC23-3609', 'MLC24-3609', 'MLC25-3609',
        'MLC31-3609', 'MLC32-3609', 'MLC41-3609', 'MLC42-3609',
        'MLC51-3609', 'MLC52-3609', 'MLC53-3609', 'MLC54-3609', 'MLC55-3609',
        'MLC61-3609', 'MLC62-3609', 'MLC63-3609'
    ],
    'central_right': [
        'MRC11-3609', 'MRC12-3609', 'MRC14-3609', 'MRC15-3609', 'MRC16-3609', 'MRC17-3609',
        'MRC21-3609', 'MRC22-3609', 'MRC23-3609', 'MRC24-3609', 'MRC25-3609',
        'MRC31-3609', 'MRC32-3609', 'MRC41-3609', 'MRC42-3609',
        'MRC51-3609', 'MRC52-3609', 'MRC53-3609', 'MRC54-3609', 'MRC55-3609',
        'MRC61-3609', 'MRC62-3609', 'MRC63-3609'
    ],
    'parietal_left': [
        'MLP11-3609', 'MLP12-3609', 'MLP21-3609', 'MLP22-3609', 'MLP23-3609',
        'MLP31-3609', 'MLP32-3609', 'MLP33-3609', 'MLP34-3609', 'MLP35-3609',
        'MLP41-3609', 'MLP42-3609', 'MLP43-3609', 'MLP44-3609', 'MLP45-3609',
        'MLP51-3609', 'MLP52-3609', 'MLP53-3609', 'MLP54-3609', 'MLP55-3609', 'MLP56-3609', 'MLP57-3609'
    ],
    'parietal_right': [
        'MRP11-3609', 'MRP12-3609', 'MRP21-3609', 'MRP22-3609', 'MRP23-3609',
        'MRP31-3609', 'MRP32-3609', 'MRP33-3609', 'MRP34-3609', 'MRP35-3609',
        'MRP41-3609', 'MRP42-3609', 'MRP43-3609', 'MRP44-3609', 'MRP45-3609',
        'MRP51-3609', 'MRP52-3609', 'MRP53-3609', 'MRP54-3609', 'MRP55-3609', 'MRP56-3609', 'MRP57-3609'
    ],
    'temporal_left': [
        'MLT11-3609', 'MLT12-3609', 'MLT13-3609', 'MLT14-3609', 'MLT15-3609', 'MLT16-3609',
        'MLT21-3609', 'MLT22-3609', 'MLT23-3609', 'MLT24-3609', 'MLT25-3609', 'MLT26-3609', 'MLT27-3609',
        'MLT31-3609', 'MLT32-3609', 'MLT33-3609', 'MLT34-3609', 'MLT35-3609', 'MLT36-3609', 'MLT37-3609',
        'MLT41-3609', 'MLT42-3609', 'MLT43-3609', 'MLT44-3609', 'MLT45-3609', 'MLT46-3609', 'MLT47-3609',
        'MLT51-3609', 'MLT52-3609', 'MLT53-3609', 'MLT54-3609', 'MLT55-3609', 'MLT56-3609', 'MLT57-3609'
    ],
    'temporal_right': [
        'MRT11-3609', 'MRT12-3609', 'MRT13-3609', 'MRT14-3609', 'MRT15-3609', 'MRT16-3609',
        'MRT21-3609', 'MRT22-3609', 'MRT23-3609', 'MRT24-3609', 'MRT25-3609', 'MRT26-3609', 'MRT27-3609',
        'MRT31-3609', 'MRT32-3609', 'MRT33-3609', 'MRT34-3609', 'MRT35-3609', 'MRT36-3609', 'MRT37-3609',
        'MRT41-3609', 'MRT42-3609', 'MRT43-3609', 'MRT44-3609', 'MRT45-3609', 'MRT46-3609', 'MRT47-3609',
        'MRT51-3609', 'MRT52-3609', 'MRT53-3609', 'MRT54-3609', 'MRT55-3609', 'MRT56-3609', 'MRT57-3609'
    ]
}

def extract_alpha_power(tfr, region_channels, freq_band=ALPHA_BAND, baseline_window=BASELINE_WINDOW):
    """
    Extract alpha power from specific sensor regions and apply baseline correction
    """
    # Get frequency indices for alpha band
    freq_mask = (tfr.freqs >= freq_band[0]) & (tfr.freqs <= freq_band[1])
    
    # Get channels that exist in the TFR data
    available_channels = [ch for ch in region_channels if ch in tfr.ch_names]
    
    if len(available_channels) == 0:
        return None
        
    # Extract data for these channels and frequency band
    ch_indices = [tfr.ch_names.index(ch) for ch in available_channels]
    
    # Average across frequencies and channels
    alpha_data = tfr.data[ch_indices, :, :]
    alpha_data = alpha_data[:, freq_mask, :].mean(axis=(0, 1))  # Average across channels and frequencies
    
    return alpha_data

def compute_lateralization_index(left_power, right_power):
    """
    Compute lateralization index: (Right - Left) / (Right + Left)
    Positive values indicate right-lateralized activity
    """
    return (right_power - left_power) / (right_power + left_power + 1e-10)  # Small epsilon to avoid division by zero

def analyze_alpha_lateralization_single_subject(subject, condition_file=None):
    """
    Analyze alpha lateralization for a single subject
    """
    print(f"Processing alpha lateralization for subject {subject}")
    
    try:
        # Load TFR data
        tfr_file = f"{tfr_dir}/sub-{subject}_tfr.pkl"
        with open(tfr_file, 'rb') as f:
            tfr = pickle.load(f)
        
        # Apply baseline correction if not already done
        tfr_copy = tfr.copy()
        tfr_copy.apply_baseline(baseline=BASELINE_WINDOW, mode='logratio')
        
        results = {}
        
        # Analyze each region pair
        region_pairs = [
            ('central_left', 'central_right'),
            ('parietal_left', 'parietal_right'),
            ('temporal_left', 'temporal_right')
        ]
        
        for left_region, right_region in region_pairs:
            # Extract alpha power for left and right regions
            left_power = extract_alpha_power(tfr_copy, regions[left_region])
            right_power = extract_alpha_power(tfr_copy, regions[right_region])
            
            if left_power is not None and right_power is not None:
                # Compute lateralization index over time
                lat_index = compute_lateralization_index(left_power, right_power)
                
                # Store results
                region_name = left_region.split('_')[0]  # 'central', 'parietal', 'temporal'
                results[region_name] = {
                    'left_power': left_power,
                    'right_power': right_power,
                    'lateralization_index': lat_index,
                    'times': tfr_copy.times
                }
                
                # Compute average values for specific time windows
                for window_name, (tmin, tmax) in STIMULUS_WINDOWS.items():
                    time_mask = (tfr_copy.times >= tmin) & (tfr_copy.times <= tmax)
                    if np.any(time_mask):
                        results[region_name][f'{window_name}_lat_index'] = lat_index[time_mask].mean()
                        results[region_name][f'{window_name}_left_power'] = left_power[time_mask].mean()
                        results[region_name][f'{window_name}_right_power'] = right_power[time_mask].mean()
        
        return results
        
    except Exception as e:
        print(f"Error processing subject {subject}: {str(e)}")
        return None

def analyze_all_subjects_lateralization(num_subjects=30):
    """
    Analyze alpha lateralization across all subjects
    """
    all_results = {}
    
    for subject in tqdm(range(1, num_subjects + 1), desc="Processing subjects"):
        result = analyze_alpha_lateralization_single_subject(subject)
        if result is not None:
            all_results[subject] = result
    
    return all_results

def plot_lateralization_timecourse(all_results, region='central'):
    """
    Plot alpha lateralization time course across subjects
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Alpha Lateralization Analysis - {region.title()} Region', fontsize=16)
    
    # Collect data across subjects
    all_lat_indices = []
    all_left_power = []
    all_right_power = []
    times = None
    
    for subject, results in all_results.items():
        if region in results:
            all_lat_indices.append(results[region]['lateralization_index'])
            all_left_power.append(results[region]['left_power'])
            all_right_power.append(results[region]['right_power'])
            if times is None:
                times = results[region]['times']
    
    if len(all_lat_indices) == 0:
        print(f"No data found for {region} region")
        return
    
    # Convert to numpy arrays
    all_lat_indices = np.array(all_lat_indices)
    all_left_power = np.array(all_left_power)
    all_right_power = np.array(all_right_power)
    
    # Plot 1: Lateralization index over time
    ax1 = axes[0, 0]
    mean_lat = all_lat_indices.mean(axis=0)
    sem_lat = stats.sem(all_lat_indices, axis=0)
    
    ax1.plot(times, mean_lat, 'k-', linewidth=2, label='Mean Lateralization')
    ax1.fill_between(times, mean_lat - sem_lat, mean_lat + sem_lat, alpha=0.3, color='gray')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Stimulus 1')
    ax1.axvline(x=1, color='blue', linestyle='--', alpha=0.5, label='Stimulus 2')
    ax1.axvline(x=3.5, color='green', linestyle='--', alpha=0.5, label='Ping')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Lateralization Index\n(Right-Left)/(Right+Left)')
    ax1.set_title('Alpha Lateralization Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Left vs Right power
    ax2 = axes[0, 1]
    mean_left = all_left_power.mean(axis=0)
    mean_right = all_right_power.mean(axis=0)
    sem_left = stats.sem(all_left_power, axis=0)
    sem_right = stats.sem(all_right_power, axis=0)
    
    ax2.plot(times, mean_left, 'b-', linewidth=2, label='Left Power')
    ax2.fill_between(times, mean_left - sem_left, mean_left + sem_left, alpha=0.3, color='blue')
    ax2.plot(times, mean_right, 'r-', linewidth=2, label='Right Power')
    ax2.fill_between(times, mean_right - sem_right, mean_right + sem_right, alpha=0.3, color='red')
    ax2.axvline(x=0, color='blue', linestyle='--', alpha=0.5)
    ax2.axvline(x=1, color='blue', linestyle='--', alpha=0.5)
    ax2.axvline(x=3.5, color='green', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Alpha Power (dB)')
    ax2.set_title('Left vs Right Alpha Power')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Lateralization in specific time windows
    ax3 = axes[1, 0]
    window_data = []
    window_labels = []
    
    for window_name in STIMULUS_WINDOWS.keys():
        window_values = []
        for subject, results in all_results.items():
            if region in results and f'{window_name}_lat_index' in results[region]:
                window_values.append(results[region][f'{window_name}_lat_index'])
        
        if len(window_values) > 0:
            window_data.append(window_values)
            window_labels.append(window_name)
    
    if len(window_data) > 0:
        ax3.boxplot(window_data, labels=window_labels)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax3.set_ylabel('Lateralization Index')
        ax3.set_title('Lateralization by Time Window')
        ax3.grid(True, alpha=0.3)
        
        # Add statistical tests
        if len(window_data) >= 2:
            for i, data in enumerate(window_data):
                t_stat, p_val = stats.ttest_1samp(data, 0)
                ax3.text(i+1, max(data), f'p={p_val:.3f}', ha='center', fontsize=8)
    
    # Plot 4: Individual subject data
    ax4 = axes[1, 1]
    for i, lat_index in enumerate(all_lat_indices[:10]):  # Show first 10 subjects
        ax4.plot(times, lat_index, alpha=0.6, linewidth=1)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='blue', linestyle='--', alpha=0.5)
    ax4.axvline(x=1, color='blue', linestyle='--', alpha=0.5)
    ax4.axvline(x=3.5, color='green', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Lateralization Index')
    ax4.set_title('Individual Subjects (first 10)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{ALPHA_DIR}/{region}_lateralization_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def compare_regions_lateralization(all_results):
    """
    Compare lateralization across different brain regions
    """
    regions_to_plot = ['central', 'parietal', 'temporal']
    
    fig, axes = plt.subplots(len(STIMULUS_WINDOWS), len(regions_to_plot), 
                            figsize=(15, 12))
    
    if len(STIMULUS_WINDOWS) == 1:
        axes = axes.reshape(1, -1)
    
    for row, (window_name, _) in enumerate(STIMULUS_WINDOWS.items()):
        for col, region in enumerate(regions_to_plot):
            ax = axes[row, col]
            
            # Collect data for this region and window
            region_data = []
            for subject, results in all_results.items():
                if region in results and f'{window_name}_lat_index' in results[region]:
                    region_data.append(results[region][f'{window_name}_lat_index'])
            
            if len(region_data) > 0:
                ax.hist(region_data, bins=10, alpha=0.7, edgecolor='black')
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                ax.axvline(x=np.mean(region_data), color='blue', linestyle='-', linewidth=2)
                
                # Statistical test
                t_stat, p_val = stats.ttest_1samp(region_data, 0)
                ax.set_title(f'{region.title()} - {window_name}\nMean={np.mean(region_data):.3f}, p={p_val:.3f}')
                ax.set_xlabel('Lateralization Index')
                ax.set_ylabel('Count')
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{ALPHA_DIR}/regions_comparison_lateralization.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_dataframe(all_results):
    """
    Create a summary dataframe for statistical analysis
    """
    data_rows = []
    
    for subject, results in all_results.items():
        for region in ['central', 'parietal', 'temporal']:
            if region in results:
                row = {'subject': subject, 'region': region}
                
                # Add time window data
                for window_name in STIMULUS_WINDOWS.keys():
                    if f'{window_name}_lat_index' in results[region]:
                        row[f'{window_name}_lateralization'] = results[region][f'{window_name}_lat_index']
                        row[f'{window_name}_left_power'] = results[region][f'{window_name}_left_power']
                        row[f'{window_name}_right_power'] = results[region][f'{window_name}_right_power']
                
                data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    df.to_csv(f"{ALPHA_DIR}/alpha_lateralization_summary.csv", index=False)
    return df

# Main analysis function
def analyze_by_stimulus_location(subject, left_trial_indices, right_trial_indices):
    """
    Analyze alpha lateralization separately for left and right stimulus presentations
    """
    try:
        # Load original epochs data to access individual trials
        epochs_file = f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif"            
        epochs = mne.read_epochs(epochs_file, preload=True)
        
        # Apply same preprocessing as in your original code
        mag_epochs = epochs.copy().pick_types(meg='mag', exclude='bads')
        mag_epochs.crop(tmin=-0.5, tmax=6)
        mag_epochs.filter(1, None)
        mag_epochs = mag_epochs.resample(200, npad='auto')
        
        results = {}
        
        for condition, trial_indices in [('left_stim', left_trial_indices), ('right_stim', right_trial_indices)]:
            # Select trials for this condition
            if len(trial_indices) == 0:
                continue
                
            condition_epochs = mag_epochs[trial_indices]
            
            # Compute TFR for this condition
            freqs_full = np.logspace(np.log10(4), np.log10(100), 27)
            freqs = freqs_full[(freqs_full < 49) | (freqs_full > 51)]
            
            tfr = mne.time_frequency.tfr_morlet(
                condition_epochs, 
                freqs=freqs, 
                n_cycles=7,
                use_fft=True, 
                return_itc=False, 
                average=True,
                decim=10,
                n_jobs=16,
                output='power'
            )
            
            # Apply baseline correction
            tfr.apply_baseline(baseline=(-0.4, -0.25), mode='logratio')
            
            # Analyze lateralization for this condition
            condition_results = {}
            region_pairs = [
                ('central_left', 'central_right'),
                ('parietal_left', 'parietal_right'),
                ('temporal_left', 'temporal_right')
            ]
            
            for left_region, right_region in region_pairs:
                left_power = extract_alpha_power(tfr, regions[left_region])
                right_power = extract_alpha_power(tfr, regions[right_region])
                
                if left_power is not None and right_power is not None:
                    lat_index = compute_lateralization_index(left_power, right_power)
                    region_name = left_region.split('_')[0]
                    
                    condition_results[region_name] = {
                        'lateralization_index': lat_index,
                        'times': tfr.times
                    }
                    
                    # Time window averages
                    for window_name, (tmin, tmax) in STIMULUS_WINDOWS.items():
                        time_mask = (tfr.times >= tmin) & (tfr.times <= tmax)
                        if np.any(time_mask):
                            condition_results[region_name][f'{window_name}_lat_index'] = lat_index[time_mask].mean()
            
            results[condition] = condition_results
            
        return results
        
    except Exception as e:
        print(f"Error analyzing stimulus location for subject {subject}: {str(e)}")
        return None

def compare_stimulus_locations(all_location_results):
    """
    Compare lateralization between left and right stimulus presentations
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Alpha Lateralization: Left vs Right Stimulus Location', fontsize=16)
    
    regions = ['central', 'parietal', 'temporal']
    windows = list(STIMULUS_WINDOWS.keys())
    
    for row, region in enumerate(regions):
        for col, window in enumerate(windows):
            ax = axes[row, col]
            
            left_stim_data = []
            right_stim_data = []
            
            for subject, results in all_location_results.items():
                if 'left_stim' in results and region in results['left_stim']:
                    if f'{window}_lat_index' in results['left_stim'][region]:
                        left_stim_data.append(results['left_stim'][region][f'{window}_lat_index'])
                
                if 'right_stim' in results and region in results['right_stim']:
                    if f'{window}_lat_index' in results['right_stim'][region]:
                        right_stim_data.append(results['right_stim'][region][f'{window}_lat_index'])
            
            if len(left_stim_data) > 0 and len(right_stim_data) > 0:
                # Create box plot
                data_to_plot = [left_stim_data, right_stim_data]
                bp = ax.boxplot(data_to_plot, labels=['Left Stim', 'Right Stim'], patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][1].set_facecolor('lightcoral')
                
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                ax.set_title(f'{region.title()} - {window}')
                ax.set_ylabel('Lateralization Index')
                ax.grid(True, alpha=0.3)
                
                # Statistical comparison
                if len(left_stim_data) == len(right_stim_data):  # Paired t-test
                    t_stat, p_val = stats.ttest_rel(left_stim_data, right_stim_data)
                    test_type = "paired"
                else:  # Independent t-test
                    t_stat, p_val = stats.ttest_ind(left_stim_data, right_stim_data)
                    test_type = "independent"
                
                ax.text(0.5, 0.95, f'{test_type} t-test\np = {p_val:.4f}', 
                       transform=ax.transAxes, ha='center', va='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{ALPHA_DIR}/stimulus_location_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def run_alpha_lateralization_analysis():
    """
    Run the complete alpha lateralization analysis
    """
    print("Starting alpha lateralization analysis...")
    
    # Analyze all subjects
    all_results = analyze_all_subjects_lateralization()
    
    # Save results
    with open(f"{ALPHA_DIR}/all_lateralization_results.pkl", 'wb') as f:
        pickle.dump(all_results, f)
    
    # Create plots for each region
    for region in ['central', 'parietal', 'temporal']:
        plot_lateralization_timecourse(all_results, region)
    
    # Compare regions
    compare_regions_lateralization(all_results)
    
    # Create summary dataframe
    df = create_summary_dataframe(all_results)
    
    print(f"Analysis complete! Results saved to {ALPHA_DIR}")
    print(f"Processed {len(all_results)} subjects successfully")
    
    return all_results, df

def run_stimulus_location_analysis(trial_info_function, stimulus_name='S1'):
    """
    Run analysis comparing left vs right stimulus presentations
    
    Parameters:
    trial_info_function: A function that takes subject ID and returns 
                        (left_trial_indices, right_trial_indices)
    stimulus_name: Which stimulus period this analysis is for ('S1' or 'S2')
    """
    print(f"Starting stimulus location analysis for {stimulus_name}...")
    
    all_location_results = {}
    
    for subject in tqdm(range(1, 31), desc=f"Processing subjects for {stimulus_name}"):
        try:
            # Get trial indices for left and right stimuli
            left_trials, right_trials = trial_info_function(subject)
            
            if len(left_trials) == 0 or len(right_trials) == 0:
                print(f"Skipping subject {subject}: insufficient trials (Left={len(left_trials)}, Right={len(right_trials)})")
                continue
            
            # Analyze this subject
            result = analyze_by_stimulus_location(subject, left_trials, right_trials, stimulus_name)
            if result is not None:
                all_location_results[subject] = result
                print(f"✅ Subject {subject}: Left={len(left_trials)}, Right={len(right_trials)}")
            else:
                print(f"❌ Subject {subject}: Analysis failed")
                
        except Exception as e:
            print(f"Error processing subject {subject}: {str(e)}")
    
    if len(all_location_results) > 0:
        # Compare conditions
        compare_stimulus_locations(all_location_results, stimulus_name)
        
        # Save results
        save_file = f"{ALPHA_DIR}/stimulus_location_results_{stimulus_name}.pkl"
        with open(save_file, 'wb') as f:
            pickle.dump(all_location_results, f)
        
        print(f"Completed analysis for {len(all_location_results)} subjects")
    else:
        print("No successful subjects processed!")
    
    return all_location_results

def analyze_by_stimulus_location(subject, left_trial_indices, right_trial_indices, stimulus_name='S1'):
    """
    Analyze alpha lateralization separately for left and right stimulus presentations
    UPDATED to work with pre-computed TFR data
    """
    try:
        # Load TFR data (same as regular analysis)
        tfr_file = f"{tfr_dir}/sub-{subject}_tfr.pkl"
        with open(tfr_file, 'rb') as f:
            tfr = pickle.load(f)
        
        print(f"Loaded TFR with shape: {tfr.data.shape}")
        
        # Apply baseline correction
        if subject in [13, 30]:
            tfr.apply_baseline((None, 0), mode='logratio')
        else:
            tfr.apply_baseline((-0.4, -0.25), mode='logratio')
        
        # Crop to analysis window (expand a bit to see context)
        if stimulus_name == 'S1':
            tfr = tfr.copy().crop(tmin=-0.2, tmax=1.2)
        else:  # S2
            tfr = tfr.copy().crop(tmin=0.8, tmax=2.2)
        
        results = {}
        
        for condition, trial_indices in [('left_stim', left_trial_indices), ('right_stim', right_trial_indices)]:
            if len(trial_indices) == 0:
                continue
            
            print(f"Processing {condition}: {len(trial_indices)} trials")
            
            # Extract data for these trials
            condition_tfr = tfr.copy()
            condition_tfr.data = condition_tfr.data[trial_indices]
            
            # Average across trials for this condition
            condition_tfr = condition_tfr.average()
            
            # Analyze lateralization for this condition
            condition_results = {}
            region_pairs = [
                ('central_left', 'central_right'),
                ('parietal_left', 'parietal_right'),
                ('temporal_left', 'temporal_right')
            ]
            
            for left_region, right_region in region_pairs:
                left_power = extract_alpha_power(condition_tfr, regions[left_region])
                right_power = extract_alpha_power(condition_tfr, regions[right_region])
                
                if left_power is not None and right_power is not None:
                    lat_index = compute_lateralization_index(left_power, right_power)
                    region_name = left_region.split('_')[0]
                    
                    condition_results[region_name] = {
                        'lateralization_index': lat_index,
                        'left_power': left_power,
                        'right_power': right_power,
                        'times': condition_tfr.times,
                        'n_trials': len(trial_indices)
                    }
                    
                    # Time window averages
                    for window_name, (tmin, tmax) in STIMULUS_WINDOWS.items():
                        time_mask = (condition_tfr.times >= tmin) & (condition_tfr.times <= tmax)
                        if np.any(time_mask):
                            condition_results[region_name][f'{window_name}_lat_index'] = lat_index[time_mask].mean()
                            condition_results[region_name][f'{window_name}_left_power'] = left_power[time_mask].mean()
                            condition_results[region_name][f'{window_name}_right_power'] = right_power[time_mask].mean()
            
            results[condition] = condition_results
            
        return results
        
    except Exception as e:
        print(f"Error analyzing stimulus location for subject {subject}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def compare_stimulus_locations(all_location_results, stimulus_name='S1'):
    """
    Compare lateralization between left and right stimulus presentations
    UPDATED with stimulus name in title and save path
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'Alpha Lateralization: Left vs Right Stimulus Location ({stimulus_name})', fontsize=16)
    
    regions = ['central', 'parietal', 'temporal']
    windows = list(STIMULUS_WINDOWS.keys())
    
    for row, region in enumerate(regions):
        for col, window in enumerate(windows):
            ax = axes[row, col]
            
            left_stim_data = []
            right_stim_data = []
            
            for subject, results in all_location_results.items():
                if 'left_stim' in results and region in results['left_stim']:
                    if f'{window}_lat_index' in results['left_stim'][region]:
                        left_stim_data.append(results['left_stim'][region][f'{window}_lat_index'])
                
                if 'right_stim' in results and region in results['right_stim']:
                    if f'{window}_lat_index' in results['right_stim'][region]:
                        right_stim_data.append(results['right_stim'][region][f'{window}_lat_index'])
            
            if len(left_stim_data) > 0 and len(right_stim_data) > 0:
                # Create box plot
                data_to_plot = [left_stim_data, right_stim_data]
                bp = ax.boxplot(data_to_plot, labels=['Left Stim', 'Right Stim'], patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][1].set_facecolor('lightcoral')
                
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                ax.set_title(f'{region.title()} - {window}')
                ax.set_ylabel('Lateralization Index')
                ax.grid(True, alpha=0.3)
                
                # Statistical comparison
                if len(left_stim_data) == len(right_stim_data):  # Paired t-test
                    t_stat, p_val = stats.ttest_rel(left_stim_data, right_stim_data)
                    test_type = "paired"
                else:  # Independent t-test
                    t_stat, p_val = stats.ttest_ind(left_stim_data, right_stim_data)
                    test_type = "independent"
                
                # Add mean values
                mean_left = np.mean(left_stim_data)
                mean_right = np.mean(right_stim_data)
                
                ax.text(0.5, 0.95, f'{test_type} t-test\np = {p_val:.4f}\nL: {mean_left:.3f}, R: {mean_right:.3f}', 
                       transform=ax.transAxes, ha='center', va='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=8)
    
    plt.tight_layout()
    save_path = f"{ALPHA_DIR}/stimulus_location_comparison_{stimulus_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comparison plot saved to {save_path}")

def test_trial_extraction():
    """Test the trial extraction for a few subjects"""
    print("Testing trial extraction...")
    
    for subject in [1, 2, 3]:  # Test first 3 subjects
        print(f"\n=== Subject {subject} ===")
        
        for stimulus in ['S1', 'S2']:
            try:
                left_trials, right_trials = get_trial_indices_by_location(subject, stimulus)
                print(f"{stimulus}: Left={len(left_trials)} trials, Right={len(right_trials)} trials")
                
                if len(left_trials) > 0 and len(right_trials) > 0:
                    print(f"  ✅ Good balance for {stimulus}")
                else:
                    print(f"  ⚠️  Imbalanced or missing trials for {stimulus}")
                    
            except Exception as e:
                print(f"  ❌ Error with {stimulus}: {str(e)}")

# Create summary function for both analyses
def create_lateralization_summary():
    """Create a comprehensive summary of all lateralization analyses"""
    
    print("\n" + "="*80)
    print("ALPHA LATERALIZATION ANALYSIS SUMMARY")
    print("="*80)
    
    # Test trial extraction first
    print("\n1. Testing trial extraction...")
    test_trial_extraction()
    
    # Basic lateralization analysis (all trials)
    print("\n2. Running basic lateralization analysis...")
    basic_results, basic_df = run_alpha_lateralization_analysis()
    
    # Location-specific analysis for both S1 and S2
    print("\n3. Running location-specific analysis...")
    
    for stimulus in ['S1', 'S2']:
        print(f"\n--- Analyzing {stimulus} stimulus presentations ---")
        
        def trial_function(subject):
            return get_trial_indices_by_location(subject, stimulus)
        
        location_results = run_stimulus_location_analysis(trial_function, stimulus)
        
        if len(location_results) > 0:
            print(f"✅ Completed {stimulus} analysis with {len(location_results)} subjects")
        else:
            print(f"❌ Failed {stimulus} analysis")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved to: {ALPHA_DIR}")
    
    return basic_results, basic_df

if __name__ == "__main__":
    # Run the analysis
    results, summary_df = run_alpha_lateralization_analysis()
    
    # Print some basic statistics
    print("\nSummary Statistics:")
    print("="*50)
    
    for region in ['central', 'parietal', 'temporal']:
        print(f"\n{region.upper()} REGION:")
        for window in STIMULUS_WINDOWS.keys():
            col_name = f'{window}_lateralization'
            if col_name in summary_df.columns:
                region_data = summary_df[summary_df['region'] == region][col_name].dropna()
                if len(region_data) > 0:
                    mean_lat = region_data.mean()
                    t_stat, p_val = stats.ttest_1samp(region_data, 0)
                    print(f"  {window}: Mean={mean_lat:.4f}, t={t_stat:.3f}, p={p_val:.4f}")