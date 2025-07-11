import numpy as np
import os
import mne
import pandas as pd
import pickle
from mne.time_frequency import tfr_multitaper
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# Configuration parameters
#HOME_DIR = '/media/headmodel/Elements/AWM4/'

HOME_DIR = '/mnt/hpc/projects/awm4/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
OUTPUT_DIR = PROCESSED_DIR + '/TFR_Multitaper'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Analysis parameters
RESAMPLE_FREQ = 200  # Hz
NUM_JOBS = 8
# BASELINE_SLOW = (-0.4, -0.25)  
# BASELINE_FAST = (-0.4, -0.125) 
TMIN = -0.5  # Include baseline period
TMAX = 5.5   # Extended to avoid edge effects
DISPLAY_TMAX = 4.5  # Actual experiment data up to 4.5s

# Frequency bands for visualization
FREQ_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'low_gamma': (30, 45),
    'high_gamma': (60, 100)
}

def get_ctf_regions(info):
    """
    Group CTF MEG sensors into brain regions based on naming conventions.
    Returns a dictionary with region names as keys and lists of channel indices as values.
    """
    # Initialize regions dictionary
    regions = {
        'frontal_left': [],
        'frontal_right': [],
        'frontal_midline': [],
        'temporal_left': [],
        'temporal_right': [],
        'central_left': [],
        'central_right': [],
        'central_midline': [],
        'parietal_left': [],
        'parietal_right': [],
        'parietal_midline': [],
        'occipital_left': [],
        'occipital_right': [],
        'occipital_midline': []
    }
    
    # Get all magnetometer channel names and indices
    mag_indices = mne.pick_types(info, meg='mag')
    ch_names = [info['ch_names'][idx] for idx in mag_indices]
    
    # Group channels by region based on CTF naming conventions
    for i, ch_name in enumerate(ch_names):
        # Skip non-MEG channels
        if not ch_name.startswith('M'):
            continue
        
        # Get the actual index from the original channels list
        ch_idx = mag_indices[i]
            
        # Extract positional information from channel name
        # CTF naming: e.g., MLF14 (Left Frontal), MRT32 (Right Temporal)
        
        # Frontal channels
        if 'LF' in ch_name:
            regions['frontal_left'].append(ch_idx)
        elif 'RF' in ch_name:
            regions['frontal_right'].append(ch_idx)
        elif 'ZF' in ch_name:
            regions['frontal_midline'].append(ch_idx)
            
        # Temporal channels
        elif 'LT' in ch_name:
            regions['temporal_left'].append(ch_idx)
        elif 'RT' in ch_name:
            regions['temporal_right'].append(ch_idx)
            
        # Central channels
        elif 'LC' in ch_name:
            regions['central_left'].append(ch_idx)
        elif 'RC' in ch_name:
            regions['central_right'].append(ch_idx)
        elif 'ZC' in ch_name:
            regions['central_midline'].append(ch_idx)
            
        # Parietal channels
        elif 'LP' in ch_name:
            regions['parietal_left'].append(ch_idx)
        elif 'RP' in ch_name:
            regions['parietal_right'].append(ch_idx)
        elif 'ZP' in ch_name:
            regions['parietal_midline'].append(ch_idx)
            
        # Occipital channels
        elif 'LO' in ch_name:
            regions['occipital_left'].append(ch_idx)
        elif 'RO' in ch_name:
            regions['occipital_right'].append(ch_idx)
        elif 'ZO' in ch_name:
            regions['occipital_midline'].append(ch_idx)
    
    # Remove empty regions
    regions = {k: v for k, v in regions.items() if v}
    
    # Print summary of regions and sensor counts
    print("CTF Sensor regions defined:")
    for region, sensors in regions.items():
        print(f"  {region}: {len(sensors)} sensors")
    
    return regions

def plot_regional_topomaps(tfr_data, regions, band_name, fmin, fmax, output_dir, subject_str):
    """
    Plot topographic maps for specific brain regions across time windows.
    
    Parameters:
    -----------
    tfr_data : mne.time_frequency.AverageTFR
        The time-frequency data object.
    regions : dict
        Dictionary with region names as keys and lists of channel indices as values.
    band_name : str
        Name of the frequency band.
    fmin, fmax : float
        Minimum and maximum frequencies for the band.
    output_dir : str
        Directory to save the plots.
    subject_str : str
        String identifier for the subject (e.g., "Subject 1" or "Grand Average (N=20)")
    """
    # Define time windows: baseline, encoding, and delay
    time_windows = [
        (-0.5, 0),    # Baseline
        (0, 2),       # Encoding
        (2, 4.5)      # Delay
    ]
    window_names = ["Baseline", "Encoding", "Delay"]
    
    # Create a figure for each region
    for region_name, ch_indices in regions.items():
        if not ch_indices:
            continue
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{subject_str} - {band_name} band ({fmin}-{fmax} Hz) - {region_name.replace('_', ' ').title()}")
        
        for i, (t_min, t_max) in enumerate(time_windows):
            # Plot topomap for this region and time window
            tfr_data.plot_topomap(
                ch_type='mag', 
                tmin=t_min, tmax=t_max,
                fmin=fmin, fmax=fmax,
                baseline=[-0.4, -0.25],  # Apply baseline correction
                mode='percent',
                vmin=-0.5, vmax=0.5,
                title=f"{window_names[i]} ({t_min}-{t_max}s)",
                axes=axes[i],
                picks=ch_indices,  # Use only channels in this region
                show=False
            )
        
        plt.subplots_adjust(top=0.9)  # Adjust for the suptitle
        
        # Save the figure
        region_filename = region_name.lower().replace(' ', '_')
        plt.savefig(f"{output_dir}/{band_name}_{region_filename}_topomap.png")
        plt.savefig(f"{output_dir}/{band_name}_{region_filename}_topomap.pdf")
        plt.close()

def plot_regional_comparison(tfr_data, regions, band_name, fmin, fmax, output_dir, subject_str):
    """
    Plot the time course of power changes in each region for a specific frequency band.
    
    Parameters are the same as plot_regional_topomaps.
    """
    plt.figure(figsize=(12, 8))
    
    # Colors for different regions
    colors = plt.cm.tab10(np.linspace(0, 1, len(regions)))
    
    # Extract the frequency band data
    freq_mask = (tfr_data.freqs >= fmin) & (tfr_data.freqs <= fmax)
    
    # Plot each region
    for (region_name, ch_indices), color in zip(regions.items(), colors):
        if not ch_indices:
            continue
            
        # Average power across frequencies in the band and across channels in this region
        region_data = tfr_data.data[ch_indices]
        region_power = np.mean(region_data[:, freq_mask, :], axis=(0, 1))
        
        # Plot region power over time
        plt.plot(tfr_data.times, region_power, 
                label=f"{region_name.replace('_', ' ').title()}",
                linewidth=2, color=color)
    
    # Add annotations
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='r', linestyle='-', label='Stimulus Onset')
    plt.axvline(2, color='g', linestyle='-', label='Delay Start')
    plt.axvspan(0, 2, color='lightblue', alpha=0.2, label='Encoding')
    plt.axvspan(2, 4.5, color='lightgreen', alpha=0.2, label='Delay')
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Power (% change from baseline)', fontsize=12)
    plt.title(f'{subject_str} - {band_name} band ({fmin}-{fmax} Hz) - Regional Comparison', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(-0.5, 5.0, 0.5))
    
    # Save the figure
    plt.savefig(f"{output_dir}/{band_name}_regional_comparison.png", dpi=300)
    plt.savefig(f"{output_dir}/{band_name}_regional_comparison.pdf")
    plt.close()

def plot_log_frequency_tfr(tfr_data, output_dir, subject_str):
    """Plot TFR with logarithmic frequency scaling to better visualize all frequency bands"""
    # Create a copy of the TFR data
    display_tfr = tfr_data.copy()
    
    # Apply baseline correction
    display_tfr.apply_baseline(baseline=[-0.4, -0.25], mode='percent')
    
    # Crop to time window of interest
    display_tfr.crop(tmin=-0.4, tmax=4.5)
    
    # Get data as numpy array
    data = display_tfr.data
    freqs = display_tfr.freqs
    times = display_tfr.times
    
    # Average across channels
    data_mean = np.mean(data, axis=0)
    
    # Create plot with logarithmic y-axis
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(times, freqs, data_mean, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    plt.yscale('log')  # Set logarithmic scale for frequency axis
    plt.colorbar(label='Power (% change)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f"{subject_str} - TF Power with Log Frequency Scale")
    
    # Add markers for important time points
    plt.axvline(x=0, color='white', linestyle='--', linewidth=1.5, label='Stimulus Onset')
    plt.axvline(x=2, color='white', linestyle=':', linewidth=1.5, label='Delay Onset')
    plt.legend(loc='upper right')
    
    # Adjust y-axis ticks to show meaningful frequencies
    plt.yticks([4, 8, 12, 20, 30, 45, 60, 80, 100])
    
    plt.savefig(f"{output_dir}/log_frequency_tfr.png", dpi=300)
    plt.savefig(f"{output_dir}/log_frequency_tfr.pdf")
    plt.close()

def combine_slow_fast_tfr(power_slow, power_fast):
    """Combine slow and fast TFR data into a single object"""
    # Convert to numpy arrays
    data_slow = power_slow.data
    data_fast = power_fast.data
    
    # Extract frequency information
    freqs_slow = power_slow.freqs
    freqs_fast = power_fast.freqs
    
    # Ensure times match (they should if from same epochs)
    if not np.array_equal(power_slow.times, power_fast.times):
        raise ValueError("Time points don't match between slow and fast TFR.")
    
    # Combine along the frequency dimension
    combined_data = np.concatenate([data_slow, data_fast], axis=1)
    
    # Sort frequencies in ascending order
    combined_freqs = np.concatenate([freqs_slow, freqs_fast])
    sort_idx = np.argsort(combined_freqs)
    combined_freqs = combined_freqs[sort_idx]
    combined_data = combined_data[:, sort_idx, :]
    
    # Create a new TFR object
    combined_power = mne.time_frequency.AverageTFR(
        info=power_slow.info,
        data=combined_data,
        times=power_slow.times,
        freqs=combined_freqs,
        nave=power_slow.nave
    )
    
    return combined_power

def process_single_subject(subject):
    """Process TF analysis for a single subject using FLUX multitaper approach"""
    print(f"Processing TF analysis for subject {subject}")
    
    subject_dir = f"{OUTPUT_DIR}/sub-{subject}"
    os.makedirs(subject_dir, exist_ok=True)
    
    try:
        # Load clean epochs
        epochs_file = f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif"
        if not os.path.exists(epochs_file):
            print(f"Epochs file not found for subject {subject}")
            return None
            
        epochs = mne.read_epochs(epochs_file, preload=True)
        print(f"Loaded {len(epochs)} epochs for subject {subject}")
        
        # Check for jump artifacts and drop affected epochs
        jumps_file = f"{PROCESSED_DIR}/ICAs/Jumps{subject}.npy"
        if os.path.isfile(jumps_file):
            jump_inds = np.load(jumps_file)
            
            if len(jump_inds) > 0:
                jump_inds = np.array(jump_inds, dtype=int)
                valid_jump_inds = jump_inds[jump_inds < len(epochs)]
                
                if len(valid_jump_inds) > 0:
                    print(f"Dropping {len(valid_jump_inds)} epochs with jump artifacts")
                    epochs.drop(valid_jump_inds, reason='jump')
        
        # Select only magnetometer channels
        mag_epochs = epochs.copy().pick_types(meg='mag')
        
        # Crop to time window of interest (including baseline and extended period)
        mag_epochs.crop(tmin=TMIN, tmax=TMAX)
        
        # Resample for efficiency
        mag_epochs = mag_epochs.resample(RESAMPLE_FREQ, npad='auto')
        
        # FLUX approach: Process slow and fast frequencies separately
        
        # 1. Process slow oscillations (2-30 Hz)
        print("Computing TFR for slow oscillations (2-30 Hz)...")
        freqs_slow = np.arange(2, 31, 1)  # Linear spacing from 2-30 Hz
        n_cycles_slow = freqs_slow / 2    # Frequency-dependent cycles
        time_bandwidth_slow = 2.0         # Time-bandwidth parameter for slow frequencies
        
        power_slow = tfr_multitaper(
            mag_epochs,
            freqs=freqs_slow,
            n_cycles=n_cycles_slow,
            time_bandwidth=time_bandwidth_slow,
            use_fft=True,
            return_itc=False,
            average=True,
            decim=1,
            n_jobs=NUM_JOBS
        )
        
        # 2. Process fast oscillations (30-100 Hz)
        print("Computing TFR for fast oscillations (30-100 Hz)...")
        freqs_fast = np.arange(30, 101, 2)  # Linear spacing from 30-100 Hz
        n_cycles_fast = 7                   # window length : n_cycles_fast / freqs_fast 
        time_bandwidth_fast = 5.0           # 5 tapers 
        
        power_fast = tfr_multitaper(
            mag_epochs,
            freqs=freqs_fast,
            n_cycles=n_cycles_fast,
            time_bandwidth=time_bandwidth_fast,
            use_fft=True,
            return_itc=False,
            average=True,
            decim=1,
            n_jobs=NUM_JOBS
        )
        
        # Apply baseline correction with different windows and using percent mode
        # power_slow.apply_baseline(BASELINE_SLOW, mode='percent')
        # power_fast.apply_baseline(BASELINE_FAST, mode='percent')
        
        # Save individual frequency range results
        with open(f"{subject_dir}/tfr_power_slow.pkl", 'wb') as f:
            pickle.dump(power_slow, f)
        with open(f"{subject_dir}/tfr_power_fast.pkl", 'wb') as f:
            pickle.dump(power_fast, f)
        
        # Combine slow and fast TFR into one object
        print("Combining slow and fast frequency results...")
        combined_power = combine_slow_fast_tfr(power_slow, power_fast)
        
        # Save combined result
        with open(f"{subject_dir}/tfr_power_combined.pkl", 'wb') as f:
            pickle.dump(combined_power, f)
        
        # Plot results
        plot_tfr_single_subject(power_slow, power_fast, combined_power, subject)
        
        print(f"Successfully processed subject {subject}")
        return power_slow, power_fast, combined_power
        
    except Exception as e:
        print(f"Error processing subject {subject}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_tfr_single_subject(power_slow, power_fast, combined_power, subject):
    """Plot TFR results for a single subject"""
    subject_dir = f"{OUTPUT_DIR}/sub-{subject}"
        
    # Plot slow oscillations TFR
    fig = plt.figure(figsize=(10, 6))
    power_slow.plot(combine='mean', baseline=[-0.4,-0.250], mode="percent",
              title=f"Subject {subject} - Slow Oscillations (2-30 Hz) - TF Power",
              tmin=-0.5, tmax=4.5,
              vmin=-0.5, vmax=0.5,
              cmap='RdBu_r',
              show=False)
    
    # Add vertical lines for important time points
    plt.axvline(x=0, color='white', linestyle='--', linewidth=1, label='Stimulus Onset')
    plt.axvline(x=2, color='white', linestyle=':', linewidth=1, label='Delay Onset')
    plt.legend(loc='upper right')
    
    plt.savefig(f"{subject_dir}/tfr_slow_oscillations.png")
    plt.close()
    
    # Plot fast oscillations TFR
    fig = plt.figure(figsize=(10, 6))
    power_fast.plot(combine='mean', baseline=[-0.4,-0.250], mode="percent",
              title=f"Subject {subject} - Fast Oscillations (30-100 Hz) - TF Power",
              tmin=-0.5, tmax=4.5,
              vmin=-0.5, vmax=0.5,
              cmap='RdBu_r',
              show=False)
    
    # Add vertical lines for important time points
    plt.axvline(x=0, color='white', linestyle='--', linewidth=1, label='Stimulus Onset')
    plt.axvline(x=2, color='white', linestyle=':', linewidth=1, label='Delay Onset')
    plt.legend(loc='upper right')
    
    plt.savefig(f"{subject_dir}/tfr_fast_oscillations.png")
    plt.close()
    
    # Plot combined oscillations TFR
    fig = plt.figure(figsize=(10, 6))
    combined_power.plot(combine='mean', baseline=[-0.4,-0.250], mode="percent",
                  title=f"Subject {subject} - Combined Oscillations (2-100 Hz) - TF Power",
                  tmin=-0.5, tmax=4.5,
                  vmin=-0.5, vmax=0.5,
                  cmap='RdBu_r',
                  show=False)
    
    # Add vertical lines for important time points
    plt.axvline(x=0, color='white', linestyle='--', linewidth=1, label='Stimulus Onset')
    plt.axvline(x=2, color='white', linestyle=':', linewidth=1, label='Delay Onset')
    plt.legend(loc='upper right')
    
    plt.savefig(f"{subject_dir}/tfr_combined_oscillations.png")
    plt.close()

    # Plot frequency band power over time
    plot_frequency_bands(combined_power, subject, subject_dir)
    
    # Plot log frequency scale TFR (added)
    plot_log_frequency_tfr(combined_power, subject_dir, f"Subject {subject}")
    
    # Get channel regions for CTF sensors
    regions = get_ctf_regions(combined_power.info)
    
    # Create regional plots for each frequency band
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        # Plot regional topomaps for this frequency band
        plot_regional_topomaps(
            combined_power, 
            regions, 
            band_name, 
            fmin, 
            fmax, 
            subject_dir, 
            f"Subject {subject}"
        )
        
        # Plot regional comparison for this frequency band
        plot_regional_comparison(
            combined_power, 
            regions, 
            band_name, 
            fmin, 
            fmax, 
            subject_dir, 
            f"Subject {subject}"
        )

def plot_frequency_bands(combined_power, subject, output_dir):
    """Plot frequency band power over time for a subject"""
    plt.figure(figsize=(10, 6))
    
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        # Find frequencies within this band
        freq_mask = (combined_power.freqs >= fmin) & (combined_power.freqs <= fmax)
        
        if not np.any(freq_mask):
            print(f"Warning: No frequencies in the {band_name} band ({fmin}-{fmax} Hz)")
            continue
        temp_tfr = combined_power.copy()
        temp_tfr.apply_baseline(baseline=[-0.4, -0.25], mode='percent')
        # Then average the baseline-corrected data
        band_power = np.mean(temp_tfr.data[:, freq_mask, :], axis=(0, 1))

        # Plot band power over time
        plt.plot(combined_power.times, band_power,
                label=f"{band_name} ({fmin}-{fmax} Hz)",
                linewidth=2)
    
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='r', linestyle='-', label='Stimulus Onset')
    plt.axvline(2, color='g', linestyle='-', label='Delay Start')
    
    # Highlight encoding and delay periods
    plt.axvspan(0, 2, color='lightblue', alpha=0.2, label='Encoding')
    plt.axvspan(2, 4.5, color='lightgreen', alpha=0.2, label='Delay')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Power (% change)')
    plt.title(f'Subject {subject} - Frequency Band Power Over Time')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/band_power_over_time.png")
    plt.close()

def compute_grand_average():
    """Compute grand average across all processed subjects using combined TFR results"""
    print("Computing grand average TFR...")
    
    # Find all subject combined TFR files
    import glob
    combined_tfr_files = sorted(glob.glob(f"{OUTPUT_DIR}/sub-*/tfr_power_combined.pkl"))
    print(f"Found {len(combined_tfr_files)} subject combined TFR files")
    
    if len(combined_tfr_files) == 0:
        print("No subject combined TFR files found. Run single-subject processing first.")
        return None
    
    # Load TFR data from all subjects
    all_subject_tfrs = []
    valid_subjects = []
    
    for tfr_file in tqdm(combined_tfr_files, desc="Loading subject data"):
        try:
            with open(tfr_file, 'rb') as f:
                power = pickle.load(f)
            
            # Extract subject ID from file path
            subject = int(tfr_file.split('sub-')[1].split('/')[0])
            
            all_subject_tfrs.append(power)
            valid_subjects.append(subject)
            print(f"Loaded combined TFR data for subject {subject}")
        except Exception as e:
            print(f"Error loading {tfr_file}: {str(e)}")
    
    if len(all_subject_tfrs) == 0:
        print("No valid subject TFR data could be loaded")
        return None
    
    # Compute grand average
    print(f"Computing grand average from {len(valid_subjects)} subjects")
    grand_avg = mne.grand_average(all_subject_tfrs)
    
    # Save grand average result
    with open(f"{OUTPUT_DIR}/grand_avg_tfr.pkl", 'wb') as f:
        pickle.dump(grand_avg, f)
    
    # Plot grand average results
    plot_tfr_grand_average(grand_avg, len(valid_subjects))

    # Also create the frequency bands over time plot
    plot_frequency_bands_over_time(grand_avg, len(valid_subjects))
    
    print("Grand average completed successfully!")
    return grand_avg

def plot_frequency_bands_over_time(grand_avg, n_subjects):
    """
    Plot the grand average time course for all frequency bands (2-100 Hz),
    showing percent change over time across all subjects between -0.5s and 4.5s.
    """
    # First ensure we're working with baseline-corrected data
    display_grand_avg = grand_avg.copy()
    display_grand_avg.apply_baseline(baseline=[-0.4, -0.25], mode="percent")
    
    # Crop to our time window of interest to avoid edge effects
    display_grand_avg.crop(tmin=-0.4, tmax=4.5)
    
    # Plot individual frequency bands
    plt.figure(figsize=(12, 7))
    
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        # Find frequencies within this band
        band_freq_mask = (display_grand_avg.freqs >= fmin) & (display_grand_avg.freqs <= fmax)
        
        if not np.any(band_freq_mask):
            print(f"Warning: No frequencies in the {band_name} band ({fmin}-{fmax} Hz)")
            continue
        
        # Average power across frequencies in the band and across all channels
        band_power = np.mean(display_grand_avg.data[:, band_freq_mask, :], axis=(0, 1))
        
        # Plot band power over time
        plt.plot(display_grand_avg.times, band_power, 
                label=f"{band_name} ({fmin}-{fmax} Hz)",
                linewidth=2)
    
    # Add annotations
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='r', linestyle='-', label='Stimulus Onset')
    plt.axvline(2, color='g', linestyle='-', label='Delay Start')
    plt.axvspan(0, 2, color='lightblue', alpha=0.2, label='Encoding')
    plt.axvspan(2, 4.5, color='lightgreen', alpha=0.2, label='Delay')
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Power (% change from baseline)', fontsize=12)
    plt.title(f'Grand Average Frequency Bands (N={n_subjects})', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(-0.5, 5.0, 0.5))
    plt.savefig(f"{OUTPUT_DIR}/all_bands_grand_avg_time_course.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/all_bands_grand_avg_time_course.pdf")
    plt.close()

def plot_tfr_grand_average(grand_avg, n_subjects):
    """Plot grand average TFR results"""
    # Create a display version cropped to our actual interest period
    display_grand_avg = grand_avg.copy().crop(tmin=TMIN, tmax=DISPLAY_TMAX)
    
    # Plot average across all channels
    fig = plt.figure(figsize=(12, 8))
    display_grand_avg.plot(combine='mean', baseline=[-0.4,-0.250], mode="percent",
                  title=f"Grand Average (N={n_subjects}) - TF Power (% change)",
                  tmin=-0.5, tmax=4.5,
                  vmin=-0.5, vmax=0.5,
                  cmap='RdBu_r',
                  show=False)
    plt.axvline(x=0, color='white', linestyle='--', linewidth=2)  # Mark stimulus onset
    plt.axvline(x=2, color='white', linestyle=':', linewidth=2)   # Mark delay onset
    plt.savefig(f"{OUTPUT_DIR}/grand_avg_tfr_all_channels.png")
    plt.savefig(f"{OUTPUT_DIR}/grand_avg_tfr_all_channels.pdf")
    plt.close()
    
    # Plot topographies for frequency bands across time windows
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        # Define time windows: baseline, encoding, and delay
        time_windows = [
            (-0.5, 0),    # Baseline
            (0, 2),       # Encoding
            (2, 4.5)      # Delay (updated to match experiment timing)
        ]
        window_names = ["Baseline", "Encoding", "Delay"]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, (t_min, t_max) in enumerate(time_windows):
            display_grand_avg.plot_topomap(
                ch_type='mag', 
                tmin=t_min, tmax=t_max,
                fmin=fmin, fmax=fmax,
                baseline=[-0.4,-0.250],
                mode='percent',
                vmin=-0.5, vmax=0.5,
                title=f"{window_names[i]} ({t_min}-{t_max}s)",
                axes=axes[i],
                show=False
            )
        
        plt.suptitle(f"Grand Average (N={n_subjects}) - {band_name} band ({fmin}-{fmax} Hz)")
        plt.savefig(f"{OUTPUT_DIR}/{band_name}_grand_avg_topomap.png")
        plt.savefig(f"{OUTPUT_DIR}/{band_name}_grand_avg_topomap.pdf")
        plt.close()
    
    # Plot frequency band power over time
    plot_frequency_bands(display_grand_avg, f"Grand Average (N={n_subjects})", OUTPUT_DIR)
    
    # Plot log frequency scale TFR (added)
    plot_log_frequency_tfr(display_grand_avg, OUTPUT_DIR, f"Grand Average (N={n_subjects})")
    
    # Get channel regions for CTF sensors
    regions = get_ctf_regions(display_grand_avg.info)
    
    # Create regional plots for each frequency band
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        # Plot regional topomaps for this frequency band
        plot_regional_topomaps(
            display_grand_avg, 
            regions, 
            band_name, 
            fmin, 
            fmax, 
            OUTPUT_DIR, 
            f"Grand Average (N={n_subjects})"
        )
        
        # Plot regional comparison for this frequency band
        plot_regional_comparison(
            display_grand_avg, 
            regions, 
            band_name, 
            fmin, 
            fmax, 
            OUTPUT_DIR, 
            f"Grand Average (N={n_subjects})"
        )

def compute_grand_average_slow():
    """Compute grand average across all processed subjects using SLOW TFR results"""
    print("Computing grand average of slow oscillations TFR...")
    
    # Find all subject slow TFR files
    import glob
    slow_tfr_files = sorted(glob.glob(f"{OUTPUT_DIR}/sub-*/tfr_power_slow.pkl"))
    print(f"Found {len(slow_tfr_files)} subject slow TFR files")
    
    if len(slow_tfr_files) == 0:
        print("No subject slow TFR files found. Run single-subject processing first.")
        return None
    
    # Load TFR data from all subjects
    all_subject_tfrs = []
    valid_subjects = []
    
    for tfr_file in tqdm(slow_tfr_files, desc="Loading subject data"):
        try:
            with open(tfr_file, 'rb') as f:
                power = pickle.load(f)
            
            # Extract subject ID from file path
            subject = int(tfr_file.split('sub-')[1].split('/')[0])
            
            all_subject_tfrs.append(power)
            valid_subjects.append(subject)
            print(f"Loaded slow TFR data for subject {subject}")
        except Exception as e:
            print(f"Error loading {tfr_file}: {str(e)}")
    
    if len(all_subject_tfrs) == 0:
        print("No valid subject TFR data could be loaded")
        return None
    
    # Compute grand average
    print(f"Computing slow oscillations grand average from {len(valid_subjects)} subjects")
    grand_avg = mne.grand_average(all_subject_tfrs)
    
    # Save grand average result
    with open(f"{OUTPUT_DIR}/grand_avg_slow_tfr.pkl", 'wb') as f:
        pickle.dump(grand_avg, f)
    
    # Plot grand average results
    plot_tfr_grand_average_slow(grand_avg, len(valid_subjects))
    
    print("Slow oscillations grand average completed successfully!")
    return grand_avg

def plot_tfr_grand_average_slow(grand_avg, n_subjects):
    """Plot grand average TFR results for slow oscillations"""
    # Create a display version cropped to our actual interest period
    display_grand_avg = grand_avg.copy().crop(tmin=TMIN, tmax=DISPLAY_TMAX)
    
    # Plot average across all channels
    fig = plt.figure(figsize=(12, 8))
    display_grand_avg.plot(combine='mean', baseline=[-0.4,-0.250], mode="percent",
                  title=f"Grand Average Slow Oscillations (N={n_subjects}) - TF Power (% change)",
                  tmin=-0.5, tmax=4.5,
                  vmin=-0.5, vmax=0.5,
                  cmap='RdBu_r',
                  show=False)
    plt.axvline(x=0, color='white', linestyle='--', linewidth=2)  # Mark stimulus onset
    plt.axvline(x=2, color='white', linestyle=':', linewidth=2)   # Mark delay onset
    plt.savefig(f"{OUTPUT_DIR}/grand_avg_slow_tfr_all_channels.png")
    plt.savefig(f"{OUTPUT_DIR}/grand_avg_slow_tfr_all_channels.pdf")
    plt.close()
    
    # Plot frequency band power over time (only including bands in the slow range)
    plot_frequency_bands_slow(display_grand_avg, f"Grand Average Slow Oscillations (N={n_subjects})", OUTPUT_DIR)
    
    # Get channel regions for CTF sensors 
    regions = get_ctf_regions(display_grand_avg.info)
    
    # Filter for only slow oscillation bands
    slow_freq_bands = {k: v for k, v in FREQ_BANDS.items() if v[0] >= 2 and v[1] <= 30}
    
    # Create regional plots for each slow frequency band
    for band_name, (fmin, fmax) in slow_freq_bands.items():
        # Plot regional topomaps for this frequency band
        plot_regional_topomaps(
            display_grand_avg, 
            regions, 
            band_name, 
            fmin, 
            fmax, 
            OUTPUT_DIR, 
            f"Grand Average Slow (N={n_subjects})"
        )
        
        # Plot regional comparison for this frequency band
        plot_regional_comparison(
            display_grand_avg, 
            regions, 
            band_name, 
            fmin, 
            fmax, 
            OUTPUT_DIR, 
            f"Grand Average Slow (N={n_subjects})"
        )

def plot_frequency_bands_slow(combined_power, subject, output_dir):
    """Plot frequency band power over time for a subject (slow oscillations only)"""
    plt.figure(figsize=(10, 6))
    
    # Filter for only slow oscillation bands
    slow_freq_bands = {k: v for k, v in FREQ_BANDS.items() if v[0] >= 2 and v[1] <= 30}
    
    for band_name, (fmin, fmax) in slow_freq_bands.items():
        # Find frequencies within this band
        freq_mask = (combined_power.freqs >= fmin) & (combined_power.freqs <= fmax)
        
        if not np.any(freq_mask):
            print(f"Warning: No frequencies in the {band_name} band ({fmin}-{fmax} Hz)")
            continue
        temp_tfr = combined_power.copy()
        temp_tfr.apply_baseline(baseline=[-0.4, -0.25], mode='percent')
        # Then average the baseline-corrected data
        band_power = np.mean(temp_tfr.data[:, freq_mask, :], axis=(0, 1))

        # Plot band power over time
        plt.plot(combined_power.times, band_power, 
                label=f"{band_name} ({fmin}-{fmax} Hz)",
                linewidth=2)
    
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='r', linestyle='-', label='Stimulus Onset')
    plt.axvline(2, color='g', linestyle='-', label='Delay Start')
    
    # Highlight encoding and delay periods
    plt.axvspan(0, 2, color='lightblue', alpha=0.2, label='Encoding')
    plt.axvspan(2, 4.5, color='lightgreen', alpha=0.2, label='Delay')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Power (% change)')
    plt.title(f'Slow Oscillations {subject} - Frequency Band Power Over Time')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/slow_band_power_over_time.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multitaper TF analysis for single subject or grand average')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--subject', type=int, help='Process a single subject with this ID')
    group.add_argument('--grand-average', action='store_true', help='Compute grand average across all processed subjects')
    group.add_argument('--grand-average-slow', action='store_true', help='Compute grand average of SLOW oscillations across all processed subjects')
    
    args = parser.parse_args()
    
    if args.subject:
        process_single_subject(args.subject)
    elif args.grand_average:
        compute_grand_average()
    elif args.grand_average_slow:
        compute_grand_average_slow()