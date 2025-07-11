import numpy as np
import os
import mne
import pandas as pd
import pickle
from mne.time_frequency import tfr_morlet
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from scipy import signal

# Configuration parameters
HOME_DIR = '/mnt/hpc/projects/awm4/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
OUTPUT_DIR = PROCESSED_DIR + '/TFR_Analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Analysis parameters
RESAMPLE_FREQ = 200  # Hz
NUM_JOBS = 8
BASELINE = (-0.5, -0.3)  # Full baseline period
TMIN = -0.5  # Include baseline period
TMAX = 5.5   # Extended to avoid edge effects
DISPLAY_TMAX = 4.5  # Actual experiment data up to 4.5s

# Frequency bands for visualization
FREQ_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'low_gamma': (30, 45),
    'high_gamma': (60, 100)
}

def apply_hanning_window(epochs):
    """Apply Hanning window to epochs data to reduce edge effects"""
    print("Applying Hanning window to reduce edge effects...")
    
    # Get data and create Hanning window matching the time dimension
    data = epochs.get_data(copy=True)  # Explicitly set copy=True
    window = signal.windows.hann(data.shape[-1])  # Use windows.hann instead of hann
    
    # Apply window along the time dimension (preserving trials and channels)
    windowed_data = data * window[None, None, :]
    
    # Create new epochs with windowed data
    windowed_epochs = mne.EpochsArray(
        windowed_data, 
        epochs.info, 
        tmin=epochs.times[0],
        events=epochs.events
    )
    
    return windowed_epochs

def process_single_subject(subject):
    """Process TF analysis for a single subject"""
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
        
        # Apply Hanning window to reduce edge effects
        windowed_epochs = apply_hanning_window(mag_epochs)
        
        # Define frequency parameters
        freqs = np.logspace(np.log10(4), np.log10(100), 30)  # Log-spaced frequencies
        n_cycles = freqs / 2  # Frequency-dependent cycles
        
        # Compute time-frequency representation
        power = tfr_morlet(
            windowed_epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            return_itc=False,
            average=True,
            decim=1,
            n_jobs=NUM_JOBS
        )
        
        # Apply baseline correction
        power.apply_baseline(BASELINE, mode='logratio')
        
        # Save the TFR result for this subject
        with open(f"{subject_dir}/tfr_power.pkl", 'wb') as f:
            pickle.dump(power, f)
        
        # Plot and save individual subject results
        plot_tfr_single_subject(power, subject)
        
        print(f"Successfully processed subject {subject}")
        return power
        
    except Exception as e:
        print(f"Error processing subject {subject}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_tfr_single_subject(power, subject):
    """Plot TFR results for a single subject"""
    subject_dir = f"{OUTPUT_DIR}/sub-{subject}"
    
    # Create a display version cropped to our actual interest period
    display_power = power.copy().crop(tmin=TMIN, tmax=DISPLAY_TMAX)
    
    # Plot TFR with power averaged across all sensors
    fig = plt.figure(figsize=(10, 6))
    display_power.plot(combine='mean', baseline=None, mode='percent',
              title=f"Subject {subject} - TF Power (% change from baseline)",
              vmin=-50, vmax=50,
              cmap='RdBu_r',
              show=False)
    
    # Add vertical lines for important time points
    plt.axvline(x=0, color='white', linestyle='--', linewidth=1, label='Stimulus Onset')
    plt.axvline(x=2, color='white', linestyle=':', linewidth=1, label='Delay Onset')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{subject_dir}/tfr_all_channels_avg.png")
    plt.close()
        
    # Plot frequency band power over time
    plt.figure(figsize=(10, 6))
    
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        # Find frequencies within this band
        freq_mask = (display_power.freqs >= fmin) & (display_power.freqs <= fmax)
        
        if not np.any(freq_mask):
            continue
        
        # Average power across frequencies in the band and across all channels
        band_power = np.mean(display_power.data[:, freq_mask, :], axis=(0, 1))
        
        # Plot band power over time
        plt.plot(display_power.times, band_power, 
                label=f"{band_name} ({fmin}-{fmax} Hz)",
                linewidth=2)
    
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='r', linestyle='-', label='Stimulus Onset')
    plt.axvline(2, color='g', linestyle='-', label='Delay Start')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (% change)')
    plt.title(f'Subject {subject} - Frequency Band Power Over Time')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{subject_dir}/band_power_over_time.png")
    plt.close()

def compute_grand_average():
    """Compute grand average across all processed subjects"""
    print("Computing grand average TFR...")
    
    # Find all subject TFR files
    import glob
    tfr_files = sorted(glob.glob(f"{OUTPUT_DIR}/sub-*/tfr_power.pkl"))
    print(f"Found {len(tfr_files)} subject TFR files")
    
    if len(tfr_files) == 0:
        print("No subject TFR files found. Run single-subject processing first.")
        return None
    
    # Load TFR data from all subjects
    all_subject_tfrs = []
    valid_subjects = []
    
    for tfr_file in tqdm(tfr_files, desc="Loading subject data"):
        try:
            with open(tfr_file, 'rb') as f:
                power = pickle.load(f)
            
            # Extract subject ID from file path
            subject = int(tfr_file.split('sub-')[1].split('/')[0])
            
            all_subject_tfrs.append(power)
            valid_subjects.append(subject)
            print(f"Loaded TFR data for subject {subject}")
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
    
    print("Grand average completed successfully!")
    return grand_avg

def plot_tfr_grand_average(grand_avg, n_subjects):
    """Plot grand average TFR results"""
    # Create a display version cropped to our actual interest period
    display_grand_avg = grand_avg.copy().crop(tmin=TMIN, tmax=DISPLAY_TMAX)
    
    # Plot average across all channels
    fig = plt.figure(figsize=(12, 8))
    display_grand_avg.plot(combine='mean', baseline=None, mode='percent',
                  title=f"Grand Average (N={n_subjects}) - TF Power (% change from baseline)",
                  vmin=-50, vmax=50,
                  cmap='RdBu_r',
                  show=False)
    plt.axvline(x=0, color='white', linestyle='--', linewidth=2)  # Mark stimulus onset
    plt.axvline(x=2, color='white', linestyle=':', linewidth=2)   # Mark delay onset
    plt.tight_layout()
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
                baseline=None,
                mode='percent',
                vmin=-50, vmax=50,
                title=f"{window_names[i]} ({t_min}-{t_max}s)",
                axes=axes[i],
                show=False
            )
        
        plt.suptitle(f"Grand Average (N={n_subjects}) - {band_name} band ({fmin}-{fmax} Hz)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{band_name}_grand_avg_topomap.png")
        plt.savefig(f"{OUTPUT_DIR}/{band_name}_grand_avg_topomap.pdf")
        plt.close()
    
    # Plot frequency band power over time
    plt.figure(figsize=(10, 8))
    
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        # Find frequencies within this band
        freq_mask = (display_grand_avg.freqs >= fmin) & (display_grand_avg.freqs <= fmax)
        
        if not np.any(freq_mask):
            print(f"Warning: No frequencies in the {band_name} band ({fmin}-{fmax} Hz)")
            continue
        
        # Average power across frequencies in the band and across all channels
        band_power = np.mean(display_grand_avg.data[:, freq_mask, :], axis=(0, 1))
        
        # Plot band power over time
        plt.plot(display_grand_avg.times, band_power, 
                label=f"{band_name} ({fmin}-{fmax} Hz)",
                linewidth=2)
    
    plt.axhline(0, color='k', linestyle='--', label='Baseline')
    plt.axvline(0, color='r', linestyle='-', label='Stimulus Onset')
    plt.axvline(2, color='g', linestyle='-', label='Delay Start')
    
    # Highlight encoding and delay periods
    plt.axvspan(0, 2, color='lightblue', alpha=0.2, label='Encoding')
    plt.axvspan(2, 4.5, color='lightgreen', alpha=0.2, label='Delay')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Power (% change)')
    plt.title(f'Grand Average (N={n_subjects}) - Frequency Band Power Over Time')
    plt.legend(loc='upper right', title='Frequency Band')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/band_power_over_time.png")
    plt.savefig(f"{OUTPUT_DIR}/band_power_over_time.pdf")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TF analysis for single subject or grand average')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--subject', type=int, help='Process a single subject with this ID')
    group.add_argument('--grand-average', action='store_true', help='Compute grand average across all processed subjects')
    
    args = parser.parse_args()
    
    if args.subject:
        process_single_subject(args.subject)
    elif args.grand_average:
        compute_grand_average()