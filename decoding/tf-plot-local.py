import numpy as np
import os
import mne
import pickle
from mne.time_frequency import tfr_morlet
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import glob

# Configuration parameters
HOME_DIR = '/mnt/hpc/projects/awm4/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
OUTPUT_DIR = PROCESSED_DIR + '/TFR_SimpleAnalysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Analysis parameters
BASELINE = (-0.5, -0.25)  # Using -500 to -250ms baseline

# Frequency bands for visualization
FREQ_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'low_gamma': (30, 45),
    'high_gamma': (60, 100)
}

def create_percentage_plots_for_subject(subject):
    """Create percentage plots for a single subject"""
    subject_dir = f"{OUTPUT_DIR}/sub-{subject}"
    tfr_file = f"{subject_dir}/tfr_power.pkl"
    
    if not os.path.exists(tfr_file):
        print(f"TFR file not found for subject {subject}")
        return False
    
    try:
        print(f"Loading TFR data for subject {subject}")
        with open(tfr_file, 'rb') as f:
            power = pickle.load(f)
        
        # Create manual TF plot
        create_tf_plot(power, subject)
        
        # Create band power plot
        create_band_power_plot(power, subject)
        
        print(f"Successfully created percentage plots for subject {subject}")
        return True
    
    except Exception as e:
        print(f"Error creating plots for subject {subject}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_tf_plot(power, subject):
    """Create time-frequency plot with percentage change"""
    subject_dir = f"{OUTPUT_DIR}/sub-{subject}"
    
    # Extract data and convert to percentage
    data = power.data.copy()  # Should have shape (n_channels, n_freqs, n_times)
    
    # Average across channels
    data_avg = np.mean(data, axis=0)
    data_avg = (np.exp(data_avg) - 1) * 100
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot TF data as image
    extent = [power.times[0], power.times[-1], power.freqs[0], power.freqs[-1]]
    plt.imshow(data_avg, aspect='auto', origin='lower', extent=extent, 
              cmap='RdBu_r', vmin=-80, vmax=80)
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('% change from baseline')
    
    # Set y-axis to log scale
    plt.yscale('log')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(f"Subject {subject} - TF Power (% change from baseline)")
    
    # Add vertical lines for important time points
    plt.axvline(x=0, color='white', linestyle='--', linewidth=1, label='Stimulus Onset')
    plt.axvline(x=2, color='white', linestyle=':', linewidth=1, label='Delay Onset')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{subject_dir}/tfr_all_channels_avg_percent.png")
    plt.close()

def create_band_power_plot(power, subject):
    """Create frequency band power over time plot"""
    subject_dir = f"{OUTPUT_DIR}/sub-{subject}"
    
    plt.figure(figsize=(10, 6))
    
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        # Find frequencies within this band
        freq_mask = (power.freqs >= fmin) & (power.freqs <= fmax)
        
        if not np.any(freq_mask):
            continue
        
        # Average power across frequencies in the band and across all channels
        band_power = np.mean(power.data[:, freq_mask, :], axis=(0, 1))
        
        band_power = (np.exp(band_power) - 1) * 100
        
        # Plot band power over time
        plt.plot(power.times, band_power,  
                label=f"{band_name} ({fmin}-{fmax} Hz)",
                linewidth=2)
    
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='r', linestyle='-', label='Stimulus Onset')
    plt.axvline(2, color='g', linestyle='-', label='Delay Start')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (% change from baseline)')
    plt.title(f'Subject {subject} - Frequency Band Power Over Time')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim([-80, 80])  # Consistent y-axis limits
    plt.tight_layout()
    plt.savefig(f"{subject_dir}/band_power_over_time_percent.png")
    plt.close()

def create_percentage_plots_for_grand_average():
    """Create percentage plots for grand average"""
    grand_avg_file = f"{OUTPUT_DIR}/grand_avg_tfr.pkl"
    
    if not os.path.exists(grand_avg_file):
        print(f"Grand average file not found: {grand_avg_file}")
        return False
    
    try:
        print("Loading grand average data")
        with open(grand_avg_file, 'rb') as f:
            power = pickle.load(f)
        
        # Count subjects
        subject_dirs = glob.glob(f"{OUTPUT_DIR}/sub-*")
        n_subjects = len(subject_dirs)
        
        # Create TF plot
        create_grand_avg_tf_plot(power, n_subjects)
        
        # Create band power plot
        create_grand_avg_band_power_plot(power, n_subjects)
        
        print("Successfully created percentage plots for grand average")
        return True
    
    except Exception as e:
        print(f"Error creating plots for grand average: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_grand_avg_tf_plot(power, n_subjects):
    """Create time-frequency plot for grand average"""
    # Extract data and convert to percentage
    data = power.data.copy()  # Should have shape (n_channels, n_freqs, n_times)
    
    # Average across channels
    data_avg = np.mean(data, axis=0)
    data_avg = (np.exp(data_avg) - 1) * 100

    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot TF data as image
    extent = [power.times[0], power.times[-1], power.freqs[0], power.freqs[-1]]
    plt.imshow(data_avg, aspect='auto', origin='lower', extent=extent, 
              cmap='RdBu_r', vmin=-80, vmax=80)
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('% change from baseline')
    
    # Set y-axis to log scale
    plt.yscale('log')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(f"Grand Average (N={n_subjects}) - TF Power (% change from baseline)")
    
    # Add vertical lines for important time points
    plt.axvline(x=0, color='white', linestyle='--', linewidth=2, label='Stimulus Onset')
    plt.axvline(x=2, color='white', linestyle=':', linewidth=2, label='Delay Onset')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/grand_avg_tfr_all_channels_percent.png")
    plt.savefig(f"{OUTPUT_DIR}/grand_avg_tfr_all_channels_percent.pdf")
    plt.close()

def create_grand_avg_band_power_plot(power, n_subjects):
    """Create band power plot for grand average"""
    plt.figure(figsize=(10, 8))
    
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        # Find frequencies within this band
        freq_mask = (power.freqs >= fmin) & (power.freqs <= fmax)
        
        if not np.any(freq_mask):
            print(f"Warning: No frequencies in the {band_name} band ({fmin}-{fmax} Hz)")
            continue
        
        # Average power across frequencies in the band and across all channels
        band_power = np.mean(power.data[:, freq_mask, :], axis=(0, 1))
        band_power = (np.exp(band_power) - 1) * 100

        # Plot band power over time
        plt.plot(power.times, band_power,  # Multiply by 100 to convert to percent
                label=f"{band_name} ({fmin}-{fmax} Hz)",
                linewidth=2)
    
    plt.axhline(0, color='k', linestyle='--', label='Baseline')
    plt.axvline(0, color='r', linestyle='-', label='Stimulus Onset')
    plt.axvline(2, color='g', linestyle='-', label='Delay Start')
    
    # Highlight encoding and delay periods
    plt.axvspan(0, 2, color='lightblue', alpha=0.2, label='Encoding')
    plt.axvspan(2, 4, color='lightgreen', alpha=0.2, label='Delay')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Power (% change from baseline)')
    plt.title(f'Grand Average (N={n_subjects}) - Frequency Band Power Over Time')
    plt.legend(loc='upper right', title='Frequency Band')
    plt.grid(True, alpha=0.3)
    plt.ylim([-80, 80])  # Consistent y-axis limits
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/band_power_over_time_percent.png")
    plt.savefig(f"{OUTPUT_DIR}/band_power_over_time_percent.pdf")
    plt.close()

def process_all_subjects():
    """Process all subjects and create percentage plots"""
    # Find all subject directories
    subject_dirs = sorted(glob.glob(f"{OUTPUT_DIR}/sub-*"))
    
    print(f"Found {len(subject_dirs)} subject directories")
    
    success_count = 0
    for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
        subject = int(subject_dir.split('sub-')[1])
        if create_percentage_plots_for_subject(subject):
            success_count += 1
    
    print(f"Successfully created percentage plots for {success_count}/{len(subject_dirs)} subjects")
    
    # Create grand average plots
    create_percentage_plots_for_grand_average()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create percentage plots for TFR data')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--subject', type=int, help='Process a single subject with this ID')
    group.add_argument('--grand-average', action='store_true', help='Process only grand average')
    group.add_argument('--all-subjects', action='store_true', help='Process all subjects and grand average')
    
    args = parser.parse_args()
    
    if args.subject:
        create_percentage_plots_for_subject(args.subject)
    elif args.grand_average:
        create_percentage_plots_for_grand_average()
    elif args.all_subjects:
        process_all_subjects()