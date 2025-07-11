import numpy as np
import os
import mne
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Using the paths from your previous code
HOME_DIR = '/media/headmodel/Elements/AWM4/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
DELAY_CONFIG = {
    'tmin': 2.0,
    'tmax': 4.7,
}

# Create directory for PSD plots
PSD_DIR = f"{PROCESSED_DIR}/PSD_Plots"
os.makedirs(PSD_DIR, exist_ok=True)

def compute_and_plot_psd(subject=None, fmin=4.0, fmax=100.0):
    """
    Compute and plot power spectral densities from epoch data.
    
    Parameters:
    -----------
    subject : str or None
        If provided, process only this subject. If None, process all subjects.
    fmin : float
        Minimum frequency to include.
    fmax : float
        Maximum frequency to include.
    """
    # Get all subjects if not specified
    if subject is None:
        epochs_files = glob.glob(f"{PROCESSED_DIR}/CutEpochs/CutData_VP*-cleanedICA-epo.fif")
        subjects = []
        for file in epochs_files:
            basename = os.path.basename(file)
            subject_id = basename.split('VP')[1].split('-')[0]
            subjects.append(subject_id)
        
        subjects = sorted(set(subjects))
        print(f"Found {len(subjects)} subjects: {subjects}")
    else:
        subjects = [subject]
    
    # Process each subject individually
    for subject_id in subjects:
        print(f"Processing subject {subject_id}...")
        
        # Load epochs data
        epochs_file = f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject_id}-cleanedICA-epo.fif"
        if not os.path.exists(epochs_file):
            print(f"File not found: {epochs_file}")
            continue
        
        try:
            epochs = mne.read_epochs(epochs_file, preload=True, verbose='WARNING')
        except Exception as e:
            print(f"Error loading epochs for subject {subject_id}: {e}")
            continue
        
        # Check for jump artifacts and drop affected epochs
        jname = f"{PROCESSED_DIR}/ICAs/Jumps{subject_id}.npy"
        if os.path.isfile(jname):
            jump_inds = np.load(jname)
            if len(jump_inds) > 0:
                jump_inds = np.array(jump_inds, dtype=int)
                valid_jump_inds = jump_inds[jump_inds < len(epochs)]
                if len(valid_jump_inds) > 0:
                    epochs.drop(valid_jump_inds, reason='jump')
        
        # Crop to delay period
        delay_epochs = epochs.copy().crop(tmin=DELAY_CONFIG['tmin'], tmax=DELAY_CONFIG['tmax'])
        
        # Create subject-specific directory for plots
        subject_dir = f"{PSD_DIR}/Subject_{subject_id}"
        os.makedirs(subject_dir, exist_ok=True)
        
        # Compute PSD
        print(f"Computing PSD for subject {subject_id}...")
        spectrum = delay_epochs.compute_psd(
            method='multitaper', 
            fmin=4, 
            fmax=100,
            picks='mag'
        )
        
        # Get the data and frequencies
        freqs = spectrum.freqs # Shape: (n_freqs,)
        psd_data = spectrum.get_data()  # Shape: (n_epochs, n_channels, n_freqs)
        
        # 1. Plot average across all sensors and epochs
        avg_psd = np.mean(psd_data, axis=(0, 1))  # Shape: (n_freqs,)
        
        plt.figure(figsize=(12, 7))
        plt.semilogy(freqs, avg_psd, color='black', linewidth=2)
        
        # Add frequency band annotations
        plt.axvspan(4, 8, alpha=0.2, color='green', label='Theta')
        plt.axvspan(8, 13, alpha=0.2, color='blue', label='Alpha')
        plt.axvspan(13, 30, alpha=0.2, color='red', label='Beta')
        plt.axvspan(30, 100, alpha=0.2, color='purple', label='Gamma')
        
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Power Spectral Density (µV²/Hz)', fontsize=12)
        plt.title(f'Subject {subject_id}: PSD During Delay Period (All Magnetometers)', fontsize=14)
        plt.xlim(fmin, fmax)
        plt.legend()
        
        # Customize y-axis formatting to show scientific notation
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        plt.savefig(f"{subject_dir}/All_Sensors_PSD.png", dpi=300)
        plt.close()
        
        # 2. Generate regional plots
        # -------------------------
        ch_names = spectrum.ch_names
        regions = {
            'Frontal': 'MLF',
            'Central': 'MLC',
            'Temporal': 'MLT',
            'Parietal': 'MLP',
            'Occipital': 'MLO',
            'Midline': 'MZ'
        }
        
        # Create a plot comparing all regions
        plt.figure(figsize=(14, 8))
        
        for region_name, region_prefix in regions.items():
            # Find channels for this region
            region_ch_indices = [i for i, name in enumerate(ch_names) if region_prefix in name]
            
            if region_ch_indices:
                # Average across EPOCHS first, then selected channels
                region_psd = np.mean(psd_data[:, region_ch_indices, :], axis=(0, 1))
                plt.semilogy(freqs, region_psd, linewidth=2.5, label=region_name)
        
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.ylabel('Power Spectral Density (µV²/Hz)', fontsize=14)
        plt.title(f'Subject {subject_id}: Regional Power Spectra', fontsize=16)
        plt.xlim(fmin, fmax)
        plt.legend(fontsize=12)
        
        # Add frequency band annotations
        plt.axvspan(4, 8, alpha=0.1, color='green')
        plt.axvspan(8, 13, alpha=0.1, color='blue')
        plt.axvspan(13, 30, alpha=0.1, color='red')
        plt.axvspan(30, 100, alpha=0.1, color='purple')
        
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        plt.savefig(f"{subject_dir}/Regional_Comparison.png", dpi=300)
        plt.close()
        
        # 3. Generate topomaps for frequency bands
        # ---------------------------------------
        try:
            fig = plt.figure(figsize=(15, 4))
            
            # Plot topomaps for each frequency band
            band_names = ['theta', 'alpha', 'beta']
            fmin_bands = [4, 8, 13]
            fmax_bands = [8, 13, 30]
            
            for i, (band, fmin_band, fmax_band) in enumerate(zip(band_names, fmin_bands, fmax_bands)):
                plt.subplot(1, 3, i+1)
                # Find frequency indices for this band
                band_indices = np.where((freqs >= fmin_band) & (freqs <= fmax_band))[0]
                
                # CRITICAL FIX: Average across EPOCHS first, then frequencies for each channel
                # This gives us the average power in this band for each channel
                band_power = np.mean(psd_data[:, :, band_indices], axis=(0, 2))
                
                # Create topomap
                mne.viz.plot_topomap(band_power, spectrum.info, axes=plt.gca(), 
                                    show=False, contours=0, sensors=True)
                plt.title(f"{band.capitalize()} ({fmin_band}-{fmax_band} Hz)")
            
            plt.tight_layout()
            plt.savefig(f"{subject_dir}/Frequency_Band_Topomaps.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error creating topomaps for subject {subject_id}: {e}")
        
        print(f"Completed processing for subject {subject_id}")
    
    print(f"PSD analysis completed. Results saved to {PSD_DIR}")

if __name__ == "__main__":
    # Run analysis with default parameters
    compute_and_plot_psd(fmin=4.0, fmax=100.0)