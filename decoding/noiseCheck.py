import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set paths
homeDir = '/media/headmodel/Elements/AWM4/'
data_path = homeDir + '/AWM4_data/raw/'
processed_path = homeDir + '/AWM4_data/processed/CutEpochs/'
metaFile = homeDir + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(metaFile)

# Get all subjects
Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])

# Create figure directory
fig_dir = homeDir + '/QC_figures/noise_removal/'
os.makedirs(fig_dir, exist_ok=True)

# Loop through all subjects
for subject_id in Subs:
    print(f"\n{'='*60}")
    print(f"Processing Subject {subject_id}")
    print('='*60)
    
    try:
        # Get file names from metadata
        sub_idx = np.where(metaInfo['Subject'] == subject_id)[0][0]
        raw_file = data_path + metaInfo['MEG_Name'].iloc[sub_idx]
        noise_file = data_path + metaInfo['Noise_Measurement'].iloc[sub_idx]
        
        # Load original raw data
        print("Loading original data...")
        raw_orig = mne.io.read_raw_ctf(raw_file, preload=False)
        
        # Load your saved preprocessed data
        print("Loading preprocessed data...")
        processed_file = processed_path + f'CutData_VP{subject_id}-raw.fif'
        raw_clean = mne.io.read_raw_fif(processed_file, preload=False)
        
        # Load empty room data
        print("Loading empty room data...")
        empty_room_raw = mne.io.read_raw_ctf(noise_file, preload=False)
        
        # Compute PSDs (only MEG channels)
        print("Computing PSDs...")
        meg_picks = mne.pick_types(raw_orig.info, meg=True)
        
        psd_orig, freqs = raw_orig.compute_psd(fmin=1, fmax=100, picks=meg_picks).get_data(return_freqs=True)
        psd_clean, _ = raw_clean.compute_psd(fmin=1, fmax=100, picks=meg_picks).get_data(return_freqs=True)
        psd_empty, _ = empty_room_raw.compute_psd(fmin=1, fmax=100, picks=meg_picks).get_data(return_freqs=True)
        
        # Average across channels
        psd_orig_mean = psd_orig.mean(axis=0)
        psd_clean_mean = psd_clean.mean(axis=0)
        psd_empty_mean = psd_empty.mean(axis=0)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Original vs Cleaned (full spectrum)
        ax1 = axes[0]
        ax1.semilogy(freqs, psd_orig_mean, 'b-', label='Original', alpha=0.7, linewidth=2)
        ax1.semilogy(freqs, psd_clean_mean, 'g-', label='Cleaned', alpha=0.7, linewidth=2)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('PSD (T²/Hz)')
        ax1.set_title('Original vs Cleaned')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Zoom on line noise frequencies
        ax2 = axes[1]
        ax2.semilogy(freqs, psd_orig_mean, 'b-', label='Original', alpha=0.7, linewidth=2)
        ax2.semilogy(freqs, psd_clean_mean, 'g-', label='Cleaned', alpha=0.7, linewidth=2)
        ax2.set_xlim([40, 110])
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('PSD (T²/Hz)')
        ax2.set_title('Zoom: 40-110 Hz (Line Noise Region)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # Mark noise frequencies
        for f in [50, 100]:
            ax2.axvline(f, color='r', linestyle='--', alpha=0.5)
        
        # Plot 3: Cleaned vs Empty room
        ax3 = axes[2]
        ax3.semilogy(freqs, psd_clean_mean, 'g-', label='Cleaned data', alpha=0.7, linewidth=2)
        ax3.semilogy(freqs, psd_empty_mean, 'r--', label='Empty room', alpha=0.7, linewidth=2)
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('PSD (T²/Hz)')
        ax3.set_title('Cleaned Data vs Empty Room')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f'Subject {subject_id}: Noise Removal Verification', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(fig_dir + f'Subject_{subject_id}_noise_removal.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print noise reduction summary
        print(f"\nNoise reduction for Subject {subject_id}:")
        noise_freqs = [50, 100, 150]
        for f_target in noise_freqs:
            idx = np.argmin(np.abs(freqs - f_target))
            orig_power = psd_orig_mean[idx]
            clean_power = psd_clean_mean[idx]
            reduction_percent = (1 - clean_power/orig_power) * 100
            print(f"  {f_target} Hz: {reduction_percent:.1f}% reduction")
            
    except Exception as e:
        print(f"ERROR with Subject {subject_id}: {str(e)}")
        continue

print(f"\n{'='*60}")
print(f"All figures saved to: {fig_dir}")
print('='*60)