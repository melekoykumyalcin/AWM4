import numpy as np
import os
import mne
from mne.time_frequency import tfr_morlet
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("morlet_processing.log"),
        logging.StreamHandler()
    ]
)

# Paths for HPC cluster
HOME_DIR = '/mnt/hpc/projects/awm4/'
#HOME_DIR = '/media/headmodel/Elements/AWM4/'

PROCESSED_DIR = HOME_DIR + 'AWM4_data/processed'
PLOT_DIR = f"{PROCESSED_DIR}/AllSensorsTFR"
os.makedirs(PLOT_DIR, exist_ok=True)
tfr_dir = f"{PROCESSED_DIR}/AllSensorsTFR/data"
os.makedirs(tfr_dir, exist_ok=True)

# Analysis parameters
RESAMPLE_FREQ = 200  # Hz
NUM_JOBS = 16  # CPU cores for MNE processing
N_PARALLEL = 30   # Number of subjects to process in parallel
NUM_SUBJECTS = 30
tmin = -0.5  # include baseline period
tmax = 6   # experiment lasts 5 seconds
DISPLAY_TMIN = 0
DISPLAY_TMAX = 5

# Frequency bands for visualization
FREQ_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'low_gamma': (30, 45),
    'high_gamma': (60, 100)
}

def process_subject(subject):
    """Process a single subject with continuous Morlet wavelet transform"""
    start_time = time.time()
    logging.info(f"Starting Morlet TFR analysis for Subject {subject}")
    
    # Create subject directory
    subject_plot_dir = f"{PLOT_DIR}/sub-{subject}"
    os.makedirs(subject_plot_dir, exist_ok=True)
    
    try:
        # Load epochs
        epochs_file = f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif"            
        epochs = mne.read_epochs(epochs_file, preload=True)
        logging.info(f"Loaded {len(epochs)} epochs for subject {subject}")
        
        # Check for jump artifacts and drop affected epochs
        jumps_file = f"{PROCESSED_DIR}/ICAs/Jumps{subject}.npy"
        if os.path.isfile(jumps_file):
            jump_inds = np.load(jumps_file)
            
            if len(jump_inds) > 0:
                jump_inds = np.array(jump_inds, dtype=int)
                valid_jump_inds = jump_inds[jump_inds < len(epochs)]
                
                if len(valid_jump_inds) > 0:
                    logging.info(f"Dropping {len(valid_jump_inds)} epochs with jump artifacts for subject {subject}")
                    epochs.drop(valid_jump_inds, reason='jump')
            
        # Mark bad channels
        if subject == 4 and 'MRO31-3609' in epochs.ch_names:
            epochs.info['bads'].append('MRO31-3609')
        elif subject == 26:
            for ch in ['MRO31-3609', 'MRO44-3609']:
                if ch in epochs.ch_names:
                    epochs.info['bads'].append(ch)

        # Process all sensors
        mag_epochs = epochs.copy().pick_types(meg='mag', exclude='bads')
        mag_epochs.crop(tmin=tmin, tmax=tmax)
        mag_epochs.filter(1, None)  # High-pass filter at 1 Hz
        mag_epochs = mag_epochs.resample(RESAMPLE_FREQ, npad='auto')
        
        # Create full logspace but remove frequencies within ±1 Hz of 50 Hz
        freqs_full = np.logspace(np.log10(4), np.log10(100), 27)
        freqs = freqs_full[(freqs_full < 49) | (freqs_full > 51)]  # Filter out 49-51 Hz
        n_cycles = 7  # Fixed 7-cycle Morlet wavelets (Kaiser 2025 paper)
        
        # Compute TFR with continuous Morlet wavelet transform
        logging.info(f"Computing Morlet wavelet transform with 7-cycle wavelets for subject {subject}...")
        tfr = tfr_morlet(
            mag_epochs, 
            freqs=freqs, 
            n_cycles=n_cycles,  
            use_fft=True, 
            return_itc=False, 
            average=False,
            decim=10,  # (200Hz/10 = 20Hz = 50ms steps) 
            n_jobs=NUM_JOBS,
            output='power'
        )
        
        # Apply baseline correction
        #tfr.apply_baseline(baseline=(-0.4, -0.25), mode='logratio')
        
        # Save TFR result
        tfr_fname = f"{tfr_dir}/sub-{subject}_tfr-nobaselinecor.h5"
        tfr.save(tfr_fname, overwrite=True)
        
        # Also save as pickle for compatibility with grand average code
        with open(f"{tfr_dir}/sub-{subject}_tfr.pkl", 'wb') as f:
            pickle.dump(tfr, f)
        
        # Plot results
        plot_tfr_single_subject(tfr, subject, subject_plot_dir)
        
        elapsed_time = time.time() - start_time
        logging.info(f"Completed Morlet TFR analysis for Subject {subject} in {elapsed_time:.2f} seconds")
        return subject, tfr
        
    except Exception as e:
        logging.error(f"Error processing subject {subject}: {str(e)}", exc_info=True)
        return subject, None

def plot_tfr_single_subject(tfr, subject, subject_dir):
    """Plot TFR results for a single subject"""

    fig = plt.figure(figsize=(10, 6))
    tfr.plot(
        picks='mag',
        combine='mean', 
        baseline=(-0.4, -0.25),
        mode='logratio', 
        tmin=DISPLAY_TMIN, 
        tmax=DISPLAY_TMAX,
        cmap='RdBu_r', 
        yscale='log', 
        show=False,
        fmin=4, 
        fmax=100,
        vmin=-0.3, 
        vmax=0.3,
        title=f'Subject {subject} - All Sensors Morlet (-0.4, -0.25 baseline)'
    )
    plt.axvline(x=0, color='white', linestyle='--', linewidth=1, label='Stimulus 1 Onset')
    plt.axvline(x=0.5, color='white', linestyle='--', linewidth=0.5, label='Stimulus Offset')
    plt.axvline(x=1, color='white', linestyle='--', linewidth=1, label='Stimulus 2 Onset')
    plt.axvline(x=1.5, color='white', linestyle='--', linewidth=0.5, label='Stimulus Offset')
    plt.axvline(x=3.5, color='white', linestyle='--', linewidth=1, label='Ping Onset')
    plt.savefig(f"{subject_dir}/tfr.png")
    plt.close()
    
def compute_grand_average():
    """Compute grand average across all processed subjects"""
    logging.info("Computing grand average TFR...")
    
    # Find all subject TFR pickle files
    import glob
    tfr_files = sorted(glob.glob(f"{tfr_dir}/sub-*_tfr.pkl"))
    logging.info(f"Found {len(tfr_files)} subject TFR files")
    
    if len(tfr_files) == 0:
        logging.warning("No subject TFR files found. Run single-subject processing first.")
        return None
    
    # Load TFR data from all subjects
    all_subject_tfrs = []
    valid_subjects = []
    
    for tfr_file in tqdm(tfr_files, desc="Loading subject data"):
        try:
            with open(tfr_file, 'rb') as f:
                tfr = pickle.load(f)
            
            # Extract subject ID from file path
            subject = int(tfr_file.split('sub-')[1].split('_')[0])
            
            all_subject_tfrs.append(tfr)
            valid_subjects.append(subject)
            logging.info(f"Loaded TFR data for subject {subject}")
        except Exception as e:
            logging.error(f"Error loading {tfr_file}: {str(e)}")
    
    if len(all_subject_tfrs) == 0:
        logging.warning("No valid subject TFR data could be loaded")
        return None
    
    # Compute grand average
    logging.info(f"Computing grand average from {len(valid_subjects)} subjects")
    grand_avg = mne.grand_average(all_subject_tfrs)
    
    # Save grand average result
    with open(f"{tfr_dir}/grand_avg_tfr.pkl", 'wb') as f:
        pickle.dump(grand_avg, f)
        
    grand_avg.save(f"{tfr_dir}/grand_avg_tfr-tfr.h5", overwrite=True)
    
    # Plot grand average results
    plot_grand_average(grand_avg, len(valid_subjects))
    
    logging.info("Grand average computation completed successfully!")
    return grand_avg

def plot_grand_average(grand_avg, n_subjects):
    """Plot grand average TFR results"""
    grand_avg_dir = f"{PLOT_DIR}/grand_average"
    os.makedirs(grand_avg_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(12, 8))
    grand_avg.plot(
        picks='mag',
        combine='mean', 
        baseline=(-0.4, -0.25),
        mode='logratio', 
        tmin=DISPLAY_TMIN, 
        tmax=DISPLAY_TMAX,
        cmap='RdBu_r', 
        yscale='log', 
        show=False,
        fmin=4, 
        fmax=100,
        vmin=-0.3, 
        vmax=0.3,
        title=f'Grand Average (N={n_subjects}) - All Sensors Morlet (-0.4, -0.25 baseline)' 
    )
    plt.axvline(x=0, color='white', linestyle='--', linewidth=1, label='Stimulus 1 Onset')
    plt.axvline(x=0.5, color='white', linestyle='--', linewidth=0.5, label='Stimulus Offset')
    plt.axvline(x=1, color='white', linestyle='--', linewidth=1, label='Stimulus 2 Onset')
    plt.axvline(x=1.5, color='white', linestyle='--', linewidth=0.5, label='Stimulus Offset')
    plt.axvline(x=3.5, color='white', linestyle='--', linewidth=1, label='Ping Onset')
    plt.savefig(f"{grand_avg_dir}/tfr.png")
    plt.close()
            
    # Create 5x6 subplot figures for all subjects
    create_all_subjects_plot()

def create_all_subjects_plot():
    """Create a 5x6 subplot figure with all subjects' TFR data"""
    
    import glob
    tfr_files = sorted(glob.glob(f"{tfr_dir}/sub-*_tfr.pkl"))
    
    # Load all subject TFRs
    all_tfrs = []
    subjects = []
    
    for tfr_file in tfr_files:
        try:
            with open(tfr_file, 'rb') as f:
                tfr = pickle.load(f)
            
            # Extract subject ID
            subject = int(tfr_file.split('sub-')[1].split('_')[0])
            subjects.append(subject)
            all_tfrs.append(tfr)
        except Exception as e:
            logging.error(f"Error loading {tfr_file}: {str(e)}")
    
    if len(all_tfrs) == 0:
        logging.warning("No TFR data found for subplot figure")
        return
    
    # Create the figure with 5 columns and 6 rows
    fig, axes = plt.subplots(6, 5, figsize=(20, 24))
    fig.suptitle(f'All Sensors - All Subjects', fontsize=16)
        
    # Loop through subjects and create subplots
    for i, (subject, tfr) in enumerate(zip(subjects, all_tfrs)):
        if i >= 30:  # Limit to 30 subjects
            break
            
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        
        tfr.plot(
            picks='mag',
            combine='mean',
            baseline=(-0.4, -0.25),
            mode='logratio',
            tmin=DISPLAY_TMIN,
            tmax=DISPLAY_TMAX,
            cmap='RdBu_r',
            yscale='log',
            show=False,
            fmin=4,
            fmax=100,
            vmin=-0.3,
            vmax=0.3,
            axes=ax,
            colorbar=False,
            title=f'S{subject}'
        )
        ax.axvline(x=0, color='white', linestyle='--', linewidth=0.5)
        ax.axvline(x=1, color='white', linestyle='--', linewidth=0.5)
        ax.axvline(x=3.5, color='white', linestyle='--', linewidth=0.5)
    
    # Fill any remaining empty subplots
    for i in range(len(subjects), 30):
        row = i // 5
        col = i % 5
        axes[row, col].set_visible(False)
    
    # Add a colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    norm = plt.Normalize(-0.3, 0.3)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='Power (logratio)')
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(f"{PLOT_DIR}/all_subjects_tfr.png", dpi=300)
    plt.close()

def process_all_subjects_parallel():
    """Process all subjects in parallel"""
    logging.info(f"Processing {NUM_SUBJECTS} subjects with {N_PARALLEL} parallel jobs")
    
    start_time = time.time()
    
    # Process all subjects in parallel
    results = Parallel(n_jobs=N_PARALLEL, verbose=10)(
        delayed(process_subject)(subject) for subject in range(1, NUM_SUBJECTS + 1)
    )
    
    # Count successful subjects
    successful = sum(1 for _, tfr in results if tfr is not None)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Processed {successful} subjects successfully in {elapsed_time:.2f} seconds")
    
    # Compute grand average
    compute_grand_average()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Morlet Wavelet TF analysis for single subject or grand average')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--subject', type=int, help='Process a single subject with this ID')
    group.add_argument('--all-subjects', action='store_true', help='Process all subjects in parallel')
    group.add_argument('--grand-average', action='store_true', help='Compute grand average across all processed subjects')
    group.add_argument('--plot-all', action='store_true', help='Create composite plots of all subjects')
    
    args = parser.parse_args()
    
    if args.subject:
        process_subject(args.subject)
    elif args.all_subjects:
        process_all_subjects_parallel()
    elif args.grand_average:
        compute_grand_average()
    elif args.plot_all:
        create_all_subjects_plot()
