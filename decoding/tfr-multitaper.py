#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merged TFR Analysis Script
Combines approaches from philipp_multitaper.py and tf-multitaper-hpc.py
- Uses JSON parameters and combined TFR computation from philipp_multitaper.py
- Uses standard MNE epochs loading and directory structure from tf-multitaper-hpc.py
- Implements trial-average baseline correction method
- Adapted for 200Hz sampling rate with 100Hz max frequency
"""

import os
import sys
import locale
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
import mne
from mne import read_epochs
from mne.time_frequency import EpochsTFR
import pickle
import json
import glob
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC uncomment if needed
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import seaborn as sns
from tqdm import tqdm

mne.set_log_level('WARNING')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ========================= CONFIGURATION CONSTANTS =========================
DEFAULT_SAMPLING_RATE = 200  # Hz (updated from 400Hz)

# Time window definitions (your constants)
TIME_WINDOWS = {
    'Full': (-0.5, 6),  # Full time window from data
    'Encoding': (0.0, 2.0),
    'Delay': (2.0, 4.5),
    'Encoding+Delay': (0.0, 4.5)
}

# Directory structure 
#HOME_DIR = '/media/headmodel/Elements/AWM4/' #local path
HOME_DIR = '/mnt/hpc/projects/awm4/' # HPC path
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
OUTPUT_DIR = PROCESSED_DIR + '/multitaper'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Analysis parameters
RESAMPLE_FREQ = 200  # Hz
NUM_JOBS = 8
BASELINE_WINDOW = (-0.3, -0.15)  # Your requested baseline window
TMIN = -0.5  # Include baseline period
TMAX = 6.0   # Your full time window
DISPLAY_TMAX = 4.5  # Actual experiment data display limit

# Brain region mapping 
REGION_MAPPING = {
    'F': 'Frontal',
    'C': 'Central', 
    'T': 'Temporal',
    'P': 'Parietal',
    'O': 'Occipital'
}

# Plotting order for regions
PLOT_ORDER = ['Frontal', 'Central', 'Temporal', 'Parietal', 'Occipital']

# Frequency bands for visualization
FREQ_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'low_gamma': (30, 45),
    'high_gamma': (60, 100)  # Updated max to 100Hz
}


def params_from_json(filename):
    """
    Load taper parameters from a json file.
    """
    params = json.load(open(filename))
    assert('foi' in params.keys())
    assert('window_length' in params.keys())
    assert('time_bandwidth' in params.keys())
    return params


def describe_and_update_taper(foi=None, window_length=None, time_bandwidth=None, **kwargs):
    """
    Print information about frequency smoothing / temporal smoothing for a set
    of taper parameters.
    """
    try:
        from tabulate import tabulate
        
        foi = np.atleast_1d(foi).astype(float)
        if len(np.atleast_1d(window_length)) == 1:
            window_length = [window_length] * len(foi)        
        window_length = np.atleast_1d(window_length).astype(float)
        if len(np.atleast_1d(time_bandwidth)) == 1:
            time_bandwidth_array = np.array([time_bandwidth] * len(foi))
        
        cycles = window_length * foi

        f_smooth = time_bandwidth_array / window_length
        
        data = list(zip(list(foi), list(cycles), list(window_length), list(f_smooth)))
        
        print(tabulate(data,
                       headers=['Freq', 'Cycles', 't. window', 'F. smooth']))
        
        tfr_params = {
            'foi': foi,
            'cycles': cycles,
            'window_length': window_length,
            'time_bandwidth': time_bandwidth,
            'f_smooth': f_smooth,
            'decim': 10
        }

        return tfr_params

    except ImportError:
        print("Tabulate not available, printing basic info:")
        for i, f in enumerate(foi):
            print(f"Freq: {f:.2f}, Time window: {window_length[i]:.3f}s")

        tfr_params = {
            'foi': foi,
            'cycles': window_length * foi,
            'window_length': window_length,
            'time_bandwidth': time_bandwidth,
            'f_smooth': time_bandwidth / window_length,
            'decim': 10
        }
        return tfr_params


def parse_channel_names(ch_names: List[str]) -> Dict[str, Dict[str, List[int]]]:
    """
    Parse MEG channel names to categorize by region and hemisphere.
    Uses philipp_multitaper.py naming convention.
    
    Channel naming convention: M + L/R + region_code + number
    Example: MLT11 = MEG Left Temporal channel 11
    
    Parameters
    ----------
    ch_names : list of str
        List of channel names
        
    Returns
    -------
    channel_groups : dict
        Dictionary with structure:
        {region: {'left': [channel_indices], 'right': [channel_indices]}}
    """
    channel_groups = {}
    
    # Initialize structure for all regions
    for region in REGION_MAPPING.values():
        channel_groups[region] = {'left': [], 'right': []}
    
    for i, ch_name in enumerate(ch_names):
        # Skip non-MEG channels
        if not ch_name.startswith('M'):
            continue
            
        # Parse channel name
        if len(ch_name) >= 3:
            hemisphere_code = ch_name[1]  # L or R
            region_code = ch_name[2]      # F, C, T, P, O
            
            # Map hemisphere
            hemisphere = 'left' if hemisphere_code == 'L' else 'right'
            
            # Map region
            if region_code in REGION_MAPPING:
                region = REGION_MAPPING[region_code]
                channel_groups[region][hemisphere].append(i)
    
    # Log channel counts
    logger.info("Channel counts by region and hemisphere:")
    for region in PLOT_ORDER:
        left_count = len(channel_groups[region]['left'])
        right_count = len(channel_groups[region]['right'])
        logger.info(f"  {region}: Left={left_count}, Right={right_count}")
    
    return channel_groups


def _apply_trialavg_baseline(epochs_tfr, tmin=-0.3, tmax=-0.15, mode='percent'):
    """
    Apply trial-average baseline correction to time-frequency data.
    
    This function computes the baseline by averaging across all epochs/trials,
    then applies percent change baseline correction to each individual epoch.
    
    Parameters
    ----------
    epochs_tfr : EpochsTFR object
        The time-frequency data to be baseline corrected.
    tmin : float, default=-0.3
        Start time of baseline period in seconds.
    tmax : float, default=-0.15
        End time of baseline period in seconds.
    mode : str, default='percent'
        Baseline correction mode
        
    Returns
    -------
    None
        Modifies epochs_tfr.data in-place with baseline corrected data.
    """
    if mode == 'percent':
        # Average over epochs to get trial-average TFR
        average_tfr = epochs_tfr.average()
        
        # Extract timepoints of the baseline period
        average_baseline_data = average_tfr.get_data(tmin=tmin, tmax=tmax)  # shape (n_sensors, n_freqs, n_times)
        
        # Compute mean over baseline timepoints
        baseline_mean = np.mean(average_baseline_data, axis=2, keepdims=True)  # shape (n_sensors, n_freqs, 1)
        
        # Apply baseline correction to each epoch using percent change from baseline
        epochs_tfr.data = (epochs_tfr.data - baseline_mean) / baseline_mean
    
    elif mode != 'percent':
        raise ValueError("Currently only 'percent' mode is implemented.")

    return None


def plot_tfr_by_regions(epochs_tfr: EpochsTFR, subject_name: str = None, 
                       save_path: str = None, figsize: Tuple[int, int] = (12, 15), cmap: str = 'RdBu_r'):
    """
    Plot time-frequency representations averaged by brain regions and hemispheres.
    """
    # Parse channel names
    channel_groups = parse_channel_names(epochs_tfr.ch_names)
    
    # Create figure and subplots
    fig, axes = plt.subplots(5, 2, figsize=figsize, sharex=True, sharey=True)

    avg_tfr = epochs_tfr.average()

    # Track if we've added a colorbar yet
    colorbar_ref = None
    
    # Plot each region
    for row, region in enumerate(PLOT_ORDER):
        for col, hemisphere in enumerate(['left', 'right']):
            ax = axes[row, col]
            
            # Get channel indices for this region and hemisphere
            channel_indices = channel_groups[region][hemisphere]
            
            if len(channel_indices) == 0:
                # No channels for this region/hemisphere - create empty plot
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12, color='gray')
                ax.set_xlim(avg_tfr.times[0], avg_tfr.times[-1])
                ax.set_ylim(avg_tfr.freqs[0], avg_tfr.freqs[-1])
            else:
                # Use MNE's plot function with picks and combine='mean'
                try:
                    im = avg_tfr.plot(
                        picks=channel_indices,
                        combine='mean',
                        axes=ax,
                        colorbar=False,  # We'll add our own colorbar
                        show=False,
                        vlim=(-0.5, 0.5),
                        cmap=cmap,
                        yscale='linear'
                    )
                    # Store reference for colorbar (from the first successful plot)
                    if colorbar_ref is None:
                        colorbar_ref = ax.images[-1] if ax.images else None
                        
                except Exception as e:
                    logger.warning(f"Error plotting {region} {hemisphere}: {e}")
                    ax.text(0.5, 0.5, 'Plot Error', transform=ax.transAxes,
                           ha='center', va='center', fontsize=10, color='red')
            
            # Add vertical line at stimulus onset (t=0)
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            ax.axvline(x=2, color='black', linestyle=':', alpha=0.7, linewidth=1)  # Delay onset
            
            # Set titles for top row and y-labels for left column
            if row == 0:
                hemisphere_title = hemisphere.capitalize()
                ax.set_title(f'{hemisphere_title} Hemisphere', fontsize=14, fontweight='bold')
            
            if col == 0:
                ax.set_ylabel(f'{region}\nFrequency (Hz)', fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel('')  # Remove y-label for right column
            
            # Clean up x-labels (only show for bottom row)
            if row < 4:
                ax.set_xlabel('')
    
    # Set x-labels for bottom row
    for col in range(2):
        axes[4, col].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    
    # Add overall title
    title = f'Time-Frequency Analysis by Brain Regions'
    if subject_name:
        title += f' - {subject_name}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout first to make room for colorbar
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.1, right=0.85)
    
    # Add colorbar if we have valid plots
    if colorbar_ref is not None:
        # Position colorbar on the right side of all plots
        cbar_ax = fig.add_axes([.87, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(colorbar_ref, cax=cbar_ax)
        cbar.set_label('Power (% change from baseline)', fontsize=12, fontweight='bold')
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        logger.info(f"Saved TFR plot to {save_path}")
    
    return fig, axes


def process_single_subject(subject, json_params_file='awm_tfr_parameters.json'):
    """
    Process TFR analysis for a single subject using merged approach.
    
    Parameters
    ----------
    subject : int
        Subject ID number
    json_params_file : str
        Path to JSON file with TFR parameters
    """
    print(f"Processing TF analysis for subject {subject}")
    
    subject_dir = f"{OUTPUT_DIR}/sub-{subject}"
    os.makedirs(subject_dir, exist_ok=True)
    
    try:
        # Load clean epochs (tf-multitaper-hpc.py approach)
        epochs_file = f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif"
        if not os.path.exists(epochs_file):
            print(f"Epochs file not found for subject {subject}")
            return None
            
        epochs = read_epochs(epochs_file, preload=True)
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
        
        # Mark bad channels
        if subject == 4 and 'MRO31-3609' in epochs.ch_names:
            epochs.info['bads'].append('MRO31-3609')
        elif subject == 26:
            for ch in ['MRO31-3609', 'MRO44-3609']:
                if ch in epochs.ch_names:
                    epochs.info['bads'].append(ch)
        if subject == 28:
            drop_idx = 63
            if drop_idx < len(epochs):
                epochs.drop(drop_idx)

        # Process all sensors
        mag_epochs = epochs.copy().pick_types(meg='mag', exclude='bads')        
        # Crop to time window of interest
        mag_epochs.crop(tmin=TMIN, tmax=TMAX)
        
        # Resample for efficiency
        mag_epochs = mag_epochs.resample(RESAMPLE_FREQ, npad='auto')
        
        # Create JSON parameter file if it doesn't exist
        if not os.path.exists(json_params_file):
            low_freqs = np.arange(1, 9 + 1, 1)  # 1-9 Hz in steps of 1
            low_freqs = low_freqs.astype(int)
            high_freqs = np.arange(10, 100 + 1, 5)  # 10-100 Hz in steps of 5 
            high_freqs = high_freqs.astype(int)

            # window lengths: 250 ms for low freqs, 100 ms for high freqs
            custom_params = {
                "decim": 10,
                "foi": [int(freq) for freq in np.concatenate((low_freqs, high_freqs))],
                "window_length": [freq for freq in np.concatenate((np.repeat(0.25, len(low_freqs)),np.repeat(0.1, len(high_freqs))))],
                "time_bandwidth": 2
            }
            
            with open(json_params_file, 'w') as f:
                json.dump(custom_params, f, indent=2)
            logger.info(f"Created JSON parameters file: {json_params_file}")
        
        # Load TFR parameters
        tfr_params = params_from_json(json_params_file)
        logger.info(f"Loaded TFR parameters from {json_params_file}")
        
        # Print and update taper information
        logger.info(f"Computing TFR for subject {subject} using JSON parameters")
        logger.info(f"Frequencies: {tfr_params['foi'][0]:.1f} - {tfr_params['foi'][-1]:.1f} Hz ({len(tfr_params['foi'])} freqs)")
        logger.info(f"Time bandwidth: {tfr_params['time_bandwidth']}")
        logger.info(f"Decimation: {tfr_params['decim']}")
        
        tfr_params = describe_and_update_taper(**tfr_params)
        
        logger.info("Computing TFR using MNE EpochsArray.compute_tfr(method='multitaper')")
        
        epochs_tfr = mag_epochs.compute_tfr(
            method='multitaper',
            freqs=tfr_params['foi'],
            n_cycles=tfr_params['cycles'],
            time_bandwidth=tfr_params['time_bandwidth'],
            use_fft=True,
            return_itc=False,
            decim=tfr_params['decim'],
            n_jobs=NUM_JOBS,
            verbose=False
        )

        # Apply trial-average baseline correction (philipp_multitaper.py method)
        baseline_tmin, baseline_tmax = BASELINE_WINDOW
        logger.info(f"Applying trial-average baseline correction ({baseline_tmin*1000} to {baseline_tmax*1000} ms)")
        
        _apply_trialavg_baseline(epochs_tfr, tmin=baseline_tmin, tmax=baseline_tmax, mode='percent')

        logger.info("TFR computation complete")
        logger.info(f"TFR data shape: {epochs_tfr.data.shape} (trials x sensors x freqs x times)")
        
        # Add metadata to the object
        epochs_tfr.comment = {
            'subject': f'Subject_{subject}',
            'sampling_rate': DEFAULT_SAMPLING_RATE,
            'tfr_params': tfr_params,
            'baseline_window': (baseline_tmin, baseline_tmax),
            'baseline_mode': 'percent',
            'description': 'Multitaper TFR with trial-average baseline correction (percent change)'
        }
        
        # Save results (both baseline corrected and raw versions)
        # Baseline corrected version
        baseline_corrected_filename = f"sub-{subject}_EpochsTFR_baseline_corrected.h5"
        baseline_corrected_path = os.path.join(subject_dir, baseline_corrected_filename)
        epochs_tfr.save(baseline_corrected_path, overwrite=True)
        
        # Create and save raw (non-baseline corrected) version
        # Recompute without baseline correction
        epochs_tfr_raw = mag_epochs.compute_tfr(
            method='multitaper',
            freqs=tfr_params['foi'],
            n_cycles=tfr_params['cycles'],
            time_bandwidth=tfr_params['time_bandwidth'],
            use_fft=True,
            return_itc=False,
            decim=tfr_params['decim'],
            n_jobs=NUM_JOBS,
            verbose=False
        )
        
        epochs_tfr_raw.comment = {
            'subject': f'Subject_{subject}',
            'sampling_rate': DEFAULT_SAMPLING_RATE,
            'tfr_params': tfr_params,
            'description': 'Multitaper TFR without baseline correction (raw power)'
        }
        
        raw_filename = f"sub-{subject}_EpochsTFR_raw.h5"
        raw_path = os.path.join(subject_dir, raw_filename)
        epochs_tfr_raw.save(raw_path, overwrite=True)
        
        logger.info(f"Saved TFR results for subject {subject}")
        logger.info(f"  - Baseline corrected (HDF5): {baseline_corrected_path}")
        logger.info(f"  - Raw power (HDF5): {raw_path}")
        
        # Create plots
        plot_single_subject_results(epochs_tfr, subject, subject_dir)
        
        print(f"Successfully processed subject {subject}")
        return epochs_tfr
        
    except Exception as e:
        print(f"Error processing subject {subject}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def plot_single_subject_results(epochs_tfr, subject, subject_dir):
    """
    Create comprehensive plots for a single subject.
    """
    plot_filename = f"sub-{subject}_TFR_regions_plot.png"
    plot_path = os.path.join(subject_dir, plot_filename)
    
    try:
        fig, axes = plot_tfr_by_regions(
            epochs_tfr=epochs_tfr,
            subject_name=f"Subject {subject}",
            save_path=plot_path,
            figsize=(12, 15)
        )
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating regional plots for subject {subject}: {e}")
    
    # 2. Average TFR plot
    avg_tfr = epochs_tfr.average()
    
    fig = plt.figure(figsize=(10, 6))
    avg_tfr.plot(combine='mean',
              title=f"Subject {subject} - TF Power (% change from baseline)",
              tmin=-0.4, tmax=4.5,
              vlim=(-0.5, 0.5),
              cmap='RdBu_r',
              show=False)
    
    # Add vertical lines for important time points
    plt.axvline(x=0, color='white', linestyle='--', linewidth=1, label='Stimulus Onset')
    plt.axvline(x=2, color='white', linestyle=':', linewidth=1, label='Delay Onset')
    plt.axvline(x=3.5, color='white', linestyle='--', linewidth=1, label='Ping')
    plt.legend(loc='upper right')
    
    plt.savefig(f"{subject_dir}/sub-{subject}_tfr_average.png", dpi=300)
    plt.close()
    
    # 3. Frequency band power over time
    plot_frequency_bands_over_time(avg_tfr, subject, subject_dir)


def plot_frequency_bands_over_time(avg_tfr, subject, subject_dir):
    """Plot frequency band power over time for a subject"""
    plt.figure(figsize=(10, 6))
    
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        # Find frequencies within this band
        freq_mask = (avg_tfr.freqs >= fmin) & (avg_tfr.freqs <= fmax)
        
        if not np.any(freq_mask):
            print(f"Warning: No frequencies in the {band_name} band ({fmin}-{fmax} Hz)")
            continue
        
        # Average power across frequencies in the band and across all channels
        band_power = np.mean(avg_tfr.data[:, freq_mask, :], axis=(0, 1))

        # Plot band power over time
        plt.plot(avg_tfr.times, band_power,
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
    plt.savefig(f"{subject_dir}/sub-{subject}_band_power_over_time.png", dpi=300)
    plt.close()


def compute_grand_average():
    """Compute grand average across all processed subjects"""
    print("Computing grand average TFR...")
    
    # Find all subject baseline corrected TFR files
    import glob
    tfr_files = sorted(glob.glob(f"{OUTPUT_DIR}/sub-*/sub-*_EpochsTFR_baseline_corrected.h5"))
    print(f"Found {len(tfr_files)} subject TFR files")
    
    if len(tfr_files) == 0:
        print("No subject TFR files found. Run single-subject processing first.")
        return None
    
    # Load TFR data from all subjects
    all_subject_tfrs = []
    valid_subjects = []
    
    for tfr_file in tqdm(tfr_files, desc="Loading subject data"):
        try:
            # Load the first (and only) TFR from the file
            epochs_tfr = mne.time_frequency.read_tfrs(tfr_file)[0]
            
            # Average across epochs to get AverageTFR
            avg_tfr = epochs_tfr.average()
            
            # Extract subject ID from file path
            subject = int(tfr_file.split('sub-')[1].split('_')[0])
            
            all_subject_tfrs.append(avg_tfr)
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
    with open(f"{OUTPUT_DIR}/grand_average/grand_avg_tfr.pkl", 'wb') as f:
        pickle.dump(grand_avg, f)
    
    # Plot grand average results
    plot_grand_average_results(grand_avg, len(valid_subjects))
    
    print("Grand average completed successfully!")
    return grand_avg


def plot_grand_average_results(grand_avg, n_subjects):
    """Plot grand average TFR results"""
    
    # 1. Overall TFR plot
    fig = plt.figure(figsize=(12, 8))
    grand_avg.plot(combine='mean',
                  title=f"Grand Average (N={n_subjects}) - TF Power (% change)",
                  tmin=-0.4, tmax=4.5,
                  vlim=(-0.5, 0.5),
                  cmap='RdBu_r',
                  show=False)
    plt.axvline(x=0, color='white', linestyle='--', linewidth=2, label='Stimulus Onset')
    plt.axvline(x=2, color='white', linestyle=':', linewidth=2, label='Delay Onset')
    plt.axvline(x=3.5, color='white', linestyle='--', linewidth=2, label='Ping')
    plt.legend(loc='upper right')
    plt.savefig(f"{OUTPUT_DIR}/grand_average/grand_avg_tfr_all_channels.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/grand_average/grand_avg_tfr_all_channels.pdf")
    plt.close()
    
    # 2. Frequency band power over time
    plot_frequency_bands_over_time(grand_avg, f"Grand Average (N={n_subjects})", OUTPUT_DIR)
    
    # 3. Topographic maps for each frequency band
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        # Define time windows: baseline, encoding, and delay
        time_windows = [
            (-0.5, 0),    # Baseline
            (0, 2),       # Encoding
            (2, 4.5)      # Delay
        ]
        window_names = ["Baseline", "Encoding", "Delay"]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, (t_min, t_max) in enumerate(time_windows):
            grand_avg.plot_topomap(
                ch_type='mag', 
                tmin=t_min, tmax=t_max,
                fmin=fmin, fmax=fmax,
                vlim=(-0.5, 0.5),
                title=f"{window_names[i]} ({t_min}-{t_max}s)",
                axes=axes[i],
                show=False
            )
        
        plt.suptitle(f"Grand Average (N={n_subjects}) - {band_name} band ({fmin}-{fmax} Hz)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/grand_average/{band_name}_grand_avg_topomap.png", dpi=300)
        plt.savefig(f"{OUTPUT_DIR}/grand_average/{band_name}_grand_avg_topomap.pdf")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multitaper TF analysis for single subject or grand average')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--subject', type=int, help='Process a single subject with this ID')
    group.add_argument('--grand-average', action='store_true', help='Compute grand average across all processed subjects')
    
    args = parser.parse_args()
    
    if args.subject:
        process_single_subject(args.subject)
    elif args.grand_average:
        compute_grand_average()