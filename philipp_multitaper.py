#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified TFR Analysis using JSON parameters and MNE EpochsArray.compute_tfr()
Saves MNE EpochsTFR objects for easier downstream analysis
Now includes plotting functionality for brain regions
"""

import os
import sys
import locale
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
import mne
from mne import EpochsArray
from mne.time_frequency import EpochsTFR
from pytictoc import TicToc
import pickle
import json
import glob
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import seaborn as sns

# ========================= CONFIGURATION CONSTANTS =========================
DEFAULT_SAMPLING_RATE = 200  # Hz

# Time window definitions
TIME_WINDOWS = {
    'Full': (-0.5, 6),  # Full time window from data
    'Encoding': (0.0, 2.0),
    'Delay': (2.0, 4.5),
    'Encoding+Delay': (0.0, 4.5)
}

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
            window_length = [window_length] * len(window_length)        
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

    
def parse_channel_names(ch_names: List[str]) -> Dict[str, Dict[str, List[int]]]:
    """
    Parse MEG channel names to categorize by region and hemisphere.
    
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

def plot_tfr_by_regions(epochs_tfr: EpochsTFR, subject_name: str = None, 
                       save_path: str = None, figsize: Tuple[int, int] = (12, 15),
                       vmin: float = -.2, vmax: float = .2, cmap: str = 'RdBu_r'):
    """
    Plot time-frequency representations averaged by brain regions and hemispheres using MNE's plot function.
    
    Parameters
    ----------
    epochs_tfr : EpochsTFR
        Time-frequency data
    subject_name : str, optional
        Subject name for title
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
    vmin, vmax : float, optional
        Color scale limits
    cmap : str
        Colormap name
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
                        vlim=(vmin, vmax),
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

def create_mne_epochs_array(neural_data, tmin=-1.75, info=None):
    """
    Create MNE EpochsArray from neural data.
    
    Parameters
    ----------
    neural_data : np.ndarray
        Neural data (n_trials, n_sensors, n_times)
    sfreq : float
        Sampling frequency
    tmin : float
        Start time of epochs
    ch_names : list, optional
        Channel names
    events : np.ndarray, optional
        Events array
        
    Returns
    -------
    epochs : mne.EpochsArray
        MNE EpochsArray object
    """
   
    
    # Create info object
    if info is None:
        raise ValueError("Info object must be provided or created.")
    
    # Create events array if not provided
    # Create EpochsArray
    epochs = EpochsArray(
        data=neural_data,
        info=info,
        tmin=tmin,
        verbose=False
    )
    
    return epochs


def save_epochs_tfr(epochs_tfr, fname):
    """
    Save EpochsTFR object.
    
    Parameters
    ----------
    epochs_tfr : mne.time_frequency.EpochsTFR
        EpochsTFR object to save
    fname : str
        Output filename
    """
    if fname.endswith('.h5') or fname.endswith('.hdf5'):
        epochs_tfr.save(fname, overwrite=True)
        logger.info(f"Saved EpochsTFR object to {fname}")
    else:
        # Save as pickle for compatibility
        with open(fname, 'wb') as f:
            pickle.dump(epochs_tfr, f)
        logger.info(f"Saved EpochsTFR object (pickle) to {fname}")


def create_cfg() -> Dict:
    """
    Create dictionary of analysis specifications for TFR analysis.
    
    Returns
    -------
    cfg : Dict
        Dictionary of analysis specifications.
    """
    cfg = {
        'ANALYSIS_PHASE': 'Encoding Delay',
        'SAMPLING_RATE': DEFAULT_SAMPLING_RATE,
        'BASE_DIR': "/home/pdeutsch/data/projects/WM_MEG_Project/WMPrecision_local/Processed",
        'TIME_WINDOWS': TIME_WINDOWS,
        'TOI_TIMES': np.arange(-1.75, 2.15, 1/100),
        'BASELINE_WINDOW': (-0.5, -0.15),  # Baseline window in seconds
        'DATA_TYPE': 'raw',  # Load raw data instead of variance
        'SOURCE_OR_SENSOR': 'sensor',
        'subCodes': subCodes,
        'NoSubs': len(subCodes),
        'validSubs': np.delete(np.array(range(len(subCodes))), [6, 9]),
        'Subject': Subject,
        'ZScore': 'sessionwise'
    }
    
    return cfg


def get_epoch_path(cfg: Dict, subject_name) -> str:
    """
    Get the base results path for TFR analysis.
    
    Parameters
    ----------
    cfg : Dict
        Configuration dictionary
        
    Returns
    -------
    results_path : str
        Base path for results
    """
    results_path = os.path.join(
        cfg['BASE_DIR'], subject_name, cfg['ANALYSIS_PHASE'], 'TFRs', cfg['SOURCE_OR_SENSOR'], cfg['DATA_TYPE'], 'Multitaper_EpochsTFR'
    )
    os.makedirs(results_path, exist_ok=True)
    return results_path


def get_plots_path(cfg: Dict, subject_name) -> str:
    """
    Get the plots path for TFR analysis.
    
    Parameters
    ----------
    cfg : Dict
        Configuration dictionary
    subject_name : str
        Subject name
        
    Returns
    -------
    plots_path : str
        Path for plots
    """
    plots_path = os.path.join(
        cfg['BASE_DIR'], subject_name, cfg['ANALYSIS_PHASE'], 'TFRs', cfg['SOURCE_OR_SENSOR'], cfg['DATA_TYPE'], 'Multitaper_EpochsTFR', 'Plots'
    )
    os.makedirs(plots_path, exist_ok=True)
    return plots_path


def check_and_load_existing_results(subject_name: str, cfg: Dict, params_hash: str):
    """
    Check if TFR results already exist and load them if they do.
    
    Parameters
    ----------
    subject_name : str
        Subject name
    cfg : Dict
        Configuration dictionary
    params_hash : str
        Hash of parameters for filename uniqueness
        
    Returns
    -------
    results_info : dict
        Dictionary containing information about existing results
    """
    results_path = get_epoch_path(cfg,subject_name)
    
    # Check for HDF5 file (native MNE format)
    hdf5_filename = f"{subject_name}_EpochsTFR_{params_hash}_{cfg['SAMPLING_RATE']}Hz.h5"
    hdf5_path = os.path.join(results_path, hdf5_filename)
    
    # Check for pickle file
    pickle_filename = f"{subject_name}_EpochsTFR_{params_hash}_{cfg['SAMPLING_RATE']}Hz.pkl"
    pickle_path = os.path.join(results_path, pickle_filename)
    
    if os.path.exists(hdf5_path):
        try:
            existing_results = mne.time_frequency.read_tfrs(hdf5_path)
            return {
                'exists': True,
                'data': existing_results,
                'path': hdf5_path
            }
        except Exception as e:
            logger.warning(f"Error loading existing HDF5 results: {e}")
    
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                existing_results = pickle.load(f)
            return {
                'exists': True,
                'data': existing_results,
                'path': pickle_path
            }
        except Exception as e:
            logger.warning(f"Error loading existing pickle results: {e}")
    
    return {'exists': False}


def _apply_trialavg_baseline(epochs_tfr, tmin=-0.5, tmax=-0.15, mode = 'percent'):
    """
    Apply trial-average baseline correction to time-frequency data.
    
    This function computes the baseline by averaging across all epochs/trials,
    then applies percent change baseline correction to each individual epoch.
    
    Parameters
    ----------
    epochs_tfr : EpochsTFR object
        The time-frequency data to be baseline corrected.
        Should have .average(), .get_data(), and .data attributes.
    tmin : float, default=-0.5
        Start time of baseline period in seconds.
    tmax : float, default=-0.15
        End time of baseline period in seconds.
        
    Returns
    -------
    None
        Modifies epochs_tfr.data in-place with baseline corrected data.
        
    Notes
    -----
    The baseline correction uses percent change from baseline:
    corrected = (original - baseline_mean) / baseline_mean
    
    The baseline is computed by:
    1. Averaging TFR data across all epochs
    2. Extracting data from baseline time window [tmin, tmax]
    3. Computing mean across baseline timepoints
    4. Applying this baseline to correct all individual epochs
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

def process_subject_tfr(subject: Subject, cfg: Dict, tfr_params: Dict) -> EpochsTFR:
    """
    Process TFR analysis for a single subject using JSON parameters and MNE EpochsArray.
    
    Parameters
    ----------
    subject : Subject
        Subject object
    cfg : Dict
        Configuration dictionary
    tfr_params : Dict
        TFR parameters from JSON file
        
    Returns
    -------
    epochs_tfr : mne.time_frequency.EpochsTFR
        TFR analysis results as EpochsTFR object
    """
    # Create parameter hash for filename uniqueness
    params_str = f"foi{len(tfr_params['foi'])}_tb{tfr_params['time_bandwidth']}_dec{tfr_params['decim']}"
    
    # Check for existing results
    existing_results_info = check_and_load_existing_results(subject.name, cfg, params_str)
    
    if existing_results_info['exists'] and existing_results_info['data'] is not None:
        logger.info(f"Loading existing TFR results for subject {subject.name}")
        return existing_results_info['data']
    
    # Load sensor-level data
    logger.info(f"Loading sensor-level data for subject {subject.name}")
    
    subject_data = pld.load_subject_data(
        subject, cfg, 
        data_type=cfg['DATA_TYPE'],  # Load raw data
        source_or_sensor=cfg['SOURCE_OR_SENSOR']  # Load sensor-level data
    )
    
    if subject_data is None:
        logger.warning(f"No sensor data loaded for {subject.name}")
        return None
    
    # Get the neural data (sensors x trials x timepoints)
    neural_data = subject_data['iem_data']
    
    # Transpose to (trials x sensors x timepoints) for MNE
    neural_data = np.transpose(neural_data, (1, 0, 2))
    
    logger.info(f"Data shape: {neural_data.shape} (trials x sensors x timepoints)")
    
    
    # Create MNE EpochsArray
    logger.info("Creating MNE EpochsArray")
    epochs = create_mne_epochs_array(
        neural_data=neural_data,
        tmin=-1.75,  # data starts at -1.75s
        info=subject_data['mne_info']  # Use the info from the loaded data
    )
    
    # Print taper information
    logger.info(f"Computing TFR for {subject.name} using JSON parameters")
    logger.info(f"Frequencies: {tfr_params['foi'][0]:.1f} - {tfr_params['foi'][-1]:.1f} Hz ({len(tfr_params['foi'])} freqs)")
    logger.info(f"Time bandwidth: {tfr_params['time_bandwidth']}")
    logger.info(f"Decimation: {tfr_params['decim']}")
    
    tfr_params = describe_and_update_taper(**tfr_params)
    
    # Compute TFR using MNE's EpochsArray.compute_tfr method
    logger.info("Computing TFR using MNE EpochsArray.compute_tfr(method='multitaper')")
    

    epochs_tfr = epochs.compute_tfr(
        method='multitaper',
        freqs=tfr_params['foi'],
        n_cycles=tfr_params['cycles'],
        time_bandwidth=tfr_params['time_bandwidth'],
        use_fft=True,
        return_itc=False,
        decim=tfr_params['decim'],
        n_jobs=20,
        verbose=False
    )

    baseline_tmin, baseline_tmax = cfg['BASELINE_WINDOW']


    # Apply baseline correction as described in the paper
    logger.info(f"Applying baseline correction ({baseline_tmin*1000} to {baseline_tmax*1000} ms)")
    # with 150 ms before stimulus onset because the maximum window length is 250 ms (i.e data 125ms before stimulus onset is affected by post-stimulus activity)

    _apply_trialavg_baseline(epochs_tfr, tmin=baseline_tmin, tmax=baseline_tmax, mode='percent')


    logger.info("TFR computation complete")
    
    logger.info(f"TFR data shape: {epochs_tfr.data.shape} (trials x sensors x freqs x times)")
    
    # Add metadata to the object
    epochs_tfr.comment = {
        'subject': subject.name,
        'sampling_rate': cfg['SAMPLING_RATE'],
        'tfr_params': tfr_params,
        'baseline_window': (baseline_tmin, baseline_tmax),
        'baseline_mode': 'percent',
        'description': 'Multitaper TFR with baseline correction (percent change) based on trial-average baseline'
    }
    
    # Save results
    results_path = get_epoch_path(cfg,subject.name)
    
    # Save as HDF5 (native MNE format)
    hdf5_filename = f"{subject.name}_EpochsTFR_{params_str}_{cfg['SAMPLING_RATE']}Hz.h5"
    hdf5_path = os.path.join(results_path, hdf5_filename)
    
    save_epochs_tfr(epochs_tfr, hdf5_path)
    
    
    # Save raw (non-baseline corrected) version as well
    raw_hdf5_filename = f"{subject.name}_EpochsTFR_Raw_{params_str}_{cfg['SAMPLING_RATE']}Hz.h5"
    raw_hdf5_path = os.path.join(results_path, raw_hdf5_filename)
    
    epochs_tfr.comment = {
        'subject': subject.name,
        'sampling_rate': cfg['SAMPLING_RATE'],
        'tfr_params': tfr_params,
        'description': 'Multitaper TFR without baseline correction (raw power)'
    }
    
    save_epochs_tfr(epochs_tfr, raw_hdf5_path)
    
    logger.info(f"Saved TFR results for {subject.name}")
    logger.info(f"  - Baseline corrected (HDF5): {hdf5_path}")
    logger.info(f"  - Raw power (HDF5): {raw_hdf5_path}")
    logger.info(f"  - Frequencies: {len(tfr_params['foi'])} ({tfr_params['foi'][0]:.1f}-{tfr_params['foi'][-1]:.1f} Hz)")
    logger.info(f"  - Time points: {epochs_tfr.times.shape[0]} (decimated by {tfr_params['decim']})")
    
    return epochs_tfr


def process_subject(actSubj: int, cfg: Dict = None, json_params_file: str = 'all_tfr150_parameters.json', 
                   create_plots: bool = True):
    """
    Process TFR analysis for a single subject using JSON parameters.
    
    Parameters
    ----------
    actSubj : int
        Subject index
    cfg : Dict, optional
        Configuration dictionary. If None, creates default config.
    json_params_file : str
        Path to JSON file with TFR parameters
    create_plots : bool
        Whether to create plots
    """
    if cfg is None:
        cfg = create_cfg()
    
    # Load parameters from JSON file
    if not os.path.exists(json_params_file):
        # Create JSON file with provided parameters (identical to the parameters reported in Wilming et al., 2020)
        
        low_freqs = np.arange(1, 9 + 1, 1)  # 1-9 Hz in steps of 1
        low_freqs = low_freqs.astype(int)
        high_freqs = np.arange(10, 151 + 1, 5)  # 10-150 Hz in steps of 5
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
        logger.info(f"Created example JSON parameters file: {json_params_file}")
    
    # Load TFR parameters
    tfr_params = params_from_json(json_params_file)
    logger.info(f"Loaded TFR parameters from {json_params_file}")
    
    # Initialize timing
    t = TicToc()
    t.tic()
    
    subject = cfg['Subject'](id=actSubj)
    logger.info(f"Processing Subject {subject.name} - TFR Multitaper Analysis (EpochsTFR)")
    
    # Process TFR
    epochs_tfr = process_subject_tfr(
        subject=subject,
        cfg=cfg,
        tfr_params=tfr_params
    )
    
    # Create plots if requested
    if create_plots and epochs_tfr is not None:
        logger.info(f"Creating TFR plots for {subject.name}")
        
        plots_path = get_plots_path(cfg, subject.name)
        plot_filename = f"{subject.name}_TFR_regions_plot.png"
        plot_path = os.path.join(plots_path, plot_filename)
        
        try:
            fig, axes = plot_tfr_by_regions(
                epochs_tfr=epochs_tfr,
                subject_name=subject.name,
                save_path=plot_path,
                figsize=(12, 15)
            )
            
            # Show plot
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating plots for {subject.name}: {e}")
    
    t.toc(f'Finished processing Subject {subject.name} - TFR Multitaper Analysis')
    
    return epochs_tfr


def load_epochs_tfr(subject_name: str, cfg: Dict = None, params_hash: str = None):
    """
    Load saved EpochsTFR object for a subject.
    
    Parameters
    ----------
    subject_name : str
        Subject name
    cfg : Dict, optional
        Configuration dictionary
    params_hash : str, optional
        Parameter hash for filename
        
    Returns
    -------
    epochs_tfr : mne.time_frequency.EpochsTFR
        Loaded EpochsTFR object
    """
    if cfg is None:
        cfg = create_cfg()
    
    if params_hash is None:
        # Try to find the most recent file
        results_path = get_epoch_path(cfg,subject_name)
        pattern = f"{subject_name}_EpochsTFR_*.h5"
        files = glob.glob(os.path.join(results_path, pattern))
        if not files:
            pattern = f"{subject_name}_EpochsTFR_*.pkl"
            files = glob.glob(os.path.join(results_path, pattern))
        
        if not files:
            logger.error(f"No TFR files found for subject {subject_name}")
            return None
        
        # Use the most recent file
        files.sort()
        filepath = files[-1]
    else:
        results_path = get_epoch_path(cfg,subject_name)
        hdf5_filename = f"{subject_name}_EpochsTFR_{params_hash}_{cfg['SAMPLING_RATE']}Hz.h5"
        filepath = os.path.join(results_path, hdf5_filename)
        
        if not os.path.exists(filepath):
            pickle_filename = f"{subject_name}_EpochsTFR_{params_hash}_{cfg['SAMPLING_RATE']}Hz.pkl"
            filepath = os.path.join(results_path, pickle_filename)
    
    # Load the file
    if filepath.endswith('.h5') or filepath.endswith('.hdf5'):
        epochs_tfr = mne.time_frequency.read_tfrs(filepath)[0]
    else:
        with open(filepath, 'rb') as f:
            epochs_tfr = pickle.load(f)
    
    logger.info(f"Loaded EpochsTFR object from {filepath}")
    return epochs_tfr


def plot_subject_from_file(subject_name: str, cfg: Dict = None, params_hash: str = None):
    """
    Load saved TFR data and create plots for a subject.
    
    Parameters
    ----------
    subject_name : str
        Subject name
    cfg : Dict, optional
        Configuration dictionary
    params_hash : str, optional
        Parameter hash for filename
    """
    if cfg is None:
        cfg = create_cfg()
    
    # Load TFR data
    epochs_tfr = load_epochs_tfr(subject_name, cfg, params_hash)
    
    if epochs_tfr is None:
        logger.error(f"Could not load TFR data for {subject_name}")
        return
    
    # Create plots
    plots_path = get_plots_path(cfg, subject_name)
    plot_filename = f"{subject_name}_TFR_regions_plot.png"
    plot_path = os.path.join(plots_path, plot_filename)
    
    try:
        fig, axes = plot_tfr_by_regions(
            epochs_tfr=epochs_tfr,
            subject_name=subject_name,
            save_path=plot_path,
            figsize=(12, 15)
        )
        
        # Show plot
        plt.show()
        
        logger.info(f"Created TFR plots for {subject_name}")
        
    except Exception as e:
        logger.error(f"Error creating plots for {subject_name}: {e}")


def main():
    """
    Main execution function - can be used to process all subjects.
    """
    cfg = create_cfg()
    
    # Process all valid subjects
    for subj_idx in cfg['validSubs']:
        try:
            process_subject(subj_idx, cfg, create_plots=True)
        except Exception as e:
            logger.error(f"Error processing subject {subj_idx}: {e}")
            continue


if __name__ == "__main__":
    # Example usage
    # Process a single subject with plots
    epochs_tfr = process_subject(0, create_plots=True)
    
    # Or just create plots for existing data
    # plot_subject_from_file('Subject01')
    
    # Load a saved EpochsTFR object
    # epochs_tfr = load_epochs_tfr('Subject01')
    
    # Or process all subjects
    # main()
