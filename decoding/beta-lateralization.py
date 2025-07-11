import numpy as np
import os
import mne
from mne.time_frequency import tfr_morlet
import matplotlib
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed
import pandas as pd
from matplotlib.animation import FuncAnimation

HOME_DIR = '/media/headmodel/Elements/AWM4/'
PROCESSED_DIR = HOME_DIR + 'AWM4_data/processed'
PLOT_DIR = f"{PROCESSED_DIR}/AllSensorsTFR"
os.makedirs(PLOT_DIR, exist_ok=True)
tfr_dir = f"{PROCESSED_DIR}/AllSensorsTFR/data"
os.makedirs(tfr_dir, exist_ok=True)

RESAMPLE_FREQ = 200  # Hz
NUM_JOBS = 16  # CPU cores for MNE processing
N_PARALLEL = 30   # Number of subjects to process in parallel
NUM_SUBJECTS = 30

DISPLAY_TMIN = 5
DISPLAY_TMAX = 5.5

FREQ_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'low_gamma': (30, 45),
    'high_gamma': (60, 100)
}

SENSORS = {
    'frontal': [
    'MLF11-3609', 'MLF12-3609', 'MLF13-3609', 'MLF14-3609',
    'MLF21-3609', 'MLF22-3609', 'MLF23-3609', 'MLF24-3609', 'MLF25-3609',
    'MLF31-3609', 'MLF32-3609', 'MLF33-3609', 'MLF34-3609', 'MLF35-3609',
    'MLF41-3609', 'MLF42-3609', 'MLF43-3609', 'MLF44-3609', 'MLF45-3609', 'MLF46-3609',
    'MLF51-3609', 'MLF52-3609', 'MLF53-3609', 'MLF54-3609', 'MLF55-3609', 'MLF56-3609',
    'MLF61-3609', 'MLF62-3609', 'MLF63-3609', 'MLF64-3609', 'MLF65-3609', 'MLF66-3609', 'MLF67-3609',
    'MRF11-3609', 'MRF12-3609', 'MRF13-3609', 'MRF14-3609',
    'MRF21-3609', 'MRF22-3609', 'MRF23-3609', 'MRF24-3609', 'MRF25-3609',
    'MRF31-3609', 'MRF32-3609', 'MRF33-3609', 'MRF34-3609', 'MRF35-3609',
    'MRF41-3609', 'MRF42-3609', 'MRF43-3609', 'MRF44-3609', 'MRF45-3609', 'MRF46-3609',
    'MRF51-3609', 'MRF52-3609', 'MRF53-3609', 'MRF54-3609', 'MRF55-3609', 'MRF56-3609',
    'MRF61-3609', 'MRF62-3609', 'MRF63-3609', 'MRF64-3609', 'MRF65-3609', 'MRF66-3609', 'MRF67-3609'
    ]
}

grand_avg = mne.time_frequency.read_tfrs(f"{tfr_dir}/grand_avg_tfr-tfr.h5")

grand_avg_dir = f"{PLOT_DIR}/sanity_check"
os.makedirs(grand_avg_dir, exist_ok=True)

# Plot left temporal sensors grand average
frontal_sensors = [ch for ch in SENSORS['frontal'] if ch in grand_avg.ch_names]

fig = plt.figure(figsize=(12, 8))
grand_avg.plot(
    picks=frontal_sensors,
    combine='mean', 
    baseline=(-0.4, -0.25),
    mode='logratio', 
    tmin=DISPLAY_TMIN, 
    tmax=DISPLAY_TMAX,
    cmap='RdBu_r', 
    yscale='log', 
    show=False,
    fmin=12, 
    fmax=30,
    vmin=-0.3, 
    vmax=0.3,
    title=f'Grand Average (N=30) - Frontal Sensors Morlet (-0.4, -0.25 baseline)' 
)
plt.axvline(x=0, color='white', linestyle='--', linewidth=1, label='Stimulus 1 Onset')
plt.axvline(x=0.5, color='white', linestyle='--', linewidth=0.5, label='Stimulus Offset')
plt.axvline(x=1, color='white', linestyle='--', linewidth=1, label='Stimulus 2 Onset')
plt.axvline(x=1.5, color='white', linestyle='--', linewidth=0.5, label='Stimulus Offset')
plt.axvline(x=3.5, color='white', linestyle='--', linewidth=1, label='Ping Onset')
plt.savefig(f"{grand_avg_dir}/frontal.png")
plt.close()

fig = plt.figure(figsize=(12, 8))
mne.viz.plot_tfr_topomap(grand_avg, baseline=(4, 4.5), tmin = DISPLAY_TMIN, tmax = DISPLAY_TMAX,
                        fmin = 12, fmax = 30, ch_type='mag', show=True, 
                        colorbar=True, sensors=False)
plt.savefig(f"{grand_avg_dir}/topo.png")

#animation 
times = np.arange(5, 5.5, 0.05)  # 50ms steps from 5 to 5.5 seconds

fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))
mag_picks = mne.pick_types(grand_avg.info, meg='mag', exclude='bads')
beta_fmin, beta_fmax = 12, 30
freqs = grand_avg.freqs
beta_freq_mask = (freqs >= beta_fmin) & (freqs <= beta_fmax)
beta_freq_indices = np.where(beta_freq_mask)[0]

def update_frame_simple(time_idx):
    ax1.clear()
    
    current_time = times[time_idx]
    
    # Check if time is within data range
    if current_time < grand_avg.times[0] or current_time > grand_avg.times[-1]:
        print(f"Warning: Time {current_time:.2f}s is outside data range "
              f"[{grand_avg.times[0]:.2f}, {grand_avg.times[-1]:.2f}]")
        return [ax1]
    
    # Get the closest time index in the data
    time_idx_data = np.argmin(np.abs(grand_avg.times - current_time))
    tfr_data = grand_avg.data[mag_picks, :, :]
    # Average over beta frequency band
    beta_data = np.mean(tfr_data[:, beta_freq_indices, time_idx_data], axis=1)
    # Create info object with only magnetometer channels
    mag_info = mne.pick_info(grand_avg.info, mag_picks)
    
    # Plot topomap for this time point
    im, _ = mne.viz.plot_topomap(
        beta_data,
        mag_info,
        axes=ax1,
        show=False,
        sensors=True,
        cmap='RdBu_r',
        vlim=(None, None),  # Auto-scale or set fixed scale
    )
    
    # Add title with frequency band info
    ax1.set_title(f'Beta Band ({beta_fmin}-{beta_fmax} Hz) - Time: {current_time:.2f}s',
                  fontsize=14, pad=20)
    
    return [ax1]

# Create animation
anim1 = FuncAnimation(fig1, update_frame_simple, frames=len(times),
                      interval=100, blit=False, repeat=True)
try:
    anim1.save(os.path.join(grand_avg_dir, 'tfr_beta_animated.gif'),
               writer='pillow', fps=5, dpi=100)
    print("Animation saved successfully!")
except Exception as e:
    print(f"Error saving animation: {e}")

plt.show()
plt.close(fig1)

# grand_avg_mag = grand_avg.copy().pick('mag')
# fig2, anim2 = grand_avg_mag.animate_topomap(
#     times=times, 
#     ch_type="mag",
#     frame_rate=10,
#     time_unit='s',
#     blit=False,
#     butterfly=True,
# )
# anim2.save(os.path.join(output_dir, 'grand_avg_mne_animation.gif'), 
#             writer='pillow', fps=10)
# print("MNE animation saved!")
# plt.close(fig2)
