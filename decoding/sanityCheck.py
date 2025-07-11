import numpy as np
import os
import mne
import pandas as pd
import pickle
from mne.time_frequency import tfr_multitaper
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Configuration parameters
HOME_DIR = '/media/headmodel/Elements/AWM4/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
OUTPUT_DIR = PROCESSED_DIR + '/TFR_SanityCheck'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Analysis parameters
RESAMPLE_FREQ = 200  # Hz
NUM_JOBS = 8
TMIN = -0.5  # Include baseline period
TMAX = 3   # Longer for good high pass filtering
DISPLAY_TMAX = 1  


# Frequency bands for visualization
FREQ_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'low_gamma': (30, 45),
    'high_gamma': (60, 100)
}


SENSORS = {
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

subject = 2    
subject_dir = f"{OUTPUT_DIR}/sub-{subject}"
os.makedirs(subject_dir, exist_ok=True)
    
epochs_file = f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif"
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

# High-pass filter here
mag_epochs.filter(1, None)  # High-pass filter at 1 Hz

# Resample for efficiency
mag_epochs = mag_epochs.resample(RESAMPLE_FREQ, npad='auto')

        
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

# Plot slow oscillations TFR
fig = plt.figure(figsize=(10, 6))
power_slow.plot(combine='mean', baseline=[-0.4,-0.25], mode="percent", 
            title=f"Subject {subject} - Slow Oscillations (2-30 Hz) - TF Power",
            tmin=-0.5, tmax=1,
            vmin=-0.5, vmax=0.5,
            cmap='RdBu_r',
            show=False)

# Add vertical lines for important time points
plt.axvline(x=0, color='white', linestyle='--', linewidth=1, label='Stimulus 1 Onset')
plt.axvline(x=0.5, color='white', linestyle='--', linewidth=1, alpha = 0.5, label='Stimulus Offset')
plt.axvline(x=1, color='white', linestyle='--', linewidth=1, label='Stimulus 2 Onset')
plt.axvline(x=1.5, color='white', linestyle='--', linewidth=1, alpha = 0.5, label='Stimulus Offset')

plt.legend(loc='upper right')

plt.savefig(f"{subject_dir}/tfr_slow_oscillations.png")
plt.close()

# Without baseline correction
fig = plt.figure(figsize=(10, 6))
power_slow.plot(combine='mean', picks = SENSORS['temporal_right'],
            title=f"Subject {subject} - 2-30 Hz - Not Baseline Corrected",
            tmin=0, tmax=2,
            #vmin=-0.5, vmax=0.5,
            cmap='RdBu_r',
            show=False)

# Add vertical lines for important time points
plt.axvline(x=0, color='white', linestyle='--', linewidth=1, label='Stimulus 1 Onset')
plt.axvline(x=0.5, color='white', linestyle='--', linewidth=1, alpha = 0.5, label='Stimulus Offset')
plt.axvline(x=1, color='white', linestyle='--', linewidth=1, label='Stimulus 2 Onset')
plt.axvline(x=1.5, color='white', linestyle='--', linewidth=1, alpha = 0.5, label='Stimulus Offset')

plt.savefig(f"{subject_dir}/tfr_slow_oscillations_wobaseline.png")
plt.close()

# Plot slow oscillations TFR
fig = plt.figure(figsize=(10, 6))
power_slow.plot(combine='mean', baseline=[-0.4,-0.25], mode="percent", dB = True, 
            title=f"Subject {subject} - Slow Oscillations (2-30 Hz) - [dB]",
            tmin=-0.5, tmax=1,
            vmin=-0.5, vmax=0.5,
            cmap='RdBu_r',
            show=False)

# Add vertical lines for important time points
plt.axvline(x=0, color='white', linestyle='--', linewidth=1, label='Stimulus Onset')
plt.axvline(x=0.5, color='white', linestyle='--', linewidth=1, label='Stimulus Offset')
plt.legend(loc='upper right')

plt.savefig(f"{subject_dir}/tfr_slow_oscillations_dB.png")
plt.close()

# slow oscillations only with temporal sensors 
fig = plt.figure(figsize=(10, 6))
power_slow.plot(picks = SENSORS['temporal_left'], combine= 'mean', baseline=[-0.4,-0.25], mode="percent",
            cmap = 'RdBu_r', yscale = 'linear', 
            title=f"Subject {subject} - Slow Oscillations (2-30 Hz) - TF Power",
            tmin=0, tmax=2,
            vmin=-0.25, vmax=0.25,
            show=False)

# Add vertical lines for important time points
plt.axvline(x=0, color='white', linestyle='--', linewidth=1, label='Stimulus 1 Onset')
plt.axvline(x=0.5, color='white', linestyle='--', linewidth=1, alpha = 0.5, label='Stimulus Offset')
plt.axvline(x=1, color='white', linestyle='--', linewidth=1, label='Stimulus 2 Onset')
plt.axvline(x=1.5, color='white', linestyle='--', linewidth=1, alpha = 0.5, label='Stimulus Offset')

plt.savefig(f"{subject_dir}/tfr_slow_oscillations_temporal_left(1).png")
plt.close()

# Plot fast oscillations TFR
fig = plt.figure(figsize=(10, 6))
power_fast.plot(combine='mean', baseline=[-0.4,-0.25], mode='percent', 
            yscale = 'log',  
            title=f"Subject {subject} - Fast Oscillations (30-100 Hz) - TF Power",
            tmin=-0.5, tmax=1,
            vmin=-0.5, vmax=0.5,
            cmap='RdBu_r',
            show=False)

# Add vertical lines for important time points
plt.axvline(x=0, color='white', linestyle='--', linewidth=1, label='Stimulus Onset')
plt.axvline(x=0.5, color='white', linestyle=':', linewidth=1, label='Stimulus Offset')
plt.legend(loc='upper right')

plt.savefig(f"{subject_dir}/tfr_fast_oscillations.png")
plt.close()



import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from mne.time_frequency.tfr import rescale


# interactive one! 
picks = mne.pick_channels(power_slow.info['ch_names'], SENSORS['temporal_left'])
data = power_slow.data[picks].mean(axis=0)  # Average across selected channels
times = power_slow.times
freqs = power_slow.freqs

baselined_data = rescale(data, power_slow.times, baseline=(-0.4, -0.25), mode='percent')

# Create plotly figure
fig = make_subplots(rows=1, cols=1)

# Add heatmap
heatmap = go.Heatmap(
    z=baselined_data,
    x=times,
    y=freqs,
    colorscale='Viridis',
    # Remove fixed zmin and zmax to allow colorbar to be interactive
    colorbar=dict(
        title='Power change (%)',
        # Add colorbar slider buttons
        lenmode='fraction',
        len=0.75,
        thickness=20,
    ),
)
fig.add_trace(heatmap)

# Add vertical lines
fig.add_vline(x=0, line=dict(color="white", width=2, dash="dash"),
              annotation_text="Stimulus 1", annotation_position="top right")
fig.add_vline(x=1, line=dict(color="white", width=2, dash="dash"),
              annotation_text="Stimulus 2", annotation_position="top right")

# Update layout
fig.update_layout(
    title=f"Subject {subject} - Slow Oscillations (2-30 Hz) - TF Power",
    xaxis_title="Time (s)",
    yaxis_title="Frequency (Hz)",
    yaxis_type="linear",
    width=1000,
    height=600,
    # Add slider for colorscale manipulation
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.7,
            y=1.15,
            showactive=True,
            buttons=[
                dict(
                    label="Default",
                    method="restyle",
                    args=[{"zmin": -0.5, "zmax": 0.5}]
                ),
                dict(
                    label="Auto",
                    method="restyle",
                    args=[{"zmin": None, "zmax": None}]
                ),
                dict(
                    label="Narrow",
                    method="restyle",
                    args=[{"zmin": -0.25, "zmax": 0.25}]
                ),
                dict(
                    label="Wide",
                    method="restyle",
                    args=[{"zmin": -1.0, "zmax": 1.0}]
                ),
            ],
        )
    ]
)

# Cut the plot at tmax = 1
fig.update_xaxes(range=[0, 2])  # Set x-axis range from tmin to tmax

# Save as interactive HTML
fig.write_html(f"{subject_dir}/tfr_slow_oscillations_temporal_left_interactive.html")

# Add this before fig.show() for a more interactive color scale
fig.update_layout(
    sliders=[
        dict(
            active=2,
            currentvalue={"prefix": "Color Range: "},
            pad={"t": 50},
            steps=[
                dict(label="-2 to 2", method="restyle", args=[{"zmin": -2, "zmax": 2}]),
                dict(label="-1 to 1", method="restyle", args=[{"zmin": -1, "zmax": 1}]),
                dict(label="-0.5 to 0.5", method="restyle", args=[{"zmin": -0.5, "zmax": 0.5}]),
                dict(label="-0.25 to 0.25", method="restyle", args=[{"zmin": -0.25, "zmax": 0.25}]),
                dict(label="Auto", method="restyle", args=[{"zmin": None, "zmax": None}]),
            ]
        )
    ]
)

# Show the figure
fig.show()