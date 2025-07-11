import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import locale
import pandas as pd

homeDir = '/media/headmodel/Elements/AWM4/'
os.chdir(homeDir)  # change current directory
metaFile = homeDir + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(metaFile)
Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1, 'Subject'])
#Subs = [1]
locale.setlocale(locale.LC_ALL, "en_US.utf8")
scalings = dict(mag=1e15)  # Convert from T to fT


# Create output directory for plots
plot_dir = os.path.join(homeDir, 'AWM4_data/processed/TemporalERFs')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# List to store all evoked objects and subject IDs
all_evokeds = []
all_subject_ids = []

# Define sensor groups as provided
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

def plot_butterfly_erfs(evoked, subject_id, save_dir=None):
    # Create a figure for this subject
    fig = plt.figure(figsize=(15, 12))
    # Plot left temporal sensors as butterfly plot
    ax1 = plt.subplot(2, 1, 1)
    left_fig = evoked.plot(picks=SENSORS['temporal_left'], exclude=[], 
                         unit=True, show=False, ylim=None, xlim=(-0.4, 5), 
                         proj=False, spatial_colors=True, time_unit='s',  
                         scalings = scalings, axes=ax1, selectable=True)
    ax1.set_title(f'Subject {subject_id} - Left Temporal Sensors (Butterfly)')
    
    # Add timepoint markers
    ymin, ymax = ax1.get_ylim()
    ax1.plot([0, 0], [ymin, ymax], 'k')
    ax1.text(0.05, ymax, 'S1')
    ax1.plot([1, 1], [ymin, ymax], 'k')
    ax1.text(1.05, ymax, 'S2')
    ax1.plot([2, 2], [ymin, ymax], 'k')
    ax1.text(2.05, ymax, 'Cue')
    ax1.plot([3.5, 3.5], [ymin, ymax], 'k')
    ax1.text(3.55, ymax, 'Ping')
    ax1.plot([4.5, 4.5], [ymin, ymax], 'k')
    ax1.text(4.55, ymax, 'Probe')
    
    # Plot right temporal sensors as butterfly plot
    ax2 = plt.subplot(2, 1, 2)
    right_fig = evoked.plot(picks=SENSORS['temporal_right'], exclude=[], 
                          unit=True, show=False, ylim=None, xlim=(-0.4, 5), 
                          proj=False, spatial_colors=True, time_unit='s', 
                          scalings = scalings, axes=ax2, selectable=True)
    ax2.set_title(f'Subject {subject_id} - Right Temporal Sensors (Butterfly)')
    
    # Add timepoint markers
    ymin, ymax = ax2.get_ylim()
    ax2.plot([0, 0], [ymin, ymax], 'k')
    ax2.text(0.05, ymax, 'S1')
    ax2.plot([1, 1], [ymin, ymax], 'k')
    ax2.text(1.05, ymax, 'S2')
    ax2.plot([2, 2], [ymin, ymax], 'k')
    ax2.text(2.05, ymax, 'Cue')
    ax2.plot([3.5, 3.5], [ymin, ymax], 'k')
    ax2.text(3.55, ymax, 'Ping')
    ax2.plot([4.5, 4.5], [ymin, ymax], 'k')
    ax2.text(4.55, ymax, 'Probe')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'sub-{subject_id}_butterfly_erfs.png'))
        plt.close()
    else:
        plt.show()
    
    return fig

#5x6 grid of butterfly plots with fixed ylim
def plot_all_subjects_butterfly(all_evokeds, subject_ids, save_dir=None):
    # Create figures for the grid - remove sharey=True to allow individual scaling
    fig_left, axes_left = plt.subplots(5, 6, figsize=(24, 20), sharex=True, sharey=False)
    fig_left.suptitle('All Subjects - Left Temporal Sensors (Butterfly)', fontsize=16)
    axes_left = axes_left.flatten()
    
    fig_right, axes_right = plt.subplots(5, 6, figsize=(24, 20), sharex=True, sharey=False)
    fig_right.suptitle('All Subjects - Right Temporal Sensors (Butterfly)', fontsize=16)
    axes_right = axes_right.flatten()
    
    # Plot each subject in the grid
    for i, (evoked, subj_id) in enumerate(zip(all_evokeds, subject_ids)):
        if i >= len(axes_left):
            break
        
        # Left temporal sensors - use fixed ylim
        evoked.plot(picks=SENSORS['temporal_left'], exclude=[], 
                   unit=True, show=False, ylim={'mag': (-350, 350)}, xlim=(-0.4, 5),
                   proj=False, spatial_colors=True, time_unit='s',
                   axes=axes_left[i], selectable=False)
        axes_left[i].set_title(f'Sub {subj_id}')
        
        # Add timepoint markers
        ymin, ymax = axes_left[i].get_ylim()  # Get the auto-determined limits
        axes_left[i].plot([0, 0], [ymin, ymax], 'k')
        axes_left[i].plot([1, 1], [ymin, ymax], 'k')
        axes_left[i].plot([2, 2], [ymin, ymax], 'k')
        axes_left[i].plot([3.5, 3.5], [ymin, ymax], 'k')
        axes_left[i].plot([4.5, 4.5], [ymin, ymax], 'k')
        
        # Right temporal sensors - use fixed ylim
        evoked.plot(picks=SENSORS['temporal_right'], exclude=[], 
                   unit=True, show=False, ylim={'mag': (-350, 350)}, xlim=(-0.4, 5),
                   proj=False, spatial_colors=True, time_unit='s',
                   axes=axes_right[i], selectable=False)
        axes_right[i].set_title(f'Sub {subj_id}')
        
        # Add timepoint markers
        ymin, ymax = axes_right[i].get_ylim()  # Get the auto-determined limits
        axes_right[i].plot([0, 0], [ymin, ymax], 'k')
        axes_right[i].plot([1, 1], [ymin, ymax], 'k')
        axes_right[i].plot([2, 2], [ymin, ymax], 'k')
        axes_right[i].plot([3.5, 3.5], [ymin, ymax], 'k')
        axes_right[i].plot([4.5, 4.5], [ymin, ymax], 'k')
    
    # Hide unused subplots
    for j in range(i+1, len(axes_left)):
        axes_left[j].axis('off')
        axes_right[j].axis('off')
    
    # Add common labels
    fig_left.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=14)
    fig_left.text(0.04, 0.5, 'Field Strength (fT)', va='center', rotation='vertical', fontsize=14)
    fig_right.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=14)
    fig_right.text(0.04, 0.5, 'Field Strength (fT)', va='center', rotation='vertical', fontsize=14)
    
    plt.tight_layout()
    
    if save_dir:
        fig_left.savefig(os.path.join(save_dir, 'all_subjects_left_butterfly.png'))
        fig_right.savefig(os.path.join(save_dir, 'all_subjects_right_butterfly.png'))
        plt.close(fig_left)
        plt.close(fig_right)
    else:
        plt.show()
    
    return fig_left, fig_right

# Function to plot ERFs for a single subject
def plot_subject_erfs(evoked, subject_id, save_dir=None):
    # Create a figure for this subject
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot left temporal sensors
    left_picks = [ch_name for ch_name in evoked.ch_names if ch_name in SENSORS['temporal_left']]
    if left_picks:
        left_data = evoked.copy().pick_channels(left_picks)
        left_data_avg = left_data.data.mean(axis=0)  # Average across channels
        times = left_data.times  # Times are in seconds in MNE
        
        # Find indices for the time range of interest (-0.4s to 5s)
        time_mask = (times >= -0.4) & (times <= 5)
        plot_times = times[time_mask]
        plot_data = left_data_avg[time_mask]
        
        axes[0].plot(plot_times, plot_data)
        axes[0].set_title(f'Subject {subject_id} - Left Temporal Sensors')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Field Strength')
        
        # Add timepoint markers
        ymin, ymax = axes[0].get_ylim()
        axes[0].plot([0, 0], [ymin, ymax], 'k')
        axes[0].text(0.05, ymax, 'S1')
        axes[0].plot([1, 1], [ymin, ymax], 'k')
        axes[0].text(1.05, ymax, 'S2')
        axes[0].plot([2, 2], [ymin, ymax], 'k')
        axes[0].text(2.05, ymax, 'Cue')
        axes[0].plot([3.5, 3.5], [ymin, ymax], 'k')
        axes[0].text(3.55, ymax, 'Ping')
        axes[0].plot([4.5, 4.5], [ymin, ymax], 'k')
        axes[0].text(4.55, ymax, 'Probe')
        
    # Plot right temporal sensors
    right_picks = [ch_name for ch_name in evoked.ch_names if ch_name in SENSORS['temporal_right']]
    if right_picks:
        right_data = evoked.copy().pick_channels(right_picks)
        right_data_avg = right_data.data.mean(axis=0)  # Average across channels
        times = right_data.times  # Times are in seconds in MNE
        
        # Find indices for the time range of interest (-0.4s to 5s)
        time_mask = (times >= -0.4) & (times <= 5)
        plot_times = times[time_mask]
        plot_data = right_data_avg[time_mask]
        
        axes[1].plot(plot_times, plot_data)
        axes[1].set_title(f'Subject {subject_id} - Right Temporal Sensors')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Field Strength')
        
        # Add timepoint markers
        ymin, ymax = axes[1].get_ylim()
        axes[1].plot([0, 0], [ymin, ymax], 'k')
        axes[1].text(0.05, ymax, 'S1')
        axes[1].plot([1, 1], [ymin, ymax], 'k')
        axes[1].text(1.05, ymax, 'S2')
        axes[1].plot([2, 2], [ymin, ymax], 'k')
        axes[1].text(2.05, ymax, 'Cue')
        axes[1].plot([3.5, 3.5], [ymin, ymax], 'k')
        axes[1].text(3.55, ymax, 'Ping')
        axes[1].plot([4.5, 4.5], [ymin, ymax], 'k')
        axes[1].text(4.55, ymax, 'Probe')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'sub-{subject_id}_temporalERFs.png'))
        plt.close()
    else:
        plt.show()
    
    return fig

# Function to create a combined plot with all subjects including joint plots
def plot_all_subjects_erfs(all_evokeds, subject_ids, save_dir=None):
    # First create the original ERF plots
    # Determine grid size for subplots
    n_rows = 5
    n_cols = 6
    
    # Create figure for left temporal sensors - remove sharey=True
    fig_left, axes_left = plt.subplots(n_rows, n_cols, figsize=(20, 16), sharex=True, sharey=False)
    fig_left.suptitle('All Subjects - Left Temporal Sensors', fontsize=16)
    axes_left = axes_left.flatten()
    
    # Create figure for right temporal sensors - remove sharey=True
    fig_right, axes_right = plt.subplots(n_rows, n_cols, figsize=(20, 16), sharex=True, sharey=False)
    fig_right.suptitle('All Subjects - Right Temporal Sensors', fontsize=16)
    axes_right = axes_right.flatten()
    
    # Plot each subject
    for i, (evoked, subj_id) in enumerate(zip(all_evokeds, subject_ids)):
        if i >= len(axes_left):  # Skip if we run out of subplots
            break
            
        # Plot left temporal sensors
        left_picks = [ch_name for ch_name in evoked.ch_names if ch_name in SENSORS['temporal_left']]
        if left_picks:
            left_data = evoked.copy().pick_channels(left_picks)
            left_data_avg = left_data.data.mean(axis=0)  # Average across channels
            times = left_data.times  # Times are in seconds in MNE
            
            # Find indices for the time range of interest (-0.4s to 5s)
            time_mask = (times >= -0.4) & (times <= 5)
            plot_times = times[time_mask]
            plot_data = left_data_avg[time_mask]
            
            axes_left[i].plot(plot_times, plot_data)
            axes_left[i].set_title(f'Sub {subj_id}')
            
            # Add timepoint markers
            ymin, ymax = axes_left[i].get_ylim()
            axes_left[i].plot([0, 0], [ymin, ymax], 'k')
            axes_left[i].plot([1, 1], [ymin, ymax], 'k')
            axes_left[i].plot([2, 2], [ymin, ymax], 'k')
            axes_left[i].plot([3.5, 3.5], [ymin, ymax], 'k')
            axes_left[i].plot([4.5, 4.5], [ymin, ymax], 'k')
        
        # Plot right temporal sensors
        right_picks = [ch_name for ch_name in evoked.ch_names if ch_name in SENSORS['temporal_right']]
        if right_picks:
            right_data = evoked.copy().pick_channels(right_picks)
            right_data_avg = right_data.data.mean(axis=0)  # Average across channels
            times = right_data.times  # Times are in seconds in MNE
            
            # Find indices for the time range of interest (-0.4s to 5s)
            time_mask = (times >= -0.4) & (times <= 5)
            plot_times = times[time_mask]
            plot_data = right_data_avg[time_mask]
            
            axes_right[i].plot(plot_times, plot_data)
            axes_right[i].set_title(f'Sub {subj_id}')
            
            # Add timepoint markers
            ymin, ymax = axes_right[i].get_ylim()
            axes_right[i].plot([0, 0], [ymin, ymax], 'k')
            axes_right[i].plot([1, 1], [ymin, ymax], 'k')
            axes_right[i].plot([2, 2], [ymin, ymax], 'k')
            axes_right[i].plot([3.5, 3.5], [ymin, ymax], 'k')
            axes_right[i].plot([4.5, 4.5], [ymin, ymax], 'k')
    
    # Hide unused subplots
    for j in range(i+1, len(axes_left)):
        axes_left[j].axis('off')
        axes_right[j].axis('off')
    
    # Add common labels
    fig_left.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=14)
    fig_left.text(0.04, 0.5, 'Field Strength', va='center', rotation='vertical', fontsize=14)
    fig_right.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=14)
    fig_right.text(0.04, 0.5, 'Field Strength', va='center', rotation='vertical', fontsize=14)
    
    plt.tight_layout()
    
    if save_dir:
        fig_left.savefig(os.path.join(save_dir, 'all_subjects_left_temporalERFs.png'))
        fig_right.savefig(os.path.join(save_dir, 'all_subjects_right_temporalERFs.png'))
        plt.close(fig_left)
        plt.close(fig_right)
    else:
        plt.show()
        
    return fig_left, fig_right

# Function to plot grand average butterfly plots
def plot_grand_avg_butterfly(all_evokeds, save_dir=None):
    # Create grand average
    grand_avg = mne.grand_average(all_evokeds)
    
    # Create a figure for grand average
    fig = plt.figure(figsize=(15, 12))
    
    # Plot left temporal sensors butterfly
    ax1 = plt.subplot(2, 1, 1)
    left_fig = grand_avg.plot(picks=SENSORS['temporal_left'], exclude=[], 
                         unit=True, show=False, ylim=None, xlim=(-0.4, 5), 
                         proj=False, spatial_colors=True, time_unit='s', 
                         scalings = scalings, axes=ax1, selectable=True)
    ax1.set_title(f'Grand Average - Left Temporal Sensors (Butterfly)')
    
    # Add timepoint markers
    ymin, ymax = ax1.get_ylim()
    ax1.plot([0, 0], [ymin, ymax], 'k')
    ax1.text(0.05, ymax, 'S1')
    ax1.plot([1, 1], [ymin, ymax], 'k')
    ax1.text(1.05, ymax, 'S2')
    ax1.plot([2, 2], [ymin, ymax], 'k')
    ax1.text(2.05, ymax, 'Cue')
    ax1.plot([3.5, 3.5], [ymin, ymax], 'k')
    ax1.text(3.55, ymax, 'Ping')
    ax1.plot([4.5, 4.5], [ymin, ymax], 'k')
    ax1.text(4.55, ymax, 'Probe')
    
    # Plot right temporal sensors butterfly
    ax2 = plt.subplot(2, 1, 2)
    right_fig = grand_avg.plot(picks=SENSORS['temporal_right'], exclude=[], 
                          unit=True, show=False, ylim=None, xlim=(-0.4, 5), 
                          proj=False, spatial_colors=True, time_unit='s', scalings=scalings,
                          axes=ax2, selectable=True)
    ax2.set_title(f'Grand Average - Right Temporal Sensors (Butterfly)')
    
    # Add timepoint markers
    ymin, ymax = ax2.get_ylim()
    ax2.plot([0, 0], [ymin, ymax], 'k')
    ax2.text(0.05, ymax, 'S1')
    ax2.plot([1, 1], [ymin, ymax], 'k')
    ax2.text(1.05, ymax, 'S2')
    ax2.plot([2, 2], [ymin, ymax], 'k')
    ax2.text(2.05, ymax, 'Cue')
    ax2.plot([3.5, 3.5], [ymin, ymax], 'k')
    ax2.text(3.55, ymax, 'Ping')
    ax2.plot([4.5, 4.5], [ymin, ymax], 'k')
    ax2.text(4.55, ymax, 'Probe')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'grand_average_butterfly.png'))
        plt.close()
    else:
        plt.show()
    
    return fig, grand_avg

# Function to plot grand average ERFs (averaged across sensors)
def plot_grand_avg_erf(grand_avg, save_dir=None):
    # Create a figure for grand average ERF
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot left temporal sensors
    left_picks = [ch_name for ch_name in grand_avg.ch_names if ch_name in SENSORS['temporal_left']]
    if left_picks:
        left_data = grand_avg.copy().pick_channels(left_picks)
        left_data_avg = left_data.data.mean(axis=0)  # Average across channels
        times = left_data.times  # Times are in seconds in MNE
        
        # Find indices for the time range of interest (-0.4s to 5s)
        time_mask = (times >= -0.4) & (times <= 5)
        plot_times = times[time_mask]
        plot_data = left_data_avg[time_mask]
        
        axes[0].plot(plot_times, plot_data, linewidth=2)
        axes[0].set_title(f'Grand Average - Left Temporal Sensors')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Field Strength (fT)')
        
        # Add timepoint markers
        ymin, ymax = axes[0].get_ylim()
        axes[0].plot([0, 0], [ymin, ymax], 'k')
        axes[0].text(0.05, ymax, 'S1')
        axes[0].plot([1, 1], [ymin, ymax], 'k')
        axes[0].text(1.05, ymax, 'S2')
        axes[0].plot([2, 2], [ymin, ymax], 'k')
        axes[0].text(2.05, ymax, 'Cue')
        axes[0].plot([3.5, 3.5], [ymin, ymax], 'k')
        axes[0].text(3.55, ymax, 'Ping')
        axes[0].plot([4.5, 4.5], [ymin, ymax], 'k')
        axes[0].text(4.55, ymax, 'Probe')
        
    # Plot right temporal sensors
    right_picks = [ch_name for ch_name in grand_avg.ch_names if ch_name in SENSORS['temporal_right']]
    if right_picks:
        right_data = grand_avg.copy().pick_channels(right_picks)
        right_data_avg = right_data.data.mean(axis=0)  # Average across channels
        times = right_data.times  # Times are in seconds in MNE
        
        # Find indices for the time range of interest (-0.4s to 5s)
        time_mask = (times >= -0.4) & (times <= 5)
        plot_times = times[time_mask]
        plot_data = right_data_avg[time_mask]
        
        axes[1].plot(plot_times, plot_data, linewidth=2)
        axes[1].set_title(f'Grand Average - Right Temporal Sensors')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Field Strength (fT)')
        
        # Add timepoint markers
        ymin, ymax = axes[1].get_ylim()
        axes[1].plot([0, 0], [ymin, ymax], 'k')
        axes[1].text(0.05, ymax, 'S1')
        axes[1].plot([1, 1], [ymin, ymax], 'k')
        axes[1].text(1.05, ymax, 'S2')
        axes[1].plot([2, 2], [ymin, ymax], 'k')
        axes[1].text(2.05, ymax, 'Cue')
        axes[1].plot([3.5, 3.5], [ymin, ymax], 'k')
        axes[1].text(3.55, ymax, 'Ping')
        axes[1].plot([4.5, 4.5], [ymin, ymax], 'k')
        axes[1].text(4.55, ymax, 'Probe')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'grand_average_erf.png'))
        plt.close()
    else:
        plt.show()
    
    return fig

# Loop through subjects
for actSubj in Subs:
    tmp_evoked_fname = f"{homeDir}/AWM4_data/processed/ERFs/VP{actSubj}_alltrial_ERFs-ave.fif"
    
    if os.path.isfile(tmp_evoked_fname):
        print(f'Processing subject {actSubj}')
        
        # Load evoked data
        tmp_evoked = mne.read_evokeds(tmp_evoked_fname)[0]
        tmp_evoked.filter(l_freq=0.1, h_freq=30, fir_design='firwin') # band pass filter (0.1–30 Hz)

        # Apply baseline correction for subjects 13 and 30
        if actSubj in [13, 30]:
            print(f'Applying baseline correction for subject {actSubj}')
            tmp_evoked.apply_baseline((None, 0))

        # Exclude sensor MRO31-3609 for subject 4 & 26 
        if actSubj in [4, 26]:
            print(f'Excluding sensor MRO31-3609 for subject {actSubj}')
            if 'MRO31-3609' in tmp_evoked.ch_names:
                if 'MRO31-3609' not in tmp_evoked.info['bads']:
                    tmp_evoked.info['bads'].append('MRO31-3609')

        if actSubj == 26:
            print(f'Excluding sensor MRO44-3609 for subject {actSubj}')
            if 'MRO44-3609' in tmp_evoked.ch_names:
                if 'MRO44-3609' not in tmp_evoked.info['bads']:
                    tmp_evoked.info['bads'].append('MRO44-3609')

        # Create individual plot for this subject
        plot_subject_erfs(tmp_evoked, actSubj, save_dir=plot_dir)
        plot_butterfly_erfs(tmp_evoked, actSubj, save_dir=plot_dir)

        # Store for combined plot
        all_evokeds.append(tmp_evoked)
        all_subject_ids.append(actSubj)
    else:
        print(f'File not found for subject {actSubj}')

# Create combined plot with all subjects
if all_evokeds:
    plot_all_subjects_erfs(all_evokeds, all_subject_ids, save_dir=plot_dir)
    plot_all_subjects_butterfly(all_evokeds, all_subject_ids, save_dir=plot_dir)
    _, grand_avg = plot_grand_avg_butterfly(all_evokeds, save_dir=plot_dir)
    plot_grand_avg_erf(grand_avg, save_dir=plot_dir)

    print(f'All plots saved to {plot_dir}')
else:
    print('No data found for plotting')