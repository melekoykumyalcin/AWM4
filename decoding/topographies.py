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
locale.setlocale(locale.LC_ALL, "en_US.utf8")

# Create output directory for plots
plot_dir = os.path.join(homeDir, 'AWM4_data/processed/topographies(2)')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# List to store all evoked objects and subject IDs
all_evokeds = []
all_subject_ids = []

# Function to create a combined topography plot for all subjects
def plot_all_subjects_topographies(all_evokeds, subject_ids, save_dir=None):
    """Create a grid showing topographies for all subjects at key time points"""
    # Define time points of interest
    time_points = [0.53, 1.53, 2.2, 3.7, 4.7]
    time_labels = ['S1', 'S2', 'Cue', 'Ping', 'Probe']
    
    # Create figure with subplots
    n_subjects = len(all_evokeds)
    n_times = len(time_points)
    
    fig = plt.figure(figsize=(20, 4 * n_subjects))
    fig.suptitle('All Subjects - Topographies at Key Time Points (Magnetometers)', fontsize=16)
    
    # Create grid of subplots
    for subj_idx, (evoked, subj_id) in enumerate(zip(all_evokeds, subject_ids)):
        for time_idx, (time_point, time_label) in enumerate(zip(time_points, time_labels)):
            # Calculate subplot position
            ax_idx = subj_idx * n_times + time_idx + 1
            ax = plt.subplot(n_subjects, n_times, ax_idx)

            if time_label == 'Cue':
                evoked.copy().apply_baseline((1.7, 2.0)).plot_topomap(
                    times=2.2, ch_type='mag', show=False,
                    axes=ax, colorbar=False, sensors=False,
                    contours=0, time_unit='s', average=0.3)

            elif time_label == 'Ping':
                evoked.copy().apply_baseline((3.2, 3.5)).plot_topomap(
                    times=3.7, ch_type='mag', show=False,
                    axes=ax, colorbar=False, sensors=False,
                    contours=0, time_unit='s', average=0.3)

            else:
                evoked.plot_topomap(times=time_point, ch_type='mag', show=False, 
                                axes=ax, colorbar=False, sensors=False, 
                                contours=0, time_unit='s', average=0.3)
            
            # Add title for first row (time labels)
            if subj_idx == 0:
                ax.set_title(f'{time_label}\n{time_point}s', fontsize=12)
            else:
                ax.set_title(f'{time_point}s', fontsize=10)
            
            # Add subject label for first column
            if time_idx == 0:
                ax.text(-0.3, 0.5, f'Sub {subj_id}', transform=ax.transAxes, 
                       rotation=90, va='center', ha='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'all_subjects_topographies.png'), dpi=150)
        plt.close()
    else:
        plt.show()
    
    return fig

# Function to create a more compact topography grid (5x6 subjects, each with 5 timepoints)
def plot_all_subjects_topographies_compact(all_evokeds, subject_ids, save_dir=None):
    """Create a compact grid showing topographies for all subjects"""
    # Define time points of interest
    time_points = [0.53, 1.53, 2.2, 3.7, 4.7]
    time_labels = ['S1', 'S2', 'Cue', 'Ping', 'Probe']
    
    # Setup grid dimensions
    n_rows = 5
    n_cols = 6
    max_subjects = n_rows * n_cols
    
    # Create figure
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('All Subjects - Topographies at Key Time Points (Magnetometers)', fontsize=16)
    
    # Create subplots for each subject
    for idx, (evoked, subj_id) in enumerate(zip(all_evokeds[:max_subjects], subject_ids[:max_subjects])):
        # Create a subplot for this subject
        row = idx // n_cols
        col = idx % n_cols
        
        # Create a sub-grid for this subject's timepoints
        for t_idx, (time_point, time_label) in enumerate(zip(time_points, time_labels)):
            # Calculate position in the overall grid
            ax_idx = row * n_cols * len(time_points) + col * len(time_points) + t_idx + 1
            ax = plt.subplot(n_rows * len(time_points), n_cols, ax_idx)
            
            # Plot topography
            if time_label == 'Cue':
                evoked.copy().apply_baseline((1.7, 2.0)).plot_topomap(
                    times=2.2, ch_type='mag', show=False,
                    axes=ax, colorbar=False, sensors=False,
                    contours=0, time_unit='s', average=0.3)

            elif time_label == 'Ping':
                evoked.copy().apply_baseline((3.2, 3.5)).plot_topomap(
                    times=3.7, ch_type='mag', show=False,
                    axes=ax, colorbar=False, sensors=False,
                    contours=0, time_unit='s', average=0.3)

            else:
                evoked.plot_topomap(times=time_point, ch_type='mag', show=False, 
                                axes=ax, colorbar=False, sensors=False, 
                                contours=0, time_unit='s', average=0.3)

            
            # Remove axis labels to save space
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add time label at the top
            if row == 0:
                ax.text(0.5, 1.1, time_label, transform=ax.transAxes, 
                       ha='center', va='bottom', fontsize=10)
            
            # Add subject ID on the left
            if t_idx == 0:
                ax.text(-0.2, 0.5, f'S{subj_id}', transform=ax.transAxes, 
                       ha='right', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'all_subjects_topographies_compact.png'), dpi=150)
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_all_subjects_topo_timepoint(all_evokeds, subject_ids, time_point, time_label, save_dir=None):
    """Create a grid showing all subjects' topographies at a specific time point"""
    # Setup grid dimensions
    n_rows = 5
    n_cols = 6

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))
    fig.suptitle(f'All Subjects - Topographies at {time_label} ({time_point}s)', fontsize=16)
    axes = axes.flatten()

    # Plot each subject
    for i, (evoked, subj_id) in enumerate(zip(all_evokeds, subject_ids)):
        if i >= len(axes):
            break

        ax = axes[i]

        # Plot topography with specific baseline for Cue and Ping
        if time_label == 'Cue':
            evoked.copy().apply_baseline((1.7, 2.0)).plot_topomap(
                times=2.2, ch_type='mag', show=False,
                axes=ax, colorbar=False, sensors=False,
                contours=0, time_unit='s', average=0.3)

        elif time_label == 'Ping':
            evoked.copy().apply_baseline((3.2, 3.5)).plot_topomap(
                times=3.7, ch_type='mag', show=False,
                axes=ax, colorbar=False, sensors=False,
                contours=0, time_unit='s', average=0.3)

        else:
            evoked.plot_topomap(times=time_point, ch_type='mag', show=False, 
                                axes=ax, colorbar=False, sensors=False, 
                                contours=0, time_unit='s', average=0.3)

        ax.set_title(f'Sub {subj_id}')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, f'all_subjects_topo_{time_label}.png'), dpi=150)
        plt.close()
    else:
        plt.show()

    return fig

# Function to plot grand average topographies
def plot_grand_avg_topographies(grand_avg, save_dir=None):
    """Plot grand average topographies at key time points"""
    # Define time points of interest
    time_points = [0.53, 1.53, 2.2, 3.7, 4.7]
    time_labels = ['S1', 'S2', 'Cue', 'Ping', 'Probe']
    
    # Create figure
    fig, axes = plt.subplots(1, len(time_points), figsize=(20, 4))
    fig.suptitle('Grand Average - Topographies at Key Time Points (Magnetometers)', fontsize=16)
    
    # Plot topography at each time point
    for idx, (time_point, time_label, ax) in enumerate(zip(time_points, time_labels, axes)):
        if time_label == 'Cue':
            grand_avg.copy().apply_baseline((1.7, 2.0)).plot_topomap(
                times=2.2, ch_type='mag', show=False,
                axes=ax, colorbar=False,
                sensors=True, contours=6, time_unit='s', average=0.3)
        elif time_label == 'Ping':
            grand_avg.copy().apply_baseline((3.2, 3.5)).plot_topomap(
                times=3.7, ch_type='mag', show=False,
                axes=ax, colorbar=False,
                sensors=True, contours=6, time_unit='s', average=0.3)
        else:
            grand_avg.plot_topomap(
                times=time_point, ch_type='mag', show=False,
                axes=ax, colorbar=False,
                sensors=True, contours=6, time_unit='s', average=0.3)
        
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'grand_average_topographies.png'), dpi=150)
        plt.close()
    else:
        plt.show()
    
    return fig

# Function to create joint plots for individual subjects
def plot_subject_joint(evoked, subject_id, save_dir=None):
    # Create joint plot with only magnetometer channels
    fig = evoked.plot_joint(times=[0, 1, 2, 3.5, 4.5], 
                           title=f'Subject {subject_id} - Joint Plot',
                           picks='mag',
                           show=False)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'sub-{subject_id}_joint_plot.png'))
        plt.close()
    else:
        plt.show()
    
    return fig

# Loop through subjects and load data
print("Loading evoked data...")
for actSubj in Subs:
    tmp_evoked_fname = f"{homeDir}/AWM4_data/processed/ERFs/VP{actSubj}_alltrial_ERFs-ave.fif"
    
    if os.path.isfile(tmp_evoked_fname):
        print(f'Loading subject {actSubj}')
        
        # Load evoked data
        tmp_evoked = mne.read_evokeds(tmp_evoked_fname)[0]
        
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

        # Store for combined plot
        all_evokeds.append(tmp_evoked)
        all_subject_ids.append(actSubj)
    else:
        print(f'File not found for subject {actSubj}')

# Create only the new topography plots
if all_evokeds:
    print("\nCreating topography plots...")
    
    # Create combined topography plots
    print("Creating combined topography plots...")
    plot_all_subjects_topographies(all_evokeds, all_subject_ids, save_dir=plot_dir)
    plot_all_subjects_topographies_compact(all_evokeds, all_subject_ids, save_dir=plot_dir)
    
    # Create topography grids for each time point
    print("Creating time-point specific topography grids...")
    time_points = [(0, 'S1'), (1, 'S2'), (2, 'Cue'), (3.5, 'Ping'), (4.5, 'Probe')]
    for time_point, time_label in time_points:
        plot_all_subjects_topo_timepoint(all_evokeds, all_subject_ids, time_point, time_label, save_dir=plot_dir)
    
    # Create grand average and its topographies
    print("Creating grand average topographies...")
    grand_avg = mne.grand_average(all_evokeds)
    plot_grand_avg_topographies(grand_avg, save_dir=plot_dir)
    
    # Create joint plots for all subjects
    print("Creating joint plots for all subjects...")
    for evoked, subj_id in zip(all_evokeds, all_subject_ids):
        plot_subject_joint(evoked, subj_id, save_dir=plot_dir)

    print(f'\nAll topography plots saved to {plot_dir}')
else:
    print('No data found for plotting')