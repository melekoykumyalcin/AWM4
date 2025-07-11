import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import mne
import pandas as pd
from matplotlib.animation import FuncAnimation

# Your existing setup
homeDir = '/media/headmodel/Elements/AWM4/'
os.chdir(homeDir)
metaFile = homeDir + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(metaFile)

# Get all subjects with FinalSample==1
Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1, 'Subject'])
print(f"Found {len(Subs)} subjects with FinalSample==1")

# Create output directory
output_dir = os.path.join(homeDir, 'AWM4_data/processed/topographies(2)')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List to store all evoked objects
all_evokeds = []

# Load all subjects' data
print("\nLoading evoked data for all subjects...")
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
        
        # Exclude bad sensors
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
        
        all_evokeds.append(tmp_evoked)
    else:
        print(f'File not found for subject {actSubj}')

# Create grand average
if all_evokeds:
    print(f"\nCreating grand average from {len(all_evokeds)} subjects...")
    grand_avg = mne.grand_average(all_evokeds)
    grand_avg = grand_avg.copy().pick('mag')


    # Time settings for animation
    times = np.arange(0, 5.0, 0.05)  # 50ms steps from 0 to 5 seconds
    
    # ANIMATION 1: Simple grand average animation
    print("\nCreating simple grand average animation...")
    
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    
    mag_picks = mne.pick_types(grand_avg.info, meg='mag', exclude='bads')

    def update_frame_simple(time_idx):
        ax1.clear()
        
        current_time = times[time_idx]
        
        # Apply baseline correction based on time period
        if 2.0 <= current_time <= 2.5:  # Around Cue time
            grand_avg_copy = grand_avg.copy().apply_baseline((1.7, 2.0))
            period_label = " (Cue baseline)"
        elif 3.5 <= current_time <= 4.0:  # Around Ping time
            grand_avg_copy = grand_avg.copy().apply_baseline((3.2, 3.5))
            period_label = " (Ping baseline)"
        else:
            grand_avg_copy = grand_avg.copy()
            period_label = ""
        
        # Get data at current time for magnetometers only
        time_idx_data = grand_avg_copy.time_as_index(current_time)[0]
        mag_data = grand_avg_copy.data[mag_picks, time_idx_data]
        
        # Create info object with only magnetometer channels
        mag_info = mne.pick_info(grand_avg_copy.info, mag_picks)
        
        # Plot topomap using low-level function
        im, _ = mne.viz.plot_topomap(
            mag_data,
            mag_info,
            axes=ax1,
            show=False,
            sensors=True,
            contours=6,
            cmap='RdBu_r'
        )
        
        # Add title
        ax1.set_title(f'Grand Average (n={len(all_evokeds)}) - Time: {current_time:.2f}s{period_label}', 
                    fontsize=14, pad=20)
        
        return [ax1]
    
    # Create and save animation
    anim1 = FuncAnimation(fig1, update_frame_simple, frames=len(times), 
                         interval=100, blit=False)
    anim1.save(os.path.join(output_dir, 'grand_avg_topomap_animation.gif'), 
               writer='pillow', fps=5)
    print("Simple grand average animation saved!")
    plt.close(fig1)
    
    # ANIMATION 2: Built-in MNE animation (if it works)
    print("\nTrying MNE's built-in animation...")
    try:
        grand_avg_mag = grand_avg.copy().pick('mag')
        fig2, anim2 = grand_avg_mag.animate_topomap(
            times=times, 
            ch_type="mag",
            frame_rate=10,
            time_unit='s',
            blit=False,
            butterfly=True,
        )
        anim2.save(os.path.join(output_dir, 'grand_avg_mne_animation.gif'), 
                   writer='pillow', fps=10)
        print("MNE animation saved!")
        plt.close(fig2)
    except Exception as e:
        print(f"MNE's animate_topomap failed: {e}")
    print(f'Created animations from {len(all_evokeds)} subjects')
    
else:
    print('No data found for creating grand average')