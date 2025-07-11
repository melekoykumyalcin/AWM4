#voice identity S2 grand average butterfly plots

# Import libraries
import os
import locale
from tqdm import tqdm
import pathlib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
plt.rcParams['figure.figsize'] = [20, 16]  # Larger figure for 4 subplots
import numpy as np
import mne
from mne.preprocessing import ICA
mne.set_log_level('warning')
import pandas as pd
from pynput.keyboard import Key, Controller

keyboard = Controller()

# Set paths
homeDir = '/media/headmodel/Elements/AWM4/'
os.chdir(homeDir) # change current directory

# Load in meta information
metaFile = homeDir + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(metaFile)
NrSessions = 1
Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
locale.setlocale(locale.LC_ALL, "en_US.utf8")

# paths and file names
data_path = homeDir +'/AWM4_data/raw/'
corrected_data = homeDir +'/AWM4_data/raw/correctTriggers/'
allFiles = metaInfo['MEG_Name']
noiseFiles = metaInfo['Noise_Measurement']
block = metaInfo['Block']
# for the Subs[:7]
corrected_files = [f.split('.')[0] + '_correct_triggers.fif' for f in allFiles]
corrected_files_series = pd.Series(corrected_files)

# saving the results
results_path = str(homeDir)+'/AWM4_data/processed/GrandAverageButterfly'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Event dictionary for S1 and S2
event_dict = {
    # S1 events
    'S1/Sp1/L1': 111, 'S1/Sp1/L2': 112, 'S1/Sp1/L3': 113, 'S1/Sp1/L4': 114,
    'S1/Sp2/L1': 121, 'S1/Sp2/L2': 122, 'S1/Sp2/L3': 123, 'S1/Sp2/L4': 124,
    'S1/Sp3/L1': 131, 'S1/Sp3/L2': 132, 'S1/Sp3/L3': 133, 'S1/Sp3/L4': 134,
    'S1/Sp4/L1': 141, 'S1/Sp4/L2': 142, 'S1/Sp4/L3': 143, 'S1/Sp4/L4': 144,
    # S2 events
    'S2/Sp1/L1': 211, 'S2/Sp1/L2': 212, 'S2/Sp1/L3': 213, 'S2/Sp1/L4': 214,
    'S2/Sp2/L1': 221, 'S2/Sp2/L2': 222, 'S2/Sp2/L3': 223, 'S2/Sp2/L4': 224,
    'S2/Sp3/L1': 231, 'S2/Sp3/L2': 232, 'S2/Sp3/L3': 233, 'S2/Sp3/L4': 234,
    'S2/Sp4/L1': 241, 'S2/Sp4/L2': 242, 'S2/Sp4/L3': 243, 'S2/Sp4/L4': 244
}

# Create lists to hold the evoked objects for both S1 and S2 periods
S1_Sp12_list = []
S1_Sp34_list = []
S1_L12_list = []
S1_L34_list = []

S2_Sp12_list = []
S2_Sp34_list = []
S2_L12_list = []
S2_L34_list = []

# Loop over subjects
for actSubj in tqdm(Subs, desc="Processing subjects"):
    print(f'Processing subject {actSubj}...')
    
    # Load cleaned and cut epochs per trial 
    try:
        CleanTrials = mne.read_epochs(str(homeDir)+'/AWM4_data/processed/CutEpochs/CutData_VP'+str(actSubj)+'-cleanedICA-epo.fif')
        
        # Get jump indices to drop if they exist
        jname = str(homeDir)+'/AWM4_data/processed/ICAs/Jumps'+str(actSubj)+'.npy'
        jump_inds = []
        if os.path.isfile(jname): 
            jump_inds = np.load(jname) 
            CleanTrials.drop(jump_inds, reason='jump')
        
        # Special case for subject 28
        if actSubj == 28:
            CleanTrials.drop(63)
        
        # Update event IDs based on the position in the trial sequence
        # Find which events correspond to S1 and S2
        actInd = (metaInfo.Subject==actSubj) & (metaInfo.Valid==1)
        if actSubj in Subs[:7]:
            actFiles = corrected_files_series[actInd]
        else:
            actFiles = allFiles[actInd]
        
        # Load events to find S1 and S2 values
        all_events = None
        for ff in range(actFiles.count()):
            if actSubj in Subs[:7]:
                fname = corrected_data + actFiles.iloc[ff]
                raw = mne.io.read_raw_fif(fname, preload=True)
            else:
                fname = data_path + actFiles.iloc[ff]
                raw = mne.io.read_raw_ctf(fname, 'truncate', True)
            
            events = mne.find_events(raw, 'UPPT001', shortest_event=1)
            if ff != 0:
                events = events[events[:, 1] == 0, :]
            
            if ff == 0:
                all_events = events
            else:
                all_events = np.concatenate((all_events, events), axis=0)
            del raw, events
        
        # Find S1 and S2 triggers
        S1_idx = [i - 1 for i in range(len(all_events[:,2])) if all_events[i,2] == 100]
        S1_values = all_events[S1_idx,2]
        
        S2_idx = [i - 1 for i in range(len(all_events[:,2])) if all_events[i,2] == 200]
        S2_values = all_events[S2_idx,2]
        
        # Handle special cases
        if actSubj == 23:
            drop_idx = 64*7
            S1_values = np.delete(S1_values, drop_idx)
            S2_values = np.delete(S2_values, drop_idx)
        
        if actSubj == 28:
            S1_values = np.delete(S1_values, 63)
            S2_values = np.delete(S2_values, 63)
        
        # Remove jump indices
        S1_values = np.delete(S1_values, jump_inds)
        S2_values = np.delete(S2_values, jump_inds)
        
        # Create copies for S1 and S2 periods
        S1Epochs = CleanTrials.copy()
        S1Epochs.crop(tmin=0, tmax=1)  # S1 period: 0-1 seconds
        S1Epochs.events[:,2] = S1_values
        S1Epochs.event_id = {k: v for k, v in event_dict.items() if k.startswith('S1')}
        
        S2Epochs = CleanTrials.copy()
        S2Epochs.crop(tmin=1, tmax=2)  # S2 period: 1-2 seconds
        S2Epochs.events[:,2] = S2_values
        S2Epochs.event_id = {k: v for k, v in event_dict.items() if k.startswith('S2')}
        
        del CleanTrials
        
        # Pick only magnetometers
        S1Epochs_mag = S1Epochs.copy().pick_types(meg='mag')
        S2Epochs_mag = S2Epochs.copy().pick_types(meg='mag')
        
        # Create averaged conditions for S1
        try:
            # Speaker conditions for S1
            S1Sp12 = S1Epochs_mag['S1/Sp1/L1', 'S1/Sp1/L2', 'S1/Sp1/L3', 'S1/Sp1/L4', 
                               'S1/Sp2/L1', 'S1/Sp2/L2', 'S1/Sp2/L3', 'S1/Sp2/L4'].average()
            S1Sp34 = S1Epochs_mag['S1/Sp3/L1', 'S1/Sp3/L2', 'S1/Sp3/L3', 'S1/Sp3/L4', 
                               'S1/Sp4/L1', 'S1/Sp4/L2', 'S1/Sp4/L3', 'S1/Sp4/L4'].average()
            
            # Location conditions for S1
            S1L12 = S1Epochs_mag['S1/Sp1/L1', 'S1/Sp2/L1', 'S1/Sp3/L1', 'S1/Sp4/L1', 
                              'S1/Sp1/L2', 'S1/Sp2/L2', 'S1/Sp3/L2', 'S1/Sp4/L2'].average()
            S1L34 = S1Epochs_mag['S1/Sp1/L3', 'S1/Sp2/L3', 'S1/Sp3/L3', 'S1/Sp4/L3', 
                              'S1/Sp1/L4', 'S1/Sp2/L4', 'S1/Sp3/L4', 'S1/Sp4/L4'].average()
            
            # Append to lists for grand averaging
            S1_Sp12_list.append(S1Sp12)
            S1_Sp34_list.append(S1Sp34)
            S1_L12_list.append(S1L12)
            S1_L34_list.append(S1L34)
        except:
            print(f"Could not create S1 averages for subject {actSubj}")
        
        # Create averaged conditions for S2
        try:
            # Speaker conditions for S2
            S2Sp12 = S2Epochs_mag['S2/Sp1/L1', 'S2/Sp1/L2', 'S2/Sp1/L3', 'S2/Sp1/L4', 
                               'S2/Sp2/L1', 'S2/Sp2/L2', 'S2/Sp2/L3', 'S2/Sp2/L4'].average()
            S2Sp34 = S2Epochs_mag['S2/Sp3/L1', 'S2/Sp3/L2', 'S2/Sp3/L3', 'S2/Sp3/L4', 
                               'S2/Sp4/L1', 'S2/Sp4/L2', 'S2/Sp4/L3', 'S2/Sp4/L4'].average()
            
            # Location conditions for S2
            S2L12 = S2Epochs_mag['S2/Sp1/L1', 'S2/Sp2/L1', 'S2/Sp3/L1', 'S2/Sp4/L1', 
                              'S2/Sp1/L2', 'S2/Sp2/L2', 'S2/Sp3/L2', 'S2/Sp4/L2'].average()
            S2L34 = S2Epochs_mag['S2/Sp1/L3', 'S2/Sp2/L3', 'S2/Sp3/L3', 'S2/Sp4/L3', 
                              'S2/Sp1/L4', 'S2/Sp2/L4', 'S2/Sp3/L4', 'S2/Sp4/L4'].average()
            
            # Append to lists for grand averaging
            S2_Sp12_list.append(S2Sp12)
            S2_Sp34_list.append(S2Sp34)
            S2_L12_list.append(S2L12)
            S2_L34_list.append(S2L34)
        except:
            print(f"Could not create S2 averages for subject {actSubj}")
    
    except Exception as e:
        print(f"Error processing subject {actSubj}: {e}")
        continue

# Create grand averages
print("Creating grand averages...")

# S1 Grand Averages
S1_Sp12_grand_avg = mne.grand_average(S1_Sp12_list)
S1_Sp34_grand_avg = mne.grand_average(S1_Sp34_list)
S1_L12_grand_avg = mne.grand_average(S1_L12_list)
S1_L34_grand_avg = mne.grand_average(S1_L34_list)

# S2 Grand Averages
S2_Sp12_grand_avg = mne.grand_average(S2_Sp12_list)
S2_Sp34_grand_avg = mne.grand_average(S2_Sp34_list)
S2_L12_grand_avg = mne.grand_average(S2_L12_list)
S2_L34_grand_avg = mne.grand_average(S2_L34_list)

# Plot S1 Grand Average Butterfly Plots (4 subplots)
print("Creating S1 butterfly plots...")
fig, axs = plt.subplots(2, 2, figsize=(20, 16))

# Speaker conditions
S1_Sp12_grand_avg.plot(axes=axs[0, 0], spatial_colors=True, show=False)
axs[0, 0].set_title('S1 - Speaker 1 & 2 (Grand Average)', fontsize=14)

S1_Sp34_grand_avg.plot(axes=axs[0, 1], spatial_colors=True, show=False)
axs[0, 1].set_title('S1 - Speaker 3 & 4 (Grand Average)', fontsize=14)

# Location conditions
S1_L12_grand_avg.plot(axes=axs[1, 0], spatial_colors=True, show=False)
axs[1, 0].set_title('S1 - Location 1 & 2 (Grand Average)', fontsize=14)

S1_L34_grand_avg.plot(axes=axs[1, 1], spatial_colors=True, show=False)
axs[1, 1].set_title('S1 - Location 3 & 4 (Grand Average)', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(results_path, 'S1_GrandAverage_Butterfly.png'), dpi=300)
plt.close()

# Plot S2 Grand Average Butterfly Plots (4 subplots)
print("Creating S2 butterfly plots...")
fig, axs = plt.subplots(2, 2, figsize=(20, 16))

# Speaker conditions
S2_Sp12_grand_avg.plot(axes=axs[0, 0], spatial_colors=True, show=False)
axs[0, 0].set_title('S2 - Speaker 1 & 2 (Grand Average)', fontsize=14)

S2_Sp34_grand_avg.plot(axes=axs[0, 1], spatial_colors=True, show=False)
axs[0, 1].set_title('S2 - Speaker 3 & 4 (Grand Average)', fontsize=14)

# Location conditions
S2_L12_grand_avg.plot(axes=axs[1, 0], spatial_colors=True, show=False)
axs[1, 0].set_title('S2 - Location 1 & 2 (Grand Average)', fontsize=14)

S2_L34_grand_avg.plot(axes=axs[1, 1], spatial_colors=True, show=False)
axs[1, 1].set_title('S2 - Location 3 & 4 (Grand Average)', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(results_path, 'S2_GrandAverage_Butterfly.png'), dpi=300)
plt.close()

print("Butterfly plots created and saved to:", results_path)