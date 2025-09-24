#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe Artifact Rejection Script
Adapted from D_ArtifactRejection.py for probe-locked epochs
Interactive ECG and EOG artifact identification based on ICA
"""

#%% Setup ####################################################################

# Import libraries
import os
import locale
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
plt.rcParams['figure.figsize'] = [10, 8]
import numpy as np
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
import pandas as pd
import scipy.stats
from pynput.keyboard import Key, Controller
keyboard = Controller()

# Set paths
homeDir = '/media/headmodel/Elements/AWM4/'
PROBE_EPOCHS_PATH = homeDir + '/AWM4_data/processed/ProbeEpochs/'
ICA_PATH = homeDir + '/AWM4_data/processed/ICAs/'
TMP_PATH = homeDir + '/AWM4_data/processed/tmp/'

os.chdir(homeDir)

# Load meta information
metaFile = homeDir + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(metaFile)
Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
locale.setlocale(locale.LC_ALL, "en_US.utf8")

# Create function for plotting overview
def show_images(images, rows = 3, titles = None):
    """Display a list of images in a single figure with matplotlib."""
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: 
        titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    cols = int(np.ceil(n_images / float(rows)))
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(rows, cols, n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    return fig

#%% Interactive artifact rejection based on ICA ############################

# Parameters
decim = 4

# Subject selection
print("Available subjects:", Subs)
actSubj = Subs[int(input("Subject no [1-" + str(len(Subs)) + "]: "))-1] 
print(f"Selected subject: {actSubj}")

# Load probe epochs (cleaned from jumps)
probe_epochs_file = PROBE_EPOCHS_PATH + f'ProbeEpochs_VP{actSubj}-epo-dropped.fif'
if not os.path.exists(probe_epochs_file):
    print(f"Error: Cleaned probe epochs not found: {probe_epochs_file}")
    print("Please run probe jump detection first.")
    exit()

print(f"Loading cleaned probe epochs...")
ProbeTrials = mne.read_epochs(probe_epochs_file)
print(f"Loaded {len(ProbeTrials)} probe epochs")

# Load raw data for artifact detection (using existing processed raw file)
raw_file = homeDir + f'/AWM4_data/processed/CutEpochs/CutData_VP{actSubj}-raw.fif'
if not os.path.exists(raw_file):
    print(f"Error: Raw data file not found: {raw_file}")
    print("Using probe epochs for raw data simulation...")
    # Create a pseudo-raw from epochs if needed
    raw = ProbeTrials.copy().load_data()
    raw = mne.concatenate_epochs([raw])
    raw = raw.to_data_frame().reset_index()
    # This is a fallback - ideally use the original raw file
else:
    raw = mne.io.read_raw_fif(raw_file, preload=True)

# Load probe-specific ICA
ica_file = ICA_PATH + f'probe_VP{actSubj}_fastica-ica.fif'
if not os.path.exists(ica_file):
    print(f"Error: Probe ICA file not found: {ica_file}")
    print("Please run probe ICA computation first.")
    exit()

print(f"Loading probe ICA...")
ica = mne.preprocessing.read_ica(ica_file)

# Set up picks
picks = mne.pick_types(ProbeTrials.info, meg=True, eeg=True, stim=False, 
                    eog=False, ref_meg=False, include=[], exclude='bads') 
title = 'Sources related to %s artifacts (red)'      

print(f"ICA loaded with {ica.n_components_} components")

#%% ECG artifact detection ##################################################

print("\n=== ECG ARTIFACT DETECTION ===")
n_max_ecg = 10

# Check available EEG channels for ECG detection
eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch]
ecg_candidates = ['EEG059-3609', 'EEG059', 'ECG']

ecg_channel = None
for candidate in ecg_candidates:
    if candidate in raw.ch_names:
        ecg_channel = candidate
        break

if ecg_channel is None:
    print("No suitable ECG channel found. Available EEG channels:", eeg_channels[:10])
    print("Skipping ECG detection or specify manual channel:")
    manual_ecg = input("Enter ECG channel name (or press Enter to skip): ").strip()
    if manual_ecg and manual_ecg in raw.ch_names:
        ecg_channel = manual_ecg

try:
    if ecg_channel:
        print(f"Using ECG channel: {ecg_channel}")
        # Generate ECG epochs with better error handling
        ecg_epochs = create_ecg_epochs(raw, ch_name=ecg_channel, tmin=-.5, tmax=.5, picks=picks)
        
        if len(ecg_epochs) == 0:
            print("No ECG epochs detected. Trying automatic detection...")
            ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5, picks=picks)  # Let MNE auto-detect
        
        if len(ecg_epochs) > 0:
            ecg_epochs.decimate(decim).apply_baseline((None,None))
            ecg_inds, scores = ica.find_bads_ecg(ecg_epochs)
            
            if len(ecg_inds) > 0:
                # Plot ECG components
                ica.plot_scores(scores, exclude=ecg_inds, title=title % 'ecg', labels='ecg')
                show_picks = np.abs(scores).argsort()[::-1][:n_max_ecg]
                ica.plot_sources(raw, show_picks, title=title % 'ecg', stop=60)
                ica.plot_components(show_picks, title=title % 'ecg', colorbar=True)
                ica.plot_sources(ecg_epochs.average(), title='ECG average')
                ica.plot_overlay(ecg_epochs.average(), exclude=ecg_inds, picks='mag')
                
                print(f"Suggested ECG components: {ecg_inds}")
                ans = input("Do you agree with ECG components? [y/n] ")
                if ans != "y":
                    ecg_inds = list(map(int, input("Please enter the ECG components (as: x y z): ").split())) 
            else:
                print("No ECG components automatically detected")
                manual_ecg_comps = input("Enter ECG components manually (or press Enter to skip): ").strip()
                if manual_ecg_comps:
                    ecg_inds = list(map(int, manual_ecg_comps.split()))
                else:
                    ecg_inds = []
        else:
            print("No ECG epochs could be created")
            ecg_inds = []
            scores = []
    else:
        print("Skipping ECG detection - no suitable channel")
        ecg_inds = []
        scores = []
    
    plt.close('all')
    
except Exception as e:
    print(f"ECG detection failed: {e}")
    ecg_inds = []
    scores = []

#%% Create ECG overview figures
if ecg_inds:
    tmpfig, ax = plt.subplots(1, len(ecg_inds))
    ica.plot_components(ecg_inds, colorbar=True, show=False, axes=ax)
    tmpfig.savefig(TMP_PATH + 'tmp1.png')

    with mne.viz.use_browser_backend("matplotlib"):
        tmpfig2 = ica.plot_sources(raw, ecg_inds, show_scrollbars=False)
    tmpfig2.savefig(TMP_PATH + 'tmp2.png')

    tmpfig3 = ica.plot_overlay(ecg_epochs.average(), exclude=ecg_inds, picks='mag', show=False)
    tmpfig3.savefig(TMP_PATH + 'tmp3.png')
else:
    # Create empty plots if no ECG components
    plt.figure(); plt.savefig(TMP_PATH + 'tmp1.png'); plt.close()
    plt.figure(); plt.savefig(TMP_PATH + 'tmp2.png'); plt.close()
    plt.figure(); plt.savefig(TMP_PATH + 'tmp3.png'); plt.close()

#%% EOG artifact detection - Horizontal (EEG060-3609) ######################

print("\n=== EOG1 ARTIFACT DETECTION (Horizontal) ===")
n_max_eog = 20

# Check available EOG channels
eog1_candidates = ['EEG060-3609', 'EEG060', 'EOG-H', 'EOG061', 'EEG061-3609']
eog1_channel = None

for candidate in eog1_candidates:
    if candidate in raw.ch_names:
        eog1_channel = candidate
        break

if eog1_channel is None:
    print("No suitable horizontal EOG channel found.")
    print("Available EEG channels (first 10):", [ch for ch in raw.ch_names if 'EEG' in ch][:10])
    manual_eog1 = input("Enter horizontal EOG channel name (or press Enter to skip): ").strip()
    if manual_eog1 and manual_eog1 in raw.ch_names:
        eog1_channel = manual_eog1

try:
    if eog1_channel:
        print(f"Using horizontal EOG channel: {eog1_channel}")
        eog_epochs1 = create_eog_epochs(raw, tmin=-.2, tmax=.2, ch_name=eog1_channel, 
                                       picks=picks, l_freq=.5, h_freq=20)
        
        if len(eog_epochs1) == 0:
            print("No horizontal EOG epochs detected")
            eog_inds1 = []
            scores_eog1 = []
        else:
            eog_epochs1.decimate(decim).apply_baseline((None, None))
            
            # Try to find bad EOG components - handle channel name issues
            try:
                eog_inds1, scores_eog1 = ica.find_bads_eog(eog_epochs1, ch_name=eog1_channel, measure='correlation')
            except:
                # Fallback: try without specifying channel name
                eog_inds1, scores_eog1 = ica.find_bads_eog(eog_epochs1, measure='correlation')
            
            if len(eog_inds1) > 0:
                ica.plot_scores(scores_eog1, exclude=eog_inds1, title=title % 'eog1', labels='eog1') 
                show_picks1 = np.abs(scores_eog1).argsort()[::-1][:n_max_eog]
                ica.plot_sources(raw, show_picks1, title=title % 'eog1')
                ica.plot_components(show_picks1, title=title % 'eog1', colorbar=True)
                ica.plot_sources(eog_epochs1.average(), title='EOG 1 average')
                ica.plot_overlay(eog_epochs1.average(), exclude=eog_inds1, picks='mag')
                
                print(f"Suggested EOG1 components: {eog_inds1}")
                ans = input("Do you agree with EOG1 components? [y/n] ")
                if ans != "y":
                    eog_inds1 = list(map(int, input("Please enter the EOG1 components (as: x y z): ").split())) 
            else:
                print("No horizontal EOG components automatically detected")
                manual_eog1_comps = input("Enter EOG1 components manually (or press Enter to skip): ").strip()
                if manual_eog1_comps:
                    eog_inds1 = list(map(int, manual_eog1_comps.split()))
                else:
                    eog_inds1 = []
    else:
        print("Skipping horizontal EOG detection")
        eog_inds1 = []
    
    plt.close('all')
    
except Exception as e:
    print(f"EOG1 detection failed: {e}")
    eog_inds1 = []

#%% Create EOG1 overview figures
if eog_inds1:
    tmpfig4, ax = plt.subplots(1, len(eog_inds1))
    ica.plot_components(eog_inds1, colorbar=True, show=False, axes=ax)
    tmpfig4.savefig(TMP_PATH + 'tmp4.png')
    
    with mne.viz.use_browser_backend("matplotlib"):
        tmpfig5 = ica.plot_sources(raw, eog_inds1, stop=120, show=False)
    tmpfig5.savefig(TMP_PATH + 'tmp5.png')
    
    if eog_epochs1: 
        tmpfig6 = ica.plot_overlay(eog_epochs1.average(), exclude=eog_inds1, picks='mag', show=False)
    else:
        plt.figure()
        tmpfig6 = plt.gca()
    tmpfig6.figure.savefig(TMP_PATH + 'tmp6.png')
else:
    # Create empty plots
    plt.figure(); plt.savefig(TMP_PATH + 'tmp4.png'); plt.close()
    plt.figure(); plt.savefig(TMP_PATH + 'tmp5.png'); plt.close()
    plt.figure(); plt.savefig(TMP_PATH + 'tmp6.png'); plt.close()

#%% EOG artifact detection - Vertical (EEG058-3609) ########################

print("\n=== EOG2 ARTIFACT DETECTION (Vertical) ===")

# Check available EOG channels for vertical
eog2_candidates = ['EEG058-3609', 'EEG058', 'EOG-V', 'EOG059', 'EEG059-3609']
eog2_channel = None

for candidate in eog2_candidates:
    if candidate in raw.ch_names:
        eog2_channel = candidate
        break

if eog2_channel is None:
    print("No suitable vertical EOG channel found.")
    print("Available EEG channels (first 10):", [ch for ch in raw.ch_names if 'EEG' in ch][:10])
    manual_eog2 = input("Enter vertical EOG channel name (or press Enter to skip): ").strip()
    if manual_eog2 and manual_eog2 in raw.ch_names:
        eog2_channel = manual_eog2

try:
    if eog2_channel:
        print(f"Using vertical EOG channel: {eog2_channel}")
        eog_epochs2 = create_eog_epochs(raw, tmin=-.2, tmax=.2, ch_name=eog2_channel, 
                                       picks=picks, l_freq=.5, h_freq=20)
        
        if len(eog_epochs2) == 0:
            print("No vertical EOG epochs detected")
            eog_inds2 = []
            scores_eog2 = []
        else:
            eog_epochs2.decimate(decim).apply_baseline((None, None))
            
            # Try to find bad EOG components - handle channel name issues
            try:
                eog_inds2, scores_eog2 = ica.find_bads_eog(eog_epochs2, ch_name=eog2_channel, measure='correlation')
            except:
                # Fallback: try without specifying channel name
                eog_inds2, scores_eog2 = ica.find_bads_eog(eog_epochs2, measure='correlation')
            
            if len(eog_inds2) > 0:
                ica.plot_scores(scores_eog2, exclude=eog_inds2, title=title % 'eog2', labels='eog2')
                show_picks2 = np.abs(scores_eog2).argsort()[::-1][:n_max_eog]
                ica.plot_sources(raw, show_picks2, title=title % 'eog2')
                ica.plot_components(show_picks2, title=title % 'eog2', colorbar=True)
                ica.plot_sources(eog_epochs2.average(), title='EOG2 average')
                ica.plot_overlay(eog_epochs2.average(), exclude=eog_inds2, picks='mag')
                
                print(f"Suggested EOG2 components: {eog_inds2}")
                ans = input("Do you agree with EOG2 components? [y/n] ")
                if ans != "y":
                    eog_inds2 = list(map(int, input("Please enter the EOG2 components (as: x y z): ").split())) 
            else:
                print("No vertical EOG components automatically detected")
                manual_eog2_comps = input("Enter EOG2 components manually (or press Enter to skip): ").strip()
                if manual_eog2_comps:
                    eog_inds2 = list(map(int, manual_eog2_comps.split()))
                else:
                    eog_inds2 = []
    else:
        print("Skipping vertical EOG detection")
        eog_inds2 = []
    
    plt.close('all')
    
except Exception as e:
    print(f"EOG2 detection failed: {e}")
    eog_inds2 = []

#%% Create EOG2 overview figures
if eog_inds2:
    tmpfig7, ax = plt.subplots(1, len(eog_inds2))
    ica.plot_components(eog_inds2, colorbar=True, show=False, axes=ax)
    tmpfig7.savefig(TMP_PATH + 'tmp7.png')
    
    with mne.viz.use_browser_backend("matplotlib"):
        tmpfig8 = ica.plot_sources(raw, eog_inds2, stop=120)
    tmpfig8.savefig(TMP_PATH + 'tmp8.png')
    
    tmpfig9 = ica.plot_overlay(eog_epochs2.average(), exclude=eog_inds2, picks='mag', show=False)
    tmpfig9.savefig(TMP_PATH + 'tmp9.png')
else:
    # Create empty plots
    plt.figure(); plt.savefig(TMP_PATH + 'tmp7.png'); plt.close()
    plt.figure(); plt.savefig(TMP_PATH + 'tmp8.png'); plt.close()
    plt.figure(); plt.savefig(TMP_PATH + 'tmp9.png'); plt.close()

#%% Create comprehensive overview figure ####################################

# Load all temporary images
image = []
titles = []
for im in range(9):
    try:
        image.append(plt.imread(TMP_PATH + f'tmp{im+1}.png'))
        if im < 3:  # ECG
            titles.extend(['ECG components', 'ECG sources', 'ECG exclusion'][im:im+1])
        elif im < 6:  # EOG1
            titles.extend(['EOG1 components', 'EOG1 sources', 'EOG1 exclusion'][im-3:im-2])
        else:  # EOG2
            titles.extend(['EOG2 components', 'EOG2 sources', 'EOG2 exclusion'][im-6:im-5])
    except:
        # Handle missing images
        image.append(np.zeros((100, 100, 3)))
        titles.append(f'Missing {im+1}')

plt.close('all')

# Create and save overview figure
sumfig = show_images(image, rows=3, titles=titles)
sumfig.savefig(ICA_PATH + f'ProbeICA_Exclusions_VP{actSubj}.pdf')

# Create component overview
compfig = ica.plot_components(range(min(20, ica.n_components_)), 
                             title='First Components', colorbar=True)
compfig.savefig(ICA_PATH + f'ProbeICA_Comps_VP{actSubj}.pdf')
plt.close('all')

#%% Combine all exclusions and save clean ICA ##############################

all_inds = list(np.unique(ecg_inds + eog_inds1 + eog_inds2))
print(f"\nFinal artifact components to exclude: {all_inds}")
print(f"Total components excluded: {len(all_inds)}")

# Apply exclusions to ICA
ica.exclude = all_inds
clean_ica_file = ICA_PATH + f'CleanProbeVP{actSubj}-ica.fif'
ica.save(clean_ica_file, overwrite=True)
print(f"Saved clean probe ICA: {clean_ica_file}")

#%% Apply clean ICA to probe epochs and save ################################

print("Applying clean ICA to probe epochs...")
ica_probe_epochs = ProbeTrials.copy() 
ica.apply(ica_probe_epochs)

# Save cleaned probe epochs
clean_epochs_file = PROBE_EPOCHS_PATH + f'ProbeEpochs_VP{actSubj}-cleanedICA-epo.fif'
ica_probe_epochs.save(clean_epochs_file, overwrite=True)
print(f"Saved ICA-cleaned probe epochs: {clean_epochs_file}")

# Print final trial counts
print(f"\nFinal cleaned probe epoch counts:")
for event_name, event_id in ica_probe_epochs.event_id.items():
    count = len(ica_probe_epochs[event_name])
    print(f"  {event_name}: {count} trials")

total_trials = len(ica_probe_epochs)
print(f"Total cleaned probe trials: {total_trials}")

plt.close('all')
print(f"\n=== PROBE ARTIFACT REJECTION COMPLETE FOR SUBJECT {actSubj} ===")