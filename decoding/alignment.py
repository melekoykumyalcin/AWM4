#!/usr/bin/env python
"""
Event Alignment Helper for TF Power Decoding

This script ensures proper alignment between TFR epochs and event labels
for the decoding analysis.
"""

import numpy as np
import mne
import os
import pandas as pd
import logging
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC

# Configure logging
logging.basicConfig(level=logging.INFO)

# Paths
HOME_DIR = '/mnt/hpc/projects/awm4/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
DATA_PATH = HOME_DIR + '/AWM4_data/raw/'
CORRECTED_DATA = HOME_DIR + '/AWM4_data/raw/correctTriggers/'

# Load metadata
metaInfo = pd.read_excel(META_FILE)
allFiles = metaInfo['MEG_Name']
corrected_files = [f.split('.')[0] + '_correct_triggers.fif' for f in allFiles]
corrected_files_series = pd.Series(corrected_files)

# Special cases
SUBJECT_SPECIAL_CASES = {
    23: {'drop_trials': [64*7]},  # 448
    28: {'drop_trials': [63]}
}

def get_aligned_events(subject, metaInfo):
    """
    Get properly aligned events for a subject, matching the epochs structure
    
    Returns:
    --------
    aligned_events : array of event codes matching the cleaned epochs
    """
    logging.info(f"Getting aligned events for subject {subject}")
    
    # Get subject files
    actInd = (metaInfo.Subject==subject) & (metaInfo.Valid==1)
    Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
    
    if subject in Subs[:7]:
        actFiles = corrected_files_series[actInd]
    else:
        actFiles = allFiles[actInd]
    
    # Load and concatenate events from raw files
    all_events = None
    reference_dev_head_t_ref = None
    
    for ff in range(actFiles.count()):
        if subject in Subs[:7]:
            fname = CORRECTED_DATA + actFiles.iloc[ff]
            raw = mne.io.read_raw_fif(fname, preload=True, verbose='ERROR')
        else:
            fname = DATA_PATH + actFiles.iloc[ff]
            raw = mne.io.read_raw_ctf(fname, 'truncate', True, verbose='ERROR')
        
        if ff == 0:
            reference_dev_head_t_ref = raw.info["dev_head_t"]
        else:
            raw.info['dev_head_t'] = reference_dev_head_t_ref
        
        events = mne.find_events(raw, 'UPPT001', shortest_event=1)
        if ff != 0:
            events = events[events[:, 1] == 0, :]
        
        if ff == 0:
            all_events = events
        else:
            all_events = np.concatenate((all_events, events), axis=0)
        del raw
    
    # Extract S1 and S2 event values
    s1_idx = [i - 1 for i in range(len(all_events[:,2])) if all_events[i,2] == 100]
    s1_values = all_events[s1_idx,2]
    
    s2_idx = [i - 1 for i in range(len(all_events[:,2])) if all_events[i,2] == 200]
    s2_values = all_events[s2_idx,2]
    
    # Combine S1 and S2 events (they appear in pairs)
    combined_events = []
    for i in range(len(s1_values)):
        combined_events.append(s1_values[i])
        if i < len(s2_values):
            combined_events.append(s2_values[i])
    
    combined_events = np.array(combined_events)
    
    # Load cleaned epochs to check how many we have
    epochs_file = f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif"
    epochs = mne.read_epochs(epochs_file, preload=False, verbose='ERROR')
    n_epochs = len(epochs)
    
    # Handle special cases - drop specific trials
    if subject in SUBJECT_SPECIAL_CASES:
        drop_trials = SUBJECT_SPECIAL_CASES[subject]['drop_trials']
        keep_mask = np.ones(len(combined_events), dtype=bool)
        for trial_idx in drop_trials:
            if trial_idx < len(keep_mask):
                keep_mask[trial_idx] = False
        combined_events = combined_events[keep_mask]
    
    # Check alignment
    if len(combined_events) != n_epochs:
        logging.warning(f"Event count mismatch for subject {subject}: "
                       f"{len(combined_events)} events vs {n_epochs} epochs")
        # Truncate to match epochs
        if len(combined_events) > n_epochs:
            combined_events = combined_events[:n_epochs]
        else:
            logging.error("Not enough events!")
            return None
    
    return combined_events

def verify_tfr_event_alignment(subject):
    """
    Verify that TFR data and events are properly aligned
    """
    # Get aligned events
    aligned_events = get_aligned_events(subject, metaInfo)
    if aligned_events is None:
        return False
    
    # Load TFR data
    tfr_file = f"{PROCESSED_DIR}/MorletTFR/data/sub-{subject}_tfr-nobaselinecor.h5"
    if os.path.exists(tfr_file):
        tfr = mne.time_frequency.read_tfrs(tfr_file)[0]
        
        # Check if number of epochs matches
        n_tfr_epochs = tfr.data.shape[0]
        n_events = len(aligned_events)
        
        if n_tfr_epochs != n_events:
            logging.error(f"Subject {subject}: TFR has {n_tfr_epochs} epochs but "
                         f"we have {n_events} events")
            return False
        
        logging.info(f"Subject {subject}: TFR and events are aligned "
                    f"({n_tfr_epochs} epochs)")
        return True
    else:
        logging.error(f"TFR file not found for subject {subject}")
        return False

def save_aligned_events(subject):
    """
    Save aligned events for easy loading in decoding script
    """
    aligned_events = get_aligned_events(subject, metaInfo)
    if aligned_events is not None:
        save_dir = f"{PROCESSED_DIR}/aligned_events/"
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(f"{save_dir}/sub-{subject}_events.npy", aligned_events)
        logging.info(f"Saved aligned events for subject {subject}")
        
        # Also save as text for inspection
        with open(f"{save_dir}/sub-{subject}_events.txt", 'w') as f:
            f.write(f"Subject {subject} Event Codes\n")
            f.write("="*30 + "\n")
            f.write(f"Total events: {len(aligned_events)}\n\n")
            
            # Count event types
            event_counts = {}
            for event in aligned_events:
                if event not in event_counts:
                    event_counts[event] = 0
                event_counts[event] += 1
            
            f.write("Event counts:\n")
            for event, count in sorted(event_counts.items()):
                f.write(f"  Event {event}: {count} occurrences\n")
            
            # List all events
            f.write("\nAll events in order:\n")
            for i, event in enumerate(aligned_events):
                f.write(f"  Trial {i:3d}: {event}\n")

def update_tf_decoding_to_use_aligned_events():
    """
    Generate updated code snippet for tf_power_decoding.py to use saved events
    """
    code_update = '''
# In process_subject_tf_decoding function, replace the event loading section with:

# Load pre-aligned events
aligned_events_file = f"{PROCESSED_DIR}/aligned_events/sub-{subject}_events.npy"
if not os.path.exists(aligned_events_file):
    logging.error(f"Aligned events file not found for subject {subject}")
    logging.info("Run event_alignment_helper.py first to generate aligned events")
    return None

events = np.load(aligned_events_file)

# The rest of the code remains the same...
'''
    
    print("\nUpdate for tf_power_decoding.py:")
    print("="*50)
    print(code_update)

def main():
    """Process all subjects to create aligned event files"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Event alignment helper')
    parser.add_argument('--subject', type=int, help='Process single subject')
    parser.add_argument('--verify', action='store_true', 
                       help='Verify TFR-event alignment')
    
    args = parser.parse_args()
    
    if args.subject:
        # Single subject
        if args.verify:
            verify_tfr_event_alignment(args.subject)
        else:
            save_aligned_events(args.subject)
    else:
        # All subjects
        Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
        
        failed_subjects = []
        for subject in Subs:
            try:
                save_aligned_events(subject)
                if args.verify:
                    if not verify_tfr_event_alignment(subject):
                        failed_subjects.append(subject)
            except Exception as e:
                logging.error(f"Failed to process subject {subject}: {str(e)}")
                failed_subjects.append(subject)
        
        if failed_subjects:
            logging.warning(f"Failed subjects: {failed_subjects}")
        else:
            logging.info("All subjects processed successfully!")
        
        # Show how to update the decoding script
        update_tf_decoding_to_use_aligned_events()

if __name__ == "__main__":
    main()