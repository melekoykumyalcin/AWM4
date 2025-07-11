import numpy as np
import os
import mne
import pandas as pd
import pickle 
from mne.time_frequency import tfr_morlet
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse

# Configuration parameters
HOME_DIR = '/mnt/hpc/projects/awm4/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
DATA_PATH = HOME_DIR + '/AWM4_data/raw/'
CORRECTED_DATA = HOME_DIR + '/AWM4_data/raw/correctTriggers/'
DELAY_CONFIG = {
    'tmin': 2.0,
    'tmax': 4.7,
}
# Decoding parameters
CV_SPLITS = 5
CV_REPEATS = 10
NUM_JOBS = 8  # Adjust to match --cpus-per-task in your job script
RESAMPLE_FREQ = 100  # Hz
BIN_SIZE = 10

# Create output directories
TF_DECODE_DIR = f"{PROCESSED_DIR}/TF_Decoding"
os.makedirs(TF_DECODE_DIR, exist_ok=True)
metaInfo = pd.read_excel(META_FILE)
    
# Event dictionary
EVENT_DICT = {
    'S1/Sp1/L1': 111, 'S1/Sp1/L2': 112, 'S1/Sp1/L3': 113, 'S1/Sp1/L4': 114,
    'S1/Sp2/L1': 121, 'S1/Sp2/L2': 122, 'S1/Sp2/L3': 123, 'S1/Sp2/L4': 124,
    'S1/Sp3/L1': 131, 'S1/Sp3/L2': 132, 'S1/Sp3/L3': 133, 'S1/Sp3/L4': 134,
    'S1/Sp4/L1': 141, 'S1/Sp4/L2': 142, 'S1/Sp4/L3': 143, 'S1/Sp4/L4': 144,
    'S2/Sp1/L1': 211, 'S2/Sp1/L2': 212, 'S2/Sp1/L3': 213, 'S2/Sp1/L4': 214,
    'S2/Sp2/L1': 221, 'S2/Sp2/L2': 222, 'S2/Sp2/L3': 223, 'S2/Sp2/L4': 224,
    'S2/Sp3/L1': 231, 'S2/Sp3/L2': 232, 'S2/Sp3/L3': 233, 'S2/Sp3/L4': 234,
    'S2/Sp4/L1': 241, 'S2/Sp4/L2': 242, 'S2/Sp4/L3': 243, 'S2/Sp4/L4': 244,
    'Cue_S1': 101, 'Cue_S2': 201
}

# Feature Definitions for Maintained Information
FEATURES = {
    'maintained_voice_identity': {
        'name': 'Maintained Voice Identity',
        'color': '#1c686b',  # Darker teal
        'description': 'Voice identity information maintained in working memory'
    },
    'maintained_location': {
        'name': 'Maintained Location',
        'color': '#cb6a3e',  # Darker orange
        'description': 'Location information maintained in working memory'
    }
}

def extract_maintained_information(subject, metaInfo):
    """
    Extract which information is maintained in working memory based on retro-cues
    
    Args:
        subject (int): Subject number
        metaInfo (DataFrame): Metadata information
    
    Returns:
        tuple: Memorized stimulus codes for each trial, original events array
    """ 
    print(f'\n=== Extracting maintained information for subject {subject} ===')
    
    try:
        # Get file information
        actInd = (metaInfo.Subject==subject) & (metaInfo.Valid==1)
        
        # Determine if subject is in early subjects (with corrected files)
        early_subject = subject in np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])[:7]
        
        if early_subject:
            actFiles = pd.Series([f.split('.')[0] + '_correct_triggers.fif' for f in metaInfo['MEG_Name']])[actInd]
        else:
            actFiles = metaInfo['MEG_Name'][actInd]
        
        # Load and concatenate events from raw files
        all_events = None
        reference_dev_head_t_ref = None
        
        # Process each file
        for ff in range(actFiles.count()):
            if early_subject:
                fname = CORRECTED_DATA + actFiles.iloc[ff]
                raw = mne.io.read_raw_fif(fname, preload=True)
            else:
                fname = DATA_PATH + actFiles.iloc[ff]
                raw = mne.io.read_raw_ctf(fname, 'truncate', True)
                
            if ff == 0:
                reference_dev_head_t_ref = raw.info["dev_head_t"]
            else:
                raw.info['dev_head_t'] = reference_dev_head_t_ref
                
            events = mne.find_events(raw, 'UPPT001', shortest_event=1)
            if ff != 0:
                events = events[events[:, 1] == 0, :]
                
            if np.any(events[:, 1] < 0):
                raise ValueError('Faulty trigger found, please inspect manually.')
                
            if ff == 0:
                all_events = events
            else:
                all_events = np.concatenate((all_events, events), axis=0)
            del raw
        
        print(f'Raw data loaded for subject {subject}')
        
        # Find all retro-cue events (101 for S1, 201 for S2)
        S1_cue_indices = np.where(all_events[:,2] == 101)[0]
        S2_cue_indices = np.where(all_events[:,2] == 201)[0]
        
        print(f"Found {len(S1_cue_indices)} S1 retro-cues and {len(S2_cue_indices)} S2 retro-cues")
        
        # Create memorized array to track which stimulus is in working memory
        memorized = np.zeros(len(all_events[:,2]))
        
        # For S1 cues, the maintained stimulus is 4 positions before
        for i in S1_cue_indices:
            if i >= 4:  # Ensure we have enough preceding events
                memorized[i - 4] = all_events[i - 4, 2]
        
        # For S2 cues, the maintained stimulus is 2 positions before
        for i in S2_cue_indices:
            if i >= 2:  # Ensure we have enough preceding events
                memorized[i - 2] = all_events[i - 2, 2]
        
        # Keep only the non-zero values (corresponding to maintained stimuli)
        memorized_values = memorized[memorized != 0]
        
        # Print diagnostics
        valid_values = [v for v in memorized_values if v in EVENT_DICT.values()]
        invalid_values = [v for v in memorized_values if v not in EVENT_DICT.values()]
        
        print(f"Maintained stimuli: {len(memorized_values)} total trials")
        print(f"Valid trial codes: {len(valid_values)}")
        if invalid_values:
            print(f"WARNING: Found {len(invalid_values)} invalid maintained values: {invalid_values}")
            
        # Count by speaker and location
        speakers = {
            'Sp1+2': sum(1 for v in memorized_values if (v//10)%10 in [1, 2]),
            'Sp3+4': sum(1 for v in memorized_values if (v//10)%10 in [3, 4])
        }
        
        locations = {
            'L1+2': sum(1 for v in memorized_values if v%10 in [1, 2]),
            'L3+4': sum(1 for v in memorized_values if v%10 in [3, 4])
        }
        
        print("\nMaintained Stimulus Distribution:")
        print(f"Speaker groups: {speakers}")
        print(f"Location groups: {locations}")
        
        return memorized_values, all_events
        
    except Exception as e:
        print(f"Error extracting maintained information for subject {subject}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def average_maintained_condition_groups(epochs, memorized_values, n_iterations=10):
    """
    Average trials within condition groups for the delay phase with multiple iterations.
    Ensures each condition group produces exactly the same number of averaged trials.
    Maintains randomization throughout the process.
    """
    # Define combined condition groups (including both S1 and S2 stimuli)
    CONDITION_GROUPS = {
        'Sp1+2/L1+2': [111, 112, 121, 122, 211, 212, 221, 222],
        'Sp1+2/L3+4': [113, 114, 123, 124, 213, 214, 223, 224],
        'Sp3+4/L1+2': [131, 132, 141, 142, 231, 232, 241, 242],
        'Sp3+4/L3+4': [133, 134, 143, 144, 233, 234, 243, 244]
    }
    
    # Get data and events
    data = epochs.get_data()
    events = memorized_values
    
    print(f"Data shape: {data.shape}")
    print(f"Memorized events shape: {len(events)}")
    print(f"Unique maintained conditions: {np.unique(events)}")
    
    # Initialize storage
    averaged_data = []
    trial_records = []
    
    # For each iteration
    for iter_idx in range(n_iterations):
        print(f"\nProcessing iteration {iter_idx + 1}/{n_iterations}")
        
        # First, analyze all condition groups to determine available trial counts
        condition_trials = {}
        condition_indices = {}
        min_trials_per_group = float('inf')
        
        for group_name, event_codes in CONDITION_GROUPS.items():
            # Get trials for each condition in the group
            condition_trials[group_name] = {code: [] for code in event_codes}
            condition_indices[group_name] = {code: [] for code in event_codes}
            
            for idx, event in enumerate(events):
                if event in event_codes:
                    condition_trials[group_name][event].append(data[idx])
                    condition_indices[group_name][event].append(idx)
            
            # Report available trials for each condition
            print(f"\nCondition Group: {group_name}")
            for code, trials in condition_trials[group_name].items():
                print(f"  Event {code}: {len(trials)} trials")
            
            # Find conditions with at least one trial
            valid_conditions = {code: trials for code, trials in condition_trials[group_name].items() if len(trials) > 0}
            
            # Skip if no valid conditions or insufficient conditions
            if not valid_conditions or len(valid_conditions) < 2:
                print(f"  Warning: Skipping group {group_name} - insufficient valid conditions")
                min_trials_per_group = 0  # Force skip this iteration
                continue
            
            # Calculate minimum number of trials across valid conditions
            group_min_trials = min(len(trials) for trials in valid_conditions.values())
            
            # Update the global minimum
            min_trials_per_group = min(min_trials_per_group, group_min_trials)
        
        # If any group has insufficient trials, skip this iteration
        if min_trials_per_group < 1:
            print(f"Skipping iteration {iter_idx + 1} - one or more groups have insufficient trials")
            continue
        
        global_used_trials = {}
        # Create trial plan - this is the key change
        # We'll create a randomized array of all the trials we plan to include
        all_averaged_trials = []
        all_averaged_labels = []
        all_trial_records = {}
        
        # For each group, prepare the trials we'll use (but don't add them sequentially yet)
        group_trials_plan = {}
        
        for group_name, group_conditions in condition_trials.items():
            valid_conditions = {code: trials for code, trials in group_conditions.items() if len(trials) > 0}
            if len(valid_conditions) < 2:
                continue
                
            group_trials = []
            group_records = []
            
            used_trials = {code: [] for code in valid_conditions.keys()}

            # Create the planned averaged trials for this group
            for trial_idx in range(min_trials_per_group):
                averaged_trial = np.zeros_like(data[0])
                trial_record = {}
                conditions_used = 0
                
                for code in valid_conditions.keys():
                    # Get pool of available trials
                    trial_pool = list(set(range(len(valid_conditions[code]))) - 
                                     set([used_trial[code] for used_trial in group_records 
                                          if code in used_trial]))
                    
                    if code not in global_used_trials:
                        global_used_trials[code] = []
                    
                    # Get pool of available trials
                    trial_pool = list(set(range(len(valid_conditions[code]))) - 
                                     set(used_trials[code]) -
                                     set(global_used_trials[code]))
                    
                    if not trial_pool:
                        continue
                        
                    # Randomly select one trial
                    selected_idx = np.random.choice(trial_pool)
                    
                    # Get the original trial index
                    original_idx = condition_indices[group_name][code][selected_idx]
                    
                    # Record which original trial was used
                    trial_record[code] = original_idx
                    
                    # Update both local and global used trials
                    used_trials[code].append(selected_idx)
                    global_used_trials[code].append(selected_idx)
                    
                    # Add to average
                    averaged_trial += valid_conditions[code][selected_idx]
                    conditions_used += 1
                
                # Skip if no conditions were used
                if conditions_used == 0:
                    continue
                
                # Complete averaging
                averaged_trial /= conditions_used
                
                # Store in plan
                group_trials.append((averaged_trial, list(CONDITION_GROUPS.keys()).index(group_name)))
                group_records.append(trial_record)
            
            # Add to group plan
            group_trials_plan[group_name] = (group_trials, group_records)
        
        # Now randomize the order in which we add trials from different groups
        all_trials = []
        all_labels = []
        all_records = []
        
        # Get equal number of trials from each group
        min_group_trials = min(len(trials) for trials, _ in group_trials_plan.values())
        
        for trial_idx in range(min_group_trials):
            # Randomly shuffle the order of groups for this position
            group_names = list(group_trials_plan.keys())
            np.random.shuffle(group_names)
            
            for group_name in group_names:
                trials, records = group_trials_plan[group_name]
                if trial_idx < len(trials):
                    trial, label = trials[trial_idx]
                    record = records[trial_idx]
                    
                    all_trials.append(trial)
                    all_labels.append(label)
                    all_records.append((group_name, record))
        
        # Check if we have enough trials and all four condition groups are represented
        if len(all_trials) > 0:
            unique_groups = np.unique(all_labels)
            if len(unique_groups) == 4:
                # Store this iteration's results
                averaged_data.append((
                    np.array(all_trials),
                    np.array(all_labels)
                ))
                
                # Create trial record for this iteration
                iter_trial_record = {'iteration': iter_idx + 1}
                for i, (group_name, record) in enumerate(all_records):
                    iter_trial_record[f"{group_name}_trial{i+1}"] = record
                
                trial_records.append(iter_trial_record)
                
                # Report condition counts for this iteration
                group_counts = {group: np.sum(np.array(all_labels) == group) for group in unique_groups}
                print(f"Iteration {iter_idx + 1} completed with {len(all_trials)} averaged trials:")
                for group, count in group_counts.items():
                    group_name = list(CONDITION_GROUPS.keys())[group]
                    print(f"  Group {group} ({group_name}): {count} trials")
            else:
                print(f"Skipping iteration {iter_idx + 1} - not all condition groups represented")
    
    # Report final results
    print(f"\nAveraging completed: {len(averaged_data)} valid iterations created")
    if len(averaged_data) > 0:
        for i, (X, y) in enumerate(averaged_data):
            print(f"Iteration {i+1}: {X.shape[0]} averaged trials")
            
            # Verify randomization - count transitions between different labels
            transitions = np.sum(np.diff(y) != 0)
            expected_transitions = len(y) - len(np.unique(y))
            transition_ratio = transitions / expected_transitions if expected_transitions > 0 else 0
            
            print(f"  Randomization check - Transition ratio: {transition_ratio:.2f}")
            print(f"  First 20 labels: {y[:20]}")
            
            # Print distribution of group codes
            unique_codes, code_counts = np.unique(y, return_counts=True)
            for code, count in zip(unique_codes, code_counts):
                group_name = list(CONDITION_GROUPS.keys())[code]
                print(f"  Group {code} ({group_name}): {count} trials")
    else:
        print("WARNING: No valid iterations were created!")
    
    return averaged_data, trial_records

def compute_tf_transforms(subject, feature_name='maintained_voice_identity'):
    """
    Compute time-frequency transforms on full epochs, apply baseline,
    then average and decode
    """
    print(f"\n=== Computing TF transforms for subject {subject}, feature {feature_name} ===")

    try:
        # First load the clean epochs with baseline period included
        clean_trials = mne.read_epochs(
            f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif",
            preload=True,
            verbose='INFO'
        )
        print(f"Loaded clean epochs: {len(clean_trials)}")
        print(f"Epochs time range: {clean_trials.times[0]} to {clean_trials.times[-1]} seconds")
        
        # Special case handling for specific subjects
        if subject == 28:
            clean_trials.drop(63)
        if subject == 23:
            clean_trials.drop(64*7)
        
        # Extract maintained information based on retro-cues
        memorized, all_events = extract_maintained_information(subject, metaInfo)
        
        if memorized is None:
            print(f"Could not extract maintained information for subject {subject}")
            return None
        
        # Update event codes to reflect maintained information
        clean_trials.events[:, 2] = memorized
        clean_trials.event_id = EVENT_DICT
        
        # Check for jump artifacts and drop affected epochs
        jname = f"{PROCESSED_DIR}/ICAs/Jumps{subject}.npy"
        if os.path.isfile(jname):
            jump_inds = np.load(jname)
            
            if len(jump_inds) > 0:
                jump_inds = np.array(jump_inds, dtype=int)
                valid_jump_inds = jump_inds[jump_inds < len(clean_trials)]
                
                if len(valid_jump_inds) > 0:
                    clean_trials.drop(valid_jump_inds, reason='jump')
                    memorized = np.delete(memorized, valid_jump_inds)
        
        # Select magnetometer channels
        mag_epochs = clean_trials.copy().pick_types(meg='mag')
        
        # Resample
        mag_epochs = mag_epochs.resample(RESAMPLE_FREQ, npad='auto')
        print(f"Epochs after resampling: {len(mag_epochs)}")
        
        # Now perform the averaging using our existing function
        memorized_values = mag_epochs.events[:, 2]
        averaged_data, trial_records = average_maintained_condition_groups(mag_epochs, memorized_values)
        
        if not averaged_data:
            print(f"No valid averaged data for subject {subject}")
            return None
        
        # Define frequency bands of interest
        freqs = np.logspace(np.log10(4), np.log10(45), 20)
        
        # Use frequency-dependent cycles (modern approach)
        n_cycles = freqs / 2
        
        # Define baseline - this should be available in clean epochs
        baseline = (-0.5, -0.2)
        
        # Store TF data for each iteration
        tf_data_list = []
        
        # Process each iteration of averaged data
        for iter_idx, (X, y) in enumerate(averaged_data):
            print(f"\nProcessing iteration {iter_idx + 1}/{len(averaged_data)}")
            
            # Create binary classification labels based on the feature
            if feature_name == 'maintained_voice_identity':
                # Speaker decoding: 0,1 (Sp1+2) vs 2,3 (Sp3+4)
                binary_y = np.where(y < 2, 0, 1)
                group_desc = "Voice Identity (0,1=Sp1+2 vs 2,3=Sp3+4)"
            else:  # maintained_location
                # Location decoding: 0,2 (L1+2) vs 1,3 (L3+4)
                binary_y = np.where(y % 2 == 0, 0, 1)
                group_desc = "Location (0,2=L1+2 vs 1,3=L3+4)"
            
            # Verify we have both classes
            unique_classes = np.unique(binary_y)
            if len(unique_classes) < 2:
                print(f"Iteration {iter_idx + 1} has only class {unique_classes[0]} after binarization. Skipping.")
                continue
            
            # Print class distribution
            print(f"{group_desc} Class distribution:")
            for cls in unique_classes:
                print(f"Class {cls}: {np.sum(binary_y == cls)} trials")
            
            # Get data dimensions
            n_trials, n_channels, n_times = X.shape
            
            # Create an info object for MNE
            # Use the same time range as the clean epochs
            full_tmin = clean_trials.times[0]  # Should be -0.5 according to your info
            
            info = mne.create_info(ch_names=[f'CH{i}' for i in range(n_channels)],
                                  sfreq=RESAMPLE_FREQ,
                                  ch_types=['misc'] * n_channels)
            
            # Create EpochsArray object from our averaged data but with the full time range
            epochs = mne.EpochsArray(X, info, tmin=full_tmin)
            
            # Compute time-frequency decomposition on FULL epochs
            print("Computing Morlet wavelet transform...")
            power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                             use_fft=True, return_itc=False, average=False,
                             decim=1, n_jobs=NUM_JOBS, picks='misc')
            
            # Apply baseline correction
            print(f"Applying baseline correction ({baseline[0]} to {baseline[1]}s)...")
            power.apply_baseline(baseline, mode='logratio')
            
            # Crop to just the delay period
            print(f"Cropping to delay period ({DELAY_CONFIG['tmin']} to {DELAY_CONFIG['tmax']}s)...")
            power.crop(tmin=DELAY_CONFIG['tmin'], tmax=DELAY_CONFIG['tmax'])
            
            # Store the power data and labels
            tf_data_list.append((power.data, binary_y, power.times, power.freqs))
            
            # Save the TF data
            tf_save_path = f"{TF_DECODE_DIR}/sub-{subject}/iter-{iter_idx}"
            os.makedirs(tf_save_path, exist_ok=True)
            with open(f"{tf_save_path}/{feature_name}_tf_data.pkl", 'wb') as f:
                pickle.dump((power.data, binary_y, power.times, power.freqs), f)
        
        return tf_data_list
    
    except Exception as e:
        print(f"Error computing TF transforms for subject {subject}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def decode_tf_data(tf_data, times, freqs, subject=None, feature_name=None, iter_idx=None):
    """
    Perform decoding on time-frequency data
    
    Parameters:
    tf_data - TF data in shape (n_trials, n_channels, n_freqs, n_times)
    times - time points
    freqs - frequency values
    
    Returns:
    scores - decoding accuracy over time
    """
    X, y, _, _ = tf_data
    n_trials, n_channels, n_freqs, n_times = X.shape
    
    # Define frequency bands of interest
    freq_bands = {
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'low_gamma': (30, 45)
    }
    
    # Create band-specific features
    band_features = {}
    for band_name, (fmin, fmax) in freq_bands.items():
        # Find frequencies within this band
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        
        if not np.any(freq_mask):
            print(f"Warning: No frequencies in the {band_name} band ({fmin}-{fmax} Hz)")
            continue
        
        # Average power across frequencies in the band, for each channel and time point
        band_data = np.mean(X[:, :, freq_mask, :], axis=2)  # (n_trials, n_channels, n_times)
        band_features[band_name] = band_data
    
    # Initialize results storage
    scores = np.zeros((len(freq_bands), n_times))
    
    # Define parameter grid for SVM
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    
    # Process each frequency band
    for band_idx, (band_name, band_data) in enumerate(band_features.items()):
        print(f"\nDecoding {band_name} band...")
        
        # Process each time point
        def process_timepoint(t_idx):
            # Extract data for this time point
            X_t = band_data[:, :, t_idx].reshape(n_trials, -1)
            
            # Setup cross-validation
            cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS // 10)
            
            # Create pipeline with scaling and SVM
            pipeline = make_pipeline(
                StandardScaler(),
                GridSearchCV(
                    SVC(kernel='linear', probability=True),
                    param_grid=param_grid,
                    cv=RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=2),
                    scoring='accuracy'
                )
            )
            
            # Perform cross-validation
            scores = []
            for train_idx, test_idx in cv.split(X_t, y):
                X_train, X_test = X_t[train_idx], X_t[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                pipeline.fit(X_train, y_train)
                score = pipeline.score(X_test, y_test)
                scores.append(score)
            
            return np.mean(scores)
        
        # Run parallel processing across time points
        time_scores = Parallel(n_jobs=NUM_JOBS)(
            delayed(process_timepoint)(t_idx) for t_idx in range(n_times)
        )
        
        scores[band_idx, :] = time_scores
    
    # Save results
    if subject is not None and feature_name is not None and iter_idx is not None:
        results_path = f"{TF_DECODE_DIR}/sub-{subject}/iter-{iter_idx}"
        os.makedirs(results_path, exist_ok=True)
        
        # Save to NumPy file
        np.savez(
            f"{results_path}/{feature_name}_decoding_results.npz",
            scores=scores,
            times=times,
            freq_bands=list(freq_bands.keys())
        )
        
        # Plot results
        plt.figure(figsize=(12, 8))
        for band_idx, band_name in enumerate(freq_bands.keys()):
            plt.plot(times, scores[band_idx], label=band_name)
        
        plt.axhline(0.5, color='k', linestyle='--', label='Chance')
        plt.xlabel('Time (s)')
        plt.ylabel('Decoding Accuracy')
        plt.title(f'Subject {subject}, Iteration {iter_idx}, {FEATURES[feature_name]["name"]}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{results_path}/{feature_name}_tf_decoding.png")
        plt.close()
    
    return scores, list(freq_bands.keys())

def process_iteration_results(iteration_results, subject, feature_name):
    """Process and save averaged results across iterations"""
    # Ensure all iterations have the same bands and time points
    bands = iteration_results[0][1]
    times = iteration_results[0][2]
    
    # Average scores across iterations
    avg_scores = np.zeros((len(bands), len(times)))
    valid_iterations = 0
    
    for scores, _, _ in iteration_results:
        if scores.shape == avg_scores.shape:
            avg_scores += scores
            valid_iterations += 1
    
    if valid_iterations > 0:
        avg_scores /= valid_iterations
        
        # Save average results
        results_path = f"{TF_DECODE_DIR}/sub-{subject}"
        os.makedirs(results_path, exist_ok=True)
        
        np.savez(
            f"{results_path}/{feature_name}_avg_results.npz",
            scores=avg_scores,
            bands=bands,
            times=times,
            n_iterations=valid_iterations
        )
        
        # Plot average results
        plt.figure(figsize=(12, 8))
        for band_idx, band_name in enumerate(bands):
            plt.plot(times, avg_scores[band_idx], label=band_name)
        
        plt.axhline(0.5, color='k', linestyle='--', label='Chance')
        plt.xlabel('Time (s)')
        plt.ylabel('Decoding Accuracy')
        plt.title(f'Subject {subject}, {FEATURES[feature_name]["name"]} (N={valid_iterations} iterations)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{results_path}/{feature_name}_avg_tf_decoding.png")
        plt.close()
        
        return {
            'scores': avg_scores,
            'bands': bands,
            'times': times,
            'n_iterations': valid_iterations
        }
    
    return None

def generate_group_level_results(subjects, features):
    """Generate group-level results for all features"""
    for feature_name in features:
        # Collect results for this feature across subjects
        all_subject_data = []
        valid_subjects = []
        
        for subject in subjects:
            result_file = f"{TF_DECODE_DIR}/sub-{subject}/{feature_name}_avg_results.npz"
            
            if os.path.exists(result_file):
                try:
                    # Load this subject's results
                    data = np.load(result_file, allow_pickle=True)
                    
                    all_subject_data.append({
                        'scores': data['scores'],
                        'bands': data['bands'],
                        'times': data['times'],
                        'n_iterations': int(data['n_iterations'])
                    })
                    
                    valid_subjects.append(subject)
                except Exception as e:
                    print(f"Error loading results for subject {subject}: {str(e)}")
        
        if not all_subject_data:
            print(f"No results found for feature {feature_name}")
            continue
        
        # Ensure all subjects have the same bands and time points
        # Use the first subject as reference
        ref_data = all_subject_data[0]
        ref_bands = np.array(ref_data['bands'])  # Convert to numpy array
        ref_times = np.array(ref_data['times'])  # Convert to numpy array
        
        # Collect scores from all subjects with matching structure
        matching_scores = []
        final_subjects = []
        
        for subject_idx, data in enumerate(all_subject_data):
            subject = valid_subjects[subject_idx]
            
            # Convert current subject's data to arrays for comparison
            current_bands = np.array(data['bands'])
            current_times = np.array(data['times'])
            
            # Check if dimensions match
            if (current_bands.size == ref_bands.size and 
                current_times.size == ref_times.size):
                
                # Check if bands match (arrays already)
                bands_match = np.array_equal(current_bands, ref_bands)
                
                # Check if times are close enough
                times_match = np.allclose(current_times, ref_times, rtol=1e-5, atol=1e-5)
                
                if bands_match and times_match:
                    matching_scores.append(data['scores'])
                    final_subjects.append(subject)
                else:
                    print(f"Subject {subject} has incompatible data structure - skipping")
            else:
                print(f"Subject {subject} has different data dimensions - skipping")
        
        if not matching_scores:
            print(f"No compatible results for feature {feature_name}")
            continue
        
        # Stack scores across subjects
        stacked_scores = np.stack(matching_scores, axis=0)  # (n_subjects, n_bands, n_times)
        
        # Compute mean and standard error across subjects
        mean_scores = np.mean(stacked_scores, axis=0)  # (n_bands, n_times)
        se_scores = np.std(stacked_scores, axis=0) / np.sqrt(len(final_subjects))
        
        # Save group results
        group_path = f"{TF_DECODE_DIR}/group"
        os.makedirs(group_path, exist_ok=True)
        
        np.savez(
            f"{group_path}/{feature_name}_group_results.npz",
            mean_scores=mean_scores,
            se_scores=se_scores,
            bands=ref_bands,
            times=ref_times,
            subjects=final_subjects
        )
        
        # Plot group results
        plt.figure(figsize=(14, 10))
        
        for band_idx, band_name in enumerate(ref_bands):
            plt.plot(ref_times, mean_scores[band_idx], 
                    label=band_name,
                    linewidth=2)
            
            # Add shaded area for standard error
            plt.fill_between(
                ref_times,
                mean_scores[band_idx] - se_scores[band_idx],
                mean_scores[band_idx] + se_scores[band_idx],
                alpha=0.2
            )
        
        plt.axhline(0.5, color='k', linestyle='--', label='Chance')
        plt.xlabel('Time (s)')
        plt.ylabel('Decoding Accuracy')
        plt.title(f'Group-Level {FEATURES[feature_name]["name"]} Time-Frequency Decoding (N={len(final_subjects)})')
        plt.legend(title='Frequency Bands')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{group_path}/{feature_name}_group_tf_decoding.png")
        plt.savefig(f"{group_path}/{feature_name}_group_tf_decoding.pdf")
        plt.close()
        
        print(f"Group results saved for feature {feature_name} with {len(final_subjects)} subjects")

# Main execution
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Time-Frequency Decoding Pipeline')
    parser.add_argument('--subject', type=int, required=True, help='Subject ID to process')
    args = parser.parse_args()
    
    # Process just this subject
    subject = args.subject
    print(f"Processing subject {subject}")
    
    # Process both features for this subject
    for feature_name in FEATURES.keys():
        try:
            # Step 1: Compute TF transforms
            tf_data_list = compute_tf_transforms(subject, feature_name)
            
            if tf_data_list is None or len(tf_data_list) == 0:
                print(f"No valid TF data for subject {subject}, feature {feature_name}")
                continue
            
            # Step 2: Process each iteration
            iteration_results = []
            for iter_idx, tf_data in enumerate(tf_data_list):
                scores, band_names = decode_tf_data(
                    tf_data, tf_data[2], tf_data[3], 
                    subject=subject,
                    feature_name=feature_name, 
                    iter_idx=iter_idx
                )
                iteration_results.append((scores, band_names, tf_data[2]))
            
            # Step 3: Average across iterations
            if iteration_results:
                process_iteration_results(iteration_results, subject, feature_name)
            
            print(f"Completed {feature_name} for subject {subject}")
        except Exception as e:
            print(f"Error processing {feature_name}: {str(e)}")
            import traceback
            traceback.print_exc()
