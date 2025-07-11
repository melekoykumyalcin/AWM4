# Optimized Delay Period Decoding Analysis with Balanced Averaging
# For Working Memory Maintenance Analysis Based on Retro-Cues

import os
import locale
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
matplotlib.use('Qt5Agg')
plt.ioff()
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = [10, 8]

import mne
mne.set_log_level('INFO')
import pandas as pd
import pickle

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from mne.decoding import cross_val_multiscore, SlidingEstimator
from joblib import Parallel, delayed



# Constants and Configuration
HOME_DIR = '/media/headmodel/Elements/AWM4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
DATA_PATH = HOME_DIR + '/AWM4_data/raw/'
CORRECTED_DATA = HOME_DIR + '/AWM4_data/raw/correctTriggers/'

# Decoding Parameters
RESAMPLE_FREQ = 100  # Hz
BIN_SIZE = 10  # 100ms bins at 100Hz
CV_SPLITS = 5
CV_REPEATS = 10
NUM_JOBS = 20  # Number of parallel jobs

# Delay Period Configuration
DELAY_CONFIG = {
    'tmin': 2.0,
    'tmax': 4.7,
    'timepoints': np.linspace(2.0, 4.7, int((4.7-2.0)*RESAMPLE_FREQ))
}

# Critical time points to mark on plots
CRITICAL_TIMEPOINTS = [3.5, 4.5]  # Gray dashed lines at these times

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
    # clean_trials = mne.read_epochs(
    #     f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif",
    #     preload=True,
    #     verbose='INFO'
    # )
    # print(f"Initial epochs loaded: {len(clean_trials)}")

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
                    
                    # ADD THESE LINES: Update both local and global used trials
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

def load_subject_data(subject, metaInfo):
    """
    Load and preprocess MEG data for delay period decoding
    
    Args:
        subject (int): Subject number
        metaInfo (DataFrame): Metadata information
    
    Returns:
        tuple: List of averaged data iterations, sampling frequency
    """
    print(f'\n=== Preparing data for subject {subject} ===')
    
    force_recompute = False
    
    try:
        # Check if there's already averaged data available
        avg_save_path = f"{PROCESSED_DIR}/averaged_data/sub-{subject}/delay(3)"
        os.makedirs(avg_save_path, exist_ok=True)
        avg_save_file = f"{avg_save_path}/averaged_data(3).pkl"
        
        if os.path.exists(avg_save_file) and not force_recompute:
            # This condition will now be skipped due to force_recompute=True
            print(f"Loading pre-averaged data for subject {subject}...")
            with open(avg_save_file, 'rb') as f:
                averaged_data = pickle.load(f)
                
            print(f"Loaded {len(averaged_data)} iterations of averaged data")
            return averaged_data, RESAMPLE_FREQ
        
        # If not available, load raw data and process
        print(f"Pre-averaged data not found, processing from raw...")
        
        # First extract maintained information based on retro-cues
        memorized, all_events = extract_maintained_information(subject, metaInfo)
        
        if memorized is None:
            print(f"Could not extract maintained information for subject {subject}")
            return None, None
        
        # Load cleaned epochs
        clean_trials = mne.read_epochs(
            f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif",
            preload=True,
            verbose='INFO'
        )
        print(f"Initial epochs loaded: {len(clean_trials)}")
        
        # Special case handling for specific subjects
        if subject == 28:
            clean_trials.drop(63)
        if subject == 23:
            clean_trials.drop(64*7)
        
        if len(memorized) != len(clean_trials.events):
            print(f"Warning: Mismatch in number of trials. Memorized: {len(memorized)}, Epochs: {len(clean_trials.events)}")

        # Update event codes to reflect maintained information
        clean_trials.events[:, 2] = memorized
        clean_trials.event_id = EVENT_DICT
        
        # Check for jump artifacts and drop affected epochs
        jname = f"{PROCESSED_DIR}/ICAs/Jumps{subject}.npy"
        if os.path.isfile(jname):
            jump_inds = np.load(jname)
            
            # Check if jump_inds is not empty before processing
            if len(jump_inds) > 0:
                # Ensure jump_inds is a valid numpy array of integers
                jump_inds = np.array(jump_inds, dtype=int)
                
                # Filter out indices that are out of bounds
                valid_jump_inds = jump_inds[jump_inds < len(clean_trials)]
                
                if len(valid_jump_inds) > 0:
                    clean_trials.drop(valid_jump_inds, reason='jump')
                    
                    # Update memorized array to match epochs
                    memorized = np.delete(memorized, valid_jump_inds)
                else:
                    print(f"No valid jump indices found within epoch range for subject {subject}")
            else:
                print(f"No jump artifacts found for subject {subject}")
        
        # Crop to delay period
        delay_epochs = clean_trials.copy()
        delay_epochs.crop(tmin=DELAY_CONFIG['tmin'], tmax=DELAY_CONFIG['tmax'])
        
        # Select magnetometer channels
        mag_epochs = delay_epochs.copy().pick_types(meg='mag')
        
        # Resample
        mag_epochs = mag_epochs.resample(RESAMPLE_FREQ, npad='auto')
        print(f"Epochs after resampling: {len(mag_epochs)}")
        
        # Now perform the averaging using our improved function
        memorized_values = mag_epochs.events[:, 2]  # Get the memorized values for each epoch
        averaged_data, trial_records = average_maintained_condition_groups(mag_epochs, memorized_values)
        
        # Save trial records
        save_path = f"{PROCESSED_DIR}/trial_records/sub-{subject}/delay(3)"
        os.makedirs(save_path, exist_ok=True)
        with open(f"{save_path}/complete_trial_record.txt", 'w') as f:
            f.write(f"Subject {subject} Delay Period Trial Record\n")
            f.write("=" * 50 + "\n")
            for iter_idx, iter_record in enumerate(trial_records):
                f.write(f"\nIteration {iter_idx + 1}\n")
                f.write("-" * 30 + "\n")
                for group_name, trials in iter_record.items():
                    if group_name != 'iteration':
                        f.write(f"\n{group_name}:\n")
                        if isinstance(trials, dict):
                            for key, values in trials.items():
                                f.write(f"  {key}: {values}\n")
        
        # Save the averaged data if we have results
        if averaged_data:
            with open(avg_save_file, 'wb') as f:
                pickle.dump(averaged_data, f)
            print(f"Saved {len(averaged_data)} iterations of averaged data to {avg_save_file}")
        
        # Return all iterations for analysis
        return averaged_data, RESAMPLE_FREQ
    
    except Exception as e:
        print(f"Error processing subject {subject}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def decode_binned_data(data, y, sfreq, subject=None):
    """
    Perform decoding on binned data with proper nested cross-validation
    
    Parameters:
    data - 3D array of shape (n_trials, n_channels, n_timepoints)
    y - array of class labels
    sfreq - sampling frequency
    subject - subject ID for tracking (optional)
    
    Returns:
    bin_scores - accuracy scores for each time bin
    bin_centers - time points corresponding to the center of each bin
    """
    n_trials, n_channels, n_timepoints = data.shape
    
    # Calculate number of bins
    n_bins = n_timepoints // BIN_SIZE
    if n_bins == 0:
        print(f"Data too short for binning: {n_timepoints} timepoints with bin size {BIN_SIZE}")
        return None, None
    
    # Pre-compute binned data to avoid redundant calculations
    binned_data = np.zeros((n_trials, n_channels, n_bins))
    bin_centers = np.zeros(n_bins)
    
    # Create bins by averaging timepoints
    for bin_idx in range(n_bins):
        bin_start = bin_idx * BIN_SIZE
        bin_end = min(bin_start + BIN_SIZE, n_timepoints)
        
        # Calculate center time point for this bin (relative to the start of the data)
        bin_center = (bin_start + bin_end) / 2 / sfreq + DELAY_CONFIG['tmin']
        bin_centers[bin_idx] = bin_center
        
        # Average data across timepoints in this bin
        binned_data[:, :, bin_idx] = np.mean(data[:, :, bin_start:bin_end], axis=2)
    
    # Define parameter grid for SVM
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    
    # Process each bin in parallel
    def process_bin_nested(bin_idx):
        """Process a single time bin with proper nested cross-validation"""
        # Extract data for this bin
        X_bin = binned_data[:, :, bin_idx].reshape(n_trials, -1)
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_bin)
        
        # Setup cross-validation schemes
        outer_cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS // 10)
        inner_cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=2)
        
        # Initialize results storage
        nested_scores = []
        best_c_values = []
        
        # Outer loop
        for train_idx, test_idx in outer_cv.split(X_scaled, y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Inner GridSearchCV
            grid_search = GridSearchCV(
                estimator=SVC(kernel='linear', probability=True),
                param_grid=param_grid,
                cv=inner_cv,
                scoring='accuracy'
            )
            
            # Fit on training data
            grid_search.fit(X_train, y_train)
            
            # Get best parameters
            best_c = grid_search.best_params_['C']
            best_c_values.append(best_c)
            
            # Create classifier with best parameters
            best_clf = SVC(kernel='linear', C=best_c, probability=True)
            best_clf.fit(X_train, y_train)
            
            # Test on held-out data
            score = best_clf.score(X_test, y_test)
            nested_scores.append(score)
        
        # Calculate average score across all outer folds
        mean_score = np.mean(nested_scores)
        
        # Calculate most common C value
        from collections import Counter
        c_counter = Counter(best_c_values)
        most_common_c = c_counter.most_common(1)[0][0]
        
        return {
            'accuracy': mean_score,
            'best_c': most_common_c,
            'success': True
        }

    # Run parallel processing across bins
    print("\nPerforming parallel nested cross-validation across bins...")
    results = Parallel(n_jobs=NUM_JOBS)(
        delayed(process_bin_nested)(bin_idx) for bin_idx in tqdm(range(n_bins))
    )
    
    # Check if any bins were successfully processed
    successful_bins = [r for r in results if r['success']]
    if not successful_bins:
        print("No bins were successfully processed")
        return None, None
    
    # Extract results
    bin_scores = np.array([r['accuracy'] for r in results])
    best_c_values = [r['best_c'] for r in results]
    
    # Save the best C values for reference if subject ID is provided
    if subject is not None:
        results_path = f"{PROCESSED_DIR}/timepoints/delay_binned(3)/params/{subject}"
        os.makedirs(results_path, exist_ok=True)
        results_data = np.column_stack((bin_centers, best_c_values))
        np.savetxt(f"{results_path}/best_c_values.txt", 
                  results_data, 
                  header="Time_bin_center Best_C_value", 
                  fmt='%.4f %f')
    
    return bin_scores, bin_centers

def decode_binned_data_iterations(averaged_data_list, sfreq, feature_name, subject=None):
    """
    Perform binned decoding across multiple iterations of averaged data
    
    Parameters:
    averaged_data_list - List of tuples (X, y) with averaged data
    sfreq - Sampling frequency
    feature_name - Name of feature being decoded
    subject - Subject ID (optional)
    
    Returns:
    mean_scores - Mean accuracy across iterations for each bin
    std_scores - Standard deviation of accuracy across iterations
    bin_centers - Center time points for each bin
    """
    if not averaged_data_list:
        print(f"No averaged data available for decoding")
        return None, None, None
        
    all_bin_scores = []
    bin_centers = None
    
    for iter_idx, (X, y) in enumerate(averaged_data_list):
        print(f"\nDecoding iteration {iter_idx + 1}/{len(averaged_data_list)}")
        
        # Check for valid data
        if X.size == 0 or len(np.unique(y)) < 2:
            print(f"Iteration {iter_idx + 1} has invalid data (empty or single class). Skipping.")
            continue
        
        # Create binary classification labels based on the condition groups
        # Group indices correspond to their position in the CONDITION_GROUPS dictionary:
        # 0: 'Sp1+2/L1+2', 1: 'Sp1+2/L3+4', 2: 'Sp3+4/L1+2', 3: 'Sp3+4/L3+4'
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
        print(f"\n{group_desc} Class distribution:")
        for cls in unique_classes:
            print(f"Class {cls}: {np.sum(binary_y == cls)} trials")
        
        # Print detailed condition distribution
        print("\nDetailed condition distribution:")
        for cond in np.unique(y):
            print(f"Condition {cond}: {np.sum(y == cond)} trials")
        
        try:
            # Decode this iteration
            iter_scores, iter_centers = decode_binned_data(X, binary_y, sfreq, 
                                                         subject=f"{subject}_iter{iter_idx}" if subject else None)
            
            if iter_scores is not None and iter_centers is not None:
                all_bin_scores.append(iter_scores)
                
                # Store centers (same for all iterations)
                if bin_centers is None:
                    bin_centers = iter_centers
            else:
                print(f"Iteration {iter_idx + 1} produced no valid results. Skipping.")
        except Exception as e:
            print(f"Error decoding iteration {iter_idx + 1}: {str(e)}")
            continue
    
    # Check if we have any valid results
    if not all_bin_scores:
        print("No valid decoding results across all iterations")
        return None, None, None
        
    # Average scores across iterations
    mean_scores = np.mean(all_bin_scores, axis=0)
    std_scores = np.std(all_bin_scores, axis=0)
    
    return mean_scores, std_scores, bin_centers

def compute_statistics(scores):
    """Compute statistical significance using permutation testing"""
    p_values = []
    significant_points = []
    
    for timepoint in range(scores.shape[1]):
        actual = scores[:, timepoint] - 0.5
        actual_mean = np.mean(actual)
        
        # Create permutation distribution
        permuted = [[np.abs(value), -np.abs(value)] for value in actual]
        
        # Permutation test
        population = []
        for _ in range(100000):
            sample = [np.random.choice(p, 1)[0] for p in permuted]
            population.append(np.mean(sample))
        
        # Calculate p-value
        p = np.sum(np.array(population) >= actual_mean) / 100000
        p_values.append(p)
        
        # Store significance
        if p <= 0.01:
            significant_points.append((timepoint, '**'))
        elif p <= 0.05:
            significant_points.append((timepoint, '*'))
    
    return p_values, significant_points

def run_delay_decoding_analysis(subjects, metaInfo):
    """
    Run comprehensive delay period decoding analysis
    
    Args:
        subjects (list): List of subject numbers
        metaInfo (DataFrame): Metadata information
    """
    # Features to decode
    features = list(FEATURES.keys())
    
    results_dict = {}
    
    # Process each subject once, running all features
    for subject in subjects:
        print(f"\n==== Processing subject {subject} ====")
        
        try:
            # Load subject data
            averaged_data, sfreq = load_subject_data(subject, metaInfo)
            
            # Process all features for this subject
            for feature_name in features:
                # Create results directories
                results_path = f"{PROCESSED_DIR}/timepoints/delay_binned(3)/{feature_name}"
                os.makedirs(results_path, exist_ok=True)
                
                # Perform binned decoding across iterations
                try:
                    mean_scores, std_scores, bin_centers = decode_binned_data_iterations(
                        averaged_data, sfreq, feature_name, subject
                    )
                    
                    # Initialize storage in results_dict if needed
                    if feature_name not in results_dict:
                        results_dict[feature_name] = {
                            'scores': [],
                            'subjects': [],
                            'bin_centers': bin_centers
                        }
                    
                    # Store results
                    results_dict[feature_name]['scores'].append(mean_scores)
                    results_dict[feature_name]['subjects'].append(subject)
                    
                    # Plot individual subject results
                    plt.figure(figsize=(12, 7))
                    plt.plot(bin_centers, mean_scores, 
                            color=FEATURES[feature_name]['color'], 
                            marker='o', markersize=4)
                    plt.fill_between(bin_centers, 
                                   mean_scores - std_scores, 
                                   mean_scores + std_scores, 
                                   alpha=0.2,
                                   color=FEATURES[feature_name]['color'])
                            
                    # Add critical timepoints as vertical lines
                    for tp in CRITICAL_TIMEPOINTS:
                        plt.axvline(x=tp, color='gray', linestyle='--', alpha=0.7)
                        
                    plt.title(f'{FEATURES[feature_name]["name"]} Delay Period Decoding - Subject {subject}')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Decoding Accuracy')
                    plt.axhline(y=0.5, color='black', linestyle='--')
                    plt.ylim(0.45, 0.7)
                    plt.tight_layout()
                    plt.savefig(f'{results_path}/sub{subject}_decoding.png')
                    plt.close()
                    
                    print(f'Completed {feature_name} decoding for subject {subject}')
                    
                    # Save individual subject results
                    subject_result = {
                        'mean_scores': mean_scores,
                        'std_scores': std_scores,
                        'bin_centers': bin_centers
                    }
                    with open(f'{results_path}/sub{subject}_results.pkl', 'wb') as f:
                        pickle.dump(subject_result, f)
                except Exception as e:
                    print(f"Error during decoding for subject {subject}, {feature_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
        except Exception as e:
            print(f"Error processing subject {subject}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # After processing all subjects, generate group results for each feature
    for feature_name, feature_data in results_dict.items():
        # Fix for nested array structure - ADDED CODE BLOCK HERE
        print(f"\nFixing data structure for {feature_name}...")
        fixed_scores = []
        
        if len(feature_data['scores']) > 0:
            print(f"Type of first score: {type(feature_data['scores'][0])}")
            print(f"Shape of first score: {np.shape(feature_data['scores'][0])}")
            
            for subject_scores in feature_data['scores']:
                # Handle different possible structures of subject_scores
                if isinstance(subject_scores, np.ndarray):
                    # If already a flat array, use as is
                    fixed_scores.append(subject_scores)
                elif isinstance(subject_scores, list) and len(subject_scores) == 1:
                    # If it's a list with one element
                    fixed_scores.append(np.array(subject_scores[0]))
                else:
                    # For any other case, try to convert to array
                    try:
                        arr = np.array(subject_scores)
                        if arr.ndim == 2 and arr.shape[0] == 1:
                            # If it's a 2D array with just one row
                            fixed_scores.append(arr[0])
                        else:
                            fixed_scores.append(arr)
                    except Exception as e:
                        print(f"Error converting scores to array: {e}")
                        continue
                        
            all_scores = np.array(fixed_scores)
            print(f"Fixed array shape: {all_scores.shape}")
        else:
            print("No scores available for this feature")
            continue
        
        valid_subjects = feature_data['subjects']
        bin_centers = feature_data['bin_centers']
        
        results_path = f"{PROCESSED_DIR}/timepoints/delay_binned(3)/{feature_name}"
        
        # Create bin labels based on the actual number of bins in the data
        bin_labels = [f'Bin_{t:.3f}s' for t in bin_centers[:all_scores.shape[1]]]
        
        # Save results to Excel
        try:
            results_df = pd.DataFrame(all_scores, index=valid_subjects, columns=bin_labels)
            results_df.to_excel(f'{results_path}/decoding_results.xlsx')
            print(f"Successfully saved results to Excel with shape {all_scores.shape}")
            
            # Calculate mean across subjects and timepoints
            mean_across_time = results_df.mean(axis=1)
            mean_across_time.to_excel(f'{results_path}/{feature_name}_mean_across_time.xlsx')
        except Exception as e:
            print(f"Error saving to Excel: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Compute significance
        try:
            p_values, significant_points = compute_statistics(all_scores)
            pd.DataFrame(p_values, columns=["p_values"]).to_csv(f'{results_path}/p_values.csv', index=False)
        except Exception as e:
            print(f"Error computing statistics: {e}")
            significant_points = []
        
        # Compute statistics
        mean_scores = np.mean(all_scores, axis=0)
        std_error = np.std(all_scores, axis=0) / np.sqrt(len(valid_subjects))
        
        # Plot group-level results
        plt.figure(figsize=(12, 7))
        plt.plot(bin_centers[:len(mean_scores)], mean_scores, 
                label=f'Mean Accuracy (N={len(valid_subjects)})', 
                color=FEATURES[feature_name]['color'],
                linewidth=2)
                
        plt.fill_between(bin_centers[:len(mean_scores)], 
                        mean_scores - std_error, 
                        mean_scores + std_error, 
                        alpha=0.2,
                        color=FEATURES[feature_name]['color'])
                        
        # Add critical timepoints as vertical lines
        for tp in CRITICAL_TIMEPOINTS:
            plt.axvline(x=tp, color='gray', linestyle='--', alpha=0.7)
        
        # Add significance markers if applicable
        if significant_points:
            sig_times = [bin_centers[tp] for tp, _ in significant_points if tp < len(bin_centers)]
            if sig_times:
                plt.plot(sig_times, [0.6] * len(sig_times), 
                        marker='*', linestyle='', color=FEATURES[feature_name]['color'])
            
        plt.title(f'Group-Level {FEATURES[feature_name]["name"]} Decoding during Delay Period')
        plt.xlabel('Time (s)')
        plt.ylabel('Decoding Accuracy')
        plt.axhline(y=0.5, color='black', linestyle='--')
        plt.ylim(0.45, 0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{results_path}/group_decoding_result.png')
        plt.savefig(f'{results_path}/group_decoding_result.pdf')
        plt.close()
    
    # Create comparison plot for all features
    if len(results_dict) > 1:
        comparison_path = f"{PROCESSED_DIR}/timepoints/delay_binned(3)/comparison"
        os.makedirs(comparison_path, exist_ok=True)
        
        plt.figure(figsize=(12, 7))
        
        for feature_name, feature_data in results_dict.items():
            # Use the same fixed_scores approach here too
            fixed_scores = []
            for subject_scores in feature_data['scores']:
                if isinstance(subject_scores, np.ndarray):
                    fixed_scores.append(subject_scores)
                elif isinstance(subject_scores, list) and len(subject_scores) == 1:
                    fixed_scores.append(np.array(subject_scores[0]))
                else:
                    try:
                        arr = np.array(subject_scores)
                        if arr.ndim == 2 and arr.shape[0] == 1:
                            fixed_scores.append(arr[0])
                        else:
                            fixed_scores.append(arr)
                    except:
                        continue
            
            all_scores = np.array(fixed_scores)
            bin_centers = feature_data['bin_centers']
            
            mean_scores = np.mean(all_scores, axis=0)
            std_error = np.std(all_scores, axis=0) / np.sqrt(all_scores.shape[0])
            
            plt.plot(bin_centers[:len(mean_scores)], mean_scores, 
                    label=f'{FEATURES[feature_name]["name"]} (N={all_scores.shape[0]})',
                    color=FEATURES[feature_name]['color'],
                    linewidth=2)
                    
            plt.fill_between(bin_centers[:len(mean_scores)], 
                            mean_scores - std_error, 
                            mean_scores + std_error, 
                            alpha=0.2,
                            color=FEATURES[feature_name]['color'])
        
        # Add critical timepoints as vertical lines
        for tp in CRITICAL_TIMEPOINTS:
            plt.axvline(x=tp, color='gray', linestyle='--', alpha=0.7)
            
        plt.axhline(y=0.5, color='black', linestyle='--', label='Chance')
        plt.xlabel('Time (s)')
        plt.ylabel('Decoding Accuracy')
        plt.title('Comparison of Maintained Information Decoding during Delay Period')
        plt.ylim(0.45, 0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{comparison_path}/feature_comparison.png')
        plt.savefig(f'{comparison_path}/feature_comparison.pdf')
        plt.close()
        
        print("\nComparison plot created successfully!")
    
    return results_dict

# Main execution block
if __name__ == "__main__":
    # Set paths and load metadata
    locale.setlocale(locale.LC_ALL, "en_US.utf8")
    
    # Load metadata
    metaInfo = pd.read_excel(META_FILE)
    
    # Get all subjects from the final sample (convert to list to avoid numpy array issues)
    all_subjects = list(np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject']))
    
    # SUBJECT SELECTION - MODIFY THESE LINES:
    # Option 1: Run a single subject (uncomment and set the subject number)
    #subjects_to_process = [1]  # Replace with your desired subject number
    
    # Option 2: Run all subjects (uncomment this line and comment the line above)
    subjects_to_process = all_subjects.copy()
    
    print(f"\nProcessing subjects: {subjects_to_process}")
    
    # Create necessary output directories
    os.makedirs(f"{PROCESSED_DIR}/timepoints/delay_binned(3)", exist_ok=True)
    os.makedirs(f"{PROCESSED_DIR}/averaged_data", exist_ok=True)
    os.makedirs(f"{PROCESSED_DIR}/trial_records", exist_ok=True)
    
    # Check existing results files to determine which subjects have been processed
    results_path = f"{PROCESSED_DIR}/timepoints/delay_binned(3)"
    
    # Initialize lists to track processed subjects
    processed_subjects = []
    
    # Check existing Excel files for each feature
    features = ['maintained_voice_identity', 'maintained_location']
    
    for feature in features:
        feature_results_file = f"{results_path}/{feature}/decoding_results.xlsx"
        
        # If results file exists, read processed subjects
        if os.path.exists(feature_results_file):
            existing_results = pd.read_excel(feature_results_file, index_col=0)
            processed_subjects.extend(list(map(int, existing_results.index)))
    
    # Remove duplicates from processed subjects
    processed_subjects = list(set(processed_subjects))
    print(f"Already processed subjects: {processed_subjects}")
    
    # Run the comprehensive delay period decoding analysis for subjects
    if len(subjects_to_process) > 0:  # Fixed condition check
        results = run_delay_decoding_analysis(subjects_to_process, metaInfo)
        
        # Append results to existing Excel sheets or create new ones
        for feature, feature_data in results.items():
            feature_results_file = f"{results_path}/{feature}/decoding_results.xlsx"
            
            # Prepare new results
            new_results_df = pd.DataFrame(
                feature_data['scores'], 
                index=feature_data['subjects'], 
                columns=[f'Bin_{t:.3f}s' for t in feature_data['bin_centers']]
            )
            
            # If existing results file exists, append to it
            if os.path.exists(feature_results_file):
                existing_results = pd.read_excel(feature_results_file, index_col=0)
                # Make sure we don't have duplicate subjects
                existing_results = existing_results.drop(index=existing_results.index.intersection(new_results_df.index))
                combined_results = pd.concat([existing_results, new_results_df])
                combined_results.to_excel(feature_results_file)
                print(f"Appended results to {feature_results_file}")
            else:
                # Create new results file
                new_results_df.to_excel(feature_results_file)
                print(f"Created new results file {feature_results_file}")
    else:
        print("No subjects to process.")
    
    print("\nDelay period analysis completed successfully!")