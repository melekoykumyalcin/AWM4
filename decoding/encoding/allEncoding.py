#averaged decoding analysis for all subjects

# Import required libraries
import os
import locale
from tqdm import tqdm
import pathlib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
plt.ioff()
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = [10, 8]
import numpy as np
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
mne.set_log_level('INFO')  # Change to 'INFO' for more detailed output
import pandas as pd
import autoreject
from autoreject import get_rejection_threshold
from pynput.keyboard import Key, Controller
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from mne.decoding import Scaler, Vectorizer, cross_val_multiscore
import seaborn as sns
from joblib import Parallel, delayed
import itertools

# Initialize keyboard controller
keyboard = Controller()

# Set paths
HOME_DIR = '/media/headmodel/Elements/AWM4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
DATA_PATH = HOME_DIR + '/AWM4_data/raw/'
CORRECTED_DATA = HOME_DIR + '/AWM4_data/raw/correctTriggers/'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'

# Load metadata
metaInfo = pd.read_excel(META_FILE)
NrSessions = 1
Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])

# Set locale
locale.setlocale(locale.LC_ALL, "en_US.utf8")

# Prepare corrected files list for early subjects
allFiles = metaInfo['MEG_Name']
corrected_files = [f.split('.')[0] + '_correct_triggers.fif' for f in allFiles]
corrected_files_series = pd.Series(corrected_files)

# Decoding parameters
RESAMPLE_FREQ = 100  # Hz
WINDOW_LENGTH_SEC = 0.1  # seconds
WINDOW_STEP_SEC = 0.01  # seconds
CV_SPLITS = 5
CV_REPEATS = 100
NUM_JOBS = 20

# Complete event dictionary
EVENT_DICT = {
    'S1/Sp1/L1': 111, 'S1/Sp1/L2': 112, 'S1/Sp1/L3': 113, 'S1/Sp1/L4': 114,
    'S1/Sp2/L1': 121, 'S1/Sp2/L2': 122, 'S1/Sp2/L3': 123, 'S1/Sp2/L4': 124,
    'S1/Sp3/L1': 131, 'S1/Sp3/L2': 132, 'S1/Sp3/L3': 133, 'S1/Sp3/L4': 134,
    'S1/Sp4/L1': 141, 'S1/Sp4/L2': 142, 'S1/Sp4/L3': 143, 'S1/Sp4/L4': 144,
    'S2/Sp1/L1': 211, 'S2/Sp1/L2': 212, 'S2/Sp1/L3': 213, 'S2/Sp1/L4': 214,
    'S2/Sp2/L1': 221, 'S2/Sp2/L2': 222, 'S2/Sp2/L3': 223, 'S2/Sp2/L4': 224,
    'S2/Sp3/L1': 231, 'S2/Sp3/L2': 232, 'S2/Sp3/L3': 233, 'S2/Sp3/L4': 234,
    'S2/Sp4/L1': 241, 'S2/Sp4/L2': 242, 'S2/Sp4/L3': 243, 'S2/Sp4/L4': 244
}

# Feature definitions
FEATURES = {
    'voice_identity': {
        'name': 'voice identity',
        'color': '#1c686b',
        'S1': {
            'low_group': [111, 112, 113, 114, 121, 122, 123, 124],
            'high_group': [131, 132, 133, 134, 141, 142, 143, 144]
        },
        'S2': {
            'low_group': [211, 212, 213, 214, 221, 222, 223, 224],
            'high_group': [231, 232, 233, 234, 241, 242, 243, 244]
        }
    },
    'location': {
        'name': 'location',
        'color': '#cb6a3e',
        'S1': {
            'low_group': [111, 121, 131, 141, 112, 122, 132, 142],
            'high_group': [113, 123, 133, 143, 114, 124, 134, 144]
        },
        'S2': {
            'low_group': [211, 221, 231, 241, 212, 222, 232, 242],
            'high_group': [213, 223, 233, 243, 214, 224, 234, 244]
        }
    }
}

# Stimulus definitions
STIMULI = {
    'S1': {
        'name': 'S1',
        'tmin': 0,
        'tmax': 1,
        'trigger': 100,
        'timepoints': np.linspace(0.0, 1.0, 95)
    },
    'S2': {
        'name': 'S2',
        'tmin': 1,
        'tmax': 2,
        'trigger': 200,
        'timepoints': np.linspace(1.0, 2.0, 95)
    }
}

# Special cases for subject handling
SUBJECT_SPECIAL_CASES = {
    23: {'block': 8, 'trial': 1, 'action': 'drop_idx', 'value': 64 * 7},
    28: {'block': 1, 'trials': [63, 64], 'action': 'drop_idx', 'value': 63}
}

def verify_balance(events, feature_name):
    """
    Verify balance between decoding groups
    Returns: Boolean indicating if groups are balanced
    """
    if feature_name == 'voice_identity':
        # Speaker decoding (Sp1+Sp2 vs Sp3+Sp4)
        low_events = [ev for ev in events[:, 2] if (ev//10)%10 in [1, 2]]
        high_events = [ev for ev in events[:, 2] if (ev//10)%10 in [3, 4]]
        group_name = "Speaker"
    else:  # location
        # Location decoding (L1+L2 vs L3+L4)
        low_events = [ev for ev in events[:, 2] if ev%10 in [1, 2]]
        high_events = [ev for ev in events[:, 2] if ev%10 in [3, 4]]
        group_name = "Location"
    
    print(f"\n{group_name} Decoding Balance Check:")
    print(f"Low group ({group_name.lower()}s 1+2): {len(low_events)} trials")
    print(f"High group ({group_name.lower()}s 3+4): {len(high_events)} trials")
    
    if len(low_events) != len(high_events):
        print(f"WARNING: Unbalanced {group_name.lower()} groups!")
        return False
    return True

def average_condition_groups(epochs, stimulus_name, n_iterations=10):
    """
    Average trials within condition groups with multiple iterations
    """
    # Define condition groups and their event codes based on stimulus
    if stimulus_name == 'S1':
        CONDITION_GROUPS = {
            'Sp1+2/L1+2': [111, 112, 121, 122],
            'Sp1+2/L3+4': [113, 114, 123, 124],
            'Sp3+4/L1+2': [131, 132, 141, 142],
            'Sp3+4/L3+4': [133, 134, 143, 144]
        }
    else:  # S2
        CONDITION_GROUPS = {
            'Sp1+2/L1+2': [211, 212, 221, 222],
            'Sp1+2/L3+4': [213, 214, 223, 224],
            'Sp3+4/L1+2': [231, 232, 241, 242],
            'Sp3+4/L3+4': [233, 234, 243, 244]
        }
    
    # Get data and events
    data = epochs.get_data()
    events = epochs.events[:, 2]
    
    # Initialize storage
    averaged_data = []
    trial_records = []
    
    # For each iteration
    for iter_idx in range(n_iterations):
        print(f"\nProcessing iteration {iter_idx + 1}/{n_iterations}")
        
        iter_averaged_epochs = []
        iter_averaged_events = []
        iter_trial_record = {'iteration': iter_idx + 1}
        
        # Process each condition group
        for group_name, event_codes in CONDITION_GROUPS.items():
            # Get trials for each condition in the group
            condition_trials = {code: [] for code in event_codes}
            condition_indices = {code: [] for code in event_codes}
            
            for idx, event in enumerate(events):
                if event in event_codes:
                    condition_trials[event].append(data[idx])
                    condition_indices[event].append(idx)
            
            # Find minimum number of trials across conditions in this group
            min_trials = min(len(trials) for trials in condition_trials.values())
            
            # Subsample equal number of trials from each condition
            group_averaged_trials = []
            used_trials = {code: [] for code in event_codes}
            
            for trial_idx in range(min_trials):
                averaged_trial = np.zeros_like(data[0])
                for code in event_codes:
                    # Randomly select one trial from this condition
                    trial_pool = list(set(range(len(condition_trials[code]))) - 
                                   set(used_trials[code]))
                    selected_idx = np.random.choice(trial_pool)
                    used_trials[code].append(selected_idx)
                    
                    # Add to average
                    averaged_trial += condition_trials[code][selected_idx]
                
                # Complete averaging
                averaged_trial /= len(event_codes)
                group_averaged_trials.append(averaged_trial)
            
            # Store the used trial indices for this group
            iter_trial_record[group_name] = {
                code: [condition_indices[code][idx] for idx in used_trials[code]]
                for code in event_codes
            }
            
            # Add averaged trials to iteration results
            iter_averaged_epochs.extend(group_averaged_trials)
            # Assign a unique event code for each averaged group
            group_code = list(CONDITION_GROUPS.keys()).index(group_name)
            iter_averaged_events.extend([group_code] * len(group_averaged_trials))
        
        # Store this iteration's results
        averaged_data.append((
            np.array(iter_averaged_epochs),
            np.array(iter_averaged_events)
        ))
        trial_records.append(iter_trial_record)
        
        # Save trial record for this iteration
        save_path = f"{PROCESSED_DIR}/trial_records/sub-{epochs.filename.split('VP')[-1].split('-')[0]}/{stimulus_name}"
        os.makedirs(save_path, exist_ok=True)
        with open(f"{save_path}/trial_record_iteration_{iter_idx+1}.txt", 'w') as f:
            f.write(f"Iteration {iter_idx + 1} Trial Record\n")
            f.write("=" * 50 + "\n")
            for group_name, trials in iter_trial_record.items():
                if group_name != 'iteration':
                    f.write(f"\n{group_name}:\n")
                    for code, indices in trials.items():
                        f.write(f"  Event {code}: {indices}\n")
    
    return averaged_data, trial_records

def decode_sliding_window(data, y, sfreq):
    """Perform sliding window decoding"""
    w_length = int(sfreq * WINDOW_LENGTH_SEC)
    w_step = int(sfreq * WINDOW_STEP_SEC)
    w_start = np.arange(-w_step * 5, data.shape[2] - w_length, w_step)
    
    scores = []
    for n in tqdm(w_start):
        # Extract window
        if n < 0:
            X = data[:, :, 0:(n + w_length)]
        else:
            X = data[:, :, n:(n + w_length)]
            
        # Setup classifier
        clf = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            SVC(probability=True)
        )
        
        # Cross-validation
        cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS)
        cv_scores = cross_val_multiscore(clf, X, y, cv=cv, n_jobs=NUM_JOBS, scoring='accuracy')
        scores.append(np.mean(cv_scores))
    
    return np.round(np.array(scores), 3)

def decode_sliding_window_iterations(averaged_data_list, sfreq, feature_name):
    """Perform sliding window decoding across multiple iterations"""
    all_scores = []
    
    for iter_idx, (X, y) in enumerate(averaged_data_list):
        print(f"\nDecoding iteration {iter_idx + 1}")
        
        # Create binary classification labels based on the feature
        if feature_name == 'voice_identity':
            # Speaker decoding: 0,1 (Sp1+2) vs 2,3 (Sp3+4)
            binary_y = np.where(y < 2, 0, 1)
            print("\nSpeaker Decoding Groups:")
            print(f"Low group (Sp1+2): {np.sum(binary_y == 0)} trials")
            print(f"High group (Sp3+4): {np.sum(binary_y == 1)} trials")
        else:  # location
            # Location decoding: 0,2 (L1+2) vs 1,3 (L3+4)
            binary_y = np.where(y % 2 == 0, 0, 1)
            print("\nLocation Decoding Groups:")
            print(f"Low group (L1+2): {np.sum(binary_y == 0)} trials")
            print(f"High group (L3+4): {np.sum(binary_y == 1)} trials")
            
        # Print detailed condition distribution
        print("\nDetailed condition distribution:")
        for cond in range(4):
            print(f"Condition {cond}: {np.sum(y == cond)} trials")
            
        scores = decode_sliding_window(X, binary_y, sfreq)
        all_scores.append(scores)
    
    # Average scores across iterations
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)
    
    return mean_scores, std_scores

def load_subject_data(subject, meta_info, feature_name, stimulus_name):
    """Load and prepare data for a single subject"""
    print(f'\n=== Starting with subject {subject} for {feature_name} decoding during {stimulus_name} ===')
    
    try:
        # 1. Load raw data and get events
        actInd = (meta_info.Subject==subject) & (meta_info.Valid==1)
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
        
        # 2. Get stimulus-specific events
        if stimulus_name == 'S2':
            stim_idx = [i - 1 for i in range(len(all_events[:,2])) if all_events[i,2] == 200]
            stim_values = all_events[stim_idx,2]
            print(f"Initial S2 events found: {len(stim_idx)}")  # Add this debug print
            print(f"Unique event values: {np.unique(stim_values)}")  # Add this debug print
        else:
            stim_idx = [i - 1 for i in range(len(all_events[:,2])) if all_events[i,2] == 100]
            stim_values = all_events[stim_idx,2]
            
        print(f"Initial number of events: {len(stim_values)}")  # Add this debug print
            
        # 3. Load cleaned epochs
        clean_trials = mne.read_epochs(
            f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif",
            preload=True,
            verbose='INFO'
        )
        print(f"Initial epochs loaded: {len(clean_trials)}")
        
        #I want to drop the the specific trials for the specific subjects
        if subject == 28:
            clean_trials.drop(63)
        if subject == 23:
            drop_idx = 64 * 7  # This is 448
            #clean_trials.drop(drop_idx)
            # Also drop the corresponding stimulus value
            stim_values = np.delete(stim_values, drop_idx)
            print(f"Dropped epoch {drop_idx} for subject 23. Remaining: {len(clean_trials)} epochs, {len(stim_values)} stim values")
                
        clean_trials.events[:,2] = stim_values
                        
        clean_trials.event_id = EVENT_DICT

        # 6. Crop time window
        stim_config = STIMULI[stimulus_name]
        stim_epochs = clean_trials.copy()
        del clean_trials
        stim_epochs.crop(tmin=stim_config['tmin'], tmax=stim_config['tmax'])
        
        # 7. Select magnetometer channels
        print("\nSelecting magnetometer channels...")
        mag_picks = mne.pick_types(stim_epochs.info, meg='mag', exclude='bads')
        mag_epochs = stim_epochs.copy().pick(mag_picks)
        print(f"Found {len(mag_picks)} magnetometer channels")
        
        # 8. Resample
        mag_epochs = mag_epochs.resample(RESAMPLE_FREQ, npad='auto')
        print(f"Epochs after resampling: {len(mag_epochs)}")

        # Create filtered event dictionary based on stimulus
        if stimulus_name == 'S1':
            relevant_event_id = {key: value for key, value in EVENT_DICT.items() 
                            if key.startswith('S1')}
        else:  # S2
            relevant_event_id = {key: value for key, value in EVENT_DICT.items() 
                            if key.startswith('S2')}

        print("\nRelevant events for", stimulus_name + ":")
        print(relevant_event_id)

        # Use the filtered event dictionary for equalization
        mag_epochs.event_id = relevant_event_id
        mag_epochs.equalize_event_counts(relevant_event_id)
        
        print(f"\nAfter equalizing - Events per condition:")
        for event_id in mag_epochs.event_id:
            n_trials = len(mag_epochs[event_id])
            print(f"{event_id}: {n_trials} trials")

        # After your balance check is successful:
        averaged_data, trial_records = average_condition_groups(mag_epochs, stimulus_name)
        
        # Save complete trial records
        save_path = f"{PROCESSED_DIR}/trial_records/sub-{subject}/{stimulus_name}"
        os.makedirs(save_path, exist_ok=True)
        with open(f"{save_path}/complete_trial_record.txt", 'w') as f:
            f.write(f"Subject {subject} Complete Trial Record\n")
            f.write("=" * 50 + "\n")
            for iter_record in trial_records:
                f.write(f"\nIteration {iter_record['iteration']}\n")
                f.write("-" * 30 + "\n")
                for group_name, trials in iter_record.items():
                    if group_name != 'iteration':
                        f.write(f"\n{group_name}:\n")
                        for code, indices in trials.items():
                            f.write(f"  Event {code}: {indices}\n")
        
        return averaged_data, mag_epochs.info['sfreq']
    
        # # 9. Setup classification groups
        # feature_config = FEATURES[feature_name][stimulus_name]
        # low_group = feature_config['low_group']
        # high_group = feature_config['high_group']
        
        # stim_values = mag_epochs.events[:, 2]
        # idx_low = [i for i in range(len(stim_values)) if stim_values[i] in low_group]
        # idx_high = [i for i in range(len(stim_values)) if stim_values[i] in high_group]
        
        # print(f"\nClassification groups:")
        # print(f"Low group trials: {len(idx_low)}")
        # print(f"High group trials: {len(idx_high)}")
        
        # if len(idx_low) < 2 or len(idx_high) < 2:
        #     print("WARNING: Insufficient trials in one or both groups")
        #     return None, None, None
        
        # # 10. Create target array and get data
        # y = np.empty(len(stim_values), dtype=int)
        # y[idx_low] = 0
        # y[idx_high] = 1
        
        # X = mag_epochs.get_data()
        # print(f"\nFinal data shape: {X.shape}")
        # print(f"Class balance: {np.sum(y==0)} vs {np.sum(y==1)}")
        
        # return X, y, mag_epochs.info['sfreq']
        
    except Exception as e:
        print(f"\nERROR processing subject {subject}:")
        print(str(e))
        import traceback
        traceback.print_exc()
        return None, None, None

def compute_statistics(scores):
    """Compute statistical significance"""
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

def save_results(all_mean_timescores, valid_subjects, timepoints, feature_name, stimulus_name, results_path):
    """Save results with detailed timepoint information, appending to existing files if they exist"""
    
    # Create DataFrame with timepoints as columns and subjects as rows
    timepoint_labels = [f"Time_{t:.3f}s" for t in timepoints]
    new_results_df = pd.DataFrame(all_mean_timescores, 
                                index=valid_subjects,
                                columns=timepoint_labels)
    
    # Path for detailed timepoint results
    detailed_path = f'{results_path}/all_{feature_name}_{stimulus_name}_byTimepoint_SVM.xlsx'
    
    # Check if file exists and append new data
    try:
        existing_df = pd.read_excel(detailed_path, index_col=0)
        # Remove any duplicate subjects from existing data
        existing_df = existing_df.drop(index=existing_df.index.intersection(new_results_df.index))
        # Combine existing and new data
        combined_df = pd.concat([existing_df, new_results_df])
    except FileNotFoundError:
        combined_df = new_results_df
    
    # Save updated detailed Excel file
    combined_df.to_excel(detailed_path)
    
    # Update mean across all timepoints
    mean_across_time = combined_df.mean(axis=1)
    mean_across_time.to_excel(f'{results_path}/all_{feature_name}_{stimulus_name}_meanAcrossTime_SVM.xlsx')
    
    return combined_df

def plot_results(scores, timepoints, feature_name, stimulus_name, save_path, significant_points=None):
    """Plot decoding results with updated data"""
    mean_accuracy = np.mean(scores, axis=0)
    std_accuracy = np.std(scores, axis=0)
    
    # Calculate confidence interval
    n_subjects = scores.shape[0]
    h = std_accuracy * 1.96 / np.sqrt(n_subjects)
    
    plt.figure(figsize=(10, 8))
    plt.plot(timepoints, mean_accuracy, color=FEATURES[feature_name]['color'])
    plt.fill_between(timepoints, mean_accuracy - h, mean_accuracy + h, 
                     color=FEATURES[feature_name]['color'], alpha=0.2)
    
    # Add significance markers
    if significant_points:
        sig_times = [tp for tp, _ in significant_points]
        if sig_times:
            earliest = timepoints[min(sig_times)]
            latest = timepoints[max(sig_times)]
            plt.hlines(y=0.6, xmin=earliest, xmax=latest, 
                      color=FEATURES[feature_name]['color'], 
                      linestyle='-', linewidth=2)
    
    plt.axhline(y=0.5, color='black', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.title(f'{feature_name.title()} decoding during {stimulus_name} (N={n_subjects})')
    
    # Save both formats, overwriting existing files
    plt.savefig(f'{save_path}/average_{feature_name}_{stimulus_name}_significance.pdf')
    plt.savefig(f'{save_path}/average_{feature_name}_{stimulus_name}_significance.png')
    plt.close()

def plot_comparison(results_dict, save_path):
    """Create comparison plots between different features/stimuli"""
    plt.figure(figsize=(12, 8))
    
    for (feature, stimulus), data in results_dict.items():
        mean_accuracy = np.mean(data, axis=0)
        std_error = np.std(data, axis=0) / np.sqrt(data.shape[0])
        timepoints = STIMULI[stimulus]['timepoints']
        
        plt.plot(timepoints, mean_accuracy, 
                label=f'{feature.title()} ({stimulus})',
                color=FEATURES[feature]['color'])
        plt.fill_between(timepoints, 
                        mean_accuracy - std_error,
                        mean_accuracy + std_error,
                        alpha=0.2,
                        color=FEATURES[feature]['color'])
    
    plt.axhline(y=0.5, color='black', linestyle='--', label='Chance')
    plt.xlabel('Time (s)')
    plt.ylabel('Decoding Accuracy')
    plt.title('Comparison of Decoding Performance')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{save_path}/feature_comparison.pdf')
    plt.savefig(f'{save_path}/feature_comparison.png')
    plt.close()

def run_analysis(subjects, meta_info, feature_name, stimulus_name):
    """Run complete analysis pipeline with improved error handling and result updating"""
    results_path = f"{PROCESSED_DIR}/timepoints/PhD/averaged/{feature_name}({stimulus_name})"
    os.makedirs(results_path, exist_ok=True)
    
    valid_scores = []
    valid_subjects = []
    
    for subject in subjects:
        try:
            # Check if subject data already exists
            existing_results_path = f'{results_path}/all_{feature_name}_{stimulus_name}_byTimepoint_SVM.xlsx'
            if os.path.exists(existing_results_path):
                existing_df = pd.read_excel(existing_results_path, index_col=0)
                if subject in existing_df.index:
                    print(f"Subject {subject} already processed, skipping...")
                    continue
            
            # Load and prepare data - now returns averaged data
            averaged_data, sfreq = load_subject_data(subject, meta_info, feature_name, stimulus_name)
            
            if averaged_data is None or sfreq is None:
                print(f"Skipping subject {subject} - invalid data")
                continue
            
            # Perform decoding with iterations
            mean_scores, std_scores = decode_sliding_window_iterations(averaged_data, sfreq, feature_name)
            
            # Store results
            valid_scores.append(mean_scores)
            valid_subjects.append(subject)
            
            # Plot individual results
            plt.figure(figsize=(10, 8))
            plt.plot(STIMULI[stimulus_name]['timepoints'], mean_scores)
            plt.fill_between(STIMULI[stimulus_name]['timepoints'], 
                           mean_scores - std_scores, 
                           mean_scores + std_scores, 
                           alpha=0.2)
            plt.xlabel('Time (s)')
            plt.ylabel('Accuracy')
            plt.title(f'{feature_name.title()} decoding for subject {subject} during {stimulus_name}\n(averaged across {len(averaged_data)} iterations)')
            plt.savefig(f'{results_path}/sub{subject}_{feature_name}_{stimulus_name}_SVM.png')
            plt.close()
            
            print(f'Completed {feature_name} decoding for subject {subject}')
            
        except Exception as e:
            print(f"Error processing subject {subject}: {str(e)}")
            continue
    
    if not valid_scores:
        print(f"No new results for {feature_name} during {stimulus_name}")
        return None, None, None
    
    # Convert to numpy array
    all_mean_timescores = np.array(valid_scores)
    
    # Save and combine with existing results
    combined_df = save_results(all_mean_timescores, 
                             valid_subjects, 
                             STIMULI[stimulus_name]['timepoints'],
                             feature_name, 
                             stimulus_name, 
                             results_path)
    
    # Compute statistics on complete dataset
    all_scores = combined_df.values
    p_values, significant_points = compute_statistics(all_scores)
    pd.DataFrame(p_values, columns=["p_values"]).to_csv(f'{results_path}/p_values.csv', index=False)
    
    # Plot updated results
    plot_results(all_scores, 
                STIMULI[stimulus_name]['timepoints'],
                feature_name, 
                stimulus_name,
                results_path,
                significant_points)
    
    return all_scores, p_values, significant_points

if __name__ == "__main__":
    # Setup
    keyboard = Controller()
    locale.setlocale(locale.LC_ALL, "en_US.utf8")
    
    # Load metadata
    metaInfo = pd.read_excel(META_FILE)
    NrSessions = 1
    Subs = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])

    # Prepare corrected files list for early subjects
    allFiles = metaInfo['MEG_Name']
    corrected_files = [f.split('.')[0] + '_correct_triggers.fif' for f in allFiles]
    corrected_files_series = pd.Series(corrected_files)
        
    # Option 1: Run all subjects
    #sub_lst = Subs.copy()
    
    # Option 2: Run a single subject (for testing)
    sub_lst = [23]  # Replace with subject number
    
    # Option 4: Run specific subjects from a list
    # sub_lst = np.array([23, 29, 30])  # Replace with your list
    
    # Option 5: Run remaining subjects after a certain point
    # completed_subjects = 27  # Number of subjects already processed
    # sub_lst = Subs[completed_subjects:]
    
    print(f"\nProcessing subjects: {sub_lst}")
    
    # Create combinations of features and stimuli !!! UNCOMMENT FOR FULL ANALYSIS !!!
    combinations = list(itertools.product(FEATURES.keys(), STIMULI.keys()))
    results_dict = {}

    # Create combinations - for special cases
    # features = ['location']  # voice_identity or location
    # stimuli = ['S2']  # S1 or S2
    # combinations = list(itertools.product(features, stimuli))
    # results_dict = {}
    
    # Run analysis for each combination
    for feature_name, stimulus_name in combinations:
        print(f"\n==== Processing {feature_name} during {stimulus_name} ====")
        scores, p_values, sig_points = run_analysis(sub_lst, metaInfo, feature_name, stimulus_name)
        if scores is not None:  # Only store if we got valid results
            results_dict[(feature_name, stimulus_name)] = scores
    
    # Create comparison plots if we have results
    if results_dict:
        os.makedirs(PROCESSED_DIR + '/timepoints/PhD/comparisons', exist_ok=True)
        plot_comparison(results_dict, PROCESSED_DIR + '/timepoints/PhD/comparisons')
        print("\nComparison plots created successfully!")
    
    print("\nAll analyses completed successfully!")