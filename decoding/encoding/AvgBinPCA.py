#binned+averaged decoding analysis for AWM4 dataset with overlapping bins

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
from sklearn.decomposition import PCA
from mne.decoding import Scaler, Vectorizer, cross_val_multiscore
import seaborn as sns
from joblib import Parallel, delayed
import itertools
import pickle

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
BIN_SIZE = 10  # data points (100ms at 100Hz)
STEP_SIZE = 1  # How many points to move for each bin (10ms step - 90% overlap)
PCA_VARIANCE_THRESHOLD = 0.99  # retain 99% of variance
CV_SPLITS = 5
CV_REPEATS = 10 #changed from 100 to 10 change later for more iterations
NUM_JOBS = -1 # -1 to use all CPUs

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
        'color': {
            'S1': '#1c686b',  # Darker teal
            'S2': '#64b1b5'   # Lighter teal
        },
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
        'color': {
            'S1': '#cb6a3e',  # Darker orange
            'S2': '#e8a87c'   # Lighter orange
        },
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

def average_condition_groups(epochs, stimulus_name, n_iterations=10, subject=None, force_recompute=False):
    
    """
    Average trials within condition groups with multiple iterations
    """

    # If subject is provided, check for saved data 
    if subject is not None:
        save_dir = f"{PROCESSED_DIR}/averaged_data/sub-{subject}/{stimulus_name}"
        os.makedirs(save_dir, exist_ok=True)
        save_file = f"{save_dir}/averaged_data.pkl"
        
        # Check if saved data exists and we're not forcing recomputation
        if os.path.exists(save_file) and not force_recompute:
            print(f"\nLoading saved averaged data for subject {subject}, {stimulus_name}...")
            with open(save_file, 'rb') as f:
                averaged_data, trial_records = pickle.load(f)
            print(f"Loaded {len(averaged_data)} iterations of averaged data")
            return averaged_data, trial_records
        
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
    if subject is not None:
        print(f"\nSaving averaged data for subject {subject}, {stimulus_name}...")
        with open(save_file, 'wb') as f:
            pickle.dump((averaged_data, trial_records), f)
    
    return averaged_data, trial_records
    
# With this function
def decode_binned_data(data, y, sfreq, subject=None):
    """
    Perform decoding on overlapping binned data with SVM hyperparameter optimization
    using GridSearchCV with 5-fold CV
    
    Parameters:
    data - 3D array of shape (n_trials, n_channels, n_timepoints)
    y - array of class labels
    sfreq - sampling frequency
    subject - subject ID for tracking (optional)
    
    Returns:
    bin_scores - accuracy scores for each time bin
    bin_centers - time points corresponding to the center of each bin
    """
    from sklearn.model_selection import GridSearchCV
    
    n_trials, n_channels, n_timepoints = data.shape
    
    # Define parameter grid for SVM
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    
    # Calculate number of bins with overlap
    n_bins = (n_timepoints - BIN_SIZE) // STEP_SIZE + 1
    bin_scores = []
    bin_centers = []
    best_c_values = []  # Track best C values for each bin
    pca_components_by_bin = []
    
    # Loop through time bins with overlap
    for bin_idx in tqdm(range(n_bins)):
        bin_start = bin_idx * STEP_SIZE
        bin_end = bin_start + BIN_SIZE
        
        # Calculate center time point for this bin
        bin_center = (bin_start + bin_end) / 2 / sfreq
        bin_centers.append(bin_center)
        
        # Extract data for this bin and average across time points within bin
        X_bin = np.mean(data[:, :, bin_start:bin_end], axis=2)
        
        # Setup outer cross-validation
        outer_cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS)
        bin_fold_scores = []
        bin_fold_c_values = []
        bin_fold_n_components = []
        
        # Outer cross-validation loop
        for train_idx, test_idx in outer_cv.split(X_bin, y):
            X_train_outer, X_test_outer = X_bin[train_idx], X_bin[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            
            # Standardize data
            scaler = StandardScaler()
            X_train_outer_scaled = scaler.fit_transform(X_train_outer)
            X_test_outer_scaled = scaler.transform(X_test_outer)
            
            # Apply PCA to training data and use same transformation for test data
            pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, svd_solver='full')
            X_train_outer_pca = pca.fit_transform(X_train_outer_scaled)
            X_test_outer_pca = pca.transform(X_test_outer_scaled)
            
            # Track number of components retained
            bin_fold_n_components.append(pca.n_components_)
            
            # Setup inner cross-validation with GridSearchCV
            inner_cv = StratifiedKFold(n_splits=5)
            
            # Create base classifier
            base_clf = SVC(kernel='linear', probability=True)
            
            # Create GridSearchCV
            grid_search = GridSearchCV(
                estimator=base_clf,
                param_grid=param_grid,
                cv=inner_cv,  # 5-fold CV
                scoring='accuracy',
                n_jobs=NUM_JOBS,  # Utilize all available CPUs
                return_train_score=False
            )
            
            # Fit grid search on outer training data
            grid_search.fit(X_train_outer_pca, y_train_outer)
            
            # Get best C value
            best_c = grid_search.best_params_['C']
            bin_fold_c_values.append(best_c)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Evaluate on outer test set
            accuracy = best_model.score(X_test_outer_pca, y_test_outer)
            bin_fold_scores.append(accuracy)
        
        # Average scores across outer folds
        bin_scores.append(np.mean(bin_fold_scores))
        
        # Find most common best C value for this bin
        from collections import Counter
        c_counter = Counter(bin_fold_c_values)
        most_common_c = c_counter.most_common(1)[0][0]
        best_c_values.append(most_common_c)

        # Average number of PCA components for this bin
        avg_components = np.mean(bin_fold_n_components)
        pca_components_by_bin.append(avg_components)

        print(f"Bin {bin_idx+1}/{n_bins} - Best C value: {most_common_c} - Avg components retained: {avg_components:.1f}")
                    
    # Save the best C values and PCA components for reference
    results_dir = f"{PROCESSED_DIR}/timepoints/PhD/bins/overlapping"
    os.makedirs(results_dir, exist_ok=True)  # Create directory if it doesn't exist
    results_data = np.column_stack((bin_centers, best_c_values, pca_components_by_bin))
    np.savetxt(f"{results_dir}/best_c_values.txt", 
            results_data, 
            header="Time_bin_center Best_C_value Avg_PCA_components", 
            fmt='%.4f %f %.1f')

    # Save per-subject component retention if subject ID is provided
    if subject is not None:
        subject_pca_dir = f"{PROCESSED_DIR}/timepoints/PhD/bins/overlapping/components_by_subject"
        os.makedirs(subject_pca_dir, exist_ok=True)
        np.savetxt(f"{subject_pca_dir}/sub-{subject}_pca_components.txt",
                np.column_stack((bin_centers, pca_components_by_bin)),
                header="Time_bin_center Avg_PCA_components",
                fmt='%.4f %.1f')
    
    return np.array(bin_scores), np.array(bin_centers)

def decode_binned_data_iterations(averaged_data_list, sfreq, feature_name, subject=None):
    """
    Perform binned decoding across multiple iterations
    
    Parameters:
    averaged_data_list - List of tuples (X, y) with averaged data
    sfreq - Sampling frequency
    feature_name - Name of feature being decoded ('voice_identity' or 'location')
    subject - Subject ID (optional)
    
    Returns:
    mean_scores - Mean accuracy across iterations for each bin
    std_scores - Standard deviation of accuracy across iterations for each bin
    bin_centers - Center time points for each bin
    """
    all_bin_scores = []
    bin_centers = None
    
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
        
        # Only pass subject ID in first iteration to avoid duplicate files
        iter_subject = subject if iter_idx == 0 else None
        
        # Decode this iteration
        iter_scores, iter_centers = decode_binned_data(X, binary_y, sfreq, subject=iter_subject)
        all_bin_scores.append(iter_scores)
        
        # Store centers (same for all iterations)
        if bin_centers is None:
            bin_centers = iter_centers
    
    # Average scores across iterations
    mean_scores = np.mean(all_bin_scores, axis=0)
    std_scores = np.std(all_bin_scores, axis=0)
    
    return mean_scores, std_scores, bin_centers

def run_analysis(subjects, meta_info, feature_name, stimulus_name, force_recompute_avg=False):
    """Run complete analysis pipeline with improved error handling and result updating"""
    results_path = f"{PROCESSED_DIR}/timepoints/PhD/bins/overlapPCA/{feature_name}({stimulus_name})"
    os.makedirs(results_path, exist_ok=True)
    
    valid_scores = []
    valid_subjects = []
    bin_centers = None
    
    for subject in subjects:
        try:
            # Check if subject data already exists
            existing_results_path = f'{results_path}/all_{feature_name}_{stimulus_name}_byBin_overlapping_SVM.xlsx'
            if os.path.exists(existing_results_path):
                existing_df = pd.read_excel(existing_results_path, index_col=0)
                if subject in existing_df.index:
                    print(f"Subject {subject} already processed, skipping...")
                    continue
            
            # Load and prepare data - now returns averaged data
            averaged_data, sfreq = load_subject_data(
                subject, 
                meta_info, 
                feature_name, 
                stimulus_name,
                force_recompute_avg=force_recompute_avg
            )
            
            if averaged_data is None or sfreq is None:
                print(f"Skipping subject {subject} - invalid data")
                continue
            
            # Perform decoding with iterations using bin-based approach
            mean_scores, std_scores, subject_bin_centers = decode_binned_data_iterations(
                averaged_data, 
                sfreq, 
                feature_name,
                subject=subject  
            )
            
            # Store results
            valid_scores.append(mean_scores)
            valid_subjects.append(subject)
            
            # Store bin centers (same for all subjects)
            if bin_centers is None:
                bin_centers = subject_bin_centers
            
            # Plot individual results
            plt.figure(figsize=(10, 8))
            plt.plot(subject_bin_centers, mean_scores, linewidth=2)
            plt.fill_between(subject_bin_centers, 
                           mean_scores - std_scores, 
                           mean_scores + std_scores, 
                           alpha=0.2)
            plt.xlabel('Time (s)')
            plt.ylabel('Accuracy')
            plt.title(f'{feature_name.title()} decoding for subject {subject} during {stimulus_name}\n(Overlapping Bins Analysis: {BIN_SIZE*10}ms bins, {STEP_SIZE*10}ms step)')
            plt.savefig(f'{results_path}/sub{subject}_{feature_name}_{stimulus_name}_overlapping_SVM.png')
            plt.close()
            
            print(f'Completed {feature_name} decoding for subject {subject}')
            
        except Exception as e:
            print(f"Error processing subject {subject}: {str(e)}")
            continue
    
    if not valid_scores:
        print(f"No new results for {feature_name} during {stimulus_name}")
        return None, None, None, None
    
    # Convert to numpy array
    all_mean_timescores = np.array(valid_scores)
    
    # Save and combine with existing results
    combined_df = save_results(all_mean_timescores, 
                             valid_subjects, 
                             bin_centers,
                             feature_name, 
                             stimulus_name, 
                             results_path)
    
    # Compute statistics on complete dataset
    all_scores = combined_df.values
    p_values, significant_points = compute_statistics(all_scores)
    pd.DataFrame(p_values, columns=["p_values"]).to_csv(f'{results_path}/p_values.csv', index=False)
    
    # Plot updated results
    plot_results(all_scores, 
                bin_centers,
                feature_name, 
                stimulus_name,
                results_path,
                significant_points)
    
    return all_scores, bin_centers, p_values, significant_points

def load_subject_data(subject, meta_info, feature_name, stimulus_name, force_recompute_avg=False):
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
            print(f"Initial S2 events found: {len(stim_idx)}")
            print(f"Unique event values: {np.unique(stim_values)}")
        else:
            stim_idx = [i - 1 for i in range(len(all_events[:,2])) if all_events[i,2] == 100]
            stim_values = all_events[stim_idx,2]
            
        print(f"Initial number of events: {len(stim_values)}")
            
        # 3. Load cleaned epochs
        clean_trials = mne.read_epochs(
            f"{PROCESSED_DIR}/CutEpochs/CutData_VP{subject}-cleanedICA-epo.fif",
            preload=True,
            verbose='INFO'
        )
        print(f"Initial epochs loaded: {len(clean_trials)}")
        
        # Drop specific trials for specific subjects
        if subject == 28:
            clean_trials.drop(63)
        if subject == 23:
            clean_trials.drop(64*7)
                
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
        averaged_data, trial_records = average_condition_groups(
            mag_epochs, 
            stimulus_name, 
            n_iterations=10,
            subject=subject,
            force_recompute=force_recompute_avg
)        
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
    
    except Exception as e:
        print(f"\nERROR processing subject {subject}:")
        print(str(e))
        import traceback
        traceback.print_exc()
        return None, None

def plot_comparison(results_dict, save_path, significance_dict=None):
    """Create comparison plots between different features/stimuli with significance markers
    
    Parameters:
    results_dict - Dictionary with (feature, stimulus) keys and (data, bin_centers) values
    save_path - Path to save the plot
    significance_dict - Dictionary with (feature, stimulus) keys and significant_points values
    """
    plt.figure(figsize=(12, 8))
    
    # Define vertical offsets for significance markers to prevent overlap
    # Start with different heights for each feature/stimulus combo
    significance_heights = {}
    current_height = 0.62
    height_step = 0.02
    
    for (feature, stimulus) in results_dict.keys():
        significance_heights[(feature, stimulus)] = current_height
        current_height += height_step
    
    # Plot each feature/stimulus data
    for (feature, stimulus), (data, bin_centers) in results_dict.items():
        mean_accuracy = np.mean(data, axis=0)
        std_error = np.std(data, axis=0) / np.sqrt(data.shape[0])
        
        color = FEATURES[feature]['color'][stimulus]
        
        # Plot the mean accuracy line
        plt.plot(bin_centers, mean_accuracy, 
                label=f'{feature.title()} ({stimulus})',
                color=color,
                linewidth=2)
        
        # Add confidence interval
        plt.fill_between(bin_centers, 
                        mean_accuracy - std_error,
                        mean_accuracy + std_error,
                        alpha=0.2,
                        color=color)
        
        # Plot significance markers if provided
        if significance_dict and (feature, stimulus) in significance_dict:
            sig_points = significance_dict[(feature, stimulus)]
            if sig_points:
                # Extract significant bin indices
                sig_bins = [bin_idx for bin_idx, _ in sig_points]
                
                if sig_bins:
                    # Get the height for this feature/stimulus
                    sig_height = significance_heights[(feature, stimulus)]
                    
                    # Group consecutive significant points
                    segments = []
                    current_segment = []
                    
                    for i in range(len(sig_bins)):
                        if i == 0 or sig_bins[i] > sig_bins[i-1] + 1:
                            if current_segment:
                                segments.append(current_segment)
                            current_segment = [sig_bins[i]]
                        else:
                            current_segment.append(sig_bins[i])
                    
                    if current_segment:
                        segments.append(current_segment)
                    
                    # Plot each segment as a horizontal line
                    for segment in segments:
                        x_start = bin_centers[segment[0]]
                        x_end = bin_centers[segment[-1]]
                        plt.plot([x_start, x_end], [sig_height, sig_height], 
                                color=color, linewidth=2, solid_capstyle='round')
                        
                        # Add asterisks at the center of the segment
                        x_center = (x_start + x_end) / 2
                        marker = next((m for idx, m in sig_points if idx == segment[len(segment)//2]), '*')
                        plt.text(x_center, sig_height + 0.01, marker, color=color, 
                                ha='center', va='bottom', fontsize=14)
    
    # Add reference lines
    plt.axhline(y=0.5, color='black', linestyle='--', label='Chance')
    plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Decoding Accuracy')
    plt.title(f'Comparison of Overlapping Bins Decoding Performance\n({BIN_SIZE*10}ms bins, {STEP_SIZE*10}ms step)')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{save_path}/feature_comparison_overlapping_bins.pdf')
    plt.savefig(f'{save_path}/feature_comparison_overlapping_bins.png')
    plt.close()

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

def plot_results(scores, bin_centers, feature_name, stimulus_name, save_path, significant_points=None):
    """Plot decoding results with updated data for overlapping bins"""
    mean_accuracy = np.mean(scores, axis=0)
    std_accuracy = np.std(scores, axis=0)
    
    # Calculate confidence interval
    n_subjects = scores.shape[0]
    h = std_accuracy * 1.96 / np.sqrt(n_subjects)
    
    plt.figure(figsize=(10, 8))
    
    # For overlapping bins, use a line plot instead of markers to show smoothness
    plt.plot(bin_centers, mean_accuracy, 
             color=FEATURES[feature_name]['color'][stimulus_name], 
             linestyle='-',
             linewidth=2)
    
    plt.fill_between(bin_centers, mean_accuracy - h, mean_accuracy + h, 
                     color=FEATURES[feature_name]['color'][stimulus_name], 
                     alpha=0.2)
    
    # Add significance markers
    if significant_points:
        for bin_idx, marker in significant_points:
            if bin_idx < len(bin_centers):  # Ensure the index is valid
                plt.text(bin_centers[bin_idx], mean_accuracy[bin_idx] + 0.02, 
                        marker, color=FEATURES[feature_name]['color'][stimulus_name], 
                        fontsize=12, ha='center')
    
    plt.axhline(y=0.5, color='black', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.title(f'{feature_name.title()} decoding during {stimulus_name} (N={n_subjects})\nOverlapping Bins Analysis ({BIN_SIZE*10}ms bins, {STEP_SIZE*10}ms step)')
    
    # Save both formats, overwriting existing files
    plt.savefig(f'{save_path}/average_{feature_name}_{stimulus_name}_overlapping_bins_significance.pdf')
    plt.savefig(f'{save_path}/average_{feature_name}_{stimulus_name}_overlapping_bins_significance.png')
    plt.close()

def save_results(all_mean_timescores, valid_subjects, bin_centers, feature_name, stimulus_name, results_path):
    """Save results with detailed bin information, appending to existing files if they exist"""
    
    # Create DataFrame with bin centers as columns and subjects as rows
    bin_labels = [f"Bin_{t:.3f}s" for t in bin_centers]
    new_results_df = pd.DataFrame(all_mean_timescores, 
                                index=valid_subjects,
                                columns=bin_labels)
    
    # Path for detailed bin results
    detailed_path = f'{results_path}/all_{feature_name}_{stimulus_name}_byBin_overlapping_SVM.xlsx'
    
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
    
    # Update mean across all bins
    mean_across_bins = combined_df.mean(axis=1)
    mean_across_bins.to_excel(f'{results_path}/all_{feature_name}_{stimulus_name}_meanAcrossBins_overlapping_SVM.xlsx')
    
    # Save bin centers for reference
    pd.DataFrame(bin_centers, columns=["bin_centers"]).to_csv(f'{results_path}/bin_centers_overlapping.csv', index=False)
    
    return combined_df

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
    sub_lst = Subs.copy()
    
    # Option 2: Run a single subject (for testing)
    # sub_lst = [1]  # Replace with subject number
    
    # Option 3: Run specific subjects from a list
    # sub_lst = np.array([24, 25, 26, 27])  # Replace with your list
    
    # Option 4: Run remaining subjects after a certain point
    # completed_subjects = 27  # Number of subjects already processed
    # sub_lst = Subs[completed_subjects:]
    
    print(f"\nProcessing subjects: {sub_lst}")
    
    # Create combinations of features and stimuli
    combinations = list(itertools.product(FEATURES.keys(), STIMULI.keys()))
    results_dict = {}
    significance_dict = {}

    # Create combinations - for special cases
    # features = ['location']  # voice_identity or location
    # stimuli = ['S2']  # S1 or S2
    # combinations = list(itertools.product(features, stimuli))
    # results_dict = {}
    # significance_dict = {}
    
    # Run analysis for each combination
    for feature_name, stimulus_name in combinations:
        print(f"\n==== Processing {feature_name} during {stimulus_name} ====")
        scores, bin_centers, p_values, sig_points = run_analysis(sub_lst, metaInfo, feature_name, stimulus_name)
        if scores is not None:  # Only store if we got valid results
            results_dict[(feature_name, stimulus_name)] = (scores, bin_centers)
            significance_dict[(feature_name, stimulus_name)] = sig_points
    
    # Create comparison plots if we have results
    if results_dict:
        comparison_path = f"{PROCESSED_DIR}/timepoints/PhD/bins/overlapPCA/comparisons"
        os.makedirs(comparison_path, exist_ok=True)
        plot_comparison(results_dict, comparison_path, significance_dict)
        print("\nComparison plots created successfully!")
    
    print("\nAll overlapping bins PCA analysis completed successfully!")