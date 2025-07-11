# Optimized Delay Period Decoding Analysis for AWM4 Dataset
# With working memory maintenance analysis based on retro-cues

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
CV_REPEATS = 100  # Reduced from 100 for better performance
NUM_JOBS = 20  # Number of parallel jobs

# Delay Period Configuration - updated to 2.0-4.7s
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
        tuple: Events array with information about which stimulus is maintained,
               and the original raw events array
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
        
        # Find indices for S1 and S2 retro-cues
        S1_idx = [i - 4 for i in range(len(all_events[:,2])) if all_events[i,2] == 101]
        S2_idx = [i - 2 for i in range(len(all_events[:,2])) if all_events[i,2] == 201]
        
        # Get the actual stimulus values for these indices
        S1_values = all_events[S1_idx,2]
        S2_values = all_events[S2_idx,2]
        
        # Create memorized array to track which stimulus is in working memory
        memorized = np.zeros(len(all_events[:,2]))
        memorized[S1_idx] = S1_values
        memorized[S2_idx] = S2_values
        memorized = memorized[memorized != 0]
        
        print(f'Extracted maintained information with {len(memorized)} trials')
        print('Unique maintained stimuli:', np.unique(memorized))
        
        return memorized, all_events
        
    except Exception as e:
        print(f"Error extracting maintained information for subject {subject}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

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
    
    try:
        # Check if there's already averaged data available
        avg_save_path = f"{PROCESSED_DIR}/averaged_data/sub-{subject}/delay(1)"
        os.makedirs(avg_save_path, exist_ok=True)
        avg_save_file = f"{avg_save_path}/averaged_data(1).pkl"
        
        if os.path.exists(avg_save_file):
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
            
            # If we can't match exactly, try to trim the longer array
            if len(memorized) < len(clean_trials.events):
                # Trim the epochs to match memorized length
                clean_trials = clean_trials[:len(memorized)]
                print(f"Trimmed epochs to {len(clean_trials)} to match memorized trials")
            elif len(memorized) > len(clean_trials.events):
                # Trim memorized to match epochs length
                memorized = memorized[:len(clean_trials.events)]
                print(f"Trimmed memorized to {len(memorized)} to match epochs")
            
            # Verify lengths match after trimming
            if len(memorized) != len(clean_trials.events):
                print("Could not align maintained stimuli with epochs. Skipping.")
                return None, None

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
        
        # Filter out cue events and keep only stimulus conditions
        stimulus_conditions = {k: v for k, v in EVENT_DICT.items() if v not in [101, 201]}

        # Equalize trial counts before averaging
        try:
            mag_epochs.equalize_event_counts(list(stimulus_conditions.keys()))
            print(f"Equalized to {len(mag_epochs)} trials per condition")
            
            if len(mag_epochs) == 0:
                print(f"WARNING: No trials remain after equalization for subject {subject}")
                return None, None
        except Exception as e:
            print(f"Error equalizing trial counts: {str(e)}")
            # Continue with unequal trials if equalization fails
            if len(mag_epochs) == 0:
                print(f"WARNING: No trials available for subject {subject}")
                return None, None
        
        # Get data and events
        data = mag_epochs.get_data()
        events = mag_epochs.events[:, 2]
        
        # Initialize storage for averaged data
        averaged_data = []
        n_iterations = 10  # Number of averaging iterations
        
        # For each iteration
        for iter_idx in range(n_iterations):
            print(f"\nProcessing iteration {iter_idx + 1}/{n_iterations}")
            
            # Group events by speaker and location
            speaker_groups = {
                'Sp1+2': [111, 112, 113, 114, 121, 122, 123, 124, 211, 212, 213, 214, 221, 222, 223, 224],
                'Sp3+4': [131, 132, 133, 134, 141, 142, 143, 144, 231, 232, 233, 234, 241, 242, 243, 244]
            }
            
            iter_data = []
            iter_events = []
            
            # Sample from each speaker group
            for group_name, group_events in speaker_groups.items():
                # Find indices for this group's events
                group_indices = np.where(np.isin(events, group_events))[0]
                
                if len(group_indices) > 0:
                    # Randomly select one trial from this group
                    selected_idx = np.random.choice(group_indices)
                    
                    iter_data.append(data[selected_idx])
                    iter_events.append(events[selected_idx])
            
            # Ensure we have trials from both speaker groups
            if len(iter_data) == 2:
                # Store this iteration's results
                averaged_data.append((
                    np.array(iter_data),
                    np.array(iter_events)
                ))
            else:
                print(f"Could not sample from both speaker groups in iteration {iter_idx + 1}")
        
        # Save the averaged data
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
    Perform decoding on binned data with optimized performance
    
    Parameters:
    data - 3D array of shape (n_trials, n_channels, n_timepoints)
    y - array of class labels
    sfreq - sampling frequency
    subject - subject ID for tracking (optional)
    
    Returns:
    bin_scores - accuracy scores for each time bin
    bin_centers - time points corresponding to the center of each bin
    """
    # Validate input data
    if data.size == 0:
        print("Empty data array provided")
        return None, None
        
    # Check for sufficient classes
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print(f"Insufficient classes for decoding. Found classes: {unique_classes}")
        return None, None
        
    # Verify each class has enough samples
    for cls in unique_classes:
        if np.sum(y == cls) < 2:  # Need at least 2 samples per class for cross-validation
            print(f"Insufficient samples for class {cls}: only {np.sum(y == cls)} samples")
            return None, None
    
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
    param_grid = {'C': [0.001, 0.01, 0.1, 1]}
    
    # Process each bin in parallel
    def process_bin(bin_idx):
        try:
            # Extract data for this bin
            X_bin = binned_data[:, :, bin_idx].reshape(n_trials, -1)  # Flatten channels
            
            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_bin)
            
            # Setup cross-validation
            cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS)
            
            # Create GridSearchCV
            grid_search = GridSearchCV(
                estimator=SVC(kernel='linear', probability=True),
                param_grid=param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=1  # Use 1 here since we're parallelizing at a higher level
            )
            
            # Fit and get results
            grid_search.fit(X_scaled, y)
            
            return {
                'accuracy': grid_search.best_score_,
                'best_c': grid_search.best_params_['C'],
                'success': True
            }
        except Exception as e:
            print(f"Error processing bin {bin_idx}: {str(e)}")
            return {
                'accuracy': 0.5,  # Default to chance level
                'best_c': 1.0,    # Default C value
                'success': False
            }
    
    # Run parallel processing across bins
    print("\nPerforming parallel decoding across bins...")
    results = Parallel(n_jobs=NUM_JOBS)(
        delayed(process_bin)(bin_idx) for bin_idx in tqdm(range(n_bins))
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
        results_path = f"{PROCESSED_DIR}/timepoints/delay_binned/params/{subject}"
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
        
        # Create binary classification labels
        if feature_name == 'maintained_voice_identity':
            # Speaker decoding: 0 (Sp1+2) vs 1 (Sp3+4)
            binary_y = np.where(y < 2, 0, 1)
        else:  # maintained_location
            # Location decoding: 0 (L1+2) vs 1 (L3+4)
            binary_y = np.where(y % 2 == 0, 0, 1)
        
        # Verify we have both classes
        unique_classes = np.unique(binary_y)
        if len(unique_classes) < 2:
            print(f"Iteration {iter_idx + 1} has only class {unique_classes[0]} after binarization. Skipping.")
            continue
            
        # Print class distribution
        print("\nClass distribution:")
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
            
            # Skip if no data could be loaded
            if averaged_data is None:
                print(f"Skipping subject {subject} - no valid data")
                continue
                
            # Process all features for this subject
            for feature_name in features:
                # Create results directories
                results_path = f"{PROCESSED_DIR}/timepoints/delay_binned(1)/{feature_name}"
                os.makedirs(results_path, exist_ok=True)
                
                # Perform binned decoding across iterations
                try:
                    mean_scores, std_scores, bin_centers = decode_binned_data_iterations(
                        averaged_data, sfreq, feature_name, subject
                    )
                    
                    # Check for valid results
                    if mean_scores is None or bin_centers is None:
                        print(f"No valid decoding results for subject {subject}, {feature_name}")
                        continue
                        
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
                    plt.ylim(0.3, 1.0)
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
        all_scores = np.array(feature_data['scores'])
        valid_subjects = feature_data['subjects']
        bin_centers = feature_data['bin_centers']
        
        results_path = f"{PROCESSED_DIR}/timepoints/delay_binned(1)/{feature_name}"
        
        # Save results to Excel
        bin_labels = [f'Bin_{t:.3f}s' for t in bin_centers]
        results_df = pd.DataFrame(all_scores, index=valid_subjects, columns=bin_labels)
        results_df.to_excel(f'{results_path}/decoding_results.xlsx')
        
        # Calculate mean across subjects and timepoints
        mean_across_time = results_df.mean(axis=1)
        mean_across_time.to_excel(f'{results_path}/{feature_name}_mean_across_time.xlsx')
        
        # Compute significance
        p_values, significant_points = compute_statistics(all_scores)
        pd.DataFrame(p_values, columns=["p_values"]).to_csv(f'{results_path}/p_values.csv', index=False)
        
        # Compute statistics
        mean_scores = np.mean(all_scores, axis=0)
        std_error = np.std(all_scores, axis=0) / np.sqrt(len(valid_subjects))
        
        # Plot group-level results
        plt.figure(figsize=(12, 7))
        plt.plot(bin_centers, mean_scores, 
                label=f'Mean Accuracy (N={len(valid_subjects)})', 
                color=FEATURES[feature_name]['color'],
                linewidth=2)
                
        plt.fill_between(bin_centers, 
                        mean_scores - std_error, 
                        mean_scores + std_error, 
                        alpha=0.2,
                        color=FEATURES[feature_name]['color'])
                        
        # Add critical timepoints as vertical lines
        for tp in CRITICAL_TIMEPOINTS:
            plt.axvline(x=tp, color='gray', linestyle='--', alpha=0.7)
        
        # Add significance markers if applicable
        if significant_points:
            sig_times = [bin_centers[tp] for tp, _ in significant_points]
            if sig_times:
                plt.plot(sig_times, [0.6] * len(sig_times), 
                        marker='*', linestyle='', color=FEATURES[feature_name]['color'])
            
        plt.title(f'Group-Level {FEATURES[feature_name]["name"]} Decoding during Delay Period')
        plt.xlabel('Time (s)')
        plt.ylabel('Decoding Accuracy')
        plt.axhline(y=0.5, color='black', linestyle='--')
        plt.ylim(0.3, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{results_path}/group_decoding_result.png')
        plt.savefig(f'{results_path}/group_decoding_result.pdf')
        plt.close()
    
    # Create comparison plot for all features
    if len(results_dict) > 1:
        comparison_path = f"{PROCESSED_DIR}/timepoints/delay_binned(1)/comparison"
        os.makedirs(comparison_path, exist_ok=True)
        
        plt.figure(figsize=(12, 7))
        
        for feature_name, feature_data in results_dict.items():
            all_scores = np.array(feature_data['scores'])
            bin_centers = feature_data['bin_centers']
            
            mean_scores = np.mean(all_scores, axis=0)
            std_error = np.std(all_scores, axis=0) / np.sqrt(all_scores.shape[0])
            
            plt.plot(bin_centers, mean_scores, 
                    label=f'{FEATURES[feature_name]["name"]} (N={all_scores.shape[0]})',
                    color=FEATURES[feature_name]['color'],
                    linewidth=2)
                    
            plt.fill_between(bin_centers, 
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
        plt.ylim(0.3, 1.0)
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
    
    # Get all subjects from the final sample
    all_subjects = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
    
    # Create necessary output directories
    os.makedirs(f"{PROCESSED_DIR}/timepoints/delay_binned(1)", exist_ok=True)
    os.makedirs(f"{PROCESSED_DIR}/averaged_data", exist_ok=True)
    os.makedirs(f"{PROCESSED_DIR}/trial_records", exist_ok=True)
    
    # Check existing results files to determine which subjects have been processed
    results_path = f"{PROCESSED_DIR}/timepoints/delay_binned(1)"
    
    # Initialize lists to track processed and unprocessed subjects
    processed_subjects = []
    unprocessed_subjects = []
    
    # Check existing Excel files for each feature
    features = ['maintained_voice_identity', 'maintained_location']
    
    for feature in features:
        feature_results_file = f"{results_path}/{feature}/decoding_results.xlsx"
        
        # If results file exists, read processed subjects
        if os.path.exists(feature_results_file):
            existing_results = pd.read_excel(feature_results_file, index_col=0)
            processed_subjects = list(map(int, existing_results.index))
        else:
            processed_subjects = []
        
        # Identify unprocessed subjects
        unprocessed_subjects = [sub for sub in all_subjects if sub not in processed_subjects]
        
        print(f"\n=== Feature: {feature} ===")
        print(f"Processed subjects: {processed_subjects}")
        print(f"Unprocessed subjects: {unprocessed_subjects}")
    
    # Run the comprehensive delay period decoding analysis for unprocessed subjects
    if unprocessed_subjects:
        results = run_delay_decoding_analysis(unprocessed_subjects, metaInfo)
        
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
                combined_results = pd.concat([existing_results, new_results_df])
                combined_results.to_excel(feature_results_file)
                print(f"Appended results to {feature_results_file}")
            else:
                # Create new results file
                new_results_df.to_excel(feature_results_file)
                print(f"Created new results file {feature_results_file}")
    else:
        print("All subjects have already been processed.")
    
    print("\nDelay period analysis completed successfully!")