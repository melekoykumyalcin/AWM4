# Sliding Window Analysis for Delay Period Data
# Reads data from existing averaged files and performs sliding window decoding

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
from collections import Counter

# Constants and Configuration
HOME_DIR = '/media/headmodel/Elements/AWM4/'
META_FILE = HOME_DIR + 'MEGNotes.xlsx'
PROCESSED_DIR = HOME_DIR + '/AWM4_data/processed/'
DATA_PATH = HOME_DIR + '/AWM4_data/raw/'
CORRECTED_DATA = HOME_DIR + '/AWM4_data/raw/correctTriggers/'

# Decoding Parameters
RESAMPLE_FREQ = 100  # Hz
WINDOW_LENGTH_SEC = 0.1  # seconds (100ms)
WINDOW_STEP_SEC = 0.01  # seconds (10ms)
CV_SPLITS = 5
CV_REPEATS = 100
NUM_JOBS = 20  # Number of parallel jobs

# Delay Period Configuration
DELAY_CONFIG = {
    'tmin': 2.0,
    'tmax': 4.7,
    'timepoints': np.linspace(2.0, 4.7, int((4.7-2.0)*RESAMPLE_FREQ))
}

# Critical time points to mark on plots
CRITICAL_TIMEPOINTS = [3.5, 4.5]  # Gray dashed lines at these times

# Complete event dictionary
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

def load_averaged_data(subject):
    """
    Load averaged data from existing files
    
    Args:
        subject (int): Subject number
    
    Returns:
        tuple: List of averaged data iterations, sampling frequency
    """
    print(f'\n=== Loading averaged data for subject {subject} ===')
    
    try:
        # Define file path for averaged data
        avg_save_path = f"{PROCESSED_DIR}/averaged_data/sub-{subject}/delay(2)"
        avg_save_file = f"{avg_save_path}/averaged_data(2).pkl"
        
        if os.path.exists(avg_save_file):
            print(f"Loading pre-averaged data for subject {subject}...")
            with open(avg_save_file, 'rb') as f:
                averaged_data = pickle.load(f)
                
            print(f"Loaded {len(averaged_data)} iterations of averaged data")
            return averaged_data, RESAMPLE_FREQ
        else:
            print(f"No averaged data found for subject {subject} at {avg_save_file}")
            return None, None
    
    except Exception as e:
        print(f"Error loading averaged data for subject {subject}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def decode_sliding_window(data, y, sfreq, subject=None):
    """
    Perform decoding using a sliding window approach with complete timepoint coverage
    
    Parameters:
    data - 3D array of shape (n_trials, n_channels, n_timepoints)
    y - array of class labels
    sfreq - sampling frequency
    subject - subject ID for tracking (optional)
    
    Returns:
    window_scores - accuracy scores for each window
    window_centers - time points corresponding to the center of each window
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
    
    # Convert window parameters from seconds to samples
    window_length = int(WINDOW_LENGTH_SEC * sfreq)
    window_step = int(WINDOW_STEP_SEC * sfreq)
    
    # Use every single timepoint as the center of a window
    n_windows = n_timepoints
    
    # Pre-allocate arrays for results
    window_scores = np.zeros(n_windows)
    window_centers = np.zeros(n_windows)
    
    # Define parameter grid for SVM
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    
    # Process each window in parallel
    def process_window_nested(time_point):
        """Process a single time point with proper nested cross-validation"""
        try:
            # Calculate window start and end with centered window around timepoint
            half_window = window_length // 2
            win_start = max(0, time_point - half_window)
            win_end = min(n_timepoints, time_point + half_window + window_length % 2)
            
            # Calculate center time for this window
            window_center = time_point / sfreq + DELAY_CONFIG['tmin']
            
            # Extract data for this window
            X_win = data[:, :, win_start:win_end].reshape(n_trials, -1)
            
            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_win)
            
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
            c_counter = Counter(best_c_values)
            most_common_c = c_counter.most_common(1)[0][0]
            
            return {
                'accuracy': mean_score,
                'best_c': most_common_c,
                'window_center': window_center,
                'success': True
            }
        except Exception as e:
            print(f"Error processing timepoint {time_point}: {str(e)}")
            return {
                'accuracy': 0.5,
                'best_c': 1.0,
                'window_center': time_point / sfreq + DELAY_CONFIG['tmin'],
                'success': False
            }
    
    # Run parallel processing across all timepoints
    print(f"\nPerforming parallel nested cross-validation across {n_windows} timepoints...")
    results = Parallel(n_jobs=NUM_JOBS)(
        delayed(process_window_nested)(time_point) for time_point in tqdm(range(n_timepoints))
    )
    
    # Check if any windows were successfully processed
    successful_windows = [r for r in results if r['success']]
    if not successful_windows:
        print("No windows were successfully processed")
        return None, None
    
    # Extract results
    for i, r in enumerate(results):
        window_scores[i] = r['accuracy']
        window_centers[i] = r['window_center']
    
    # Save results if subject ID is provided
    if subject is not None:
        results_path = f"{PROCESSED_DIR}/timepoints/delay_sliding_window/params/{subject}"
        os.makedirs(results_path, exist_ok=True)
        
        # Save accuracy and time centers
        accuracy_data = np.column_stack((window_centers, window_scores))
        np.savetxt(f"{results_path}/sliding_window_results.txt", 
                  accuracy_data, 
                  header="Time_window_center Accuracy", 
                  fmt='%.4f %.4f')
        
        # Save the best C values
        c_values = [r['best_c'] for r in results]
        c_data = np.column_stack((window_centers, c_values))
        np.savetxt(f"{results_path}/best_c_values.txt", 
                  c_data, 
                  header="Time_window_center Best_C_value", 
                  fmt='%.4f %f')
    
    return window_scores, window_centers

def decode_sliding_window_iterations(averaged_data_list, sfreq, feature_name, subject=None):
    """
    Perform sliding window decoding across multiple iterations of averaged data
    
    Parameters:
    averaged_data_list - List of tuples (X, y) with averaged data
    sfreq - Sampling frequency
    feature_name - Name of feature being decoded
    subject - Subject ID (optional)
    
    Returns:
    mean_scores - Mean accuracy across iterations for each window
    std_scores - Standard deviation of accuracy across iterations
    window_centers - Center time points for each window
    """
    if not averaged_data_list:
        print(f"No averaged data available for decoding")
        return None, None, None
        
    all_window_scores = []
    window_centers = None
    
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
            iter_scores, iter_centers = decode_sliding_window(X, binary_y, sfreq, 
                                               subject=f"{subject}_iter{iter_idx}" if subject else None)
            
            if iter_scores is not None and iter_centers is not None:
                all_window_scores.append(iter_scores)
                
                # Store centers (same for all iterations)
                if window_centers is None:
                    window_centers = iter_centers
            else:
                print(f"Iteration {iter_idx + 1} produced no valid results. Skipping.")
        except Exception as e:
            print(f"Error decoding iteration {iter_idx + 1}: {str(e)}")
            continue
    
    # Check if we have any valid results
    if not all_window_scores:
        print("No valid decoding results across all iterations")
        return None, None, None
        
    # Average scores across iterations
    all_window_scores = np.array(all_window_scores)
    mean_scores = np.mean(all_window_scores, axis=0)
    std_scores = np.std(all_window_scores, axis=0)
    
    return mean_scores, std_scores, window_centers

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

def run_sliding_window_analysis(subjects):
    """
    Run sliding window decoding analysis on subjects
    
    Args:
        subjects (list): List of subject numbers
    """
    # Features to decode
    features = list(FEATURES.keys())
    
    results_dict = {}
    
    # Process each subject
    for subject in subjects:
        print(f"\n==== Processing subject {subject} with sliding window approach ====")
        
        try:
            # Load pre-averaged data
            averaged_data, sfreq = load_averaged_data(subject)
            
            # Skip if no data could be loaded
            if averaged_data is None:
                print(f"Skipping subject {subject} - no averaged data found")
                continue
                
            # Process all features for this subject
            for feature_name in features:
                # Create results directories
                results_path = f"{PROCESSED_DIR}/timepoints/delay_sliding_window/{feature_name}"
                os.makedirs(results_path, exist_ok=True)
                
                # Perform sliding window decoding across iterations
                try:
                    mean_scores, std_scores, window_centers = decode_sliding_window_iterations(
                        averaged_data, sfreq, feature_name, subject
                    )
                    
                    # Check for valid results
                    if mean_scores is None or window_centers is None:
                        print(f"No valid decoding results for subject {subject}, {feature_name}")
                        continue
                        
                    # Initialize storage in results_dict if needed
                    if feature_name not in results_dict:
                        results_dict[feature_name] = {
                            'scores': [],
                            'subjects': [],
                            'window_centers': window_centers
                        }
                    
                    # Store results
                    results_dict[feature_name]['scores'].append(mean_scores)
                    results_dict[feature_name]['subjects'].append(subject)
                    
                    # Plot individual subject results
                    plt.figure(figsize=(12, 7))
                    plt.plot(window_centers, mean_scores, 
                            color=FEATURES[feature_name]['color'], 
                            linewidth=2)
                    plt.fill_between(window_centers, 
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
                    # Set y-axis limits to 0.3-0.7
                    plt.ylim(0.3, 0.7)
                    plt.tight_layout()
                    plt.savefig(f'{results_path}/sub{subject}_decoding.png')
                    plt.close()
                    
                    print(f'Completed {feature_name} decoding for subject {subject}')
                    
                    # Save individual subject results
                    subject_result = {
                        'mean_scores': mean_scores,
                        'std_scores': std_scores,
                        'window_centers': window_centers
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
        window_centers = feature_data['window_centers']
        
        results_path = f"{PROCESSED_DIR}/timepoints/delay_sliding_window/{feature_name}"
        
        # Save results to Excel
        time_labels = [f'Time_{t:.3f}s' for t in window_centers]
        results_df = pd.DataFrame(all_scores, index=valid_subjects, columns=time_labels)
        results_df.to_excel(f'{results_path}/decoding_results.xlsx')
        
        # Calculate mean across subjects and timepoints
        mean_across_time = results_df.mean(axis=1)
        mean_across_time.to_excel(f'{results_path}/{feature_name}_mean_across_time.xlsx')
        
        # Calculate p-values and significant points
        p_values, significant_points = compute_statistics(all_scores)
        pd.DataFrame(p_values, columns=["p_values"]).to_csv(f'{results_path}/p_values.csv', index=False)
        
        # Compute statistics for group results
        mean_scores = np.mean(all_scores, axis=0)
        std_error = np.std(all_scores, axis=0) / np.sqrt(len(valid_subjects))
        
        # Plot group-level results
        plt.figure(figsize=(12, 7))
        plt.plot(window_centers, mean_scores, 
                label=f'Mean Accuracy (N={len(valid_subjects)})', 
                color=FEATURES[feature_name]['color'],
                linewidth=2)
                
        plt.fill_between(window_centers, 
                        mean_scores - std_error, 
                        mean_scores + std_error, 
                        alpha=0.2,
                        color=FEATURES[feature_name]['color'])
                        
        # Add critical timepoints as vertical lines
        for tp in CRITICAL_TIMEPOINTS:
            plt.axvline(x=tp, color='gray', linestyle='--', alpha=0.7)
        
        # Add significance markers if applicable
        if significant_points:
            sig_times = [window_centers[tp] for tp, _ in significant_points]
            if sig_times:
                plt.plot(sig_times, [0.6] * len(sig_times), 
                        marker='*', linestyle='', color=FEATURES[feature_name]['color'])
            
        plt.title(f'Group-Level {FEATURES[feature_name]["name"]} Decoding during Delay Period')
        plt.xlabel('Time (s)')
        plt.ylabel('Decoding Accuracy')
        plt.axhline(y=0.5, color='black', linestyle='--')
        # Set y-axis limits to 0.3-0.7
        plt.ylim(0.3, 0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{results_path}/group_decoding_result.png')
        plt.savefig(f'{results_path}/group_decoding_result.pdf')
        plt.close()
    
    # Create comparison plot for all features
    if len(results_dict) > 1:
        comparison_path = f"{PROCESSED_DIR}/timepoints/delay_sliding_window/comparison"
        os.makedirs(comparison_path, exist_ok=True)
        
        plt.figure(figsize=(12, 7))
        
        for feature_name, feature_data in results_dict.items():
            all_scores = np.array(feature_data['scores'])
            window_centers = feature_data['window_centers']
            
            mean_scores = np.mean(all_scores, axis=0)
            std_error = np.std(all_scores, axis=0) / np.sqrt(all_scores.shape[0])
            
            plt.plot(window_centers, mean_scores, 
                    label=f'{FEATURES[feature_name]["name"]} (N={all_scores.shape[0]})',
                    color=FEATURES[feature_name]['color'],
                    linewidth=2)
                    
            plt.fill_between(window_centers, 
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
        # Set y-axis limits to 0.3-0.7
        plt.ylim(0.3, 0.7)
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
    
    # Create necessary output directories
    os.makedirs(f"{PROCESSED_DIR}/timepoints/delay_sliding_window", exist_ok=True)
    
    # SUBJECT SELECTION - MODIFY THESE LINES:
    # Option 1: Run a single subject (uncomment and set the subject number)
    # subjects_to_process = [1]  # Replace with your desired subject number
    
    # Option 2: Run all subjects (uncomment this line and comment the line above)
    # Load metadata to get all subjects
    metaInfo = pd.read_excel(META_FILE)
    all_subjects = list(np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject']))
    subjects_to_process = all_subjects.copy()
    
    print(f"\nProcessing subjects: {subjects_to_process} using sliding window approach")
    
    # Check existing results files to determine which subjects have been processed
    results_path = f"{PROCESSED_DIR}/timepoints/delay_sliding_window"
    
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
    
    # Filter out already processed subjects
    # Comment this out if you want to reprocess subjects
    # subjects_to_process = [s for s in subjects_to_process if s not in processed_subjects]
    
    # Run the sliding window decoding analysis
    if len(subjects_to_process) > 0:
        results = run_sliding_window_analysis(subjects_to_process)
        print("\nSliding window analysis completed successfully!")
    else:
        print("No subjects to process.")