#encoding S2#

# Import libraries
import os
import locale
import pathlib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
plt.ioff()
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = [10, 8]
import numpy as np#, h5py
#import scipy
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
mne.set_log_level('warning')
import pandas as pd
import autoreject
from autoreject import get_rejection_threshold
from pynput.keyboard import Key, Controller
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from mne.decoding import Scaler, Vectorizer, cross_val_multiscore, SlidingEstimator
import numpy as np
import seaborn as sns


keyboard = Controller()

# Set paths
homeDir = '/media/headmodel/Elements/AWM4/'
os.chdir(homeDir) # change current directory

# Load in meta information
metaFile = homeDir + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(metaFile)
NrSessions = 1
Subs       = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject'])
#Subs = Subs[20:21] #for testing purposes !!!
locale.setlocale(locale.LC_ALL, "en_US.utf8")

# paths and file names
data_path   = homeDir +'/AWM4_data/raw/'
corrected_data = homeDir +'/AWM4_data/raw/correctTriggers/'
allFiles    = metaInfo['MEG_Name']
noiseFiles  = metaInfo['Noise_Measurement']
block = metaInfo['Block']
# for the Subs[:7]
corrected_files = [f.split('.')[0] + '_correct_triggers.fif' for f in allFiles]
corrected_files_series = pd.Series(corrected_files)
# saving the results
results_path = str(homeDir)+'/AWM4_data/processed/timepoints/S2encoding(10)'
if not os.path.exists(results_path):
    os.makedirs(results_path)
all_mean_timescores = np.zeros((len(Subs), 10)) # change the number of timepoints according to sampling rate 
all_mean_acrossalltimes = np.zeros(len(Subs))

#actSubj = Subs[int(input("Subject no [1-26]: "))-1] 
actSubj = False # change into False if you want to loop over all subjects
sub_lst = Subs.copy()
if actSubj:
    sub_lst = sub_lst[(actSubj-1): actSubj ]

# Loop over subjects to get all the raw data/events as well as the cleaned epochs:
for actSubj in sub_lst:
    actSubj_idx = np.where(Subs == actSubj)[0][0]
    print(f'starting with subject {actSubj}...')
    actInd        = (metaInfo.Subject==actSubj) & (metaInfo.Valid==1)
    if actSubj in Subs[:7]:
        actFiles      = corrected_files_series[actInd]
    else:
        actFiles      = allFiles[actInd]
    actNoiseFiles = noiseFiles[actInd] 
    for ff in range(actFiles.count()):
        if actSubj in Subs[:7]:
            fname = corrected_data + actFiles.iloc[ff]
            raw = mne.io.read_raw_fif(fname, preload=True)
        else:
            fname = data_path + actFiles.iloc[ff]
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
        #for each ff, there will be a different events array, create an events array for every ff 
        if ff == 0:
            all_events = events
        else:
            all_events = np.concatenate((all_events, events), axis=0)
        del raw, events

    print(f'raw data loaded...')

    #according to the cue, which S1 or S2 value it is kept in WM                
    S2_idx = [i - 1 for i in range(len(all_events[:,2])) if all_events[i,2] == 200]
    S2_values = all_events[S2_idx,2]
    if actSubj == 23:
        drop_idx = 64*7
        S2_values = np.delete(S2_values, drop_idx)

    #which trials are lowpitch and which are highpitch
    idx_lowpitch = [i for i in range(len(S2_values)) if S2_values[i] in [211, 212, 213, 214, 221, 222, 223, 224]]
    idx_highpitch = [i for i in range(len(S2_values)) if S2_values[i] in [231, 232, 233, 234, 241, 242, 243, 244]]

    # load cleaned and cut epochs per trial 
    CleanTrials = mne.read_epochs(str(homeDir)+'/AWM4_data/processed/CutEpochs/CutData_VP'+str(actSubj)+'-cleanedICA-epo.fif')
    
    event_dict  = {'S2/Sp1/L1': 211, 'S2/Sp1/L2': 212, 'S2/Sp1/L3': 213, 'S2/Sp1/L4': 214,
                'S2/Sp2/L1': 221, 'S2/Sp2/L2': 222, 'S2/Sp2/L3': 223, 'S2/Sp2/L4': 224,
                'S2/Sp3/L1': 231, 'S2/Sp3/L2': 232, 'S2/Sp3/L3': 233, 'S2/Sp3/L4': 234,
                'S2/Sp4/L1': 241, 'S2/Sp4/L2': 242, 'S2/Sp4/L3': 243, 'S2/Sp4/L4': 244}

    CleanTrials.events[:,2] = S2_values
    CleanTrials.event_id = event_dict

    # drop epochs with jump      
    jname = str(homeDir)+'/AWM4_data/processed/ICAs/Jumps'+str(actSubj)+'.npy'
    if os.path.isfile(jname): 
        jump_inds = np.load(jname) 
        CleanTrials.drop(jump_inds,reason='jump')

    # from memorized, drop the values that corresponds the index from jump_inds
    S2_values = np.delete(S2_values, jump_inds)

    #create delay epochs 
    S2Epochs = CleanTrials.copy() 
    del CleanTrials
    S2Epochs.crop(tmin=1, tmax=2) 

    # getting only the magnotometers and resampling it to 10 Hz 
    S2Epochs_mag = S2Epochs.copy().pick_types(meg='mag')
    S2Epochs_mag = S2Epochs_mag.resample(10, npad='auto')  # resample to 10 Hz to speed up decoding
    
    # equalize the number of trials in each condition
    S2Epochs_mag.equalize_event_counts(S2Epochs_mag.event_id)
    resampled_data = S2Epochs_mag.get_data()
    S2_values = S2Epochs_mag.events[:,2]

    idx_lowpitch = [i for i in range(len(S2_values)) if S2_values[i] in [211, 212, 213, 214, 221, 222, 223, 224]]
    idx_highpitch = [i for i in range(len(S2_values)) if S2_values[i] in [231, 232, 233, 234, 241, 242, 243, 244]]


    # decoding starts 

    print(f'setup for decoding...')
    #create x and y  
    y = np.empty(len(S2_values), dtype=int)  
    #lowpitch as 0 and highpitch as 1 
    y[idx_lowpitch] = 0
    y[idx_highpitch] = 1
    X = resampled_data

    #classifier
    clf = make_pipeline(StandardScaler(), 
                    SVC(probability=True))

    #cross validation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1000, random_state=None) 
    scoring = 'accuracy'

    #decoding over every single timepoint
    print(f'decoding over every single timepoint...')
    time_decoder = SlidingEstimator(clf, scoring=scoring, n_jobs=20, verbose=False)
    time_scores = cross_val_multiscore(time_decoder, X, y, cv=cv, n_jobs=20, verbose=False) 

    print(f'saving results...')
    # Mean scores across cross-validation splits, for each time point.
    mean_timescores = np.mean(time_scores, axis=0)
    all_mean_timescores[actSubj_idx,:] = mean_timescores

    # Mean score across all time points.
    mean_acrossalltimes = round(np.mean(time_scores), 3)
    all_mean_acrossalltimes[actSubj_idx] = mean_acrossalltimes

    # round to 3 decimal places
    mean_acrossalltimes = np.round(mean_acrossalltimes, 3)
    all_mean_acrossalltimes = np.round(all_mean_acrossalltimes, 3)

    print(f'finished with subject {actSubj}...')

    np.savetxt(results_path + '/all_encodingS2(10)_mean_timescores_SVM.txt', all_mean_timescores)
    np.savetxt(results_path + '/all_encodingS2(10)_mean_acrossalltimes_SVM.txt', all_mean_acrossalltimes)

    # save as excel file with pandas
    df = pd.DataFrame(all_mean_timescores)
    df.to_excel(results_path + '/all_encodingS2(10)_mean_timescores_SVM.xlsx')

    df = pd.DataFrame(all_mean_acrossalltimes)
    df.to_excel(results_path + '/all_encodingS2(10)_mean_acrossalltimes_SVM.xlsx')

    # #plot the results from each timepoint with timepoints being on the x axis and accuracy on the y axis
    # plt.plot(np.linspace(1.0, 2.0, 10), mean_timescores)
    # #plt.axhline(y=0.5, color='r', linestyle='--')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Accuracy')
    # plt.title('Decoding accuracy over time')
    # plt.savefig(results_path + f'/sub{actSubj}_encodingS2(10)_mean_timescores_SVM.png')
    # plt.close()

# I want to average the results over all subjects and show the standard deviation
plt.plot(np.linspace(0.0, 1.0, 10), np.mean(all_mean_timescores, axis=0))
plt.xlabel('Time (s)')
plt.ylabel('Accuracy')
plt.title('Decoding accuracy over time')
plt.fill_between(np.linspace(0.0, 1.0, 10), np.mean(all_mean_timescores, axis=0) - np.std(all_mean_timescores, axis=0), np.mean(all_mean_timescores, axis=0) + np.std(all_mean_timescores, axis=0), alpha=0.2)
plt.axvline(x=0.5, color='r', linestyle='--')
plt.savefig(results_path + '/average_encodingS2.png')
plt.close()

#significance tests for each timepoint
all_mean_timescores = pd.read_excel(results_path + '/all_encodingS2(10)_mean_timescores_SVM.xlsx')

# Assuming mean_scores is the relevant DataFrame from earlier
mean_scores = all_mean_timescores.iloc[:, 1:]  # Skip the first column if it's an index

# Initialize lists to store p-values and significant points
p_values = []
significant_points = []

# Calculate p-values and determine significance
for timepoint in range(mean_scores.shape[1]):
    actual = mean_scores.iloc[:, timepoint] - 0.5  # Adjust for chance level (0.5)
    actual_mean = np.mean(actual)

    permuted = []
    for value in actual:
        permuted.append([np.abs(value), -np.abs(value)])  # Permutation distribution

    # Perform permutation test
    population = []
    for i in range(100000):  # Number of permutations
        permutation_sample = [np.random.choice(permuted[s], 1)[0] for s in range(len(permuted))]
        population.append(np.mean(permutation_sample))

    # Calculate p-value
    p = np.sum(population >= actual_mean) / 100000
    p_values.append(p)

    # Store significant points for annotation
    if 0.01 < p < 0.05:
        significant_points.append((timepoint, '*'))
    elif p <= 0.01:
        significant_points.append((timepoint, '**'))

# Save p-values to a CSV file
p_values_df = pd.DataFrame(p_values, columns=["p_values"])
p_values_df.to_csv(results_path + '/p_values.csv', index=False)

# Plot the average accuracy over time
timepoints = np.linspace(0.0, 1.0, mean_scores.shape[1])
mean_accuracy = np.mean(mean_scores, axis=0)
std_accuracy = np.std(mean_scores, axis=0)

plt.plot(timepoints, mean_accuracy, marker='o', color='b')  # Add dots (markers) on the line
plt.fill_between(timepoints, mean_accuracy - std_accuracy, mean_accuracy + std_accuracy, alpha=0.2)
plt.xlabel('Time (s)')
plt.ylabel('Accuracy')
plt.title('Decoding accuracy over time')
plt.axvline(x=0.5, color='r', linestyle='--')  # Timepoint of the cue

# Annotate significant timepoints with asterisks on top of the dots
for tp, sig in significant_points:
    plt.text(timepoints[tp], mean_accuracy[tp] + 0.01, sig, color='blue', fontsize=12, ha='center')

# Save the figure
plt.savefig(results_path + '/average_encodingS2_with_significance.png')
plt.close()


# Prepare data for violin plot using all_mean_timescores
data = pd.DataFrame(all_mean_timescores, columns=[f'Time_{i+1}' for i in range(10)])
data['Subject'] = Subs

# Melt the DataFrame to have a long-form format suitable for seaborn
data_long = pd.melt(data, id_vars=['Subject'], var_name='Time', value_name='Accuracy')

# Plot the violin plot
plt.figure(figsize=(12, 8))
sns.violinplot(x='Subject', y='Accuracy', data=data_long, inner='quartile')
plt.xlabel('Subjects')
plt.ylabel('Decoding Accuracy')
plt.title('Decoding Performance Across Subjects for S2')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(results_path + '/S2decoding_violinplot.png')
plt.show()

