#melek's trial and error script#

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
results_path = str(homeDir)+'/AWM4_data/processed/timepoints'
if not os.path.exists(results_path):
    os.makedirs(results_path)
all_mean_timescores = np.zeros((len(Subs), 250))
all_mean_acrossalltimes = np.zeros(len(Subs))

#actSubj = Subs[int(input("Subject no [1-26]: "))-1] 
actSubj = False
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
    actBlock = block[actInd]
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
    S1_idx = [i - 4 for i in range(len(all_events[:,2])) if all_events[i,2] == 101]
    S2_idx = [i - 2 for i in range(len(all_events[:,2])) if all_events[i,2] == 201]
    S1_values = all_events[S1_idx,2]
    S2_values = all_events[S2_idx,2]

    #which condition is in WM right now according to their index in the event array
    memorized = np.zeros(len(all_events[:,2]))
    memorized[S1_idx] = S1_values
    memorized[S2_idx] = S2_values
    memorized = memorized[memorized != 0]

    #which trials are lowpitch and which are highpitch
    idx_lowpitch = [i for i in range(len(memorized)) if memorized[i] in [111, 112, 113, 114,121, 122, 123, 124, 211, 212, 213, 214, 221, 222, 223, 224]]
    idx_highpitch = [i for i in range(len(memorized)) if memorized[i] in [131, 132, 133, 134, 141, 142, 143, 144, 231, 232, 233, 234, 241, 242, 243, 244]]

    # load cleaned and cut epochs per trial 
    CleanTrials = mne.read_epochs(str(homeDir)+'/AWM4_data/processed/CutEpochs/CutData_VP'+str(actSubj)+'-cleanedICA-epo.fif')
    
    event_dict  = {'S1/Sp1/L1': 111, 'S1/Sp1/L2': 112, 'S1/Sp1/L3': 113, 'S1/Sp1/L4': 114,
                'S1/Sp2/L1': 121, 'S1/Sp2/L2': 122, 'S1/Sp2/L3': 123, 'S1/Sp2/L4': 124,
                'S1/Sp3/L1': 131, 'S1/Sp3/L2': 132, 'S1/Sp3/L3': 133, 'S1/Sp3/L4': 134,
                'S1/Sp4/L1': 141, 'S1/Sp4/L2': 142, 'S1/Sp4/L3': 143, 'S1/Sp4/L4': 144,
                'S2/Sp1/L1': 211, 'S2/Sp1/L2': 212, 'S2/Sp1/L3': 213, 'S2/Sp1/L4': 214,
                'S2/Sp2/L1': 221, 'S2/Sp2/L2': 222, 'S2/Sp2/L3': 223, 'S2/Sp2/L4': 224,
                'S2/Sp3/L1': 231, 'S2/Sp3/L2': 232, 'S2/Sp3/L3': 233, 'S2/Sp3/L4': 234,
                'S2/Sp4/L1': 241, 'S2/Sp4/L2': 242, 'S2/Sp4/L3': 243, 'S2/Sp4/L4': 244}

    CleanTrials.events[:,2] = memorized
    CleanTrials.event_id = event_dict

    # drop epochs with jump      
    jname = str(homeDir)+'/AWM4_data/processed/ICAs/Jumps'+str(actSubj)+'.npy'
    if os.path.isfile(jname): 
        jump_inds = np.load(jname) 
        CleanTrials.drop(jump_inds,reason='jump')

    # from memorized, drop the values that corresponds the index from jump_inds
    memorized = np.delete(memorized, jump_inds)

    #create delay epochs 
    delayEpochs = CleanTrials.copy() 
    del CleanTrials, memorized
    delayEpochs.crop(tmin=2, tmax=4.5)

    # event_id  = {'S1/Sp1/L1': 111, 'S1/Sp1/L2': 112, 'S1/Sp1/L3': 113, 'S1/Sp1/L4': 114,
    #             'S1/Sp2/L1': 121, 'S1/Sp2/L2': 122, 'S1/Sp2/L3': 123, 'S1/Sp2/L4': 124,
    #             'S1/Sp3/L1': 131, 'S1/Sp3/L2': 132, 'S1/Sp3/L3': 133, 'S1/Sp3/L4': 134,
    #             'S1/Sp4/L1': 141, 'S1/Sp4/L2': 142, 'S1/Sp4/L3': 143, 'S1/Sp4/L4': 144,
    #             'S2/Sp1/L1': 211, 'S2/Sp1/L2': 212, 'S2/Sp1/L3': 213, 'S2/Sp1/L4': 214,
    #             'S2/Sp2/L1': 221, 'S2/Sp2/L2': 222, 'S2/Sp2/L3': 223, 'S2/Sp2/L4': 224,
    #             'S2/Sp3/L1': 231, 'S2/Sp3/L2': 232, 'S2/Sp3/L3': 233, 'S2/Sp3/L4': 234,
    #             'S2/Sp4/L1': 241, 'S2/Sp4/L2': 242, 'S2/Sp4/L3': 243, 'S2/Sp4/L4': 244,
    #             'Cue_S1': 101, 'Cue_S2': 201, 'Ping': 254, 
    #             'Delay/S1/Cue': 100, 'Delay/S1/Ping': 200, 'Delay/S2/Cue': 250, 'Delay/S2/Ping': 251,
    #             'Probe/SpNM/LoNM': 150, 'Probe/SpNM/LoM': 151, 'Probe/SpM/LoNM': 160, 'Probe/SpM/LoM': 161,
    #             'lowpitch': [111, 112, 113, 114,121, 122, 123, 124, 211, 212, 213, 214, 221, 222, 223, 224],
    #             'highpitch' : [131, 132, 133, 134, 141, 142, 143, 144, 231, 232, 233, 234, 241, 242, 243, 244]}
    #delay = {'Cue_S1': 101, 'Cue_S2': 201} #seems like I dont need this 

    # getting only the magnotometers and resampling it to 100 Hz 
    delayEpochs_mag = delayEpochs.copy().pick_types(meg='mag')
    delayEpochs_mag = delayEpochs_mag.resample(100, npad='auto')  # resample to 100 Hz to speed up decoding
    
    # equalize the number of trials in each condition
    delayEpochs_mag.equalize_event_counts(delayEpochs_mag.event_id)
    resampled_data = delayEpochs_mag.get_data()
    # get the events array
    memorized = delayEpochs_mag.events[:,2]

    idx_lowpitch = [i for i in range(len(memorized)) if memorized[i] in [111, 112, 113, 114,121, 122, 123, 124, 211, 212, 213, 214, 221, 222, 223, 224]]
    idx_highpitch = [i for i in range(len(memorized)) if memorized[i] in [131, 132, 133, 134, 141, 142, 143, 144, 231, 232, 233, 234, 241, 242, 243, 244]]

    # decoding starts 

    print(f'setup for decoding...')
    #create x and y  
    y = np.empty(len(memorized), dtype=int)  
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

    # scores = cross_val_score(clf, X=X, y=y, cv=cv, scoring=scoring, n_jobs=20)
    # # Mean and standard deviation of roc_auc across cross-validation runs.
    # roc_auc_mean = round(np.mean(scores), 3)
    # roc_auc_std = round(np.std(scores), 3)
    # AUC_meansd.append([roc_auc_mean, roc_auc_std])


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

    np.savetxt(results_path + '/all_mean_timescores_SVM.txt', all_mean_timescores)
    np.savetxt(results_path + '/all_mean_acrossalltimes_SVM.txt', all_mean_acrossalltimes)

    # save as excel file with pandas
    df = pd.DataFrame(all_mean_timescores)
    df.to_excel(results_path + '/all_mean_timescores_SVM.xlsx')

    df = pd.DataFrame(all_mean_acrossalltimes)
    df.to_excel(results_path + '/all_mean_acrossalltimes_SVM.xlsx')

    #plot the results from each timepoint with timepoints being on the x axis and accuracy on the y axis
    plt.plot(np.arange(0.0, 2.50, 0.01), mean_timescores)
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.title('Decoding accuracy over time')
    plt.savefig(results_path + f'/sub{actSubj}_mean_timescores_SVM.png')
    plt.close()



    # %%
