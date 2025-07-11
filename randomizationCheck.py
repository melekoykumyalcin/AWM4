import numpy as np
import pandas as pd
# import imshow
import matplotlib.pyplot as plt
import mne

homeDir = '/media/headmodel/J/AWM4/'
inputfile = homeDir + '/InputMEG/'
metaFile = homeDir + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(metaFile)
NrSessions = 1
Subs       = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject']) # 1-26
Subs_Codes = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'SubjectCode']) # S51-S76
# drop all the 'S' in the subject codes
Sub_Codes = [int(sub[1:]) for sub in Subs_Codes] # 51-76

#%% Load and cut data per subject and session ################################

# paths and file names
data_path   = homeDir +'/AWM4_data/raw/'
allFiles    = metaInfo['MEG_Name']
corrected_files = [f.split('.')[0] + '_correct_triggers.fif' for f in allFiles]
# how can I change the type of corrected_files from a list to a pandas series?
corrected_files_series = pd.Series(corrected_files)
#Subs       = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject']) # 1-26
Sub_Codes = [51,53,54,52,57,56,62]
S1_conditions_MEG_by_block = np.zeros((np.max(Sub_Codes)+1, 4,4,8))
S1_conditions_MEG_overall = np.zeros((np.max(Sub_Codes)+1, 4,4))

# Loop over subjects:
for actSubj in Subs:
    if actSubj in Subs[1:7]:
        actInd        = (metaInfo.Subject==actSubj) & (metaInfo.Valid==1)
        actFiles      = corrected_files_series[actInd]
        for ff in range(actFiles.count()):
            fname = data_path + actFiles.iloc[ff]
            #raw   = mne.io.read_raw_ctf(fname,'truncate',True) 
            raw   = mne.io.read_raw_fif(fname, preload=True) 
            #get events
            if actSubj == 1 and ff == 0:
                events    = mne.find_events(raw, 'UPPT001',shortest_event=1)
                events = events[8:,:]
            else:
                events    = mne.find_events(raw, 'UPPT001', shortest_event=1)
                events= events[events[:,1]==0,:]
            if np.any(events[:,1]<0):
                raise ValueError('faulty trigger found, please inspect manually.')    # check fo corrupted triggers 
            S1_triggers = [111, 112, 113, 114, 121, 122, 123, 124, 131, 132, 133, 134, 141, 142, 143, 144]
                # within events, read the third column only for S1_triggers
            S1_events = events[np.isin(events[:,2], S1_triggers), :]
            # only keep the third column of S1_events
            S1_events = S1_events[:,2]
            S1_events = np.array([np.array([int(str(ev)[1]), int(str(ev)[2])]) for ev in S1_events])
            #make S1_events sorted out in a way as follows [[p,l] for p in np.unique(data.PitchS1) for l in np.unique(data.LocS1)] 
            S1_events = np.array(sorted(S1_events, key=lambda x: (x[0], x[1])))
            #save first array as tmp_PitchS1_MEG and second array as tmp_LocS1_MEG
            for comb in S1_events:
        
                tmp_PitchS1_MEG = comb[0]
                tmp_LocS1_MEG = comb[1]

                comb_trials = (S1_events[:,0] == tmp_PitchS1_MEG) & (S1_events[:,1] == tmp_LocS1_MEG)
                nTrials = np.sum(comb_trials)

                S1_conditions_MEG_by_block[actSubj, tmp_PitchS1_MEG-1, tmp_LocS1_MEG-1, ff] = nTrials

        # sum over all blocks and write to S1_conditions_MEG_overall
        S1_conditions_MEG_overall[actSubj,:,:] = np.sum(S1_conditions_MEG_by_block[actSubj,:,:,:], axis=2)


S1_conditions_input = np.zeros((np.max(Sub_Codes)+1, 4,4))

for sub_code in Sub_Codes:
    if sub_code == [53, 54, 52, 57, 56, 62]:
        fnameinput = inputfile + f'Input_AWM4_Exp1_MEG_VP{sub_code}.txt'
        # read file with columns seperated by spaces
        # header in second row
        data = pd.read_csv(fnameinput, sep=' ', header=1)

        # drop all rows with Block == 0
        data = data.loc[data.Block != 0]

        # drop all rows with TargetPos == 2
        data = data.loc[data.TargetPos != 2]

        # get all combinations of PitchS1 and LocS1
        S1_combs = [[p,l] for p in np.unique(data.PitchS1) for l in np.unique(data.LocS1)]

        # count the number of trials for each combination of PitchS1 and LocS1 in data
        for comb in S1_combs:
            tmp_PitchS1 = comb[0]
            tmp_LocS1 = comb[1]

            comb_trials = ((data.PitchS1 == tmp_PitchS1) & (data.LocS1 == tmp_LocS1))

            nTrials = np.sum(comb_trials)
            S1_conditions_input[sub_code, tmp_PitchS1-1, tmp_LocS1-1] = nTrials



# please plot imshow for S1_conditions_input and S1_conditions_MEG_overall
# for each subject in a row

fig,axes = plt.subplots(len(Sub_Codes),2, figsize=(10,50))

for subCode, actSubj in zip(Sub_Codes, Subs):
    im = axes[actSubj-1,0].imshow(S1_conditions_input[subCode,:,:], vmin = 20, vmax = 50)

    # Loop over data dimensions and create text annotations.
    for i in range(4):
        for j in range(4):
            text = axes[actSubj-1,0].text(j, i, S1_conditions_input[subCode,i,j],
                        ha="center", va="center", color="w")
            
    axes[actSubj-1,1].imshow(S1_conditions_MEG_overall[actSubj,:,:], vmin = 20, vmax = 50)
    # Loop over data dimensions and create text annotations.
    for i in range(4):
        for j in range(4):
            text = axes[actSubj-1,1].text(j, i, S1_conditions_MEG_overall[actSubj,i,j],
                        ha="center", va="center", color="w")

fig = plt.figure()
plt.title(f'VP{sub_code}')
plt.imshow(S1_conditions_input[sub_code,:,:], vmin = 20, vmax = 50)
plt.colorbar()
plt.show()

