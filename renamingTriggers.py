'''
Plan:
1. load the input_file
2. only select trials from current block (1st column)
3. load events

iterate through every trial (every row) in the input file
get the index of trials (2nd column)
find the index of the current trial in the events
event_idx = np.where(events[:,0] == input[idx_trial,0])

current_trigger = 200 + 10*input[idx_trial,5] + input[idx_trial,6]
current_trigger = int(f"2{input[idx_trial,5}{input[idx_trial,6}")

change it in the events file

'''
import numpy as np
import pandas as pd
# import imshow
import matplotlib.pyplot as plt
import mne

#input files
homeDir = '/media/headmodel/J/AWM4/'
inputfile = homeDir + '/InputMEG/'
metaFile = homeDir + 'MEGNotes.xlsx'
metaInfo = pd.read_excel(metaFile)
NrSessions = 1
Subs       = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'Subject']) # 1-26
Subs_Codes = np.unique(metaInfo.loc[metaInfo.FinalSample==1,'SubjectCode']) # S51-S76
Sub_Codes = [int(sub[1:]) for sub in Subs_Codes] # 51-76
Sub_Codes = [51,53,54,52,57,56,62]
#loading events 
data_path   = homeDir +'AWM4_data/raw/'
allFiles    = metaInfo['MEG_Name']

for subCode, actSubj in zip(Sub_Codes, Subs):
    if actSubj == 1:
        #getting the new triggers from input files
        fnameinput = inputfile + f'Input_AWM4_Exp1_MEG_VP{subCode}.txt'
        # read file with columns seperated by spaces
        # header in second row
        data = pd.read_csv(fnameinput, sep=' ', header=1)
        data = data.loc[data.Block != 0]
        new_s2l2 = [data.PitchS2, data.LocS2]
        new_s2l2T = np.transpose(new_s2l2)
        new_trigger = [200 + 10*new_s2l2T[:,0] + new_s2l2T[:,1]]
        new_trigger = np.transpose(new_trigger)

        actInd        = (metaInfo.Subject==actSubj) & (metaInfo.Valid==1)
        actFiles      = allFiles[actInd]
        for b, ff in zip(np.unique(data.Block), range(actFiles.count())):
            if b == 1 and ff == 0:
                fname = data_path + actFiles.iloc[ff]
                # fname_correct_triggers 
                fname_correct_triggers = data_path + actFiles.iloc[ff]
                fname_correct_triggers = fname_correct_triggers.split('.')
                fname_correct_triggers = fname_correct_triggers[0] + '_correct_triggers.fif'
                raw   = mne.io.read_raw_ctf(fname,'truncate',True) 
                #get events
                if actSubj == 1 and ff == 0:
                    events    = mne.find_events(raw, 'UPPT001',shortest_event=1)
                    events = events[8:,:]
                else:
                    events    = mne.find_events(raw, 'UPPT001', shortest_event=1)
                    events= events[events[:,1]==0,:]
                if np.any(events[:,1]<0):
                    raise ValueError('faulty trigger found, please inspect manually.')    # check fo corrupted triggers 

                # changing 134 to 234 in the events file
                onethreefour_idx = np.where(events[:,2] == [134])
                #if there are no 134 triggers, then skip the following code untill the trialNo
                if onethreefour_idx[0].size != 0:
                    onethreefour_idx = np.array(onethreefour_idx)
                    onethreefour_idxT = np.transpose(onethreefour_idx)
                    wrong_idx = [onethreefour_idxT[i] for i in range(len(onethreefour_idxT)) if i % 2 != 0]
                    wrong_idx = np.array(wrong_idx)
                    events[wrong_idx,2] = 234
                else:
                    continue
            
                # getting s2 values
                trialNo = np.arange(22,65)
                trialNo = list(trialNo)
                triggerlist = list(events[:,2])
                trial_idx = [triggerlist.index(i) for i in trialNo]
                s2_idx = [i + 3 for i in trial_idx]
                print(s2_idx)
                s2_values = events[s2_idx,2]
                print(s2_values)

                #divide the new_trigger into blocks
                new_trigger_block = np.split(new_trigger, 8)
                # with a slicing of [:, :, 0]
                new_trigger_block = np.array(new_trigger_block)[:, :, 0]
                new_trigger_blockT = np.transpose(new_trigger_block)
                
                #changing the triggers in the events file
                #for t in range(data.Trial.shape[0]):
                    # inputNo = np.arange(2,45)
                    # trial_idx = [triggerlist.index(i) for i in inputNo]
                    # s2_idx = [i + 3 for i in trial_idx]
                    # s2_values = events[s2_idx,2]
                    # every column in the new_trigger_block_b is a block. the first column is the first block
                    # the second column is the second block and so on
                    # please read the new_trigger_blockT to understand the order of the blocks and if b=0 take new_trigger_blockT[:,0]
                    # if b=1 take new_trigger_blockT[:,1] and so on
                new_trigger_block_b = new_trigger_blockT[:,b-1]
                #I want to delete the first 21 rows of new_trigger_block_b
                # from the new_trigger_block_b, I want to have from the second row to the 44th row
                new_trigger_block_b = new_trigger_block_b[1:44]
                events[s2_idx,2] = new_trigger_block_b
                print(events[s2_idx,2])

                #save
                raw.add_events(events, 'UPPT001', replace=True)
                raw.save(fname_correct_triggers)