%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Auditory Working Memory 4 Feature Binding - Voice Stimuli              %
%  remember one stimulus with 2 features (speaker&location)               %
%  only speaker task-relevant                                             %
%  match/nonmatch task with probe that has same or diff location          %
%  MEG version including trigger and optional eye tracking                %
%  Cora Fischer, 01.03.2024                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set Paths & request participants information

clearvars
cd(fullfile('C:\STIMCODE_TO_RUN\STIM_Cora_Fischer\AWM4\Experiment'))
datapath  = fullfile(cd,'/Output');
soundpath = fullfile(cd,'/Sounds');

prompt      = {'VP-Nr: ', 'StartBlock: ', 'Uebung?:' };
NrFiles     = numel(dir(sprintf('%s/Output_AWM4_Exp1_MEG_VP*.txt',datapath)));
defAns      = {sprintf('%d',NrFiles+51),'1','1'};
CodeEingabe = inputdlg(prompt,'VP-Code',1,defAns);
SubjNr      = str2double(CodeEingabe(1));
StartBlock  = str2double(CodeEingabe(2));
Practice    = str2double(CodeEingabe(3))==1;
vpCode      = sprintf('AWM4_Exp1%d',SubjNr);

SC          = 1;%menu('External soundcard?','yes','no');
Sylls       = 1;%menu('Single syllable?','yes','no');
KbName('UnifyKeyNames')
pauseKey = KbName('p');
stopKey  = KbName('s');

%% Measurement with or without eye tracker? %%%%%%%%%%%%%
% dummy or real eye tracker? - dummy for programming purposes and difficult
% ET subjects

ET    = menu('Eye Tracking?','yes','no');
dummy = ET-1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Psychtoolbox Stuff
InitializePsychSound;
screens = Screen('Screens',2);
if numel(screens) > 1
    screenindex = 1;
else 
    screenindex = 0;
end
rect = Screen('Rect', screenindex, 0);
% rect = [0 0 1680 1050];
%    ShowCursor
%     Screen('CloseAll');
[w, ~] = Screen('OpenWindow', screenindex, 1);
fps = 120;

%% Load Sounds

% syll 11
LowSpeaker1  = 'M07uh';
HighSpeaker1 = 'W29uh';
% syll 2
LowSpeaker2  = 'M07oo';
HighSpeaker2 = 'W29oo';
% syll 3
LowSpeaker3  = 'M07uw';
HighSpeaker3 = 'W29uw';
[xM,fs] = audioread(fullfile(soundpath,sprintf('Morph_%s_%s_Step%i.wav',LowSpeaker1,HighSpeaker1,1))); % fs: sampling rate of sound input in Hz
duration    = numel(xM)/fs;                                % length of the sound in ms                                 
nrchannels  = 2;                                           % input sound channels, typically 2 (left and right ear)
t           = 1:numel(xM);                                 % sampling points of sound
npts        = length(t);                                   % number of sampling points
steps       = 101;   

% load in sounds
% sounds = nan(3,steps,length(xM));
sounds = cell(3,steps);
for sn = 1:steps
   [sounds{1,sn},~] = audioread(fullfile(soundpath,sprintf('Morph_%s_%s_Step%i.wav',LowSpeaker1,HighSpeaker1,sn)));
   [sounds{2,sn},~] = audioread(fullfile(soundpath,sprintf('Morph_%s_%s_Step%i.wav',LowSpeaker2,HighSpeaker2,sn)));
   [sounds{3,sn},~] = audioread(fullfile(soundpath,sprintf('Morph_%s_%s_Step%i.wav',LowSpeaker3,HighSpeaker3,sn)));
end

% determine target sounds
Anchors       = linspace(1,steps,6);
TargetSpeaker = Anchors(2:end-1);
% NMSpeaker     = [TargetSpeaker-10;TargetSpeaker-5;TargetSpeaker+5;TargetSpeaker+10]'; -subs 20-23
NMSpeaker     = [TargetSpeaker-12;TargetSpeaker-9;TargetSpeaker+9;TargetSpeaker+12]'; 

% create ping
noise1 = (.95*(mean([sounds{1,[21,41,61,81]}],2))/max(abs(mean([sounds{1,[21,41,61,81]}],2))))';
noise2 = (.95*(mean([sounds{2,[21,41,61,81]}],2))/max(abs(mean([sounds{1,[21,41,61,81]}],2))))';
noise3 = (.95*(mean([sounds{3,[21,41,61,81]}],2))/max(abs(mean([sounds{1,[21,41,61,81]}],2))))';

desDuration  = .1; % in seconds
p1           = noise1(round(size(noise1,2)*.4):round(size(noise1,2)*.4)+fs*desDuration);
%p2           = noise2(round(size(noise2,2)*.4):round(size(noise2,2)*.4)+fs*desDuration); 
%p3           = noise3(round(size(noise3,2)*.4):round(size(noise3,2)*.4)+fs*desDuration); 
ping         = .95*(p1)/max(abs(p1));
durationPing = numel(ping)/fs;                                % length of the sound in ms                                 


% task-irrelevant feature (location)
NrLocs           = 4;
Locs             = [-39,-13,13,39];
LocDist          = [-45,-35,35,45]; % changed in comp. to Pilot 3

c                = 343; % m/s
U                = .57; % head circumference in cm
r                = U/(2*pi); 

%% rise/fall - for ping

rftime                 = 0.05;
env                    = ones(1,size(p1,2));
risepts                = round(rftime*fs);
envxstepsize           = pi/(risepts-1);
envx                   = -pi:envxstepsize:0;
env(1:risepts)         =((cos(envx))+1)/2;
env(end-risepts+1:end) = fliplr(((cos(envx))+1)/2);

% test ping sound:
% y=[ping.*env;ping.*env];
% PsychPortAudio('FillBuffer', soundhandle, y);
% PsychPortAudio('Start', soundhandle, 1, 0, 1);

%% Stimulation Parameters

featureCols  = [0 240 190; 240 140 0];
fixCols      = [155 155 155; 255 255 255];
fixWidth     = 2;
fixLength    = 20;
feedbackCols = [0 255 0; 255 0 0];
cueLength    = fixLength;
screenCenter = [rect(3)/2 rect(4)/2];
% Cues: 
Cue = [];

% run('attenuation_test');
 

%% Experiment Parameters

NrFeat       = 1; % nr of different features
Features     = {'Speaker';'Position'};
RefStim      = TargetSpeaker;
NrRefStim    = numel(RefStim); % nr of ref stimuli per feature
NrWMStim     = 2;


%% Timing parameter

INTITIALWAIT     = 2;
T_ITI            = 1.5;
T_Stim           = .5;%duration; - rounded up to fit all the stims ad to mirror the prev exps
T_ISI            = T_Stim;
T_PrePingDelay   = 1;%T_Stim;
T_Ping           = durationPing;
T_PostPingDelay  = 1-T_Ping;%T_Stim;
T_Cue            = .5;
T_Feedback       = .5;
T_Resp           = 1.5; % max time for response


%% Load Input & prepare output

inData = importdata(fullfile(cd,'Input',sprintf('Input_AWM4_Exp1_MEG_VP%d.txt',SubjNr)));
if StartBlock == 1
    OutputHeader = {'VP', num2str(SubjNr), ' Output_AWM4_Exp1_MEG ', datestr(now)};
    dlmwrite(sprintf('%s//Output_AWM4_Exp1_MEG_VP%d.txt',datapath,SubjNr),OutputHeader,'delimiter','','newline','pc','-append');
end
ExpMat = inData.data;

NrTrials              = length(ExpMat);
NrTrialsExp           = length(ExpMat(ExpMat(:,1)>0,:));
NrTrialsPract         = length(ExpMat(ExpMat(:,1)==0,:));
NrBlocks              = max(ExpMat(:,1));
NrTrialsPerBlock      = NrTrialsExp/NrBlocks;


%% initialize soundhandle

% for some reason this way of finding the external fireface soundcard does
% not work anymore (03.01.2024)
% in MEG MPI PC, the "old way" works
devi = PsychPortAudio('GetDevices',3);
% devi = PsychPortAudio('GetDevices');
for ii=1:length(devi)
    devi(ii).HostAudioAPIName;
    fprintf('%d %s %s',[devi(ii).DeviceIndex ' ' devi(ii).HostAudioAPIName ' ' devi(ii).DeviceName]);
    fprintf('\n');
end

if SC == 1
    mydevice = devi(strcmp({devi(:).DeviceName},'ASIO Fireface USB')).DeviceIndex; % Get usb soundcard...
    %mydevice = 30;%devi(strcmp({devi(:).DeviceName},'ASIO Fireface USB')).DeviceIndex; % Get usb soundcard...
else
    mydevice = [];
end

% soundhandle    = PsychPortAudio('Open', mydevice, [], 3, fs, nrchannels);
% y              = zeros(2,npts);
% %y=[xM';xM']; %to test audio output
% PsychPortAudio('FillBuffer', soundhandle, y);
% PsychPortAudio('Start', soundhandle, 1, 0, 1);


%% Setup Triggers
% identify output port
ioObj = io64;
status = io64(ioObj);
address = hex2dec('FFF8'); % look up MEG output port

% trigger library
trigger                  = [];
trigger.block.start      = 98;
trigger.block.no         = 71:78;                                 % Blocks 1-8 
trigger.block.practice   = 70;                                    % practice block 
trigger.block.end        = 99;
trigger.trial.no         = 1:64;                                  % Trials 1-64
trigger.stim.S1          = [111 112 113 114; 121 122 123 124; 131 132 133 134; 141 142 143 144]; %S1,SpeakerIdentity,Location
trigger.stim.S2          = [211 212 213 214; 221 222 223 224; 231 232 233 234; 241 242 243 244]; %S2,SpeakerIdentity,Location
trigger.cue.start        = [101;201];                             % S1/S2 as target
trigger.ping.start       = 254;                                   % ping sound
trigger.delay.start      = [100;200;250;251];                     % after S1; after S2, after Cue, after Ping
trigger.probe.start      = [150 151; 160 161];                    % second number: 5-Speaker Nonmatch, 6-Speaker Match; third number: 0-Location Nonmatch, 1-Location Match
trigger.probe.value      = [154:159;164:169;174:179;184:189];     % second number: location ID (1-4 == 5-8);  Third number: speaker probe distance (0-4 == 4-8)
trigger.response.start   = 199;
trigger.response.value   = 190:192;                               % 190 == Nonmatch, 191 ==Match, 192 == too slow
trigger.response.correct = 195:196;                               % 195 == incorrect, 196 == correct


%% Eyetracking setup

if dummy == true
    Eyelink('InitializeDummy') 
    mouseInsteadOfGaze = 1; % control gaze cursor using mouse instead
    
else
    Eyelink('Initialize')
    mouseInsteadOfGaze = 0;
end
KbName('UnifyKeyNames')     % enables cross-platform key ids. el=EyelinkInitDefaults() sets el.keysCahed to 0 on windows
                            % this causes EyelinkDoTrackerSetup() and
                            % EyelinkDoDriftCorrect() to fail, because they
                            % call EyelinkGetKey(), which depends on cached
                            % key id's. 

% Eyelink Initialization & Tracker Setup
% Provide Eyelink with details about the graphics environment
% and perform some initializations. The information is returned in a
% structure that also contains useful defaults and control codes (e.g.
% tracker state bit and Eyelink key values).
el = EyelinkInitDefaults(w);

% make sure that we get gaze data from Eyelink
ETStatus = Eyelink('command','link_sample_data = LEFT,RIGHT,GAZE,AREA,GAZERES,HREF,PUPIL,STATUS,INPUT');
if ETStatus ~= 0
    error('link_sample_data error, status: %d',status);
end
    
% adjust eyetracker default options & update them
el.backgroundcolour         = [0 0 0]; % same as experiment background color
el.foregroundcolour         = [0 0 0]; % same as experiment background color
el.msgfontcolour            = fixCols(2,:); 
el.imgtitlecolour           = fixCols(2,:);
el.calibrationtargetcolour  = fixCols(2,:);
el.targetbeep               = 0; % turn off beep sounds during calibration
el.feedbackbeep             = 0; % turn off beep sounds after calibration

% Reduce FOV for eyetracker (to avoid edges, which might be hard to
% calibrate) & set up other eyetracking variables
Eyelink('command','calibration_area_proportion = .75 .75');
Eyelink('command','validation_area_proportion = .75 .75');
Eyelink('command','calibration_type = HV5'); % 5-point calibration - simpler routine bcs it's an auditory study
Eyelink('command','sample_rate = 1000');
Eyelink('command','enable_automatic_calibration = NO'); % manual calibration
% Eyelink('command','file_event_filter = LEFT, RIGHT, FIXATION, SACCADE, BLINK, MESSAGE, BUTTON');
% Eyelink('command','file_sample_data = LEFT, RIGHT, GAZE, AREA, STATUS');
% Eyelink('command','link_event_filter = LEFT, RIGHT, FIXATION, SACCADE, BLINK, BUTTON');
% Eyelink('command','link_sample_data = LEFT, RIGHT, GAZE, AREA');
% Eyelink('command','screen_pixel_coords = 0,0,%d,%d',rect(3),rect(4));
Eyelink('command','pupil_size_diameter = YES');
[v,vs] = Eyelink('GetTrackerVersion');
EyelinkUpdateDefaults(el);

%% Experiment's Loop
% Start Experiment:
% Screen('Preference', 'DefaultFontName', 'Helvetica', 'SkipSyncTests', 1);
Screen('BlendFunction', w, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
HideCursor(w);
Screen('TextFont',w,'Arial Unicode MS');
Screen('TextStyle',w,0);
Screen('TextSize',w,21);
StopScript = false;

% initialize response
KbQueueCreate
KbQueueStart

% initialize eye tracker
EyelinkInit(dummy,1); % this line needs to be run once in order to present the calibration targets

if Practice
    % Welcome Screen:
%     WelcomeText = sprintf(['Nachfolgend werden Ihnen in jedem Durchgang zwei Toene praesentiert.\n\n'...
%         'Ihre Aufgabe ist es, sich die TONHOEHE der Toene zu merken. Nach einer kurzen Pause zeigt\n\n'...
%         'Ihnen eine Zahl an, ob Sie sich den ersten ("1") oder den zweiten ("2") Ton merken sollen.\n\n'... 
%         'Diesen sollen Sie nach einer weiteren kurzen Pause mit einem Vergleichston vergleichen. \n\n'...
%         'Hat der Vergleichston die GLEICHE Tonhoehe wie der Merkton, druecken Sie bitte die \n\n'...
%         'LINKE Maustaste. Hat der Vergleichston eine ANDERE Tonhoehe, druecken Sie bitte die \n\n'...
%         'RECHTE Maustaste. Antworten Sie bitte so SCHNELL und PRAEZISE wie moeglich. Nachdem\n\n'...
%         'Sie Ihre Antwort abgegeben haben, erhalten Sie ein Feedback. Erscheint ein HAKEN, \n\n'...
%         'haben Sie richtig geantwortet. \n\n'...
%         'Erscheint ein KREUZ, war Ihre Antwort leider falsch.\n\n'...
%         'Wenn Sie keine Fragen mehr haben, folgt nun ein Uebungsblock. Starten Sie die\n\n'...
%         'Uebungsdurchgaenge mit beliebiger Maustaste']);
   WelcomeText = sprintf(['Nachfolgend werden Ihnen in jedem Durchgang zwei gesprochene Merkreize praesentiert.\n\n'...
        'Ihre Aufgabe ist es sich die Stimmen beider Reize zu merken. Nach einer kurzen Pause zeigt\n\n'...
        'Ihnen eine Zahl an, ob Sie sich den ersten ("1") oder den zweiten ("2") Merkreiz merken sollen.\n\n'... 
        'Diesen sollen Sie nach einer weiteren kurzen Pause mit einem Vergleichsreiz vergleichen. \n\n'...
        'Wird der Vergleichsreiz von der GLEICHEN Stimme gesprochen wie der Merkreiz, druecken Sie bitte die \n\n'...
        'LINKE Taste. Wird der Vergleichsreiz von einer ANDEREN Stimme gesprochen, druecken Sie bitte die \n\n'...
        'RECHTE Taste. Antworten Sie bitte so SCHNELL und PRAEZISE wie moeglich. Nachdem\n\n'...
        'Sie Ihre Antwort abgegeben haben, erhalten Sie ein Feedback. Erscheint ein HAKEN, \n\n'...
        'haben Sie richtig geantwortet. \n\n'...
        'Erscheint ein KREUZ, war Ihre Antwort leider falsch.\n\n'...
        'Wenn Sie keine Fragen mehr haben, startet nun das Experiment.\n\n'...
        ' \n\n'...
        'In the following experiment, you will be presented with two spoken stimuli in each trial.\n\n'...
        'Your task is to remember the two voices of the stimuli. After a short break, a number will show\n\n'...
        'you whether you should remember the first ("1") or the second ("2") stimulus.\n\n'... 
        'You will have to compare this stimulus to a target stimulus after another short break. \n\n'...
        'If the target stimulus is spoken by the SAME voice as the memory stimulus, please press the\n\n'...
        'LEFT button. If the target stimulus is spoken by a DIFFERENT voice than the memory stimulus, please press\n\n'...
        'the RIGHT button. Please answer as FAST and ACCURATE as possible. After you have given your\n\n'...
        'answer, you will receive feedback. If a TICK appears, your answer was correct. \n\n'...
        'If a CROSS appears, your answer was incorrect.\n\n'...
        'If you have no further questions,the experiment will start now.\n\n'...
        'trials with any button.\n\n']);
    DrawFormattedText(w, WelcomeText, 'center', 'center', [255 255 255]);
    Screen('Flip', w);
    %[~, ~, PauseButtons] = GetMouse();
    [pressed, keyPress] = KbQueueCheck;
    button = KbName(keyPress);
    while ~any(strcmp(button,'9('))
        [pressed, keyPress] = KbQueueCheck;
        button = KbName(keyPress);
    end
    %     while any(PauseButtons) % Warte bis Maustaste wieder losgelassen wurde
%         [~, ~, PauseButtons] = GetMouse();
%     end
end
trialEnd = GetSecs;


% Trialloop:
responseCheck = false;
ExpStart      = GetSecs;
if Practice
    blocksToDo    = [0 StartBlock:NrBlocks];
else
    blocksToDo    = StartBlock:NrBlocks;
end

for bb = blocksToDo
    
    if ~dummy
        ETText = sprintf(['Pause.\n\nEye Tracker Kalibirierung startet gleich...\n\n'...
                          'Break.\n\nEye tracker calibration starts soon...']);
        DrawFormattedText(w, ETText, 'center', 'center', [255 255 255]);
        Screen('Flip', w);
        WaitSecs(2);
    end
        
    % Eyetracking output
    edfFile = sprintf('S%d_B%d.edf',SubjNr,bb); % name of remote file to create - name has to be very short due to eyelink restrictions
    ETFileStatus = Eyelink('OpenFile',edfFile);
    if ETFileStatus ~=0
        error('openfile error, status: %d',status)
    end
    
    % Write general information in edf file
    Eyelink('Message', 'Eyetracker: %s',vs);
    Eyelink('Message', 'BEGIN OF DESCRIPTIONS');
    Eyelink('Message', 'SUBJECT ID: %d',SubjNr);
    Eyelink('Message', 'DISPLAY SIZE: %d %d',rect(3),rect(4));
    Eyelink('Message', 'FRAMERATE: %d',fps);
    Eyelink('Message', 'END OF DESCRIPTIONS');
    
    % Calibrate eye tracker before each block
    if ~dummy % only execute the next lines if we're not in dummy mode
        % Calibrate the eye tracker using the standard calibration routine
        EyelinkDoTrackerSetup(el,KbName('c')); 
%         Eyelink('StartSetup')
        % do a final check of calibration using driftcorrection
        EyelinkDoDriftCorrect(el)
    end
    
    % record eye tracking data
    Eyelink('StartRecording')
    Eyelink('CheckRecording')
    WaitSecs(0.1); % wait a moment to ensure eyelink has started recording
    [~, ~] = KbQueueCheck;
    % break before practice block
    if bb == 0 % practice block
        PauseText = sprintf(['Der Uebungsblock startet gleich\n\n'...
                            'The practice block starts soon']);
    % Block Pause screen:
    else
        PauseText = sprintf(['Pause.\n\nDer %d. Block startet gleich\n\n'...
                             'Break.\n\nThe %d. block starts soon.'],bb,bb);
    end
    DrawFormattedText(w, PauseText, 'center', 'center', [255 255 255]);
    Screen('Flip', w);
    [pressed, keyPress] = KbQueueCheck;
    button = KbName(keyPress);
    while  ~any(strcmp(button,'9('))
        [pressed, keyPress] = KbQueueCheck;
        button = KbName(keyPress);
    end
%     [~, ~, PauseButtons]=GetMouse();
%     while ~any(PauseButtons) 
%         [~, ~, PauseButtons]=GetMouse();
%     end
%     while any(PauseButtons) % Warte bis Maustaste wieder losgelassen wurde
%         [~, ~, PauseButtons]=GetMouse();
%     end  
    Screen('FillRect',w,[0 0 0]);
    Screen('Flip',w,0,1);
%     WaitSecs(1);
    
    % Start the experimental block
    Screen('FillRect',w,[0 0 0]);
    Screen('Flip',w,0,1);
    % Trigger & ET Message
    io64(ioObj,address,trigger.block.start)
    io64(ioObj,address,0)
    Eyelink('Message', int2str(trigger.block.start))
    WaitSecs(.05); % to avoid trigger intereference
    if bb > 0 % real blocks block
        io64(ioObj,address,trigger.block.no(bb))
        io64(ioObj,address,0)   
        Eyelink('Message', int2str(trigger.block.no(bb)))
    else
        io64(ioObj,address,trigger.block.practice)
        io64(ioObj,address,0)
        Eyelink('Message', int2str(trigger.block.practice))
    end
    
    % present fix circle and wait before the start of the first trial
    Screen('FrameOval',w,fixCols(1,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
    Screen('Flip',w,0,1);
    WaitSecs(INTITIALWAIT); 
    
    % trialloop
    for tr = (find(ExpMat(:,1)==bb))'
        
        % Record Time:
        trialStart = GetSecs;
        
        % Trigger & ET Message
        if bb>0
            trNo = ExpMat(tr,2);
        else
            trNo = ExpMat(tr,3);
        end
        io64(ioObj,address,trigger.trial.no(trNo))
        io64(ioObj,address,0)
        Eyelink('Message', int2str(trigger.trial.no(trNo)))
        
        % Input Matrix columns:
        % 1:'Block' 2:'Trial' 3:'BlockTrial' 4:'SpeakerS1' 5:'LocS1' 6: 'SpeakerS2'
        % 7:'LocS2' 8:'TargetPos' 9:'SpeakerMatch' 10:'ProbeDistance'
        % 11:'LocMatch' 12:'ProbeLoc'

        % Precalculations:
        if Sylls == 1
            actSyll = Sylls;
        else
            actSyll = randi(3,1);
        end
        soundS1   = sounds{actSyll,TargetSpeaker(ExpMat(tr,4))}';
        soundS2   = sounds{actSyll,TargetSpeaker(ExpMat(tr,6))}';
        actTarget = ExpMat(tr,8);
        
        % open soundhandle:
        soundhandle    = PsychPortAudio('Open', mydevice, [], 3, fs, nrchannels);
        
        % probe pitch match or non match? (relevant feature)
        ProbeMatch = ExpMat(tr,9);
        if ~ProbeMatch
            % implement stim distances
            if actTarget ==1
                NMID = NMSpeaker(ExpMat(tr,4),ExpMat(tr,10));
            else
                NMID = NMSpeaker(ExpMat(tr,6),ExpMat(tr,10));
            end
            soundProbe   = sounds{actSyll,NMID}';
        else
            if actTarget ==1
                soundProbe = soundS1;
            else
                soundProbe = soundS2;
            end
        end
        
        LocMatch = ExpMat(tr,11);
        LocPos   = Locs(ExpMat(tr,12));
  
        % Stimulus creation S1:           
        S1_angle = Locs(ExpMat(tr,5))*((2*pi)/360);
        ITD = (r/c)*(S1_angle+sin(S1_angle));
        if sign(ITD)==1
            ITD_l = zeros(1,abs(round(ITD*fs)));
            ITD_r = [];
        else
            ITD_r = zeros(1,abs(round(ITD*fs)));
            ITD_l = [];
        end

        % create sound:
        base_sound  = ((soundS1/max(abs(soundS1))));  % normalize to 1
        left_sound  = [ITD_l base_sound ITD_r];
        right_sound = [ITD_r base_sound ITD_l];
        y = [left_sound; right_sound];

        PsychPortAudio('FillBuffer', soundhandle, y);
        WaitSecs('UntilTime', trialStart+T_ITI);

        % Stimulus Presentation:
        Screen('FillRect',w,[0 0 0]);
        Screen('FrameOval',w,fixCols(2,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
        Screen('Flip',w,0,1);

        % Play:
%         PsychPortAudio('FillBuffer', soundhandle, y);
        S1Play  = PsychPortAudio('Start', soundhandle, 1, 0, 1);
        S1Start = GetSecs;
        % Trigger & ET Message
        io64(ioObj,address,trigger.stim.S1(ExpMat(tr,4),ExpMat(tr,5))) % first index: speaker (1-4), second index: location (1-4)
        io64(ioObj,address,0)
        Eyelink('Message', int2str(trigger.stim.S1(ExpMat(tr,4),ExpMat(tr,5))))

        % ISI
        Screen('FillRect',w,[0 0 0]);
        Screen('FrameOval',w,fixCols(2,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
        WaitSecs('UntilTime', S1Start+T_Stim);
        ISIStart = Screen('Flip',w,0,1);
        % Trigger & ET Message
        io64(ioObj,address,trigger.delay.start(1))
        io64(ioObj,address,0)
        Eyelink('Message', int2str(trigger.delay.start(1)))
        
        % Stimulus creation S2:           
        S2_angle = Locs(ExpMat(tr,7))*((2*pi)/360);
        ITD = (r/c)*(S2_angle+sin(S2_angle));
        if sign(ITD)==1
            ITD_l = zeros(1,abs(round(ITD*fs)));
            ITD_r = [];
        else
            ITD_r = zeros(1,abs(round(ITD*fs)));
            ITD_l = [];
        end

        % create sound:
        base_sound  = ((soundS2/max(abs(soundS2))));  % normalize to 1
        left_sound  = [ITD_l base_sound ITD_r];
        right_sound = [ITD_r base_sound ITD_l];
        y = [left_sound; right_sound];
     
        PsychPortAudio('FillBuffer', soundhandle, y);
        WaitSecs('UntilTime', ISIStart+T_ISI);

        % Stimulus Presentation:
        Screen('FillRect',w,[0 0 0]);
        Screen('FrameOval',w,fixCols(2,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
        Screen('Flip',w,0,1);

        % Play:
%         PsychPortAudio('FillBuffer', soundhandle, y);
        S2Play  = PsychPortAudio('Start', soundhandle, 1, 0, 1);
        S2Start = GetSecs;
        % Trigger & ET Message
        io64(ioObj,address,trigger.stim.S2(ExpMat(tr,6),ExpMat(tr,7))) % first index: speaker (1-4), second index: location (1-4)
        io64(ioObj,address,0)
        Eyelink('Message', int2str(trigger.stim.S2(ExpMat(tr,6),ExpMat(tr,7))))
        
        % PreCueDelay
        Screen('FillRect',w,[0 0 0]);
        Screen('FrameOval',w,fixCols(2,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
        WaitSecs('UntilTime', S2Start+T_Stim);
        PreCueDelayStart = Screen('Flip',w,0,1);
        % Trigger & ET Message
        io64(ioObj,address,trigger.delay.start(2))
        io64(ioObj,address,0)
        Eyelink('Message', int2str(trigger.delay.start(2)))
        
        % Present cue
        Screen('FillRect',w,[0 0 0]);
        cueText = num2str(actTarget);
        DrawFormattedText(w, cueText, 'center', 'center', [255 255 255]);
        WaitSecs('UntilTime', PreCueDelayStart+T_ISI);
        CueStart = Screen('Flip',w,0,1);     
        % Trigger & ET Message
        io64(ioObj,address,trigger.cue.start(actTarget))
        io64(ioObj,address,0)
        Eyelink('Message', int2str(trigger.cue.start(actTarget)))
        
        % Postcue/PrePing Delay
        Screen('FillRect',w,[0 0 0]);
        Screen('FrameOval',w,fixCols(2,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
        WaitSecs('UntilTime', CueStart+T_Cue);
        PrePingDelayStart = Screen('Flip',w,0,1);
        % Trigger & ET Message
        io64(ioObj,address,trigger.delay.start(3))
        io64(ioObj,address,0)
        Eyelink('Message', int2str(trigger.delay.start(3)))
        
        % Play ping sound
        y = [ping.*env;ping.*env];
        PsychPortAudio('FillBuffer', soundhandle, y);        
        WaitSecs('UntilTime', PrePingDelayStart+T_PrePingDelay);
        PingPlay = PsychPortAudio('Start', soundhandle, 1, 0, 1);
        PingStart = GetSecs;
        % Trigger & ET Message
        io64(ioObj,address,trigger.ping.start)
        io64(ioObj,address,0)
        Eyelink('Message', int2str(trigger.ping.start))
        
        % PostPingDelay
        Screen('FillRect',w,[0 0 0]);
        Screen('FrameOval',w,fixCols(2,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
        WaitSecs('UntilTime', PingStart+T_Ping);
        PostPingDelayStart = Screen('Flip',w,0,1);
        % Trigger & ET Message
        io64(ioObj,address,trigger.delay.start(4))
        io64(ioObj,address,0)
        Eyelink('Message', int2str(trigger.delay.start(4)))

        % create sound Probe:
        Probe_angle = LocPos*((2*pi)/360);
        
        ITD = (r/c)*(Probe_angle+sin(Probe_angle));
        if sign(ITD)==1
            ITD_l = zeros(1,abs(round(ITD*fs)));
            ITD_r = [];
        else
            ITD_r = zeros(1,abs(round(ITD*fs)));
            ITD_l = [];
        end
        base_sound  = ((soundProbe/max(abs(soundProbe))));  % normalize to 1
        left_sound  = [ITD_l base_sound ITD_r];
        right_sound = [ITD_r base_sound ITD_l];
        y = [left_sound; right_sound];

        PsychPortAudio('FillBuffer', soundhandle, y);      
        WaitSecs('UntilTime', PostPingDelayStart+T_PostPingDelay);

        % Stimulus Presentation Probe:
        Screen('FillRect',w,[0 0 0]);
        if ExpMat(tr,9)==1
            Screen('FrameOval',w,fixCols(2,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
        else
            Screen('FrameOval',w,fixCols(2,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
        end
        Screen('Flip',w,0,1);
        
        % Play:
        ProbePlay  = PsychPortAudio('Start', soundhandle, 1, 0, 1);
        
        ProbeStart = GetSecs;

        % Input Matrix columns:
        % 1:'Block' 2:'Trial' 3:'BlockTrial' 4:'SpeakerS1' 5:'LocS1' 6: 'SpeakerS2'
        % 7:'LocS2' 8:'TargetPos' 9:'SpeakerMatch' 10:'ProbeDistance'
        % 11:'LocMatch' 12:'ProbeLoc'

        % Trigger & ET Message
        io64(ioObj,address,trigger.probe.start(ProbeMatch+1,LocMatch+1)) % first index: speaker Nonmatch vs Match; second index: location Nonmatch vs Match
        io64(ioObj,address,0)
        Eyelink('Message', int2str(trigger.probe.start(ProbeMatch+1,LocMatch+1)))
        [~, ~] = KbQueueCheck; % flush button presses
        WaitSecs(.05); % to avoid trigger intereference
        io64(ioObj,address,trigger.probe.value(ExpMat(tr,12),ExpMat(tr,10)+1)) % first index: probe location; second index: speaker match distance (0=perfect match, 1=)
        io64(ioObj,address,0)
        Eyelink('Message', int2str(trigger.probe.value(ExpMat(tr,12),ExpMat(tr,10)+1)))
        WaitSecs('UntilTime', ProbeStart+T_Stim);

        % Response:
        respStart = GetSecs();
        actResp = 0;
        response  = false;
        resp = [];
        [pressed, keyPress] = KbQueueCheck;
        button = KbName(keyPress);
        if any(any(strcmp(button,'1!')) || any(strcmp(button,'2@')))
            response = true;
        end

        while response == false && actResp< T_Resp
            respAct = GetSecs();
            actResp = respAct-respStart;
            [pressed, keyPress] = KbQueueCheck;
            button = KbName(keyPress);

            % Check for response during loop:
            if any(any(strcmp(button,'1!')) || any(strcmp(button,'2@')))
                response = true;
            end       
        end
        
        respEnd = GetSecs();
        if response == true
            if ~iscell(button)
                resp  = str2double(button(1)); 
            else % if more than one button was pressed verz rapidly: only first answer is valid
                actButton = button{1};
                resp  = str2double(actButton(1)); 
            end
        end
        % Trigger & ET Message
        io64(ioObj,address,trigger.response.start) 
        io64(ioObj,address,0)
        Eyelink('Message', int2str(trigger.response.start))

        % Feedback:
        if resp==2
           resp=0; % convert button press 2 to nonmatch (0)
        elseif isempty(resp)
           resp = 2; % too slow
        end
        % Trigger & ET Message
        io64(ioObj,address,trigger.response.value(resp+1)) 
        io64(ioObj,address,0)
        Eyelink('Message', int2str(trigger.response.value(resp+1)))
        
        corr = resp == ExpMat(tr,9);
        Screen('FillRect',w,[0 0 0]);
        if resp == ExpMat(tr,9)
%             feedbackCol = [0 255 0];
            Screen('DrawLines',w,[rect(3)/2-fixLength/2,rect(3)/2,rect(3)/2,rect(3)/2+fixLength/2;rect(4)/2,rect(4)/2+fixLength/2,rect(4)/2+fixLength/2,rect(4)/2-fixLength/2],fixWidth*2,fixCols(2,:));
        else
%             feedbackCol = [255 0 0];
            Screen('DrawLines',w,[rect(3)/2-fixLength/2,rect(3)/2+fixLength/2,rect(3)/2-fixLength/2,rect(3)/2+fixLength/2;rect(4)/2-fixLength/2,rect(4)/2+fixLength/2,rect(4)/2+fixLength/2,rect(4)/2-fixLength/2],fixWidth*2,fixCols(2,:));
        end
        
%         Screen('FillOval',w,feedbackCol,[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2]);
        FeedbackStart = Screen('Flip', w);
        WaitSecs('UntilTime', FeedbackStart+T_Feedback);
        Screen('FrameOval',w,fixCols(1,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
        Screen('Flip', w);
        % Trigger & ET Message
        io64(ioObj,address,trigger.response.correct(int8(corr)+1))
        io64(ioObj,address,0)
        Eyelink('Message', int2str(trigger.response.correct(int8(corr)+1)))
        
        % Record Time:
        trialEnd = GetSecs;
        
        % Close soundhandle
        PsychPortAudio('Close',soundhandle);

        % Output:
        TrialOutputVector = [ExpMat(tr,:) resp corr respEnd-respStart trialEnd-trialStart];
        dlmwrite(sprintf('%s/Output_AWM4_Exp1_MEG_VP%d.txt',datapath,SubjNr),TrialOutputVector,'delimiter',' ','newline','pc','-append');

        % Manual Break:
        [~,~,pkey] = KbCheck;
        if any(find(pkey)==pauseKey)    % if pressing 'P' after feedback: break
            EndText = sprintf(['Pause.\n\n'...
                               'Break.']);
            Screen('FillRect',w,[0 0 0]);
            DrawFormattedText(w, EndText, 'center', 'center', [255 255 255]);
            Screen('Flip', w);
            [pressed, keyPress] = KbQueueCheck;
            button = KbName(keyPress);
            while  ~any(strcmp(button,'9('))
                [pressed, keyPress] = KbQueueCheck;
                button = KbName(keyPress);
            end
            
%             [~, ~, PauseButtons]=GetMouse();
%             while ~any(PauseButtons)
%                 [~, ~, PauseButtons]=GetMouse();
%             end
%             while any(PauseButtons)
%                 [~, ~, PauseButtons]=GetMouse();
%             end
            Screen('FillRect',w,[0 0 0]);
            Screen('Flip',w,0,1);
            WaitSecs(1);
            Screen('FrameOval',w,fixCols(1,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
            Screen('Flip',w,0,1);
            WaitSecs(INTITIALWAIT);
        end
        
        % Forced Stop:
        [~,~,skey] = KbCheck;
        if any(find(skey)==stopKey)    % if pressing 'S' after feedback: stop script
%             PsychPortAudio('Close',soundhandle);
            % End screen:
            EndText = sprintf('Geschafft!\n\nDone!');
            DrawFormattedText(w, EndText, 'center', 'center', [255 255 255]);
            Screen('Flip', w);
            [pressed, keyPress] = KbQueueCheck;
            button = KbName(keyPress);
            while  ~any(strcmp(button,'9('))
                [pressed, keyPress] = KbQueueCheck;
                button = KbName(keyPress);
            end
%             [~, ~, PauseButtons]=GetMouse();
%             while ~any(PauseButtons)
%                 [~, ~, PauseButtons]=GetMouse();
%             end
%             while any(PauseButtons) % Warte bis Maustaste wieder losgelassen wurde
%                 [~, ~, PauseButtons]=GetMouse();
%             end
            Priority(0);
            ShowCursor
%             Screen('CloseAll');
            StopScript = true;
        end
        if StopScript == true
            break
        end
    end
    
    if StopScript == true
        break
    end
    
    % inform the MEG & ET that block is over
    io64(ioObj,address,trigger.block.end)
    io64(ioObj,address,0)
    Eyelink('Message', int2str(trigger.block.end))
    Screen('FillRect',w,[0 0 0]);
    LastFlipTime = Screen('Flip',w,0,1);

    % end eyetracking & save file
    Eyelink('StopRecording')
    Eyelink('CloseFile')
    Eyelink('ReceiveFile', edfFile, fullfile(datapath,edfFile))
end

% End of Experiment:   
if ~StopScript
%     PsychPortAudio('Close',soundhandle);
    % End screen:
    EndText = sprintf('Geschafft!\n\nDone!');
    DrawFormattedText(w, EndText, 'center', 'center', [255 255 255]);
    Screen('Flip', w);
    [pressed, keyPress] = KbQueueCheck;
    button = KbName(keyPress);
    while  ~any(strcmp(button,'9('))
        [pressed, keyPress] = KbQueueCheck;
        button = KbName(keyPress);
    end

    Priority(0);
    ShowCursor
    
end
Screen('CloseAll');