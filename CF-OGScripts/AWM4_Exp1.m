%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Auditory Working Memory 4 Feature Binding - Voice Stimuli              %
%  remember one stimulus with 2 features (speaker&location)               %
%  only speaker task-relevant                                             %
%  match/nonmatch task with probe that has same or diff location          %
%  >> Experimental script (behavioral)                                    %
%  Cora Fischer, 03.01.2024                                               %
%  v1.1: added ping sound (23.01.2024)                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Set Paths & request participants information

cd(fullfile('C:\Users\IMP\Desktop\AWM4\Experiment'))
datapath  = fullfile(cd,'/Output');
soundpath = fullfile(cd,'/Sounds');

prompt      = {'VP-Nr: ', 'StartBlock: ', 'Uebung?:' };
NrFiles     = numel(dir(sprintf('%s/Output_AWM4_Exp1_VP*.txt',datapath)));
defAns      = {sprintf('%d',NrFiles+1),'1','1'};
CodeEingabe = inputdlg(prompt,'VP-Code',1,defAns);
SubjNr      = str2double(CodeEingabe(1));
StartBlock  = str2double(CodeEingabe(2));
Practice    = str2double(CodeEingabe(3))==1;
vpCode      = sprintf('AWM4_Exp1%d',SubjNr);

SC          = menu('External soundcard?','yes','no');
Sylls       = 1;%menu('Single syllable?','yes','no');
KbName('UnifyKeyNames')
pauseKey = KbName('p');
stopKey  = KbName('s');

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


%% Load Sounds
% syll 1
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

inData = importdata(fullfile(cd,'Input',sprintf('Input_AWM4_Exp1_VP%d.txt',SubjNr)));
if StartBlock == 1
    OutputHeader = {'VP', num2str(SubjNr), ' Output_AWM4_Exp1 ', datestr(now)};
    dlmwrite(sprintf('%s//Output_AWM4_Exp1_VP%d.txt',datapath,SubjNr),OutputHeader,'delimiter','','newline','pc','-append');
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
% devi = PsychPortAudio('GetDevices',3);
devi = PsychPortAudio('GetDevices');
for ii=1:length(devi)
    devi(ii).HostAudioAPIName;
    % fprintf('%d %s %s',[devi(ii).DeviceIndex ' ' devi(ii).HostAudioAPIName ' ' devi(ii).DeviceName]);
    % fprintf('\n');
end

if SC == 1
    %mydevice = devi(strcmp({devi(:).DeviceName},'ASIO Fireface USB')).DeviceIndex; % Get usb soundcard...
    mydevice = 5;%devi(strcmp({devi(:).DeviceName},'ASIO Fireface USB')).DeviceIndex; % Get usb soundcard...
else
    mydevice = [];
end

% when the device changes, uncomment 181 and 182 to get the device index; 
% doubleclick devii in MATLAB from the variables 
% should look like ...ASI0 fireface USB, find the one that has NO input TWO output 
% WASAPI 

% soundhandle    = PsychPortAudio('Open', mydevice, [], 0, fs, nrchannels);
% y              = zeros(2,npts);
% % y=[xM';xM']; %to test audio output
% PsychPortAudio('FillBuffer', soundhandle, y);
% PsychPortAudio('Start', soundhandle, 1, 0, 1);

%% Experiment's Loop
% Start Experiment:
[w, ~]           = Screen('OpenWindow', screenindex, 1);
% Screen('Preference', 'DefaultFontName', 'Helvetica', 'SkipSyncTests', 1);
Screen('BlendFunction', w, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
HideCursor(w);
Screen('TextFont',w,'Arial Unicode MS');
Screen('TextStyle',w,0);
Screen('TextSize',w,21);
StopScript = false;
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
        'LINKE Maustaste. Wird der Vergleichsreiz von einer ANDEREN Stimme gesprochen, druecken Sie bitte die \n\n'...
        'RECHTE Maustaste. Antworten Sie bitte so SCHNELL und PRAEZISE wie moeglich. Nachdem\n\n'...
        'Sie Ihre Antwort abgegeben haben, erhalten Sie ein Feedback. Erscheint ein HAKEN, \n\n'...
        'haben Sie richtig geantwortet. \n\n'...
        'Erscheint ein KREUZ, war Ihre Antwort leider falsch.\n\n'...
        'Wenn Sie keine Fragen mehr haben, folgt nun ein Uebungsblock. Starten Sie die\n\n'...
        'Uebungsdurchgaenge mit beliebiger Maustaste.\n\n'...
        ' \n\n'...
        'In the following experiment, you will be presented with two spoken stimuli in each trial.\n\n'...
        'Your task is to remember the two voices of the stimuli. After a short break, a number will show\n\n'...
        'you whether you should remember the first ("1") or the second ("2") stimulus.\n\n'... 
        'You will have to compare this stimulus to a target stimulus after another short break. \n\n'...
        'If the target stimulus is spoken by the SAME voice as the memory stimulus, please press the\n\n'...
        'LEFT mouse button. If the target stimulus is spoken by a DIFFERENT voice than the memory stimulus, please press\n\n'...
        'the RIGHT mouse button. Please answer as FAST and ACCURATE as possible. After you have given your\n\n'...
        'answer, you will receive feedback. If a TICK appears, your answer was correct. \n\n'...
        'If a CROSS appears, your answer was incorrect.\n\n'...
        'If you have no further questions, we will proceed with a practice round. Please start the practice\n\n'...
        'trials with any mouse button.\n\n']);
    DrawFormattedText(w, WelcomeText, 'center', 'center', [255 255 255]);
    Screen('Flip', w);
    [~, ~, PauseButtons] = GetMouse();
    while ~any(PauseButtons)
        [~, ~, PauseButtons] = GetMouse();
    end
    while any(PauseButtons) % Warte bis Maustaste wieder losgelassen wurde
        [~, ~, PauseButtons] = GetMouse();
    end
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
    
    % break before practice block
    if bb == 0 % practice block
        PauseText = sprintf(['Starten Sie den Uebungsblock mit beliebiger Maustaste\n\n'...
                            'Start the practice block with any mouse key']);
    % Block Pause screen:
    else
        PauseText = sprintf(['Pause.\n\nStarten Sie den %d. Block mit beliebiger Maustaste\n\n'...
                             'Break.\n\nStart the %d. block with any mouse key.'],bb,bb);
    end
    DrawFormattedText(w, PauseText, 'center', 'center', [255 255 255]);
    Screen('Flip', w);
    [~, ~, PauseButtons]=GetMouse();
    while ~any(PauseButtons) 
        [~, ~, PauseButtons]=GetMouse();
    end
    while any(PauseButtons) % Warte bis Maustaste wieder losgelassen wurde
        [~, ~, PauseButtons]=GetMouse();
    end  
    Screen('FillRect',w,[0 0 0]);
    Screen('Flip',w,0,1);
    WaitSecs(1);
    Screen('FrameOval',w,fixCols(1,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
    Screen('Flip',w,0,1);
    WaitSecs(INTITIALWAIT); 
    
    
    for tr = (find(ExpMat(:,1)==bb))'
        
        % Record Time:
        trialStart = GetSecs;
        
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
        
        % open soundhandle
        soundhandle    = PsychPortAudio('Open', mydevice, [], 0, fs, nrchannels);

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
        PsychPortAudio('FillBuffer', soundhandle, y);
        S1Play  = PsychPortAudio('Start', soundhandle, 1, 0, 1);
        S1Start = GetSecs;

        % ISI
        Screen('FillRect',w,[0 0 0]);
        Screen('FrameOval',w,fixCols(2,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
        WaitSecs('UntilTime', S1Start+T_Stim);
        ISIStart = Screen('Flip',w,0,1);
        
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
        PsychPortAudio('FillBuffer', soundhandle, y);
        S2Play  = PsychPortAudio('Start', soundhandle, 1, 0, 1);
        S2Start = GetSecs;
        
        % PreCueDelay
        Screen('FillRect',w,[0 0 0]);
        Screen('FrameOval',w,fixCols(2,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
        WaitSecs('UntilTime', S2Start+T_Stim);
        PreCueDelayStart = Screen('Flip',w,0,1);
        
        % Present cue
        Screen('FillRect',w,[0 0 0]);
        cueText = num2str(actTarget);
        DrawFormattedText(w, cueText, 'center', 'center', [255 255 255]);
        WaitSecs('UntilTime', PreCueDelayStart+T_ISI);
        CueStart = Screen('Flip',w,0,1);     
        
        % Postcue/PrePing Delay
        Screen('FillRect',w,[0 0 0]);
        Screen('FrameOval',w,fixCols(2,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
        WaitSecs('UntilTime', CueStart+T_Cue);
        PrePingDelayStart = Screen('Flip',w,0,1);
        
        % Play ping sound
        y = [ping.*env;ping.*env];
        PsychPortAudio('FillBuffer', soundhandle, y);        
        WaitSecs('UntilTime', PrePingDelayStart+T_PrePingDelay);
        PingPlay = PsychPortAudio('Start', soundhandle, 1, 0, 1);
        PingStart = GetSecs;
        
        % PostPingDelay
        Screen('FillRect',w,[0 0 0]);
        Screen('FrameOval',w,fixCols(2,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
        WaitSecs('UntilTime', PingStart+T_Ping);
        PostPingDelayStart = Screen('Flip',w,0,1);

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
        % if ExpMat(tr,9)==1
        Screen('FrameOval',w,fixCols(2,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
        % else
            % Screen('FrameOval',w,fixCols(2,:),[rect(3)/2-fixLength/2,rect(4)/2-fixLength/2,rect(3)/2+fixLength/2,rect(4)/2+fixLength/2],fixWidth,fixWidth);
        % end
        Screen('Flip',w,0,1);

        % Play:
        ProbePlay  = PsychPortAudio('Start', soundhandle, 1, 0, 1);
        ProbeStart = GetSecs;
        WaitSecs('UntilTime', ProbeStart+T_Stim);

        % Response:
        respStart = GetSecs();
        [~, ~, Mb] = GetMouse();
        response  = false;
        if any(Mb == 1)
            response = true;
        end
        actResp = 0;

        while response == false && actResp< T_Resp

            respAct = GetSecs();
            actResp = respAct-respStart;
            [~, ~, Mb] = GetMouse();

            % Check for response during loop:
            if any(Mb([1 3]) == 1)
                response = true;
            end       

        end
        respEnd = GetSecs();
        resp    = find(Mb); 

        % Feedback:
        if resp==3
           resp=0;
        elseif isempty(resp)
           resp = 2; % too slow
        end
        
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
        
        % Record Time:
        trialEnd = GetSecs;

        % Close soundhandle
        PsychPortAudio('Close', soundhandle);

        % Output:
        TrialOutputVector = [ExpMat(tr,:) resp corr respEnd-respStart trialEnd-trialStart actSyll]; % actSyll added 25.01.
        dlmwrite(sprintf('%s/Output_AWM4_Exp1_VP%d.txt',datapath,SubjNr),TrialOutputVector,'delimiter',' ','newline','pc','-append');

        % Manual Break:
        [~,~,pkey] = KbCheck;
        if any(find(pkey)==pauseKey)    % if pressing 'P' after feedback: break
            EndText = sprintf(['Pause.\n\n Zum Fortfahren beliebige Maustaste druecken.\n\n'...
                               'Break.\n\n To proceed, press any mouse key.']);
            Screen('FillRect',w,[0 0 0]);
            DrawFormattedText(w, EndText, 'center', 'center', [255 255 255]);
            Screen('Flip', w);
            [~, ~, PauseButtons]=GetMouse();
            while ~any(PauseButtons)
                [~, ~, PauseButtons]=GetMouse();
            end
            while any(PauseButtons)
                [~, ~, PauseButtons]=GetMouse();
            end
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
            % PsychPortAudio('Close',soundhandle);
            % End screen:
            EndText = sprintf('Geschafft!\n\nDone!');
            DrawFormattedText(w, EndText, 'center', 'center', [255 255 255]);
            Screen('Flip', w);
            [~, ~, PauseButtons]=GetMouse();
            while ~any(PauseButtons)
                [~, ~, PauseButtons]=GetMouse();
            end
            while any(PauseButtons) % Warte bis Maustaste wieder losgelassen wurde
                [~, ~, PauseButtons]=GetMouse();
            end
            Priority(0);
            ShowCursor
            Screen('CloseAll');
            StopScript = true;
        end
        if StopScript == true
            break
        end
    end
    
    if StopScript == true
        break
    end
end

% End of Experiment:   
if ~StopScript
    % PsychPortAudio('Close',soundhandle);
    % End screen:
    EndText = sprintf('Geschafft!\n\nDone!');
    DrawFormattedText(w, EndText, 'center', 'center', [255 255 255]);
    Screen('Flip', w);
    [~, ~, PauseButtons]=GetMouse();
    while ~any(PauseButtons)
        [~, ~, PauseButtons]=GetMouse();
    end
    while any(PauseButtons) % Warte bis Maustaste wieder losgelassen wurde
        [~, ~, PauseButtons]=GetMouse();
    end

    Priority(0);
    ShowCursor
    Screen('CloseAll');
end
