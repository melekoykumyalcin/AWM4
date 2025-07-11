%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Auditory Working Memory 4 Feature Binding - Voice Stimuli              %
%  remember one stimulus with 2 features (speaker&location)               %
%  only speaker task-relevant                                             %
%  match/nonmatch task with probe that has same or diff location          %
%  >> Analysis script                                                     %
%  Cora Fischer, 15.01.2024                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Set Paths & request participants information

cd('/media/headmodel/J/AWM4behavioral/Experiment')
datapath    = fullfile(cd,'/Output');
NrSub       = numel(dir(sprintf('%s/Output_AWM4_Exp1_VP*.txt',datapath)));
SubFiles    = dir(sprintf('%s/Output_AWM4_Exp1_VP*.txt',datapath));

%% Load data
Accuracy = zeros(NrSub,2,2);
RT       = zeros(NrSub,2,2); %reaktionszeiten
NMAcc    = zeros(NrSub,2,2); %nonmatch accuracy
NMRT     = zeros(NrSub,2,2);
PTAcc    = zeros(NrSub,2,2,4);
PTRT     = zeros(NrSub,2,2,4);
ACCHalf  = zeros(NrSub,2);
RTHalf   = zeros(NrSub,2);
allSubsName = char(zeros(NrSub,4));

%%
for actSubj = 1:NrSub
    outData = importdata(fullfile(cd,'Output',SubFiles(actSubj).name));
    anData  = outData.data(outData.data(:,1)~=0,:);
    allSubsName(actSubj,:) =  SubFiles(actSubj).name(end-7:end-4);   
    if allSubsName(actSubj,1) == '_'
        allSubsName(actSubj,:) = sprintf('%s%d%s',allSubsName(actSubj,2:end-1),0,allSubsName(actSubj,end));
    end
    % exclude incorrectly recorded trials:
    if any(sum(isnan(anData),2)>0)
       anData([find(sum(isnan(anData),2)>0),find(sum(isnan(anData),2)>0)-1],:) = nan;
    end
    anData = anData(~isnan(anData(:,1)),:);
%     fprintf('%d-%d\n',actSubj,size(anData,1))   
    
    % output logic:
    % 1:'Block' 2:'Trial' 3:'BlockTrial' 4:'SpeakerS1' 5:'LocS1' 6: 'SpeakerS2'
    % 7:'LocS2' 8:'TargetPos' 9:'SpeakerMatch' 10:'ProbeDistance'
    % 11:'LocMatch' 12:'ProbeLoc' 13:Answer (1:Match, 0:Nonmatch, 2:too slow) 
    % 14:Correct(1)/Incorrect(0) 15:RT 16:Trialduration new 16: Syllable
    % (1-3)
    
    
    %% Data
    for SpeakerMNM = [1,0]
        for LocMNM = [1,0]
            Accuracy(actSubj,SpeakerMNM+1,LocMNM+1) = mean(anData(anData(:,9) == SpeakerMNM & anData(:,11) == LocMNM,14))*100; 
%             RT(actSubj,PitchMNM+1,RippleMNM+1)       = mean(anData(anData(:,9) == PitchMNM & anData(:,11) == RippleMNM,15)); 
            RT(actSubj,SpeakerMNM+1,LocMNM+1)       = mean(anData(anData(:,9) == SpeakerMNM & anData(:,11) == LocMNM & anData(:,14) == 1,15)); % RT only for correct responses
            if SpeakerMNM == 1
               pitchSteps = [1,4];
            else
               pitchSteps = [2,3];
            end
            NMAcc(actSubj,SpeakerMNM+1,LocMNM+1) = mean(anData(any(anData(:,10) == pitchSteps,2) & anData(:,11) == LocMNM,14))*100;
            NMRT(actSubj,SpeakerMNM+1,LocMNM+1)  = mean(anData(any(anData(:,10) == pitchSteps,2) & anData(:,11) == LocMNM,15));
            for pitch = 1:4
                PTAcc(actSubj,SpeakerMNM+1,LocMNM+1,pitch) = mean(anData(anData(:,4) == pitch & anData(:,9) == SpeakerMNM & anData(:,11) == LocMNM,14))*100;
                PTRT(actSubj,SpeakerMNM+1,LocMNM+1,pitch)  = mean(anData(anData(:,4) == pitch & anData(:,9) == SpeakerMNM & anData(:,11) == LocMNM,15)); 
            end
        end
    end
%     fprintf('%d\n',actSubj)
    ACCHalf(actSubj,:) = [mean(anData(1:round(end/2),14))*100;mean(anData(round(end/2)+1:end,14))*100];
    RTHalf(actSubj,:) = [mean(anData((anData(1:round(end/2),14) == 1),15));mean(anData((anData(round(end/2)+1:end,14) == 1),15))]; 
    
end

idx = 1:NrSub;
%% Request Subjects to highlight:

%[indx,tf] = listdlg('PromptString',{'Select subjects to highlight.',...
       %'To select several subjects, press ctrl (strg).','','',''},'ListString',allSubsName);

% %% ANOVA
% % t = table(repmat((1:NrSub)',4,1),[Accuracy(:,2,2);Accuracy(:,1,2);Accuracy(:,2,1);Accuracy(:,1,1)],'VariableNames',{'subjects','Accuracy'});
% datamat        = zeros(NrSub,2,2);
% datamat(:,1,:) = Accuracy(:,1,:); %(Versuchsperson,Pitch Match/Nonmatch,Loc Match/Nonmatch)
% datamat(:,2,:) = Accuracy(:,2,:);
% tbl = simple_mixed_anova(datamat,[],{'Speaker','Location'}); %Accuracy
% 
% datamat2        = zeros(NrSub,2,2);
% datamat2(:,1,:) = RT(:,1,:);
% datamat2(:,2,:) = RT(:,2,:);
% tbl2 = simple_mixed_anova(datamat2,[],{'Speaker','Location'}); %Reaktionszeiten

%% Plotting II
figure(1);clf;set(gcf,'Color','w');hold on
RippleCols = [.7,.3,.3;.3,.7,.3];
fw='normal';
subplot(2,3,1);cla;hold on
title({sprintf('a) Accuracy - Speaker Match vs Nonmatch - %d Subjects',NrSub);' '})
set(gca,'xlim',[0,3.5])
xticks([1,2.5])
xticklabels({'Speaker Match','Speaker Nonmatch'})
set(gca,'ylim',[20,100])
ylabel('Accuracy [%]')

subplot(2,3,4);cla;hold on
title({sprintf('b) Reaction Time (correct responses) - Speaker Match vs Nonmatch - %d Subjects',NrSub);' '})
set(gca,'xlim',[0,3.5])
xticks([1,2.5])
xticklabels({'Speaker Match','Speaker Nonmatch'})
set(gca,'ylim',[.0,.75])
ylabel('Reaction Time [s]')
plotPos = [2.75,1.25;2.25,.75];

% subplot(2,3,2);cla;hold on
% set(gca,'xtick',[],'ytick',[],'xcolor',[1 1 1],'ycolor',[1 1 1]);
% text(0,.9,'ANOVA Accuracy: Speaker*Location','FontWeight','bold')
% text(0,.8,'Main effect Speaker Match vs Nonmatch:')
% if tbl{3,5}<.05
%     fw = 'bold';
% else
%     fw='normal';
% end
% text(0,.7,sprintf(' p = %.3f',tbl{3,5}),'FontWeight',fw)
% text(0,.6,'Main effect Location Match vs Nonmatch:')
% if tbl{5,5}<.05
%     fw = 'bold';
% else
%     fw='normal';
% end
% text(0,.5,sprintf(' p = %.3f',tbl{5,5}),'FontWeight',fw)
% text(0,.4,'Interaction Speaker*Location Match vs Nonmatch:')
% if tbl{7,5}<.05
%     fw = 'bold';
% else
%     fw='normal';
% end
% text(0,.3,sprintf(' p = %.3f',tbl{7,5}),'FontWeight',fw)
% 
% subplot(2,3,5);cla;hold on
% set(gca,'xtick',[],'ytick',[],'xcolor',[1 1 1],'ycolor',[1 1 1]);
% text(0,.9,'ANOVA Reaction Time: Speaker*Location','FontWeight','bold')
% text(0,.8,'Main effect Speaker Match vs Nonmatch:')
% if tbl2{3,5}<.05
%     fw = 'bold';
% else
%     fw='normal';
% end
% text(0,.7,sprintf(' p = %.3f',tbl2{3,5}),'FontWeight',fw)
% text(0,.6,'Main effect Location Match vs Nonmatch:')
% if tbl2{5,5}<.05
%     fw = 'bold';
% else
%     fw='normal';
% end
% text(0,.5,sprintf(' p = %.3f',tbl2{5,5}),'FontWeight',fw)
% text(0,.4,'Interaction Speaker*Location Match vs Nonmatch:')
% if tbl2{7,5}<.05
%     fw = 'bold';
% else
%     fw='normal';
% end
% text(0,.3,sprintf(' p = %.3f',tbl2{7,5}),'FontWeight',fw)

subplot(2,3,3);cla;hold on
title({'c) Accuracy - Speaker Nonmatch easy vs difficult';' '})
set(gca,'xlim',[0,3.5])
xticks([1,2.5])
xticklabels({'Easy','Difficult'})
set(gca,'ylim',[15,100])
ylabel('Accuracy [%]')

subplot(2,3,6);cla;hold on
title({'d) Reaction Time (correct responses) - Speaker Nonmatch easy vs difficult';' '})
set(gca,'xlim',[0,3.5])
xticks([1,2.5])
xticklabels({'Easy','Difficult'})
set(gca,'ylim',[.0,.75])
ylabel('Reaction Time [s]')

for SpeakerMNM = [1,0]
    subplot(2,3,1)
    line(plotPos(:,SpeakerMNM+1),[Accuracy(:,SpeakerMNM+1,1),Accuracy(:,SpeakerMNM+1,2)]','color',[.5,.5,.5])
    if numel(indx) ~= NrSub
        line(plotPos(:,SpeakerMNM+1),[Accuracy(indx,SpeakerMNM+1,1),Accuracy(indx,SpeakerMNM+1,2)]','color','k','Linewidth',2)
    end
    subplot(2,3,4)
    line(plotPos(:,SpeakerMNM+1),[RT(:,SpeakerMNM+1,1),RT(:,SpeakerMNM+1,2)]','color',[.5,.5,.5])
    if numel(indx) ~= NrSub
        line(plotPos(:,SpeakerMNM+1),[RT(indx,SpeakerMNM+1,1),RT(indx,SpeakerMNM+1,2)]','color','k','Linewidth',2)
    end
    
     if SpeakerMNM == 1
        pitchSteps = [1,4];
    else
        pitchSteps = [2,3];
     end
    subplot(2,3,3)
    line(plotPos(:,SpeakerMNM+1),[NMAcc(:,SpeakerMNM+1,1),NMAcc(:,SpeakerMNM+1,2)]','color',[.5,.5,.5])
    if numel(indx) ~= NrSub
        line(plotPos(:,SpeakerMNM+1),[NMAcc(indx,SpeakerMNM+1,1),NMAcc(indx,SpeakerMNM+1,2)]','color','k','Linewidth',2)
    end
    subplot(2,3,6)
    line(plotPos(:,SpeakerMNM+1),[NMRT(:,SpeakerMNM+1,1),NMRT(:,SpeakerMNM+1,2)]','color',[.5,.5,.5])
    if numel(indx) ~= NrSub
        line(plotPos(:,SpeakerMNM+1),[NMRT(indx,SpeakerMNM+1,1),NMRT(indx,SpeakerMNM+1,2)]','color','k','Linewidth',2)
    end
    for LocMNM = [1,0]
        subplot(2,3,1)
        meanAcc = mean(Accuracy(:,SpeakerMNM+1,LocMNM+1));
        stdAcc  = std(Accuracy(:,SpeakerMNM+1,LocMNM+1))/sqrt(NrSub);
        PL2(1,LocMNM+1,:) = plot(plotPos(LocMNM+1,SpeakerMNM+1),meanAcc,'MarkerFaceColor',RippleCols(LocMNM+1,:),'MarkerEdgeColor',RippleCols(LocMNM+1,:),'Marker','square','MarkerSize',10);
        errorbar(plotPos(LocMNM+1,SpeakerMNM+1),meanAcc,stdAcc,'color',RippleCols(LocMNM+1,:))
        line(xlim,[50,50],'Color','k')
        line(xlim,[60,60],'Color','m')
        if SpeakerMNM == 1 && LocMNM==1
            text(2,45,'overall NM accuracy:')
            text(2,40,sprintf('%.2f%%',mean(mean(squeeze(mean(Accuracy(:,1,:),2)),2))))
        end

        subplot(2,3,4)
        meanRT = mean(RT(:,SpeakerMNM+1,LocMNM+1)); 
        stdRT  = std(RT(:,SpeakerMNM+1,LocMNM+1))/sqrt(NrSub);
        PL2(2,LocMNM+1,:) = plot(plotPos(LocMNM+1,SpeakerMNM+1),meanRT,'MarkerFaceColor',RippleCols(LocMNM+1,:),'MarkerEdgeColor',RippleCols(LocMNM+1,:),'Marker','square','MarkerSize',10);
        errorbar(plotPos(LocMNM+1,SpeakerMNM+1),meanRT,stdRT,'color',RippleCols(LocMNM+1,:))
        
        subplot(2,3,3)
        meanAcc = mean(NMAcc(:,SpeakerMNM+1,LocMNM+1));
        stdAcc  = std(NMAcc(:,SpeakerMNM+1,LocMNM+1))/sqrt(NrSub);
        PL(3,LocMNM+1,:) = plot(plotPos(LocMNM+1,SpeakerMNM+1),meanAcc,'MarkerFaceColor',RippleCols(LocMNM+1,:),'MarkerEdgeColor',RippleCols(LocMNM+1,:),'Marker','square','MarkerSize',10);
        errorbar(plotPos(LocMNM+1,SpeakerMNM+1),meanAcc,stdAcc,'color',RippleCols(LocMNM+1,:))
        line(xlim,[50,50],'Color','k')
        line(xlim,[60,60],'Color','m')
        if LocMNM==1
           text(plotPos(LocMNM+1,SpeakerMNM+1),45,sprintf('NM accuracy: %.2f%%',mean(mean(NMAcc(:,SpeakerMNM+1,:)))))
        end

        subplot(2,3,6)
        meanRT = mean(NMRT(:,SpeakerMNM+1,LocMNM+1)); 
        stdRT  = std(NMRT(:,SpeakerMNM+1,LocMNM+1))/sqrt(NrSub);
        PL(4,LocMNM+1,:) = plot(plotPos(LocMNM+1,SpeakerMNM+1),meanRT,'MarkerFaceColor',RippleCols(LocMNM+1,:),'MarkerEdgeColor',RippleCols(LocMNM+1,:),'Marker','square','MarkerSize',10);
        errorbar(plotPos(LocMNM+1,SpeakerMNM+1),meanRT,stdRT,'color',RippleCols(LocMNM+1,:))
       
    end       
end

subplot(2,3,1)
pos = get(gca,'Position');
legend([PL2(1,2,:),PL2(1,1,:)],'Location Match','Location Nonmatch','Location','northeastoutside')
legend('boxoff')
set(gca,'Position',[pos(1:2),pos(3)-.05,pos(4)]) 

subplot(2,3,4)
pos = get(gca,'Position');
legend([PL2(2,2,:),PL2(2,1,:)],'Location Match','Location Nonmatch','Location','northeastoutside')
legend('boxoff')
set(gca,'Position',[pos(1:2),pos(3)-.05,pos(4)])  


