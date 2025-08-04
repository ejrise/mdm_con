
clear all;
close all;
clearvars;

setenv('TZ', 'America/Los_Angeles');
addpath(genpath('/path/to/subfolders/'));
%% Step 0: Define Files, Define Variables

subfold = [{'sub-01'}];
% run in serial (0) or parallel (1)
runInParallel=1;

if runInParallel
    cluster=parcluster('local'); 
    job = createJob(cluster);
end

%% Task 

tic
try
    
    for iSub = 1:length(subfold)
        sjNum = subfold(iSub);
        sjNum = sjNum{1};
            if runInParallel
                createTask(job,@lr_runmydtd_hhs,0,{sjNum}); %Task code must be a Matlab function
            else
                lr_runmydtd_hhs(sjNum); %Task Code must be a matlab function
            end
    end
   
    
    if runInParallel
        submit(job)
    end
    
catch e

    message = sprintf('The identifier was:\n%s.\nThe message was:\n%s', e.identifier, e.message);
    
end
