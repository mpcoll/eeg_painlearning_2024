%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analyse SCR using PsPM
% Take the output scr_prepare_pspm.py and model SCR responses
% using PsPM toolbox.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Main function to run
function fearcond_pspmSCR()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Import data from raw files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%
% Clear workspace

clc;
clear;

% Get files path
p.rawpath = 'derivatives';
p.outpath = 'derivatives';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Process SCR with PSPM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Print a plot for each trial
p.pspm.plot_model = 1;

% % Use qa on data
p.pspm.cleanscr = 1;

% WARNING this turns off matlab figures for all scripts.
% if you stop script before the end, you need to turn this on
% manually
set(groot,'defaultFigureVisible','off') % Turn off figure popup


subdirs = dir(fullfile(p.rawpath, 'sub-*'));
subdirs = {subdirs.name};

disp([num2str(length(subdirs)) ' subjects were found']);


% % Run
fearcond_pspmSCR_analyses(subdirs, p.rawpath, p.pspm.plot_model,...
                           p.pspm.cleanscr)

% Collect all data in a single file and save
outpath = fearcond_collect_pspmdat(p.outpath);

exit;

end


% Subjects to run
function outpath = fearcond_pspmSCR_analyses(subdirs, scrpath, plot_model, cleanscr)

% list all subjects in scr path


for s = 1:length(subdirs)

    %Sub name
    subname = subdirs{s};

    % Path where data is
    sub_path = fullfile(scrpath, subname, 'scr');


    % Load physio data from bids data
    info = load(fullfile(sub_path, [subname '_scr_pspm.mat']));


    % Make sure all numbers are in double format
    vars = fieldnames(info);
    for v = 1:length(vars)
        type = eval(['class(info.' vars{v} ');']);
        if strcmp(type, 'int64')
            eval(['info.' vars{v} '= double(info.' vars{v} ');']);
        end
    end

    % Add sub info to struct
    info.sub_path = sub_path;
    info.subname = subname;


    % Run modelling of single trials using PsPM

    % Onsets of interest are cue onsets (everywhere to go from ms to s)
    info.ons_int{1, 1} = info.cue_onsets /info.srate;
    info.dur_int{1,1} = 1;

    %Onsets of no interest: shock, fix, rating, pause onsets
    info.ons_no_int{1, 1} = info.shock_onsets/info.srate;
    info.ons_no_int{1, 2} = info.fix_onsets/info.srate;
    info.ons_no_int{1, 3} = info.rat_onsets/info.srate;
    info.ons_no_int{1, 4} = info.pause_onsets/info.srate;
    info.ons_no_int{1, 5} = 1; % Start at time 0

    info.dur_no_int{1, 1} = 0;
    info.dur_no_int{1, 2} = info.fix_durations/info.srate;
    info.dur_no_int{1, 3} = info.rat_durations/info.srate;
    info.dur_no_int{1, 4} = info.pause_durations/info.srate;
    info.dur_no_int{1, 5} = info.dur_no_int{1, 2}(1)-1;



    % Condition names
    info.names = {'CS_target', 'CS_nointerest', 'shock', 'fix', ...
                  'ratings', 'pause', 'recordstart'};

    % Trim -1000 ms before first fix and 10000 after last to remove potential
    % long durations before / after recording
    info.trim_zone = [1,...
                      info.ons_int{1, 1}(end) + 5];

    pspm_single_trial(info, plot_model, cleanscr)

end

end



function pspm_single_trial(info, plot_model, cleanscr)

cd(info.sub_path);

% Import the data from the python exported txt file
global settings;
if isempty(settings), pspm_init; end
D{1} = [info.subname '_scr_data.txt'];
datatype = 'txt';
import{1}.channel = 1;
import{1}.type = 'scr';
import{1}.sr = info.srate;
import{1}.transfer = 'none';
options.overwrite = 1;
pspm_import(D, datatype, import, options);
clear D

% Trim the data to keep only duration of interest
imported_filename = ['pspm_' info.subname '_scr_data.mat'];
D = imported_filename;
from = info.trim_zone(1);
to = info.trim_zone(2);
reference = 'file';
options.overwrite = 1;
pspm_trim(D, from, to, reference, options);
clear D;

filename = fullfile(info.sub_path, ['t' imported_filename]);

% If clean data, use a simple quality correction on scr based on
% default max=60 min=0.05 slope=10
if cleanscr
    disp('Cleaning scr')
    load(filename)

    figure;
    % plot raw
    plot(data{1}.data)
    hold on

    % Censor values
    [sts, data{1}.data] = pspm_simple_qa(data{1}.data, info.srate);
    infos.percdatacleaned = sum(isnan(data{1}.data))/length(data{1}.data)*100;
    disp(['Cleaned ' num2str(infos.percdatacleaned) ' % of data'])
    % Interpolate
    options.extrapolate = 1;
    [sts, data{1}.data] = pspm_interpolate(data{1}.data, options);


    % Plot cleaned over raw and save
    plot(data{1}.data)
    h = gcf;
    saveas(h(1), fullfile(info.sub_path,...
                          ['DataCleaning.png']),...
                          'png')
    close;
    filename = fullfile(info.sub_path, ['ct' imported_filename]);
    save(filename, 'data', 'infos')
end


% Run the GLM
clear data infos
models = cell(length(info.ons_int), 1);
for i= 1:length(info.ons_int{1,1})

    % For each trial, put the onset of the trial first and all others
    % afterwards
    if i == 1   % If first trial
    info.onsets = {info.ons_int{1,1}(i) info.ons_int{1,1}(i+1:end)};
    elseif i == length(info.ons_int{1,1})  % if last trial
    info.onsets = {info.ons_int{1,1}(i) info.ons_int{1,1}(1:end-1)};
    else % other trials
    info.onsets = {info.ons_int{1,1}(i) info.ons_int{1,1}([1:i-1 i+1:end])};
    end

    info.durations = {info.dur_int{1,1} info.dur_int{1,1}};

    % Put onsets of no interest after the ones of interest
    if length(info.ons_no_int) > 0
        for j = 1:length(info.ons_no_int)
            info.onsets = [info.onsets info.ons_no_int{1,j}];
            info.durations{1,2+j} =  info.dur_no_int{1,j};
        end
    end


    % predefined basis functions:
    % 'scrf' provides a canonical skin conductance response function
    % 'scrf1' adds the time derivative
    % 'scrf2' adds time dispersion derivative
    % 'FIR' provides 30 post-stimulus timebins of 1 s duration

    % Specify model
    % Directory for model
%     modeldir = fullfile(info.sub_path, 'SCR_GLM_analyses',...
%                  ['model_response_#' num2str(i)]);

%     mkdir(modeldir);

    cond.names = info.names;
    cond.onsets = info.onsets;
    cond.durations = info.durations;

    % Model parameters
    model.datafile = filename;
    model.modelfile = fullfile(info.sub_path, 'model.mat');
    model.timing = cond;  % onsets
    model.timeunits = 'seconds';  % time unit
    model.norm = 1;  % Normalize
    model.channel = 0;
    model.bf = struct('fhandle', @pspm_bf_scrf, 'args' , 1);
    model.info = info;
    % model.filter not specified, will use default 0.05-5Hz
    options.overwrite = 1;
    save(fullfile(info.sub_path, 'model.mat'), '-struct', 'model')


    % Run the model
    pspm_glm(model, options);

    %display model
    %scr_rev_glm(modelfile); %section that is displaying the model files
    display(['Running model ' num2str(i) ' on ' ...
              num2str(length(info.ons_int{1,1}))]); %what we have replaced so that can run faster

    % Evaluate the model with contrasts
    connames{1} = 'target_interest';  % Main targets
    convec{1} = [1 0 zeros(1,length(info.ons_no_int))];
    connames{2} = 'other_interest';   % Nuisance targets
    convec{2} = [0 1 zeros(1,length(info.ons_no_int))];

    for j = 1:length(info.ons_no_int)
        connames{2+j} = ['no_interest' num2str(j)];
        convec{2+j} = [ zeros(1,1+j) 1];
    end

    deletecon = 1;
    datatype = 'recon';
    pspm_con1(model.modelfile, connames, convec, datatype, deletecon);
    load(model.modelfile); % Load the glm (do not save for each trial to save space)

    if plot_model

        close all
        % Make fig dir if does not exist
        figdir = fullfile(info.sub_path, 'figures');
        if ~exist(figdir, 'dir')
            mkdir(figdir)
        end

        % Make all plots

        % These plots are mostly the same for all trials, just produce once
        % for speed (TODO make this an option in the function input)
        if i == 1
            figure('visible','off');
            fig = pspm_rev_glm(model.modelfile, model, 1);
            h = gcf;
            saveas(h(1), fullfile(figdir,...
                                  ['DesignMatrix_trial' num2str(i) '.png']),...
                                  'png');



            pspm_rev_glm(model.modelfile, glm, 2);
            h = gcf;
            saveas(h(1), fullfile(figdir,...
                                  ['DesignOrtho_trial' num2str(i) '.png']),...
                                  'png');
            close;

            pspm_rev_glm(model.modelfile, glm, 3);
            h = gcf;
            saveas(h(1), fullfile(figdir,...
                                  ['ModelFit_trial' num2str(i) '.png']),...
                                  'png');
            close;

            pspm_rev_glm(model.modelfile, glm, 5);
            h = gcf;
            print(fullfile(figdir,...
                           ['EstResponse_trial' num2str(i) '.png']),...
                  '-dpng', '-r50')
        end




        close all;

    end

    % Collect response
    model.response = glm.con(1).con;
    models{i} = model;



    close all;
end

% Save models (not evaluated to save space)
save(fullfile(info.sub_path, 'models.mat'), 'models')

% Remove single model file and intermediate data
% delete(fullfile(info.sub_path, 'model.mat'))
% delete(fullfile(info.sub_path, 'pspm_sub-32_scr_data.mat'))
% delete(fullfile(info.sub_path, 'tpspm_sub-32_scr_data.mat'))
% delete(fullfile(info.sub_path, 'cpspm_sub-32_scr_data.mat'))

% Save responses
responses = [];
for i=1:length(models)
    responses = [responses; models{i}.response];
end
save(fullfile(info.sub_path, [info.subname '_scr_responses.mat']),...
                              'responses')

% Turn back on figure display
set(groot,'defaultFigureVisible','on')

% Plot average responses
figure;
boxplot(responses, info.conditions)
h = get(0,'children');
saveas(h(1), fullfile(figdir,...
                      ['BoxplotAlltrials.png']),...
                      'png');
close;

figure;
plot(responses)
h = get(0,'children');
saveas(h(1), fullfile(figdir,...
                      ['LinePlotAlltrials.png']),...
                      'png');
close;


end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Collect data from SCR analyses in a single file
% @ 2019, MP Coll
% Take the output from fearcond_pspmSCR_analyses.m and makes a single
% data file for modelling
% scrpath = where is the modelled data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function outpath = fearcond_collect_pspmdat(scrpath)



% list all subjects in scr path
subdirs = dir(fullfile(scrpath, 'sub-*'));
subdirs = {subdirs.name};
disp([num2str(length(subdirs)) ' subjects were found']);


% Loop models
% Collect in a cell variable to export to csv
alldatacsv = {};

% Also collect in an array of structures which should be easier to use for later
% matlab scripts
data = {};
for s = 1:length(subdirs)

    % make empty cell array for sub data
    subdata ={'sub', 'trial', 'response', 'condition', 'cue', 'cuenum', 'shock'};
    substruct = struct();

    % Load sub data
    load(fullfile(scrpath, subdirs{s}, 'scr', [subdirs{s} '_scr_responses.mat']));

    % Load trial info
    load(fullfile(scrpath, subdirs{s}, 'scr', [subdirs{s} '_scr_pspm.mat']));

    for t = 1:length(responses)
       % In a cell variable to export to csv
       subdata{end+1, 1} = str2num(subdirs{s}(1,5:end));
       subdata{end, 2} = t;
       subdata{end, 3} = responses(t);
       subdata{end, 4} = conditions(t, :);
       subdata{end, 5} = cue(t, :);
       % Add bool for shock to make it easy later
       if strcmp(conditions(t,:), 'CSplusSI')
           subdata{end, 7} = 1;
       else
          subdata{end, 7} = 0;
       end
    end


    % In a structure
    substruct.sub = str2num(subdirs{s}(1,5:end));
    substruct.trials = [subdata{2:end, 2}];
    substruct.response = [subdata{2:end, 3}];
    substruct.condition = {subdata{2:end, 4}};
    substruct.cue = {subdata{2:end, 5}};
    substruct.shock = [subdata{2:end, 7}];

    % Recode cue as numeric in alphabetical order
    cues = unique(substruct.cue);

    for c = 1:length(substruct.cue)
        substruct.cuenum(c) = strmatch(substruct.cue{c}, cues);
        subdata{c+1, 6} = substruct.cuenum(c);
    end


    % Append to data cell variable
    if s == 1 % Keep header for first
    alldatacsv = [alldatacsv ; subdata(1:end, :)];
    else
    alldatacsv = [alldatacsv ; subdata(2:end, :)];
    end
    data{s} =  substruct;

end

% Save files

cell2csv(fullfile(scrpath, 'task-fearcond_scr_glmresp_processed.csv'), alldatacsv)

outpath = fullfile(scrpath, 'task-fearcond_scr_glmresp_processed.csv');
save(fullfile(scrpath, 'task-fearcond_scr_glmresp_processed'), 'data')

disp(['All data sucessfuly imported and saved to ' scrpath])
end


function cell2csv(fileName, cellArray, separator, excelYear, decimal)
% % Writes cell array content into a *.csv file.
% %
% % CELL2CSV(fileName, cellArray[, separator, excelYear, decimal])
% %
% % fileName     = Name of the file to save. [ e.g. 'text.csv' ]
% % cellArray    = Name of the Cell Array where the data is in
% %
% % optional:
% % separator    = sign separating the values (default = ',')
% % excelYear    = depending on the Excel version, the cells are put into
% %                quotes before they are written to the file. The separator
% %                is set to semicolon (;)  (default = 1997 which does not change separator to semicolon ;)
% % decimal      = defines the decimal separator (default = '.')
% %
% %         by Sylvain Fiedler, KA, 2004
% % updated by Sylvain Fiedler, Metz, 06
% % fixed the logical-bug, Kaiserslautern, 06/2008, S.Fiedler
% % added the choice of decimal separator, 11/2010, S.Fiedler
% % modfiedy and optimized by Jerry Zhu, June, 2014, jerryzhujian9@gmail.com
% % now works with empty cells, numeric, char, string, row vector, and logical cells.
% % row vector such as [1 2 3] will be separated by two spaces, that is "1  2  3"
% % One array can contain all of them, but only one value per cell.
% % 2x times faster than Sylvain's codes (8.8s vs. 17.2s):
% % tic;C={'te','tm';5,[1,2];true,{}};C=repmat(C,[10000,1]);cell2csv([datestr(now,'MMSS') '.csv'],C);toc;
%% Checking for optional Variables
if ~exist('separator', 'var')
    separator = ',';
end
if ~exist('excelYear', 'var')
    excelYear = 1997;
end
if ~exist('decimal', 'var')
    decimal = '.';
end
%% Setting separator for newer excelYears
if excelYear > 2000
    separator = ';';
end
% convert cell
cellArray = cellfun(@StringX, cellArray, 'UniformOutput', false);
%% Write file
datei = fopen(fileName, 'w');
[nrows,ncols] = size(cellArray);
for row = 1:nrows
    fprintf(datei,[sprintf(['%s' separator],cellArray{row,1:ncols-1}) cellArray{row,ncols} '\n']);
end
% Closing file
fclose(datei);
% sub-function
function x = StringX(x)
    % If zero, then empty cell
    if isempty(x)
        x = '';
    % If numeric -> String, e.g. 1, [1 2]
    elseif isnumeric(x) && isrow(x)
        x = num2str(x);
        if decimal ~= '.'
            x = strrep(x, '.', decimal);
        end
    % If logical -> 'true' or 'false'
    elseif islogical(x)
        if x == 1
            x = 'TRUE';
        else
            x = 'FALSE';
        end
    % If matrix array -> a1 a2 a3. e.g. [1 2 3]
    % also catch string or char here
    elseif isrow(x) && ~iscell(x)
        x = num2str(x);
    % everthing else, such as [1;2], {1}
    else
        x = 'NA';
    end
    % If newer version of Excel -> Quotes 4 Strings
    if excelYear > 2000
        x = ['"' x '"'];
    end
end % end sub-function
end % end function
