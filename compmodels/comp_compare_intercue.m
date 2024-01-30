function comp_compare_intercue()
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Paths
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;clear;

% Specify paths
% Paths for toolboxes
p.HGF_path = 'Documents/MATLAB/HGF';
p.VBA_path = 'Documents/MATLAB/VBA-toolbox-master';

% Custom tapas models path
p.customfuncpath = ['code/compmodels'];
% Where is the data
p.datafile =  ['derivatives/task-fearcond_scr_glmresp_processed.mat'];
% Ouput path
m.path = 'derivatives/computational_models';
if ~exist(m.path, 'dir')
   mkdir(m.path)
end

% Add toolbox and functions to path
addpath(p.HGF_path);
addpath(genpath(p.VBA_path));
addpath(genpath(p.customfuncpath));

% Name to give to comparison file
p.comparison_name = 'comp_intercue_';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Models to run using names below or 'all'
p.to_run =  {};


% Models to compare using VBA (if empty compares model that were ran)
% Use this to compare models without running any
p.to_compare =  {'null_binary',...
                 'RW_intercue', 'PH_intercue_vhat_assoc',...
                 'HGF2_intercue_vhat_sa1hat'};
% Families (IN ORDER OF APPEARANCE BELOW)
p.comp_families = {};


% General model parameters
m.optim = 'tapas_quasinewton_optim_config'; % Optimisation function

% Exclude participants
p.substoremove = {31, 35, 42, 55};

% Design
m.ignoreshocks = 1; % Ignore the shock trials in the response model
m.removeoutliers = 3; % Remove outliers in response data (Z score threshold)


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prepare data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load data
d = load(p.datafile);
m.data = d.data;

% Run options
debug = 0; % Just run a couple of participants to test
if debug == 1
    m.data = m.data(1:3);
end

% Exclude participants
keep = ones(1, length(m.data));
for i = 1:length(m.data)
    for rems = 1:length(p.substoremove)
        if strcmp(num2str(cell2mat(p.substoremove(rems))), num2str(m.data{i}.sub))
            keep(i) = 0;
        end
    end
end

m.data = m.data(logical(keep));

% Add cspplus for null model
for s = 1:length(m.data)
    m.data{s}.csplus = zeros(length(m.data{s}.trials), 1);
    m.data{s}.csplus([strmatch('CSplusSI', m.data{s}.condition); strmatch('CSplus ', m.data{s}.condition)]) = 1;
end

% Array to collect LME for each model
L_vba = [];
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODELS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
models = {}; % Init empty cell
m.simulate = 0;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Null models
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m.HGF=0; % Not an HGF
m.name = 'null_binary'; % Name to save under
m.prc_model = 'null_binary_config'; % Perceptual model function
m.resp_model = 'resp_RW_vhat_config'; % Response model function
m.tolineplot = {'state'}; % Parameters to plot
m.tolineplot_labels = {'cspplus'}; % Labels for line plots
m.resp_param = {'be0', 'be1', 'ze'}; % Response paramters
m.perc_param = {}; % Perceptual paramters
m.u = "[m.data{s}.shock', m.data{s}.csplus]";  % Data input
models{end+1} = m;


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % RW no intercue
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simplest model
m.HGF=0; % Not an HGF
m.name = 'RW_nointercue'; % Name to save under
m.prc_model = 'RW_nointercue_config'; % Perceptual model function
m.resp_model = 'resp_RW_vhat_config'; % Response model function
m.tolineplot = {'vhat'}; % Parameters to plot
m.tolineplot_labels = {'Expectation'}; % Labels for line plots
m.resp_param = {'be0', 'be1', 'ze'}; % Response paramters
m.perc_param = {'v_0', 'al'}; % Perceptual paramters
m.u = "[m.data{s}.shock', m.data{s}.cuenum']";  % Data input
models{end+1} = m;


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % RW intercue
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m.HGF=0; % Not an HGF
m.name = 'RW_intercue'; % Name to save under
m.prc_model = 'RW_intercue_config'; % Perceptual model function
m.resp_model = 'resp_RW_vhat_config'; % Response model function
m.tolineplot = {'vhat'}; % Parameters to plot
m.tolineplot_labels = {'Expectation'}; % Labels for line plots
m.resp_param = {'be0', 'be1', 'ze'}; % Response paramters
m.perc_param = {'v_0', 'al'}; % Perceptual paramters
m.u = "[m.data{s}.shock', m.data{s}.cuenum']";  % Data input
models{end+1} = m;


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % PH no intercue
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m.HGF = 0;
m.name = 'PH_nointercue';
m.prc_model = 'PH_nointercue_config';
m.resp_model = 'resp_PH_vhat_assoc_config';
m.tolineplot = {'vhat', 'a'};
m.tolineplot_labels = {'Expectation', 'Associativity'};
m.resp_param = {'be0', 'be1', 'be2', 'ze'};
m.perc_param = {'v_0', 'al', 'a_0', 'ga'};
m.u = "[m.data{s}.shock', m.data{s}.cuenum']";
models{end+1} = m;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PH intercue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m.HGF = 0;
m.name = 'PH_intercue_vhat_assoc';
m.prc_model = 'PH_intercue_config';
m.resp_model = 'resp_PH_vhat_assoc_config';
m.tolineplot = {'vhat', 'a', 'da'};
m.tolineplot_labels = {'Expectation', 'Associativity', 'Prediction error'};
m.resp_param = {'be0', 'be1', 'be2', 'ze'};
m.perc_param = {'v_0', 'al', 'a_0', 'ga'};
m.u = "[m.data{s}.shock', m.data{s}.cuenum']";
models{end+1} = m;



% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %2 levels HGF global options
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m.HGF = 1;
m.hgflevels = 2;
m.simulate = 0;
m.tolineplot = {'vhat', 'sa1hat', 'sa2hat', 'da1', 'da2'};
m.tolineplot_labels = {'Expectation (vhat)', ...
    'Irreducible uncertainty (sa1hat)', ...
    'Estimation uncertainty (sa2hat)',...
    'Prediction error lv1 (da1)', ...
    'Prediction error lv2 (da2)'};
m.sim_prc_model = 'HGF_2levels_config_sim'; % Model for simulation
m.prc_model = 'HGF_2levels_intercue_config';
m.perc_param = {'om2'};
m.u = "[m.data{s}.shock', m.data{s}.cuenum']";


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2 levels HGF no interecue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2 levels, uncertainty only, 1 predictor no intercue

%___________________________________________________
% Simulation
% m.name = 'HGF2_nointercue_sim';
% m.prc_model = 'HGF_2levels_config_sim';
% m.resp_model = 'tapas_bayes_optimal_binary_config';
% m.simulate = 1;
% models{end+1} = m;

m.name = 'HGF2_nointercue_vhat_sa1hat';
m.prc_model = 'HGF_2levels_config';
m.resp_model = 'HGF_vhat_sa1hat_config';
m.resp_param = {'be0', 'be1', 'be2', 'ze'};
m.simulate = 0;
models{end+1} = m;


% %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 2 levels HGF intercue
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulation
% m.name = 'HGF2_intercue_sim';
% m.prc_model = 'HGF_2levels_intercue_config_sim';
% m.resp_model = 'tapas_bayes_optimal_binary_config';
% m.simulate = 1;
% models{end+1} = m;

m.name = 'HGF2_intercue_vhat_sa1hat';
m.prc_model = 'HGF_2levels_intercue_config';
m.resp_model = 'HGF_vhat_sa1hat_config';
m.resp_param = {'be0', 'be1', 'be2', 'ze'};
m.simulate = 0;
models{end+1} = m;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run/Compare selected models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run

names = {};
for mod = 1:length(models) % Loop all models
    names{end+1} = models{mod}.name;
end

% Run models
if sum(strcmp(p.to_run, 'all'))

    for mod = 1:length(models) % Loop all models
        % Run if in torun or 'all
        m = fearcond_fit_HGF(models{mod});
        % Keep LME if not simulation
        if ~m.simulate
            L_vba = [L_vba, m.LMEs];
        end
    end
end

if ~isempty(p.to_run)
    for n=1:length(p.to_run)
        m = fearcond_fit_HGF(models{strcmp(p.to_run{n}, names)});
    end
    if ~m.simulate
        L_vba = [L_vba, m.LMEs];
    end
end

% Compare if more than 2 models
if size(L_vba, 2) > 1
    if sum(strcmp(p.to_run, 'all'))
        fearcond_compare_models(L_vba', m.path, names, p.comp_families, p.comparison_name)
    else
        fearcond_compare_models(L_vba', m.path, p.to_run, p.comp_families, p.comparison_name)
    end
end


% Compare without running
if ~isempty(p.to_compare)

    if sum(strcmp(p.to_compare, 'all'))
        compnames = {};
        for mod = 1:length(models)
            load(fullfile(m.path, models{mod}.name, [models{mod}.name '_data']))
            L_vba = [L_vba, m.LMEs];
            compnames{end+1} = models{mod}.name;
        end
    else

        compnames = p.to_compare;
        for mod=1:length(p.to_compare)
            n = strcmp(p.to_compare{mod}, names);
            load(fullfile(m.path, models{n}.name, [models{n}.name '_data']))
            L_vba = [L_vba, m.LMEs];
        end

    end

    fearcond_compare_models(L_vba', m.path, compnames, p.comp_families, p.comparison_name)

end

exit;

end

%%
function m = fearcond_fit_HGF(m)

% Make output dir
outpath = fullfile(m.path, m.name);
if ~exist(outpath, 'dir')
    mkdir(outpath)
end

% Fit model for each subject
m.subfit = {};

% Collect LMEs, AICs, BICs, in a single column for model comparison
m.LMEs = nan(length(m.data), 1);
m.BICs = nan(length(m.data), 1);
m.AICs = nan(length(m.data), 1);

% Loop subject
for s = 1:length(m.data)

    % Extract inputs and response
    if isfield(m, 'u')
        u = eval(m.u);  % If special input;
    else
        u = m.data{s}.shock';
    end

    y = m.data{s}.response';


    tic

    % Ignore shock trials in response function
    if m.ignoreshocks
        y(u(:,1) == 1) = nan;
    end


    if m.removeoutliers ~= 0
        z = abs((y-nanmean(y))/nanstd(y));
        y(z > m.removeoutliers) = nan;
        %m.data{s}.response = y';
        m.data{s}.nremoved = sum(z > m.removeoutliers);
    else
        m.data{s}.nremoved = 0;
    end

    if ~m.simulate % IF not simulation
        disp(['Fitting model ' m.name ' for sub ' num2str(s) ' out of '...
            num2str(length(m.data))])
        m.subfit{s} = tapas_fitModel(y,...
            u,...  % Inputs
            m.prc_model,... % Perceptual model
            m.resp_model, ... % Response model
            m.optim);

        % GEt model fits
        SSE = nansum(m.subfit{s}.optim.res.^2);
        m.subfit{s}.optim.SSE = SSE;
        m.LMEs(s) = m.subfit{s}.optim.LME;
        m.BICs(s) = m.subfit{s}.optim.BIC;
        m.AICs(s) = m.subfit{s}.optim.AIC;

        %         clc;
        %         disp(m.LMEs(:))

    else %Estimate priors for HGF
        disp(['Simulating model ' m.name ' for sub ' num2str(s) ' out of '...
            num2str(length(m.data))])
        sim{s} = tapas_fitModel([],...
            u,...  % Inputs
            m.prc_model,...
            'tapas_bayes_optimal_binary_config', ...
            m.optim);
        minparam(s, :) = sim{s}.optim.argMin;
        clc;
        %         disp(minparam(:, :))
    end
    toc


end


if m.simulate
    disp('Average parameters for bayes ideal observer')
    mean(minparam)
    m.sim_average_min_params = mean(minparam);
    m.sim_min_params = minparam;
    m.sim = sim;
    tapas_bayesian_parameter_average(sim{:})
    save(fullfile(m.path, m.name, [m.name '_data']), 'm')

else
    % Bayesian averaging across participants
    % m.bpa = tapas_bayesian_parameter_average(m.subfit{:});
    % Save model
    save(fullfile(m.path, m.name, [m.name '_data']), 'm')

    % Collect data in a table
    m = HGF_data2table(m);

    % Save model
    save(fullfile(m.path, m.name, [m.name '_data']), 'm')

end

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Put data in table
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function m = HGF_data2table(m)
allsubs = [];
allcond = {};

for s = 1:length(m.data)
    % Data
    sub = repmat(m.data{s}.sub, length(m.data{s}.response), 1);

    % Get predicted response

    pred = m.subfit{s}.optim.yhat;
    % get data
    trial = (1:length(pred))';
    scr = m.data{s}.response';
    cue = m.data{s}.cuenum';
    cond = m.data{s}.condition';

    % Add block number
    block = [];
    for b = 1:7
        if b == 1
            block = [block; repmat(b, 36, 1)];
        else
            block = [block; repmat(b, 72, 1)];
        end
    end

    % Rename conditions and add double conditions for plots
    cond2 = cell(max(trial), 1);
    cond_plot = cell(max(trial), 1);
    cond_plot2 = cell(max(trial), 1);

    startcues = unique(cue(1:36));

    for c = 1:length(cond)

        switch cond{c}
            case 'CSminus '
                cond2{c} = 'CS-1';
                cond_plot{c} = ['CS-1_' num2str(block(c)) ...
                    '/CS-2_' num2str(block(c)+1) '_' num2str(find(cue(c) == startcues))];
                cond_plot2{c} = cond_plot{c};
            case 'CSminus2'
                cond2{c} = 'CS-2';
                if block(c) == 2
                    cond_plot{c} = ['CS-1_' num2str(block(c)-1) ...
                        '/CS-2_' num2str(block(c)) '_' num2str(find(cue(c) == startcues))];
                else
                    cond_plot{c} = ['CS-1_' num2str(block(c)-1) ...
                        '/CS-2_' num2str(block(c))];
                end
                cond_plot2{c} = cond_plot{c};

            case 'CSnaif1 '
                cond2{c} = 'CS-1';
                cond_plot{c} = ['CS-1_' num2str(block(c)) ...
                    '/CS-2_' num2str(block(c)+1)];
                cond_plot2{c} = cond_plot{c};
            case 'CSnaif2 '
                cond2{c} = 'CS-2';
                cond_plot{c} = ['CS-1_' num2str(block(c)-1) ...
                    '/CS-2_' num2str(block(c))];
                cond_plot2{c} = cond_plot{c};
            case 'CSplus  '
                cond2{c} = 'CS+';
                cond_plot{c} = ['CS+_' num2str(block(c)) ...
                    '/CS-E_' num2str(block(c)+1)];
                cond_plot2{c} = cond_plot{c};
            case 'CSplusSI'
                cond2{c} = 'CS++';
                cond_plot{c} = ['CS++_' num2str(block(c)) ...
                    '/CS-E_' num2str(block(c)+1)];
                cond_plot2{c} = ['CS+_' num2str(block(c)) ...
                    '/CS-E_' num2str(block(c)+1)];
%             case 'CSplusSI'
%                 cond2{c} = 'CS+';
%                 cond_plot{c} = ['CS+_' num2str(block(c)) ...
%                     '/CS-E_' num2str(block(c)+1)];
            case 'CSeteint'
                cond2{c} = 'CS-E';
                cond_plot{c} = ['CS+_' num2str(block(c)-1) ...
                    '/CS-E_' num2str(block(c))];
                cond_plot2{c} = cond_plot{c};

        end
    end

    ucond_plot = unique(cond_plot);
    count = ones(length(ucond_plot), 1);
    trial_within = cell(max(trial), 1);
    trial_within_wb = cell(max(trial), 1);

    for i = 1:length(cond_plot)

        % Reset count for each block
        if i > 1 && block(i-1) ~= block(i)
            count = ones(length(ucond_plot), 1);
        end

        % Where is it
        trial_within{i} = count(strmatch(cond_plot{i}, ucond_plot));

        % update
        count(strmatch(cond_plot{i}, ucond_plot)) = count(strmatch(cond_plot{i}, ucond_plot)) + 1;

        switch block(i)
            case 1
                trial_within_wb{i} = trial_within{i};
            otherwise
                trial_within_wb{i} = trial_within{i} + (18*(block(i)-1));
        end

    end

    ucond_plot = unique(cond_plot2);
    count = ones(length(ucond_plot), 1);
    trial_within_wcs = cell(max(trial), 1);
    trial_within_wb_wcs = cell(max(trial), 1);
    for i = 1:length(cond_plot)

        % Reset count for each block
        if i > 1 && block(i-1) ~= block(i)
            count = ones(length(ucond_plot), 1);
        end

        % Where is it
        trial_within_wcs{i} = count(strmatch(cond_plot2{i}, ucond_plot));

        % update
        count(strmatch(cond_plot2{i}, ucond_plot)) = count(strmatch(cond_plot2{i}, ucond_plot)) + 1;

        switch block(i)
            case 1
                trial_within_wb_wcs{i} = trial_within_wcs{i};
            otherwise
                trial_within_wb_wcs{i} = trial_within_wcs{i} + (18*(block(i)-1));
        end

    end


    % Get trialwise estimations for parameters that will be pltted
    m.traj_names = m.tolineplot;
    traj_data = nan(length(pred), length(m.traj_names)-1);

    if m.HGF
        % IF HGF, extract cue trajectory

        mu1hat = m.subfit{s}.traj.muhat(:, 1);
        sa2 = m.subfit{s}.traj.sa(:, 2);
        mu2 = m.subfit{s}.traj.mu(:, 2);

        m.subfit{s}.traj.mu1hat = mu1hat;
        m.subfit{s}.traj.vhat = mu1hat; % Duplicate for plots
        m.subfit{s}.traj.sa1hat = mu1hat.*(1-mu1hat);
        m.subfit{s}.traj.sa2hat = tapas_sgm(mu2, 1).*(1 -tapas_sgm(mu2, 1)).*sa2;
        m.subfit{s}.traj.da1 = m.subfit{s}.traj.da(:, 1);
        m.subfit{s}.traj.da2 = m.subfit{s}.traj.da(:, 2);

        for p = 1:length(m.traj_names)
            traj_data(:, p) = m.subfit{s}.traj.(m.traj_names{p});
        end

    else% Non HGF
        for p = 1:length(m.traj_names)
            traj_data(:, p) = m.subfit{s}.traj.(m.traj_names{p});
        end
    end


    % Get all parameters value
    % Perceptual parameters

    if ~m.HGF
        m.perc_param = fieldnames(m.subfit{s}.p_prc);
        m.perc_param = m.perc_param(1:end-2); % Remove summary
        perc_data = nan(length(pred), length(m.perc_param));
        for p = 1:length(m.perc_param)
            perc_data(:, p) = repmat(m.subfit{s}.p_prc.(m.perc_param{p}), length(pred), 1);
        end
    else % For HGF
        alllevels = {};
        alldata = [];
        m.perc_param = fieldnames(m.subfit{s}.p_prc);
        m.perc_param = m.perc_param(1:end-2); % Remove summary
        for p = 1:length(m.perc_param)
            for l = 1:length(m.subfit{s}.p_prc.(m.perc_param{p}))
                alllevels = [alllevels, {[m.perc_param{p}, '_' num2str(l)]}];
                alldata = [alldata, m.subfit{s}.p_prc.(m.perc_param{p})(l)];
            end
        end

        m.perc_param = alllevels';
        perc_data = nan(length(pred), length(m.perc_param));
        for p = 1:length(m.perc_param)
            perc_data(:, p) = repmat(alldata(p), length(pred), 1);
        end

    end

    % Response parameters
    m.resp_param = fieldnames(m.subfit{s}.p_obs);
    m.resp_param = m.resp_param(1:end-2); % Remove summary
    resp_data = nan(length(pred), length(m.resp_param));
    for p = 1:length(m.resp_param)
        resp_data(:, p) = repmat(m.subfit{s}.p_obs.(m.resp_param{p}), length(pred), 1);
    end


    % Get all model fits
    AIC = repmat(m.subfit{s}.optim.AIC, length(pred), 1);
    BIC = repmat(m.subfit{s}.optim.BIC, length(pred), 1);
    LME = repmat(m.subfit{s}.optim.LME, length(pred), 1);
    SSE = repmat(nansum(m.subfit{s}.optim.res.^2), length(pred), 1);
    nremoved = repmat(m.data{s}.nremoved, length(pred), 1);


    % Put all in same array
    allsubs = [allsubs; sub, trial, cue, block, scr, pred, traj_data,...
        perc_data, resp_data, AIC, BIC, LME, SSE, nremoved];
    allcond = [allcond; [cond, cond2, cond_plot, cond_plot2,...
               trial_within, trial_within_wb, ...
               trial_within_wcs, trial_within_wb_wcs]];

end

tablehead = [{'sub', 'trial', 'cue', 'block', 'scr', 'pred'}, m.traj_names,...
    m.perc_param', m.resp_param', {'AIC', 'BIC', 'LME', 'SSE', 'nremoved',...
    'cond_original', 'cond', 'cond_plot', 'cond_plot2', 'trial_within',...
    'trial_within_wb', 'trial_within_wcs', 'trial_within_wb_wcs'}];

tabledata = [num2cell(allsubs), allcond];


m.tabdata = cell2table(tabledata, 'VariableNames', tablehead);
writetable(m.tabdata, fullfile(m.path, m.name, [m.name , '_data.csv']))

end


function fearcond_compare_models(L, path, modnames, families, comparison_name)

% Plots parameters
p.resolution = '-r400';  % Figure resolution

options.modelNames = modnames;
if ~isempty(families)
    options.families = families;
end

[posterior,out] = VBA_groupBMC(L, options) ;
out.pep = (1-out.bor)*out.ep + out.bor/length(out.ep);

print(fullfile(path, 'Model_comparison.png'),...
    '-dpng', p.resolution)
save(fullfile(path, [comparison_name 'VBA_model_comp']), 'posterior', 'out')
close;
figure;
L_plot = L';
boxplot(L_plot,'Labels', modnames)
hold on

for i = 1:length(modnames)
    scatter(ones(size(L_plot, 1), 1).*(1+(rand(size(L_plot, 1), 1)-0.5)/5) + i -1,...
        L_plot(:,i), 100, 'k','filled', 'MarkerFaceAlpha', 0.4);
    hold on
    text(i +0.2 , mean(L_plot(:, i)), num2str(round(mean(L_plot(:, i)), 2)),...
        'FontSize', 20)
    hold on
end

print(fullfile(path, 'Models_LMEs_box.png'),...
    '-dpng', p.resolution)

close;
end
