% fearcond_multilevel_mediation(X, Y, M, varagin)%

% output:
% -------------------------------------------------------------------------
% outputs from the canlab mediation toolbox with plots and data
%

% Required input
% -------------------------------------------------------------------------
% 'X'               Name of the column in the data file to use as the X var
% (string)
% 'Y'               Name of the column in the data file to use as the Y var
% (string)
% 'M'               Name of the column in the data file to use as the
% Mediator (string)
%
% Optional input
% -------------------------------------------------------------------------
% 'LM2'             Name(s) of the column(s) to use as 2nd level moderator
% (cellstring)

function run_all()

run_mediation('da_z', 'n2p2_z', 'nfr_auc_z')
run_mediation('da_z', 'gamma_z', 'nfr_auc_z')
run_mediation('da_z', 'ratings_z', 'nfr_auc_z')
exit;
end

function run_mediation(X, Y, M, varagin)

% If more than 3 var in, 4th is a cell of second level moderators
if nargin > 3
   L2M = varagin;
else
   L2M = {};
end

% Data path

p.datadir =  '/media/mp/Crucial X8/2023_painlerarning_validate_R2Rpain/derivatives/statistics/pain_responses';



%Load tabled data
data = readtable(fullfile(p.datadir, 'pain_responses_medata.csv'));
% Mark bad trials
data.n2p2_z(data.good_eeg == 0) = NaN;
data.tf(data.good_eeg == 0) = NaN;

% Get sub numbers
subs = unique(data.sub);

% Get data using string input
Xin = eval(['data.' X]); Yin = eval(['data.' Y]); Min = eval(['data.' M]); 

if ~isempty(L2M)
    for l = 1:length(L2M)
      l2 = L2M{l};
      L2Min{l} = eval(['data.' L2M{l}]);
    end

end

% Put each participant's timeseries in cells for mediation toolbox
[Xm, Ym, Mm] = deal(cell(length(subs), 1));
[L2m] = deal(zeros(length(subs), length(L2M)));
keep = ones(1, length(subs));

for s = 1:length(subs)
  
    % Get data for this sub
    Xm{s} = Xin(strcmp(data.sub, subs(s)));
    Ym{s} = Yin(strcmp(data.sub, subs(s)));
    Mm{s} = Min(strcmp(data.sub, subs(s)));
    
    for l = 1:length(L2M)
       vals = L2Min{l}(strcmp(data.sub, subs(s)));
       L2m(s, l) = vals(1);
       % Remove sub with missing values on moderator
       if isnan(vals(1))
          keep(s) = 0;
       else
          keep(s) = 1;
       end
    end
    
    % keep only non nan on all vars
    idx = find((isnan(Xm{s}) + isnan(Ym{s}) + isnan(Mm{s})) == 0);

    Xm{s} = Xm{s}(idx);
    Ym{s} = Ym{s}(idx);
    Mm{s} = Mm{s}(idx);

    % Standardize
    Xm{s} = zscore(Xm{s});
    Ym{s} = zscore(Ym{s});
    Mm{s} = zscore(Mm{s});

end

% Remove part with missing values (should not happen)
Xm = Xm(logical(keep));
Ym = Ym(logical(keep));
Mm = Mm(logical(keep));
L2m = L2m(logical(keep), :);

if sum(keep) < length(subs)
    disp('Some participants had missing values and were removed:')
    disp(subs(keep == 0))
    disp(['Running mediation on ' num2str(length(subs) - sum(keep == 0)) ' participants'])
    crashhere
end

% Run the mediation

mkdir(fullfile( p.datadir, 'mediation', ['X_' X '-M_' M '-Y_' Y]))
cd(fullfile( p.datadir, 'mediation', ['X_' X '-M_' M '-Y_' Y]))
[paths, stats] = mediation(Xm', Ym', Mm', 'verbose', 'boot', 'plots',...
                           'bootsamples', 10000, 'names', {X, Y, M},...
                           'doCIs', 'dosave', 'L2M', L2m, 'hierarchical');

save('mediation_results.mat', 'paths', 'stats', 'X', 'Y', 'M', 'Xm', 'Ym', 'Mm')

close all

end
