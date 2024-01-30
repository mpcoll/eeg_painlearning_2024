function [traj, infStates] = PH_cuespec(r, p, varargin)
% NULL MODEL 1 if CS+, 0 if CS-

% Transform paramaters back to their native space if needed
if ~isempty(varargin) && strcmp(varargin{1},'trans')
    p = null_binary_transp(r, p);
end


% Store in data structure
traj.state = r.u(:,2);

% Create matrix (in this case: vector) needed by observation model
infStates = [traj.state];

return;
