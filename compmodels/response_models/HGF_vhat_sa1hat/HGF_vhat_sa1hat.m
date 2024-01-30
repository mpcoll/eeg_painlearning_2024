function [logp, yhat, res] = HGF_vhat_sa1hat(r, infStates, ptrans)
% Calculates the log-probability of log-reaction times y (in units of log-ms) according to the
% linear log-RT model developed with Louise Marshall and Sven Bestmann
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2014-2016 Christoph Mathys, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Transform parameters to their native space
be0  = ptrans(1);
be1  = ptrans(2);
be2  = ptrans(3);
ze   = exp(ptrans(4));

% Initialize returned log-probabilities, predictions,
% and residuals as NaNs so that NaN is returned for all
% irregualar trials
n = size(infStates,1);
logp = NaN(n,1);
yhat = NaN(n,1);
res  = NaN(n,1);

% Weed irregular trials out from responses and inputs
y = r.y(:,1);
y(r.irr) = [];

u = r.u(:,1);
u(r.irr) = [];


% Extract trajectories of interest from infStates
mu1hat = infStates(:,1,1);
sa1hat = infStates(:,1,2);
sa2hat = infStates(:,2,2);
sa3hat = infStates(:,3,2);

mu2    = infStates(:,2,3);
mu3    = infStates(:,3,3);

sa2    = infStates(:,2,4);
sa3    = infStates(:,3,4);

% % Expectation
% % ~~~~~~~~
m1hreg = mu1hat;
m1hreg(r.irr) = [];

% Irreduceble uncertainty
% sa1hat : uncertainty of predictions at the first level on trial k. Because beliefs at the
% first level take the form of a Bernoulli distribution. Intuitively, this form of
% uncertainty represents an individual�s estimate of the entropy of the environment
% at that moment in time; that is, how surprising they expect things to be. We refer to
% it as irreducible uncertainty.
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bernv = mu1hat.*(1-mu1hat);
bernv(r.irr) = [];

% Estimation uncertainty (aka informational uncertainty,
% ambiguity) - sa2hat this is a form of informational uncertainty on trial k, representing lack of
% knowledge about the current stimulus:outcome relationship. Over time and in a
% stable environment, this uncertainty would fall to zero as the probabilities
% underlying the task are learned. In volatile environments, however, this is not the
% case. In the HGF, this form of uncertainty is approximately equivalent to a timevarying
% learning rate, used to update beliefs quickly when they are uncertain and
% slowly when they are supported by plentiful evidence. We refer to it as estimation
% uncertainty.
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
inferv = tapas_sgm(mu2, 1).*(1 -tapas_sgm(mu2, 1)).*sa2; % transform down to 1st level
inferv(r.irr) = [];

% Volatility uncertainty (sa3hat) This can also be considered a form of estimation uncertainty, this time over
% the volatility of the environment at trial k. Again, it controls the speed of learning
% about volatility, weighting prediction errors from the probability space at the
% second level. We refer to it as volatility uncertainty.
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% pv = tapas_sgm(mu2, 1).*(1-tapas_sgm(mu2, 1)).*exp(mu3); % transform down to 1st level
% pv(r.irr) = [];
pv = tapas_sgm(mu2, 1).*(1-tapas_sgm(mu2, 1)).*sa3; % transform down to 1st level
pv(r.irr) = [];


% Calculate predicted scr
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
predscr = be0 + be1.* m1hreg + be2.* bernv;

% Calculate log-probabilities for non-irregular trials
% Note: 8*atan(1) == 2*pi (this is used to guard against
% errors resulting from having used pi as a variable).
reg = ~ismember(1:n,r.irr);
logp(reg) = -1/2 .* log(8 * atan(1) .* ze) - (y - predscr).^2 ./ (2 .* ze);
yhat(reg) = predscr;
res(reg) = y - predscr;

return;
