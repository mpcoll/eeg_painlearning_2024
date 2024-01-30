function [pvec, pstruct] = PH_nointercue_transp(r, ptrans)
% MODIFIED HGF TOOLBOX FUNCTION BY MP COLL, 2019 for FEAR COND EXPERIMENT

% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

pvec    = NaN(1,length(ptrans));
pstruct = struct;

pvec(1)       = tapas_sgm(ptrans(1),1); % v_0
pstruct.v_0   = pvec(1);
pvec(2)       = tapas_sgm(ptrans(2),1); % alpha
pstruct.al    = pvec(2);
pvec(3)       = tapas_sgm(ptrans(3),1); % a_0
pstruct.a_0   = pvec(3);
pvec(4)       = tapas_sgm(ptrans(4),1); % ga
pstruct.g     = pvec(4);


return;
