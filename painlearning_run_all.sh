#!/bin/bash

# REQUIREMENTS
# Python environment created from requirements.txt
# R environment with lme4

# Matlab with search path including:
# PSPM
# VBA toolbox
# HGF toolbox
# Mediation toolbox + CanlabCore

shopt -s expand_aliases
alias matlab=/usr/local/MATLAB/R2020a/bin/matlab


# SCR
## Prepare data
python scr/scr_prepare_pspm.py
## Process with PSPM
matlab -r "run scr/scr_glm_pspm.m"

# Preprocess EEG
python eeg/eeg_import_clean.py || exit

## ERP
python eeg/eeg_erps.py || exit

## TFR
python eeg/eeg_tfr.py || exit

# EMG
python emg/emg_nfr.py

# Computational models
## Simulation for HGF priors
matlab -r "run compmodels/comp_fitmodels_HGFsim.m"
## Fit all models
matlab -r "run compmodels/comp_fitmodels.m"
## Compare models
matlab  -r "run compmodels/comp_compare_families.m"
matlab  -r "run compmodels/comp_compare_intercue.m"


# Collect all data in a frame
python collect_all_data.py || exit

# Stats
python stats/stats_erps_modelbased.py || exit
python stats/stats_erps_modelfree.py || exit
python stats/stats_tfr_modelfree.py || exit
python stats/stats_tfr_modelbased.py || exit
python stats/stats_pain_responses.py || exit
Rscript stats/stats_scr_lmer.R
# Multilevel mediation
matlab -r "run stats/stats_multilevel_mediation.m"

# Figures
python figures/figure_compmodels.py || exit
python figures/figure_erps_modelbased.py || exit
python figures/figure_erps_modelfree.py || exit
python figures/figure_painresponses.py || exit
python figures/figure_tfr_modelfree.py || exit
python figures/figure_tfr_modelbased.py || exit
