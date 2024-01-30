import bioread
import convert_eprime as convep
import pandas as pd
import numpy as np
import os
from numpy import argwhere as nwhere
from os.path import join as opj
from scipy import signal
import shutil
import json
import mne

# Where is the raw data
rawpath = '/data/raw'
# Where to create the bids dataset
bidsout = '/data/source'

# Where to create the bids dataset
derivativesout = '/data/derivatives'

if not os.path.exists(bidsout):
    os.mkdir(bidsout)


if not os.path.exists(derivativesout):
    os.mkdir(derivativesout)


# Helperunction to write to json files
def writetojson(outfile, path, content):
    """
    Helper to write a dictionnary to json
    """
    data = os.path.join(path, outfile)
    if not os.path.isfile(data):
        with open(data, 'w') as outfile:
            json.dump(content, outfile)
    else:
        print('File ' + outfile + ' already exists.')


# sub to run
suball = os.listdir(opj(rawpath, 'eeg'))
suball = [s.replace('.bdf', '') for s in suball]
suball = [s.replace('sj', '') for s in suball]


#########################################################################
# Global files
#########################################################################
# Create description files
dataset_description = {"Name": "Fear conditioning EEG",
                       "BIDSVersion": "1.3.0",
                       "Authors": ["MP Coll", "Z Walden", "M Roy",
                                   "P Rainville", "PA Bourgoin"],
                       "EthicsApprovals": "CRIUGM ethics committee",
                       }

writetojson('dataset_description.json', bidsout, dataset_description)

# Participants file
partvar = {"age": {"Description": "age in years nan if not reported)"},
           "ismale": {"Description": "1 if male nan if not reported)"},
           "bdi_total": {"Description": "total score on the Beck Depression Inventory-II (BDI)"},
           "bpi_pss": {"Description": "score on Brief Pain Inventory (BPI) severity scale"},
           "bpi_pint": {"Description":
                        "score on BPI pain interfercence scale"},
           "pcs_rum": {"Description": "score on rumination scale Pain catastrophizing Scale (PCS)"},
           "pcs_mag": {"Description": "score on magnitude scale PCS"},
           "pcs_hel": {"Description": "score on helplessness scale PCS"},
           "pcs_tot": {"Description": "total score on PCS"},
           "stait_state": {"Description": "score on State Trait Anxiety Inventoy(STAI)-state"},
           "stait_trait": {"Description": "score on STAI-trait"},
           "ffm_obs": {"Description":
                       "score on the observing scale of Five-Facet mindfulness scale (FFM)"},
           "ffmq_desc": {"Description":
                         "score on the describing scale of FFM"},
           "ffmq_aware": {"Description":
                          "score on the observation scale of FFM"},
           "ffmq_njudg": {"Description":
                          "score on the observation scale of FFM"},
           "ffmq_nonrea": {"Description":
                           "score on the observation scale of FFM"},
           "ffmq_total": {"Description":
                          "total score on FFM"},
           }

writetojson('participants.json', bidsout, partvar)

# Load questionnaire and socio files
part = pd.read_csv(opj(rawpath, 'questionnaires',
                       'fearcond_allquestionnaires.csv'), sep='\t')

part = part.rename(columns={'part': 'participant_id'})
part_id = ['sub-' + str(p) for p in part.participant_id]
part['participant_id'] = part_id
part.to_csv(opj(bidsout, 'participants.tsv'), sep='\t', index=False)

#########################################################################
# EEG raw -> EEG bids
#########################################################################

# Task description
fcond = {"TaskName": "Fear conditioning",
         "TaskDescription": """
                             Participants saw faces and some faces were
                             probabilistically paired with an electrical shock.
                             """,
         "Instructions": "XXX",
         "InstitutionName": "CRIUGM",
         "EEGChannelCount": 66,
         "EOGChannelCount": 3,
         "ECGChannelCount": 0,
         "TriggerChannelCount": 1,
         "EEGPlacementScheme": "10 percent system",
         "EEGReference": "FCz",
         "EEGGround": "AFz",
         "SamplingFrequency": [1024, 2048],
         "SoftwareFilters": None,
         "PowerlineFreuqency": 60,
         "Manufacturer": "BioSemi",
         "RecordingType": "continuous"}


def make_chan_file(raw):
    """
    Helper function to create a channel description file from raw eeg data
    """
    types = []
    notes = []
    for c in raw.info['ch_names']:

        if 'EOG' in c:
            types.append('EOG')
            notes.append('')
        elif 'ECG' in c:
            types.append('ECG')
            notes.append('')
        elif c == 'EXG1':
            types.append('EEG')
            notes.append('M1')
        elif c == 'EXG2':
            types.append('EEG')
            notes.append('M2')
        elif c == 'EXG3':
            types.append('EOG')
            notes.append('HEOGL')
        elif c == 'EXG4':
            types.append('EOG')
            notes.append('HEOGR')
        elif c == 'EXG5':
            types.append('EOG')
            notes.append('VEOGL')
        elif c == 'EXG':
            types.append('unused')
            notes.append('')
        elif c == 'Status':
            types.append('triggers')
            notes.append('')
        else:
            types.append('EEG')
            notes.append('')

    chan_desc = pd.DataFrame(data={'name': raw.info['ch_names'], 'type': types,
                                   'units': 'microV', 'notes': notes})
    return chan_desc


def make_event_file(events, raw):
    """
    Helper function to create an event description file from raw data
    """

    # Event dict with code : [name, duration]
    events_dict = {"Description": """
                                  This task had multiple markers for each
                                  event. For each trigger value (key),
                                  the accompanying list describes the:
                                  [main event name, duration,
                                  'detailed trigger name',
                                  'detailed trial info'].
                                  """,
                   254: ['trialendnovas', 0, 'nan', 'trialendnovas'],
                   255: ['shock', 0.03, 'shock', 'shock'],
                   94: ['CS-1', 1, 'CS-1', 'CS-1b1'],
                   95: ['CS-2', 1, 'CS-2', 'CS-2b1'],
                   96: ['CS-1', 1, 'CS-1', 'CS-1'],
                   97: ['CS-2', 1, 'CS-2', 'CS-2'],
                   98: ['CS+', 1, 'CS+', 'CS++'],
                   99: ['CS+', 1, 'CS+', 'CS+'],
                   100: ['CS-E', 1, 'CS-E', 'CS-E']
                   }

    for i in range(1, 73):
        events_dict[i] = ['trial_' + str(i) + '_stimonset', 0,
                          'nan', 'trialflag']

    for i in range(73, 80):
        events_dict[i] = ['block_' + str(i-72), 0, 'nan', 'blockflag']

    for i in range(245, 254):
        events_dict[i] = ['vas_trial_' + str(i-244), 10, 'VAS', 'VAS']

    for i in range(191, 209):
        events_dict[i] = ['cs-2_trial_' + str(i-190), 0, 'nan',
                          'CS-2_trialflag']

    for i in range(209, 227):
        events_dict[i] = ['cs+_trial_' + str(i-208), 0, 'nan',
                          'CS+_trialflag']

    for i in range(173, 191):
        events_dict[i] = ['cs-1_trial_' + str(i-172), 0, 'nan',
                          'CS-1_trialflag']

    for i in range(119, 137):
        events_dict[i] = ['cs-1a_b1_trial_' + str(i-118), 0, 'nan',
                          'CS-1a_trialflag']

    for i in range(101, 119):
        events_dict[i] = ['cs-1b_b1_trial_' + str(i-100), 0, 'nan',
                          'CS-1b_trialflag']

    for i in range(227, 245):
        events_dict[i] = ['cs-e_trial_' + str(i-226), 0, 'nan',
                          'CS-E_trialflag']

    for i in range(80, 94):
        events_dict[i] = ['stimid_' + str(i), 0, 'nan', 'stimulus_id']

    for i in range(137, 152):
        events_dict[i] = ['cs-2b_b2_trial_' + str(i-136), 0, 'nan', 'CS-2a']

    for i in range(155, 170):
        events_dict[i] = ['cs-2a_b2_trial_' + str(i-154), 0, 'nan', 'CS-2b']

    onsets, trial_type, duration, value, sample = [], [], [], [], []
    trial_info, trial_info_first = [], []
    for e in events:
        if e[2] < 1000:  # Some part have weird values at the end
            onsets.append(e[0]/raw.info['sfreq'])
            trial_type.append(events_dict[e[2]][2])
            duration.append(events_dict[e[2]][1])
            trial_info.append(events_dict[e[2]][0])
            trial_info_first.append(events_dict[e[2]][3])
            sample.append(e[0])
            value.append(e[2])

    eventfile = pd.DataFrame(data={'onsets': onsets, 'duration': duration,
                                   'trial_type': trial_type,
                                   'trigger_info': trial_info,
                                   'sample': sample,
                                   'trigger': value,
                                   'trial_info_firstblock': trial_info_first})

    return eventfile, events_dict


# Loop subject and copy/create file
ratings = pd.read_csv('/data/raw/behavioural/painratings.csv')

for sub in suball:

    # Create folder
    outpath = opj(bidsout, 'sub-' + sub)
    if not os.path.exists(outpath):
        os.mkdir(outpath)
        os.mkdir(opj(outpath, 'eeg'))

    # find eegrawfile
    eegrawfile = [s for s in os.listdir(opj(rawpath, 'eeg')) if sub in s][0]
    # Copy the EEG raw file
    shutil.copy(opj(rawpath, 'eeg', eegrawfile),
                opj(outpath, 'eeg', 'sub-' + sub + '_task-fearcond_eeg.bdf'))

    writetojson('sub-' + sub + '_task-fearcond_eeg' + '.json',
                opj(outpath, 'eeg'),
                fcond)

    eegraw = mne.io.read_raw_bdf(opj(rawpath, 'eeg', eegrawfile))

    chanfile = make_chan_file(eegraw)
    chanfile.to_csv(opj(outpath, 'eeg', 'sub-' + sub
                        + '_task-fearcond_channels.tsv'),
                    sep='\t')

    events, events_dict = make_event_file(mne.find_events(eegraw), eegraw)

    ratdata = ratings[sub]

    events['painrating'] = 'nan'
    ratframe = np.asarray(events['painrating'])
    ratframe[events.trial_type == 'shock'] = list(ratdata)
    events['painrating'] = ratframe
    events.to_csv(opj(outpath, 'eeg', 'sub-' + sub
                      + '_task-fearcond_events.tsv'),
                  sep='\t', index=False)

writetojson('task-fearcond_events.json', bidsout, events_dict)


#########################################################################
# Physio data BIOPAC -> BIDS
#########################################################################

for sub in suball:

    outpath = opj(bidsout, 'sub-' + sub)
    derivout = opj(derivativesout, 'sub-' + sub)
    if not os.path.exists(derivout):
        os.mkdir(derivout)
        os.mkdir(opj(derivout, 'scr'))

    data = bioread.read(opj(rawpath, 'physio', 'S' + sub + '_testing.acq'))

    # Get shock trigger onsets in SCR
    scrdat = data.channels[0].data
    emgdat = data.channels[1].data
    rmsemgdat = data.channels[6].data
    trigdat = data.channels[5].data

    # Get # samples /second
    sec = data.channels[0].samples_per_second
    rmsemgsec = data.channels[6].samples_per_second
    trigsec = data.channels[5].samples_per_second
    emgsec = data.channels[1].samples_per_second

    # Correct split files for part 25 that were apparently recorded with
    # a different sampling rate
    if sub == '25':
        data1 = bioread.read(opj(rawpath, 'physio', 'S'
                                 + sub + '_testing.acq'))

        data2 = bioread.read(opj(rawpath, 'physio', 'S'
                                 + sub + '_testing2.acq'))

        print('Resampling scr')
        decim_factor = [int(data2.channels[0].samples_per_second)
                        / int(data1.channels[0].samples_per_second)][0]
        scrdat2 = signal.decimate(data2.channels[0].data,
                                  int(decim_factor))
        scrdat = np.append(data1.channels[0].data, scrdat2)
        sec = data1.channels[0].samples_per_second

        print('Resampling emg')
        decim_factor = [int(data1.channels[1].samples_per_second)
                        / int(data2.channels[1].samples_per_second)][0]
        emgdat1 = signal.decimate(data1.channels[1].data,
                                  int(decim_factor))
        emgdat = np.append(emgdat1, data2.channels[1].data)
        emgsec = data2.channels[1].samples_per_second

        print('Resampling rmsemg')
        decim_factor = [int(data1.channels[6].samples_per_second)
                        / int(data2.channels[6].samples_per_second)][0]
        rmsemgdat1 = signal.decimate(data1.channels[6].data.astype(float),
                                     int(decim_factor))
        rmsemgdat = np.append(rmsemgdat1, data2.channels[6].data)
        rmsemgsec = data2.channels[6].samples_per_second

        print('Resampling triggers')
        decim_factor = [int(data2.channels[5].samples_per_second)
                        / int(data1.channels[5].samples_per_second)][0]
        trigdat2 = signal.decimate(data2.channels[5].data,
                                   int(decim_factor))
        trigdat2 = trigdat2
        trigdat = np.append(data1.channels[5].data, trigdat2)
        trigsec = data1.channels[5].samples_per_second

    # Resample everything to 1000 Hz
    if sec != 1000:
        if not (sec/1000).is_integer():
            up = 25000/sec
            scrdat = signal.resample_poly(scrdat.astype('float'),
                                          up=int(up), down=25)
        else:
            scrdat = signal.decimate(scrdat.astype('float'),
                                     int(sec/1000))
        sec = 1000

    if rmsemgsec != 1000:

        if not (rmsemgsec/1000).is_integer():
            # Upsample and downsample
            up = 25000/rmsemgsec
            rmsemgdat = signal.resample_poly(rmsemgdat.astype('float'),
                                             up=int(up), down=25)
        else:
            rmsemgdat = signal.decimate(rmsemgdat.astype('float'),
                                        int(rmsemgsec/1000))
        # Keep same number of samples as scr
        rmsemgdat = rmsemgdat[:len(scrdat)]
        rmsemgsec = 1000

    if emgsec != 1000:
        if not (emgsec/1000).is_integer():
            up = 25000/emgsec
            emgdat = signal.resample_poly(emgdat.astype('float'),
                                          up=int(up), down=25)
        else:
            emgdat = signal.decimate(emgdat.astype('float'),
                                     int(emgsec/1000))

        emgdat = emgdat[:len(scrdat)]
        emgsec = 1000

    if trigsec != 1000:
        if not (trigsec/1000).is_integer():
            up = 25000/trigsec
            trigdat = signal.resample_poly(trigdat.astype('float'),
                                           up=int(up), down=25)
        else:
            trigdat = signal.decimate(trigdat.astype('float'),
                                      int(trigsec/1000))
        trigdat = trigdat[:len(scrdat)]
        trigsec = 1000

    # Find triggers
    potential_peaks = nwhere(trigdat > 0.5*np.max(trigdat[int(5*sec):]))
    trigloc = [int(potential_peaks[0])]
    for idx, p in enumerate(potential_peaks[1:]):
        if potential_peaks[idx+1] - potential_peaks[idx+1-1] > 5*sec:
            trigloc.append(int(p))

    # Onset in ms
    shock_ons_acq = np.round(np.asarray(trigloc)/(sec/1000))

    # ___________________________________________________________________
    # Load eprime data
    convep.text_to_csv(opj(rawpath, 'eprime', 'Fear3-' + sub + '-1.txt'),
                       out_file=opj(rawpath, 'eprime',
                                    'Fear3-' + sub + '-1.csv'))

    edata = pd.read_csv(opj(rawpath, 'eprime', 'Fear3-' + sub + '-1.csv'))

    # Get shock onsets
    shock_ons_ep = list(edata['Shockdur.OnsetTime'].dropna())

    # Get cue onsets
    cue_ons_ep = list(edata['cue.OnsetTime'].dropna())
    len(cue_ons_ep)
    len(shock_ons_ep)

    # Get cue id
    cue = list(edata['cue'].dropna())
    len(cue)
    cue = [c.replace('.jpg', '') for c in cue]
    cue = cue[0:468]  # Keep only main task

    # Get condition
    condition = list(edata['Procedure'].dropna())
    list(edata.columns)
    condition = condition[1:469]  # Keep only main task

    # Get block
    block = list(edata['BLOC'].dropna())

    # Get Fixations onsets
    fix_ons_ep = list(edata['fix.OnsetTime'].dropna())

    # Get Fixations durations
    fix_dur = list(edata['fix.OnsetToOnsetTime'][1:].dropna())

    # Get rating onsets
    rat_ons_ep = list(edata['Rating.OnsetTime'][1:].dropna())

    # Get rating durations
    rat_dur = list(edata['Rating.OnsetToOnsetTime'][1:].dropna())

    # Get pause onsets
    pause_ons_ep = [[list(edata['pause1.OnsetTime'].dropna())[0]]
                    + [list(edata['pause2.OnsetTime'].dropna())[0]]
                    + [list(edata['pause3.OnsetTime'].dropna())[0]]
                    + [list(edata['pause4.OnsetTime'].dropna())[0]]][0]

    # Get pause durations
    pause_dur = []
    for p in pause_ons_ep:
        diff = [f - p for f in fix_ons_ep]
        diff = min([d for d in diff if d > 0])
        pause_dur.append(int(diff))

    # ___________________________________________________________________
    # Time lock all events to the SCR. First cue == 0
    # Get cue onset for first shock
    fs_cueshock_ep = cue_ons_ep[int(nwhere(np.isclose(cue_ons_ep,
                                                      shock_ons_ep[0]-1000,
                                                      atol=50)))]
    fs_cueshock_acq = shock_ons_acq[0]-1000

    # Time lock onsets to scr data
    cue_ons_acq = (cue_ons_ep - (fs_cueshock_ep - fs_cueshock_acq))
    rat_ons_acq = (rat_ons_ep - (fs_cueshock_ep - fs_cueshock_acq))
    p_ons_acq = (pause_ons_ep - (fs_cueshock_ep - fs_cueshock_acq))
    fix_ons_acq = (fix_ons_ep - (fs_cueshock_ep - fs_cueshock_acq))

    if sub == '37':
        # For this participant, it looks like acqknowledge was paused.
        # Move trials after this pause

        # Find where is the pause
        cue_after_pause = np.argmax(np.diff(cue_ons_ep)) + 1
        cue_time_before_pause = cue_ons_ep[cue_after_pause-1]

        # For trials after pause remove the difference of the difference
        # between the shock onsets
        shock_pos_pause_ep = np.argmax(np.diff(shock_ons_ep))
        shock_dur_pause_ep = np.diff(shock_ons_ep)[shock_pos_pause_ep]
        shock_dur_pause_acq = np.diff(shock_ons_acq)[shock_pos_pause_ep]
        offset_pauses = shock_dur_pause_ep - shock_dur_pause_acq

        # Remove the offset to all trials after pause
        for t in range(cue_after_pause-1, len(cue_ons_ep)):
            cue_ons_acq[t] = cue_ons_acq[t] - offset_pauses

        for t, val in enumerate(rat_ons_ep):
            if val > cue_time_before_pause:
                rat_ons_acq[t] = rat_ons_acq[t] - offset_pauses

        for t, val in enumerate(pause_ons_ep):
            if val > cue_time_before_pause:
                p_ons_acq[t] = p_ons_acq[t] - offset_pauses

        for t, val in enumerate(fix_ons_ep):
            if val > cue_time_before_pause:
                fix_ons_acq[t] = fix_ons_acq[t] - offset_pauses

        # Correct also shock ons_ep to pass sanity check
        for t, val in enumerate(shock_ons_ep):
            if shock_ons_ep[t] > cue_time_before_pause:
                shock_ons_ep[t] = shock_ons_ep[t] - offset_pauses

    if len(shock_ons_acq) > 54:
        # For some participants, there were additional triggers in ACQ
        # Keep only acq triggers that are in EP as well

        # Get EP onsets in ACQ time
        shock_ons_ep2 = shock_ons_ep - (shock_ons_ep[0] - shock_ons_acq[0])

        # Remove the ACQ onsets not in EP
        shock_ons_acq_good = []
        for o in shock_ons_acq:
            diff = shock_ons_ep2 - o
            where = nwhere(abs(diff) < 50)
            if where.size != 0:
                shock_ons_acq_good.append(True)
            else:
                shock_ons_acq_good.append(False)

        shock_ons_acq = shock_ons_acq[shock_ons_acq_good]

    len(shock_ons_acq)
    # ___________________________________________________________________
    # Sanity check: check if difference bettwen eprime and acq onsets match
    tolerance = 50  # flag anything over 50 ms

    if sub != '25':  # Skip 25 cause recording was paused before trigger 50
        diff_check = nwhere(abs(np.diff(shock_ons_acq)
                                - np.diff(shock_ons_ep))
                            > tolerance)

        # Flag if not enough triggers or too far appart
        if diff_check.size != 0:
            raise Exception('Something is wrong in the triggers')
        else:
            trig_flag = 'Triggers look ok'
            print(trig_flag)

    # Resample back to data and make int for pspm
    cue_ons_acq = (cue_ons_acq*sec/1000).astype(int)
    rat_ons_acq = (rat_ons_acq*sec/1000).astype(int)
    p_ons_acq = (p_ons_acq*sec/1000).astype(int)
    fix_ons_acq = (fix_ons_acq*sec/1000).astype(int)
    shock_ons_acq = (shock_ons_acq*sec/1000).astype(int)

    rat_dur = (np.asarray(rat_dur)*sec/1000).astype(int)
    pause_dur = (np.asarray(pause_dur)*sec/1000).astype(int)
    fix_dur = (np.asarray(fix_dur)*sec/1000).astype(int)

    # Put all in a single DF

    # Save physio data to BIDS

    physio = pd.DataFrame(data={'seconds': np.arange(1, len(scrdat)+1)/1000,
                                'sample': np.arange(1, len(scrdat)+1),
                                'scr': scrdat,
                                'emg': emgdat,
                                'rmsemg': rmsemgdat,
                                'trigger': trigdat})

    samp = np.arange(1, len(scrdat)+1)
    events = np.where(np.isin(samp,  cue_ons_acq), 'cue', 'nan')
    events = np.where(np.isin(samp,  rat_ons_acq), 'vas', events)
    events = np.where(np.isin(samp,  p_ons_acq), 'pause', events)
    events = np.where(np.isin(samp,  fix_ons_acq), 'fix', events)
    events = np.where(np.isin(samp,  shock_ons_acq), 'shock', events)
    durations = np.zeros(events.shape)
    durations[np.where(events == 'pause')[0]] = pause_dur
    durations[np.where(events == 'vas')[0]] = rat_dur
    durations[np.where(events == 'fix')[0]] = fix_dur
    durations[np.where(events == 'cue')[0]] = 1000
    durations[np.where(events == 'shock')[0]] = 30
    cond_type = pd.Series(['nan'] * len(events))
    cond_type.shape
    cond_type[np.where(events == 'cue')[0]] = condition
    physio['events'] = events
    physio['duration'] = durations/1000
    physio['duration_samp'] = durations
    physio['condition'] = cond_type

    # Save to CSV
    physio.to_csv(opj(outpath, 'eeg', 'sub-' + sub
                      + '_task-fearcond_physio.tsv'),
                  sep='\t', index=False)

    #########################################################################
    # Plots and save data as .mat to make it easier to process in the PSMP
    # toolbox
    ##########################################################################

    # Average raw SCR for all conditions
    # Create epochs
    # Combine metadata in pandas
    metedat = pd.DataFrame({'trial': list(range(1, 469)),
                            'cue': cue,
                            'block': block,
                            'condition': condition})

    # Rename conditions
    metedat['condition2'] = metedat['condition'].copy()
    metedat['condition2'] = np.where(metedat['condition2'] == 'CSminus',
                                     'CS-1', metedat['condition2'])
    metedat['condition2'] = np.where(metedat['condition2'] == 'CSminus2',
                                     'CS-2', metedat['condition2'])
    metedat['condition2'] = np.where(metedat['condition2'] == 'CSnaif1',
                                     'CSa-1', metedat['condition2'])
    metedat['condition2'] = np.where(metedat['condition2'] == 'CSnaif2',
                                     'CSa-2', metedat['condition2'])
    metedat['condition2'] = np.where(metedat['condition2'] == 'CSplus',
                                     'CS+', metedat['condition2'])
    metedat['condition2'] = np.where(metedat['condition2'] == 'CSplusSI',
                                     'CS+S', metedat['condition2'])
    metedat['condition2'] = np.where(metedat['condition2'] == 'CSeteint',
                                     'CS-E', metedat['condition2'])

    metedat['condition3'] = metedat['condition'].copy()
    metedat['condition3'] = np.where(metedat['condition3'] == 'CSminus',
                                     'CS-1', metedat['condition3'])
    metedat['condition3'] = np.where(metedat['condition3'] == 'CSminus2',
                                     'CS-2', metedat['condition3'])
    metedat['condition3'] = np.where(metedat['condition3'] == 'CSnaif1',
                                     'CS-1', metedat['condition3'])
    metedat['condition3'] = np.where(metedat['condition3'] == 'CSnaif2',
                                     'CS-2', metedat['condition3'])
    metedat['condition3'] = np.where(metedat['condition3'] == 'CSplus',
                                     'CS+', metedat['condition3'])
    metedat['condition3'] = np.where(metedat['condition3'] == 'CSplusSI',
                                     'CS+', metedat['condition3'])
    metedat['condition3'] = np.where(metedat['condition3'] == 'CSeteint',
                                     'CS-E', metedat['condition3'])

    # Add other events to eeg events file
    events = pd.read_csv(opj(outpath, 'eeg', 'sub-' + sub
                             + '_task-fearcond_events.tsv'),
                         sep='\t')

    # Add condition labels to trial start
    trial_type_eprime = [0]*len(events['trigger_info'])
    trial_cues = [0]*len(events['trigger_info'])
    trial_cond_4 = [0]*len(events['trigger_info'])
    trial_type = [0]*len(events['trigger_info'])
    count = 0
    for idx, stim in enumerate(events['trigger_info']):
        if 'stimonset' in str(stim):
            trial_type_eprime[idx] = condition[count]
            trial_cues[idx] = cue[count]
            trial_type[idx] = list(metedat['condition2'])[count]
            trial_cond_4[idx] = list(metedat['condition3'])[count]
            count += 1
        else:
            trial_type_eprime[idx] = 'nan'
            trial_cues[idx] = 'nan'
            trial_type[idx] = 'nan'
            trial_cond_4[idx] = 'nan'

    events['trial_type_eprime'] = trial_type_eprime
    events['trial_cues'] = trial_cues
    events['trial_type'] = trial_type
    events['trial_cond4'] = trial_cond_4

    events.to_csv(opj(outpath, 'eeg', 'sub-' + sub
                      + '_task-fearcond_events.tsv'),
                  sep='\t', index=False)

# Save physio json file
scrdesc = {
    "SamplingFrequency": 1000,
    "StartTime": 'Not synced with EEG',
    "Columns": ["seconds", 'sample', "scr", 'emg', 'rmsemg',
                "trigger",
                "events", 'duration', 'duration_samp'],
    "Units": ["s", 'sample', 'mS', 'mV', 'rms(mV)', 'nan', 'nan',
              's', 'sample']
}


writetojson('task-fearcond_physio.json', bidsout, scrdesc)
