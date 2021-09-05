"""
Calculate HRV based on Moolah approach for a compound siganl
usage:
    nkhrv input_ecg_file sampling_rate epoch_size
    sample:  nkhrv test-ecg-signal.csv 1000 30
    The output file is 'hrv_'+input_file_name
    sampling_rate default is 1000
    epoch_size default is 30
"""

"""
Release note
29 Aug 2021 by MK
    1. created function create_best_signal_by_removal
    2. removed signal cleaning from moolah_calc_hrv
    3. remove calculating HRV using traditional method from main body
30 Aug 2021 by MK
    1. Summarize calculations based on selected features
    2. Add calculation for MeanHR based on 60000/MeanRR
    3. Create command-line interface so that by giving input signal produce output features - better for testing
           nkhrv input_ecg_file sampling_rate epoch_size
           nkhrv test-ecg-signal.csv 1000 30
           The output file is 'hrv_'+input_file_name
    4. Change file name to nk-hrv from nk-hrv-compound
    5. Use nk.hrv function to calculate all categories of features instead of separate hrv_time, hrv_frequency and hrv_nonlinear
    6. Make the code full parametric with default values
    7. Comment not needed functions such as: clean_epochs, create_best_signal_by_selection
    8. Bug fix in create_best_signal_by_removal (when removing no-quality epochs)
4 Sep 2021 by MK
    1. Add features from NK: HRV_VLF, HRV_LFn, HRV_HFn, HRV_LFHF, HRV_HTI
    2. Create new features Min_HR, Max_HR
    3. Reorder feature accoding to NK features
    4. Group features to Time-Domain, Frequency-Domain, Nonlinear, Other
    5. Add Meta data: Version and date, User, filename, sampling-rate, epoch-size, data-lengh, start-time, end-time, duration-of-analysis
    6. Bug fix in function moolah_calc_hrv inorder if only exist one excellent epoch, then it can be usable len(ecg_best) >= (sampling_rate * epoch_size)
5 Sep 2021 by MK
    1. Bug fix in find_min_max_HR when the sum(rri) is less than 1 minute
"""

# Load the NeuroKit package and other useful packages
import neurokit2 as nk
from neurokit2.hrv.hrv_utils import _hrv_get_rri, _hrv_sanitize_input
import pandas as pd
import numpy as np
import sys
import getpass
import time


version = "Moolah-Neurokit2-HRV-Ananlysis 1.0 - 4 Sep 20121"
selected_features = ["*** Time Domain Features", "HRV_RMSSD", "HRV_MeanNN", "HRV_SDNN", "HRV_HTI",
                    "*** Frequency Domain Features", "HRV_VLF", "HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_LFn", "HRV_HFn",
                    "*** Nonlinear Featues", "HRV_SD1", "HRV_SD2", "HRV_SD1SD2", "HRV_SI"]


# find min and max HR from rri
def find_min_max_HR(rri):
    # if sum(rri) < 60000 (1 minute)
    s = sum(rri)
    if s < 60000:
        min_hr = max_hr = 60000 * len(rri) / s
        return min_hr, max_hr
    # if sum(rri) >= 60000 (1 minute)
    beats = 0
    beats_list = np.array([], dtype=np.int8)
    time_len = 0.0
    for rr in rri:
        if time_len+rr >= 60000.0:
            beats_list = np.append(beats_list,[beats])
            print("---time len is ", time_len, "---next rr is ", rr, "---beats is ", beats)
            beats = 1
            time_len = rr
        else:
            beats += 1
            time_len += rr
    min_hr = np.min(beats_list)
    max_hr = np.max(beats_list)
    return min_hr, max_hr

# Calculate hrv metrics for ECG signal
def calc_hrv(data, sampling_rate=1000):
    # Find peaks
    peaks, info = nk.ecg_peaks(data, sampling_rate=sampling_rate, correct_artifacts=False)
    # Calculate features
    hrv_indices = nk.hrv(peaks, sampling_rate=sampling_rate, show=False)
    # Extract selected features
    selected_indices = pd.DataFrame(columns=['feature', 'value'])
    for feature in selected_features:
        if feature in hrv_indices.columns:
            feature_value = hrv_indices.loc[:, feature][0]
            selected_indices = selected_indices.append({'feature':feature, 'value':feature_value}, ignore_index=True)
        else:
            selected_indices = selected_indices.append({'feature':"", 'value':""}, ignore_index=True)
            selected_indices = selected_indices.append({'feature':feature, 'value':"Neurokit2"}, ignore_index=True)
    
    # Add other features
    selected_indices = selected_indices.append({'feature':"", 'value':""}, ignore_index=True)
    selected_indices = selected_indices.append({'feature': "*** Other Features", 'value':"Moolah"}, ignore_index=True)
    selected_indices = selected_indices.append({'feature':'MeanHR', 'value': 60000/hrv_indices.loc[:, "HRV_MeanNN"][0]}, ignore_index=True)
    # extract rri then min, max of HR
    peaks_s = _hrv_sanitize_input(peaks)
    rri = _hrv_get_rri(peaks_s, sampling_rate=sampling_rate, interpolate=False)
    min_HR, max_HR = find_min_max_HR(rri)
    selected_indices = selected_indices.append({'feature':'MinHR', 'value': min_HR}, ignore_index=True)
    selected_indices = selected_indices.append({'feature':'MaxHR', 'value': max_HR}, ignore_index=True)
        
    print(selected_indices)
    return selected_indices


# Calculate hrv metrics using Moolah approach
# removed cleaning the epochs
def moolah_calc_hrv(ecg_signal, sampling_rate=1000, epoch_size=30):
    epochs = partition_signal(ecg_signal, sampling_rate, epoch_size)
    quality = assess_quality(epochs, sampling_rate)
    ecg_best = create_best_signal_by_removal(epochs, quality, sampling_rate)
    moolah_hrv = pd.DataFrame()
    if len(ecg_best) >= (sampling_rate * epoch_size):
        moolah_hrv = calc_hrv(ecg_best, sampling_rate)
    return moolah_hrv


# Partition ecg_signal to n second epochs and report number of epochs created
def partition_signal(signal, sampling_rate=1000, epoch_size=30):
    len_epoch = sampling_rate * epoch_size
    strt = 0
    epochs = []
    while strt < len(signal):
        end = strt + len_epoch
        if end > len(signal):
           end = len(signal)
        epochs.append(signal[strt:end])
        strt += len_epoch
    print(len(epochs), "epochs created.")
    return epochs


"""
# clean the epochs based on specified method
# if method is "none" no cleaning is done
def clean_epochs(epochs, method):
    if method != "none":
        for i in range(len(epochs)):
            epochs[i] = nk.ecg_clean(epochs[i], sampling_rate=sampling_rate, method = method)
        print(len(epochs), "epochs cleaned.")
    else:
        print("no epochs cleaned")
    return epochs
"""


# For each epoch assess its quality
# if epoch is less that 30 second the quality would not be Excellent
def assess_quality(epochs, sampling_rate=1000):
    quality = []
    for i in range(len(epochs)):
        quality.append(nk.ecg_quality(epochs[i], sampling_rate=sampling_rate, method="zhao2018", approach="fuzzy"))
    print("result of assessing quality of epochs: ", quality)
    return quality


# Create best signal by removing non-excellent epochs
def create_best_signal_by_removal(epochs, quality, sampling_rate=1000):
    # remove epoch which their quality is not excellent
    i = 0
    for q in quality:
        if q.lower() != "excellent":
            del epochs[i]
            i -= 1
        i += 1

    # build the signal based on remaining epochs
    best_signal = np.array([])
    for epoch in epochs:
        best_signal = np.append(best_signal, epoch)
    print("Created best signal with the len:", len(best_signal))
    
    quality = nk.ecg_quality(best_signal, sampling_rate=sampling_rate, method="zhao2018", approach="fuzzy")
    print("Quality of the best signal is:", quality)
    return best_signal


"""
# create best signal by Selection of last n_selection consecutive epochs with Excellent Quality
def create_best_signal_by_selection(epochs, quality, n_selection):
    # first find the index of last n consecutive "Excellent" items in quality list
    sq = ""
    for q in quality:
        if q.lower() == "excellent":
            sq += "E"
        elif q.lower() == "barely acceptable":
            sq += "B"
        elif q.lower() == "unacceptable":
            sq += "U"
    reverse_sq = sq[::-1]
    ser = re.search("EEE", reverse_sq)
    if ser:
        i = len(sq) - ser.span()[0] - 1
        index = []
        for j in range(n_selection):
            index.append(i-n_selection+j+1)
    else:
        print("Not enough Excellent epochs.")
        return([])  
    
    # now build the best signal based on found indexes
    best_signal = np.array([])
    for i in index:
        best_signal = np.append(best_signal, epochs[i])
    print("Created best signal with the len:", len(best_signal))
    
    quality = nk.ecg_quality(best_signal, sampling_rate=sampling_rate, method="zhao2018", approach="fuzzy")
    print("Quality of the best signal is:", quality)
    return best_signal
"""


def nkhrv(input_signal_file, sampling_rate=1000, epoch_size=30):
    # Read Data
    dat = pd.read_csv(input_signal_file)
    print(dat.head())  # Print first 5 rows
    ecg_signal = dat.iloc[:,0]
    uname = getpass.getuser()

    # Calculate hrv metrics on ECG signal compound using moolah method
    start = time.time()
    df_hrv_moolah = moolah_calc_hrv(ecg_signal, sampling_rate=sampling_rate, epoch_size=epoch_size)
    end = time.time()

    # Add meta data
    df_meta = pd.DataFrame(columns=['feature', 'value'])
    df_meta = df_meta.append({'feature':"Version", 'value': version}, ignore_index=True)
    df_meta = df_meta.append({'feature':"User", 'value': uname}, ignore_index=True)
    df_meta = df_meta.append({'feature':"file name", 'value': input_signal_file}, ignore_index=True)
    df_meta = df_meta.append({'feature':"Sampling Rate", 'value': sampling_rate}, ignore_index=True)
    df_meta = df_meta.append({'feature':"Epoch Size (s)", 'value': epoch_size}, ignore_index=True)
    df_meta = df_meta.append({'feature':"Sample size", 'value': len(ecg_signal)}, ignore_index=True)
    df_meta = df_meta.append({'feature':"Start time", 'value': time.ctime(start)}, ignore_index=True)
    df_meta = df_meta.append({'feature':"end time", 'value': time.ctime(end)}, ignore_index=True)
    df_meta = df_meta.append({'feature':"Duration (s)", 'value': end-start}, ignore_index=True)
    df_meta = df_meta.append({'feature':"", 'value': ""}, ignore_index=True)
    df_hrv_moolah = df_meta.append(df_hrv_moolah)

    out_file = "hrv_" + input_signal_file
    df_hrv_moolah.to_csv(out_file, header=False, index=False)


# for test
nkhrv("ecg_compound_3min_1000hz_hr60.csv", 1000, 40)

"""
if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: nkhrv ecg_file sampling_rate=1000 epoch_size=30")
        sys.exit()
    if len(sys.argv) == 2:
        nkhrv(sys.argv[1])
    elif len(sys.argv) == 3:
        nkhrv(sys.argv[1], int(sys.argv[2]))
    else:
        nkhrv(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
"""

