# This is to test if we strat from a clean RRi with both NK and Kubios, are the HRV metrics the same?
import neurokit2 as nk
import pandas as pd
from neurokit2.hrv.hrv_utils import _hrv_get_rri, _hrv_sanitize_input
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images

# Get data
data = pd.read_csv("ecg_clean_5min_1000hz_hr60.csv")
print(data.head())  # Print first 5 rows

# Find peaks
peaks, info = nk.ecg_peaks(data.iloc[:,0], sampling_rate=1000, correct_artifacts=False)
print(peaks.head())

# Sanitize input because hrv_time does this
peaks_s = _hrv_sanitize_input(peaks)
print(peaks_s[0:5])

# Compute R-R intervals (also referred to as NN) in milliseconds
rri = _hrv_get_rri(peaks_s, sampling_rate=1000, interpolate=False)
print(rri[0:5])

# write rri into csv file
df = pd.DataFrame(rri)
print(df.head())
df.to_csv("rri_ecg_clean_5min_1000hz_hr60.csv", header=None, index=None)

# Extract time domain features
hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False, silent=False)
df = hrv_time.transpose()
df.loc[df.shape[0]] = [""]
df.to_csv("nk_hrv_rri_ecg_clean_5min_1000hz_hr60.csv", header=["time domain, nk-hrv"])

# Extract freq domain features
hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False, silent=False)
df = hrv_freq.transpose()
df.loc[df.shape[0]] = [""]
df.to_csv("nk_hrv_rri_ecg_clean_5min_1000hz_hr60.csv", mode='a', header=["freq domain, nk-hrv"])

# Extract non-liner features
hrv_non = nk.hrv_nonlinear(peaks, sampling_rate=1000, show=False)
df = hrv_non.transpose()
df.append(["   "])
df.loc[df.shape[0]] = [""]
df.to_csv("nk_hrv_rri_ecg_clean_5min_1000hz_hr60.csv", mode='a', header=["nonlinear domain, nk-hrv"])
