# generate noisy signal with artifacts
import warnings
warnings.filterwarnings('ignore')

# Load NeuroKit and other useful packages
import neurokit2 as nk
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

plt.rcParams['figure.figsize'] = [10, 6]  # Bigger images


# Generate noise free ecg signal
ecg_clean = nk.ecg_simulate(duration=30, noise=0, heart_rate=60, heart_rate_std=0, random_state=10, sampling_rate=1000)

# add noise to the signal
ecg_noisy = nk.signal_distort(ecg_clean, sampling_rate=1000, 
    noise_shape = 'laplace', noise_amplitude=0.7, noise_frequency=100, 
    powerline_amplitude=0.02, powerline_frequency=50,
    artifacts_amplitude=0.7, artifacts_frequency=50, artifacts_number=5, 
    linear_drift=True, random_state=10)

# create compund siganl
ecg_comp = np.append(ecg_clean, ecg_clean)
ecg_comp = np.append(ecg_comp, ecg_noisy)
ecg_comp = np.append(ecg_comp, ecg_clean)
ecg_comp = np.append(ecg_comp, ecg_clean)
ecg_comp = np.append(ecg_comp, ecg_clean)

#, ecg_noisy, ecg_clean, ecg_clean, ecg_clean)

# draw signals
ecg_df1 = pd.DataFrame({"ECG_CLEAN" : ecg_clean, "ECG_Noisy" : ecg_noisy})
ecg_df1.plot(subplots=True)
ecg_df2 = pd.DataFrame({"ECG_Compound" : ecg_comp})
ecg_df2.plot(subplots=True)

# evaluate quality
q_clean = nk.ecg_quality(ecg_clean, sampling_rate=1000, method="zhao2018", approach="fuzzy")
print('ecg clean quality: ' + q_clean)
q_noisy = nk.ecg_quality(ecg_noisy, sampling_rate=1000, method="zhao2018", approach="fuzzy")
print('ecg noisy quality: ' + q_noisy)
q_comp = nk.ecg_quality(ecg_comp, sampling_rate=1000, method="zhao2018", approach="fuzzy")
print('ecg compound quality: ' + q_comp)


# write signals for second stage
ecg_df2.to_csv("ecg_compound.csv", header=False, index=False)
print("Finished writing signal data.")
