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
ecg_clean = nk.ecg_simulate(duration=300, noise=0, heart_rate=60, heart_rate_std=0, random_state=10, sampling_rate=1000)

ecg_df = pd.DataFrame({"ECG_FREE_Noise" : ecg_clean})
ecg_df.to_csv("ecg_clean_5min_1000hz_hr60.csv", index=False, header=False)

ecg_df.plot()
plt.savefig('ecg_clean_5min_1000hz_hr60.pdf')

print("Finished writing clean signal data.")

