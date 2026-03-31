'''
Model with function to load artefact data from Excel .xls and .xlsx files and splits 
data in separate vectors to process. Written for KT3401 - Assignment Artefact Detection

'''

import os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "KT3401_AFdata_2025")
folder = "Calibratie"
filename = "D01Cal.xlsx"
fs = 100

# #%% Clear system
# from IPython import get_ipython
# # Clear all variables (IPython/Jupyter)
# get_ipython().magic('reset -sf')
# import matplotlib.pyplot as plt
# # Close all figures
# plt.close('all')
# import os
# # Clear the console
# os.system('cls' if os.name == 'nt' else 'clear')

#%% Import modules
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def read_Artefacts(path, folder, filename, fs):
    """
    Inputs: 
    path: string to the path with data folders
    folder: string with the name of the folder with the dataset
    filename: string with name of the file, incl. extension
    fs: sampling rate (in Hz)

    Outputs: 
    t: time vector based on length of signal (to use instead of Time)
    ABP, CVP: vectors of arterial blood pressure and central venous pressure, with same length as t
    """
    filepath = os.path.join(path, folder, filename)
    print(f"Attempting to read file at: {filepath}")  # Debug print statement
    
    # Read the Excel file
    try:
        raw = pd.read_excel(filepath, sheet_name=0, header=None)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None
    
    raw = pd.read_excel(filepath, sheet_name=0, header=None)
    
    # Skip the first two rows
    raw = raw.iloc[2:, :]
    
    # Convert the data to a numpy array
    data = raw.to_numpy()
    
    # Allocate imported array to column variable names
    ABP = pd.to_numeric(data[:, 1], errors = 'coerce')
    CVP = pd.to_numeric(data[:, 2], errors = 'coerce')
    
    # Create time vector
    t = np.arange(1/fs, len(ABP)/fs + 1/fs, 1/fs)

    return t, ABP, CVP

resolution = 1   # seconds per spectrogram window
frange = [0, 10] # frequency range of interest (Hz)

t, ABP, CVP = read_Artefacts(path, folder, filename, fs)

# Spectrogram
frequencies, times, Sxx = spectrogram(ABP, fs, nperseg=int(resolution * fs), noverlap=0, nfft=int(fs * resolution))
Sxx_db = 10 * np.log10(Sxx)  # omzetten naar decibel
Amplitude = np.mean(np.abs(Sxx[(frequencies >= frange[0]) & (frequencies <= frange[1]), :]), axis=0)

fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle(f"{folder} - {filename}")

ax1.plot(t, ABP, label="ABP")
ax1.plot(t, CVP, label="CVP")
ax1.set_title("Arterial Blood Pressure (ABP) & Central Venous Pressure (CVP)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Pressure (mmHg)")
ax1.legend()

im = ax3.pcolormesh(times, frequencies, Sxx_db, shading="auto", cmap="viridis")
ax3.set_ylim(frange)
ax3.set_title("ABP Spectrogram")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Frequency (Hz)")
fig.colorbar(im, ax=ax3, label="Power (dB)")

plt.tight_layout()
plt.show()
