#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from functions import *
import pickle
#%%
folder ="XRD"
filename= "XRD_mittma_00012_slowscans.txt" 
filepath = os.path.join(folder, filename)
data_slow_FL = pd.read_csv(filepath, sep="\t", header=1, decimal=".", usecols=[0,1], names=["2theta", "intensity"])
data_slow_BL = pd.read_csv(filepath, sep="\t", header=1, decimal=".", usecols=[2,3], names=["2theta", "intensity"])
#datas = [data_slow_FL, data_slow_BL]

datas = {}
datas["FL"] =data_slow_FL
datas["BL"] =data_slow_BL
for key, data in datas.items():
    print(key, data)
#%%

fig,axes =plt.subplots(2,1,figsize= (12,8))
i=0
for key, data in datas.items():
    ax = axes[i]
    ax.plot(data["2theta"], data["intensity"], label = key)
    ax.legend()
    i+=1

# %%
dataRangeMin = 0
dataRangeMax = len(data)
filterstrength = 10 # higher number = more noise is removed by the filter
peakprominence = 200 # higher number = only peaks with prominence above this value are considered
peakwidth = 5

dfpeaks_list={}
dfdata_list = {}

for key, data in datas.items():
    
    peaks, dataCorrected = initial_peaks(data, dataRangeMin, dataRangeMax, filterstrength,
                                         peakprominence, peakwidth,
                                         withplots = True, plotscale = 'log')
    df_peaks = pd.DataFrame(peaks)
    dfpeaks_list[key] = df_peaks
    #dfpeaks_list.append(df_peaks)
    display(df_peaks)

    col_theta = data.columns.values[::2]
    col_counts = data.columns.values[1::2]
    peaks_theta = peaks.columns.values[::2]

    data_out = data.copy()


    for i in range(0, len(col_theta)):
        cut_intensity=[]

        two_theta = data[col_theta[i]]
        intensity = data[col_counts[i]]
        idx_range = np.where(two_theta >= 20+2)[0][0] #set how many degrees you want to cut around each

        # Cut data around peaks
        for j in range(len(intensity)):
            if data[col_theta[i]][j] in peaks[peaks_theta[i]].values:
                start_index = max(0, j-idx_range)
                end_index = min(len(data), j+idx_range)
                data_out[col_counts[i]][start_index:end_index] = np.nan #cut data intensity around peaks in data_out

        # Cut data around Si peak
        idx_Si = np.where((two_theta >= 60) & (two_theta<= 70))[0]
        data_out[col_counts[i]][idx_Si] = np.nan

        cut_intensity = data_out[col_counts[i]]

        # Smooth the data for better peak detection
        smoothed_intensity = savgol_filter(intensity, window_length=10, polyorder=3)
        # Filter out NaN values (they exist because we cut the data) before fitting
        mask = ~np.isnan(cut_intensity)
        filtered_two_theta = two_theta[mask]
        filtered_intensity = intensity[mask]

        # Perform polynomial fitting with filtered data
        background_poly_coeffs = np.polyfit(filtered_two_theta, filtered_intensity, 4)
        background = np.polyval(background_poly_coeffs, two_theta)

        # Subtract background
        corrected_intensity = smoothed_intensity - background

        df = pd.DataFrame({"2theta":two_theta, "Corrected Intensity": corrected_intensity, "Background": background, "Raw data": intensity})
        dfdata_list[key] = df
        display(df)

        #plot
        plt.figure()
        #coord= data_out.columns.get_level_values(0).unique()[i]
        plt.plot(two_theta, intensity, label='Original Data')
        plt.plot(filtered_two_theta, filtered_intensity, label='filtered Data')
        plt.plot(two_theta, background, label='Background, order=4', linestyle='--')
        plt.plot(two_theta, corrected_intensity, label='Corrected Data')
        #plt.title('XRD data at {}'.format())
        plt.legend()
        plt.show()


# %%
print(dfdata_list) 
# %%

fig,axes =plt.subplots(2,1,figsize= (12,8))
i=0
for key, data in dfdata_list.items():
    ax = axes[i]
    ax.plot(data["2theta"], data["Corrected Intensity"], label = key)
    ax.legend()
    i+=1
plt.savefig(os.path.join(folder, "XRD_mittma_00012_slowscans.png"), dpi=300, bbox_inches='tight')    
# %%
with open(os.path.join(folder, "XRD_mittma_00012_slowscans_data.pkl"), "wb") as f:
    pickle.dump(dfdata_list, f)
# %%
for key, data in dfdata_list.items():
    print(key)
    display(data)
# %%
with open(os.path.join(folder, "XRD_mittma_00012_slowscans_peaks.pkl"), "wb") as f:
    pickle.dump(dfpeaks_list, f)
# %%
