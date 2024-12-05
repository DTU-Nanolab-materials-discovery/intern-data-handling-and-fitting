#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import pickle

# %%
folder = r"C:\Users\s222531\OneDrive - Danmarks Tekniske Universitet\MASTER_PROJECT_PHOSPHOSULFIDES\results\Raman_241007"
files = os.listdir(folder)
files = [f for f in files if f.endswith('.txt')]
print(files)

dict_0012 = []
dict_0015 = []
dict_0019 = []
dict_0011 = []
dict_0022 = []
dict_0023 = []
for f in files:
    if f.split('_')[1] == '0011':
        dict_0011.append(f)
    if f.split('_')[1] == '0012':
        dict_0012.append(f)
    if f.split('_')[1] == '0015':
        dict_0015.append(f)
    if f.split('_')[1] == '0019':
        dict_0019.append(f)
    if f.split('_')[1] == '0022':
        dict_0022.append(f)
    if f.split('_')[1] == '0023':
        dict_0023.append(f)

dicts = [dict_0011, dict_0012, dict_0015, dict_0019, dict_0022, dict_0023]
dict_Cu3PS4 = [dict_0011, dict_0012, dict_0015]
# %% -------------------------- create overview plots --------------------------
for dict in dicts: 
    k=0
    fig= plt.figure()
    for f in dict:
        label = f.split('_')[3].split('.')[0]
        data = pd.read_csv(os.path.join(folder, f), sep='\t', skiprows=1, names=['#Wave', '#Intensity'])
        # print(data)
        plt.plot(data['#Wave'], data['#Intensity']+k*10000, label=label)
        k+=1
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel('Wavenumber (cm-1)')
    plt.ylabel('Intensity')
    plt.title(f'Raman spectra of mittma_{f.split("_")[1]}')
    # plt.savefig(os.path.join(folder, f'mittma_{f.split("_")[1]}.png'), dpi=300)
# %% -------------------------- Cu3PS4 --------------------------
data_RT= pd.read_csv(os.path.join(folder, 'mittma_0011_BR_5sx5aq br1.txt'), sep='\t', skiprows=1, names=['#Wave', '#Intensity'])
data_400= pd.read_csv(os.path.join(folder, 'mittma_0012_FL_5perc.txt'), sep='\t', skiprows=1, names=['#Wave', '#Intensity'])
data_500= pd.read_csv(os.path.join(folder, 'mittma_0015_BL_5perc tr.txt'), sep='\t', skiprows=1, names=['#Wave', '#Intensity'])
data= {'RT':data_RT, '400': data_400, '500': data_500}
fig= plt.figure()
k=0
for d in data.keys():
    plt.plot(data[d]['#Wave'], (k*0.2)+(data[d]['#Intensity']/max(data[d]['#Intensity'])), label= d)
    k+=1
plt.legend()
plt.xlabel('Wavenumber (cm-1)')
plt.ylabel('Intensity')
plt.title('Raman spectra of Cu3PS4')

# %%  subplots 
fig,ax= plt.subplots(3,1, figsize=(10,6))
ax[0].plot(data_RT['#Wave'], data_RT['#Intensity']/25, label= 'RT')
ax[1].plot(data_400['#Wave'], data_400['#Intensity']+2000, label= '400')
ax[2].plot(data_500['#Wave'], data_500['#Intensity']+4000, label= '500')
ax[2].set_xlabel('Wavenumber (cm-1)')
for a in ax: 
    a.legend()
    a.set_ylabel('Intensity / s (a.u.)')
    a.set_xlim(-50, 1000)

ax[0].set_title('Raman spectra of Cu3PS4')
# fig.savefig(os.path.join(folder, 'Cu3PS4_Raman.png'), dpi=300)
# %% 
fig,ax= plt.subplots(3,1, figsize=(10,6))
i=0
labels= ['RT', '400', '500']
calibrate= [25, 1, 1]
for d in dict_Cu3PS4: 
    Temp = labels[i]
    calib = calibrate[i]
    for f in d: 
        label = f.split('_')[3].split('.')[0]
        try:
            label= label.split(' ')[1]
        except:
            pass
        data = pd.read_csv(os.path.join(folder, f), sep='\t', skiprows=1, names=['#Wave', '#Intensity'])
        ax[i].plot(data['#Wave'], data['#Intensity']/ calib, label= Temp +' - '+ label)
    ax[i].legend()
    ax[i].set_ylabel('Intensity / s (a.u.)')
    ax[i].set_xlim(-50, 1000)
    i+=1

ax[2].set_xlabel('Wavenumber (cm-1)')
ax[0].set_title('Raman spectra of Cu3PS4')
fig.savefig(os.path.join(folder, 'all_Cu3PS4_Raman.png'), dpi=300)

# %% --------- find peak positions ----------------

def find_peak(data, range):
    "data is a pandas dataframe with columns #Wave and #Intensity, range is a tuple with the range of interest"
    idx = np.where((data['#Wave'] > range[0]) & (data['#Wave'] < range[1]))[0]
    peak = data['#Intensity'].iloc[idx].idxmax()
        
    return data['#Wave'].iloc[peak], data['#Intensity'].iloc[peak]

# %%
x,y = find_peak(data_500, (300, 400))
print(x,y)
# %%
peaks_Cu3PS4 = pd.DataFrame(columns=['file', 'Peak pos', 'Intensity', 'Peak Si', 'Intensity Si'])
Temps = ['RT', '400', '500']
i=0
for d in dict_Cu3PS4: 
    Temp = Temps[i]
    for f in d: 
        label = f.split('_')[3].split('.')[0]
        try:
            label= label.split(' ')[1]
        except:
            pass
        data = pd.read_csv(os.path.join(folder, f), sep='\t', skiprows=1, names=['#Wave', '#Intensity'])
        Si_x, Si_y = find_peak(data, (500, 600))
        peak_pos, intensity = find_peak(data, (300, 400))
        peaks = pd.DataFrame.from_dict({'file': Temp +'_'+label, 'Peak pos': peak_pos, 'Intensity': intensity, 'Peak Si': Si_x, 'Intensity Si': Si_y}, orient='index').T
        peaks_Cu3PS4 = pd.concat([peaks_Cu3PS4, peaks], ignore_index=True)
    i+=1
        
# %%

# peaks_Cu3PS4 = pd.DataFrame(columns=['file', 'Peak 282', 'Intensity 282', 'Peak 299', 'Intensity 299', 'Peak 319', 'Intensity 319', 'Peak 389', 'Intensity 389'])
peaks_Cu3PS4_only = pd.DataFrame()
expected_peaks = [282, 299, 319, 389, 519]
Temps = ['RT', '400', '500']
i=0
for d in dict_Cu3PS4: 
    Temp = Temps[i]
    for f in d: 
        label = f.split('_')[3].split('.')[0]
        try:
            label= label.split(' ')[1]
        except:
            pass
        data = pd.read_csv(os.path.join(folder, f), sep='\t', skiprows=1, names=['#Wave', '#Intensity'])
        
        peaks_dict={'file': Temp +'_'+label}
        for j in range(0, len(expected_peaks)):
            peak_pos, intensity = find_peak(data, (expected_peaks[j]-10, expected_peaks[j]+10))
            peaks_dict[f'Peak {expected_peaks[j]}']= peak_pos
            peaks_dict[f'Intensity {expected_peaks[j]}']= intensity

        peaks = pd.DataFrame.from_dict(peaks_dict, orient='index').T
        peaks_Cu3PS4_only = pd.concat([peaks_Cu3PS4_only, peaks], ignore_index=True)
    i+=1
peaks_Cu3PS4_only.to_csv(os.path.join(folder, 'peaks_Cu3PS4.csv'), index=False)


# %% now let's look at mittma 0019

folder = r"Z:\P110143-phosphosulfides-Andrea\Data\Samples\mittma_0019_Cu\Raman\BR"
files = os.listdir(folder)
files = [f for f in files if f.endswith('.txt')]
i=0
for f in files:
    try:
        label = f.split('_')[4].split('.')[0]
    except:
        label = f.split('_')[3].split('.')[0]
    data = pd.read_csv(os.path.join(folder, f), sep='\t', skiprows=1, names=['#Wave', '#Intensity'])
    plt.plot(data['#Wave'], (data['#Intensity']/25)+ 300*i, label= f.split('_')[2] +' '+label)
    i+=1
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('Wavenumber (cm-1)')
plt.ylabel('Intensity / s (a.u.)')
plt.xlim(300,600)
# %%

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Lorentzian function definition
def lorentzian(x, amp, cen, width):
    return amp * width**2 / ((x - cen)**2 + width**2)

# Linear background function definition
def linear(x, a, b):
    return a * x + b

# Combined function for Lorentzian peaks + linear background
def lorentzian_with_background(x, *params):
    num_peaks = int(len(params) / 3)
    y = np.zeros_like(x)
    for i in range(num_peaks):
        amp = params[3 * i]
        cen = params[3 * i + 1]
        width = params[3 * i + 2]
        y += lorentzian(x, amp, cen, width)
    return y

# Main function to fit background and Lorentzian peaks
def fit_raman_spectrum(df, wavenum_range, peak_centers, init_guess=None):
    """
    Fit a linear background and Lorentzian peaks to Raman spectrum data.

    Parameters:
    - df: pd.DataFrame with columns ['#Wave', '#Intensity'] for wavenumber and intensity
    - wavenum_range: tuple (min_wavenum, max_wavenum) for linear background fitting
    - peak_centers: list of peak centers (in wavenumbers) to fit Lorentzian peaks
    - init_guess: list of initial guesses for the Lorentzian peaks in the form 
                  [amplitude1, center1, width1, amplitude2, center2, width2, ...]
                  If None, defaults will be provided.
    Returns:
    - popt: optimized parameters for Lorentzian peaks
    - background_params: optimized parameters for the linear background
    """
    x = df['#Wave'].values
    y = df['#Intensity'].values

    # Step 1: Fit the linear background in the specified wavenumber range
    mask = (x >= wavenum_range[0]) & (x <= wavenum_range[1])
    x_bg = x[mask]
    y_bg = y[mask]

    # Fit a linear function to the selected background region
    background_params, _ = curve_fit(linear, x_bg, y_bg)
    
    # Subtract the linear background from the original data
    y_bg_fit = linear(x_bg, *background_params)
    y_corrected = y_bg - y_bg_fit

    # Step 2: Fit the Lorentzian peaks
    if init_guess is None:
        # Generate default initial guesses for Lorentzian peaks: amplitude, center, and width
        init_guess = []
        for center in peak_centers:
            amp_guess = max(y_corrected)  # Guess the amplitude based on the peak height
            width_guess = 10  # A reasonable default width guess
            init_guess.extend([amp_guess, center, width_guess])

    # Fit Lorentzian peaks to the background-corrected spectrum
    popt, _ = curve_fit(lorentzian_with_background, x_bg, y_corrected, p0=init_guess)
    
    # Plot the data and the fit
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Original Data', color='blue')
    plt.plot(x_bg, y_bg_fit, label='Fitted Background', linestyle='--', color='red')
    plt.plot(x_bg, y_corrected + y_bg_fit, label='Lorentzian Fit', linestyle='-', color='green')
    plt.xlabel('Wavenumber')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()
    
    return popt, background_params

# Example usage:
# Assuming `df` is your pandas dataframe with columns '#Wave', '#Intensity'
# wavenum_range = (min_wavenum, max_wavenum) for linear background fitting
# peak_centers = [center1, center2, ...] for Lorentzian peaks
# popt, bg_params = fit_raman_spectrum(df, (1000, 1200), [1050, 1100, 1150])
#%%

df= pd.read_csv(os.path.join(folder, 'mittma_0019_BR_5perc_5sx5aq bl.txt'), sep='\t', skiprows=1, names=['#Wave', '#Intensity'])
wn_range= (350,580)
peak_centers = [428, 519]
popt, bg_params = fit_raman_spectrum(df, wn_range, peak_centers)

# %%
