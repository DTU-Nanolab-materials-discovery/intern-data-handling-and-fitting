#%%
from functions import *

# %% ##############################################
##############################################

#grid = measurement_grid(5,10,32,72,-16,-40) # mittma_00012_first50
grid = measurement_grid(5,5,32,32,-16,-36) # for one sample FR (mittma 00015)

#filename = "XRD_0003_5points.txt"
#filename = "XRD-mittma_00012_first50.txt" 
folder = r"O:\Nlab\Public\DCH-plasma\phosphosulfides_students\Students\Giulia\01_Characterization\XRD\raw data"
filename = "XRD_mittma_00015_FR_map.txt" 
filename = os.path.join(folder, filename)

initial_data, coords = read_XRD(filename, grid, n = 0, separator = "\t")
plot_grid(coords,grid)

which =initial_data.keys()[0:8]
data= initial_data[which]
display(data)
#%%

dataRangeMin = 0
dataRangeMax = len(data)
filterstrength = 10 # higher number = more noise is removed by the filter
peakprominence = 300 # higher number = only peaks with prominence above this value are considered
peakwidth = 8

peaks, dataCorrected = initial_peaks(data, dataRangeMin, dataRangeMax, filterstrength,
                                         peakprominence, peakwidth,
                                         withplots = True, plotscale = 'log')
#%%
print(peaks)

#%%
data_out = XRD_background(data,peaks, cut_range=1, order=4, withplots= True, Si_cut=False)


# %% define the peak to fit
data_to_fit_x = data_out[('-16.0,-4.0',              '2θ (°)')]
data_to_fit_y = data_out[('-16.0,-4.0',              'Corrected Intensity')]

cut_range=1 
peak_pos = peaks[('-16.0,-4.0',         '2θ (°)')][0]
idx = np.where((data_to_fit_x >= peak_pos-cut_range) & (data_to_fit_x<=  peak_pos+cut_range))[0]
x_range = data_to_fit_x[idx].values
y_range = data_to_fit_y[idx].values
plt.plot(x_range, y_range)


#%%
###### CHATGPT SUGGESTION FOR FITTING 1 PEAK ######

import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import PseudoVoigtModel

def fit_single_peak(x, y):
    
    # Initialize the Pseudo-Voigt model
    model = PseudoVoigtModel()

    # Estimate initial parameters: amplitude, center, and sigma
    params = model.guess(y, x=x)

    # Perform the fit
    fit_result = model.fit(y, params, x=x)

    # Extract the fitted parameters
    amplitude = fit_result.params['amplitude'].value
    center = fit_result.params['center'].value
    sigma = fit_result.params['sigma'].value
    fraction = fit_result.params['fraction'].value
    height = fit_result.params['height'].value

    # Calculate FWHM from sigma and fraction
    # FWHM for PseudoVoigt is calculated by combining Lorentzian and Gaussian contributions
    gamma = sigma / np.sqrt(2 * np.log(2))  # Convert sigma to gamma for Gaussian part
    fwhm = (1 - fraction) * (2 * gamma) + fraction * (2 * sigma)  # Linear combination

    return fit_result, amplitude, fwhm, center, fraction, height

#%%
# Example usage:
fit_result, amplitude, fwhm, center, fraction, height = fit_single_peak(x_range, y_range)

print(f"Amplitude: {amplitude:.2f}")
print(f"FWHM: {fwhm:.2f}")
print(f"Center: {center:.2f}")
print(f"Gaussian/Lorentzian Fraction: {fraction:.2f}")
print(f"Height: {height:.2f}")

# Plot the data and fitted profile
plt.figure(figsize=(8, 6))
plt.plot(x_range, y_range, 'bo', label='Data')
plt.plot(x_range, fit_result.best_fit, 'r-', label='Fit')
plt.xlabel('2θ')
plt.ylabel('Intensity')
plt.title('Single Peak Fit with PseudoVoigt Model')
plt.legend()
plt.show()


# %%

def fit_two_related_peaks(x, y):

    # Initialize two Pseudo-Voigt models with prefixes to distinguish parameters
    model1 = PseudoVoigtModel(prefix='p1_')
    model2 = PseudoVoigtModel(prefix='p2_')

    # Estimate initial parameters for the first peak
    params = model1.guess(y, x=x)
    
    # Extract initial guesses
    amplitude1 = params['p1_amplitude'].value
    center1 = params['p1_center'].value
    sigma1 = params['p1_sigma'].value
    fraction1 = params['p1_fraction'].value

    # Set constraints for the second peak based on the provided relations
    #xpeak2 = 2 * np.arcsin((0.154439 / 0.1540562) * np.sin(center1 / 2))
    xpeak2= (360/np.pi)* np.arcsin((0.154439 / 0.1540562) * np.sin(center1*np.pi /360))
    
    params.add('p2_center', expr='(360/pi)* arcsin((0.154439 / 0.1540562) * sin(p1_center*pi /360))')
    params.add('p2_amplitude', expr='0.5 * p1_amplitude')
    params.add('p2_sigma', expr='1 * p1_sigma')
    params.add('p2_fraction', expr='1 * p1_fraction')

    # Create a combined model by summing the two models
    combined_model = model1 + model2

    # Perform the fit
    fit_result = combined_model.fit(y, params, x=x)

    # Extract the fitted parameters for both peaks
    amplitude1 = fit_result.params['p1_amplitude'].value
    center1 = fit_result.params['p1_center'].value
    sigma1 = fit_result.params['p1_sigma'].value
    fraction1 = fit_result.params['p1_fraction'].value

    amplitude2 = fit_result.params['p2_amplitude'].value
    center2 = fit_result.params['p2_center'].value
    sigma2 = fit_result.params['p2_sigma'].value
    fraction2 = fit_result.params['p2_fraction'].value

    # Calculate FWHM for both peaks
    gamma1 = sigma1 / np.sqrt(2 * np.log(2))  # Convert sigma to gamma for Gaussian part
    fwhm1 = (1 - fraction1) * (2 * gamma1) + fraction1 * (2 * sigma1)

    gamma2 = sigma2 / np.sqrt(2 * np.log(2))
    fwhm2 = (1 - fraction2) * (2 * gamma2) + fraction2 * (2 * sigma2)

    return fit_result, amplitude1, fwhm1, center1, fraction1, amplitude2, fwhm2, center2, fraction2




# %%
# Example usage:
fit_result, amplitude1, fwhm1, center1, fraction1, amplitude2, fwhm2, center2, fraction2 = fit_two_related_peaks(x_range, y_range)

print(f"Peak 1 - Amplitude: {amplitude1:.2f}, FWHM: {fwhm1:.2f}, Center: {center1:.2f}, Fraction: {fraction1:.2f}")
print(f"Peak 2 - Amplitude: {amplitude2:.2f}, FWHM: {fwhm2:.2f}, Center: {center2:.2f}, Fraction: {fraction2:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(x_range, y_range, 'bo', label='Data')
plt.plot(x_range, fit_result.best_fit, 'r-', label='Fit')
plt.xlabel('2θ')
plt.ylabel('Intensity')
plt.title('Two Related Peaks Fit with PseudoVoigt Models')
plt.legend()
plt.show()
# %%
#now try to make a function that fits this peak for all the coordinates in the df

def xrd_fit_peak(data, peaks, peak_angle, cut_range, withplots=False):

    dat_theta = data.iloc[:,data.columns.get_level_values(1)=='2θ (°)']
    dat_counts = data.iloc[:,data.columns.get_level_values(1)=='Corrected Intensity']

    print(dat_theta, dat_counts)

    for i in range(0, len(dat_theta.columns)):

      


# %%
data=data_out.copy()
cut_range=1
peak_angle = 28.27

dat_theta = data.iloc[:,data.columns.get_level_values(1)=='2θ (°)']
dat_counts = data.iloc[:,data.columns.get_level_values(1)=='Corrected Intensity']

plt.figure(figsize=(8, 6))

for i in range(0, len(dat_theta.columns)):
    data_to_fit_x = dat_theta[dat_theta.columns[i]]
    data_to_fit_y = dat_counts[dat_counts.columns[i]]

    #peak_pos = peaks[dat_theta.columns[i]][peak_idx]

    idx = np.where((data_to_fit_x >= peak_angle-cut_range) & (data_to_fit_x<=  peak_angle+cut_range))[0]
    x_range = data_to_fit_x[idx].values
    y_range = data_to_fit_y[idx].values

    fit_result, amplitude1, fwhm1, center1, fraction1, amplitude2, fwhm2, center2, fraction2 = fit_two_related_peaks(x_range, y_range)

    print(dat_theta.columns[i][0])
    print(f"Peak 1 - Amplitude: {amplitude1:.2f}, FWHM: {fwhm1:.2f}, Center: {center1:.2f}, Fraction: {fraction1:.2f}")
    print(f"Peak 2 - Amplitude: {amplitude2:.2f}, FWHM: {fwhm2:.2f}, Center: {center2:.2f}, Fraction: {fraction2:.2f}")

    plt.plot(x_range, y_range, 'o', label='Data '+str(dat_theta.columns[i][0]))
    plt.plot(x_range, fit_result.best_fit, '-', label='Fit ' +str(dat_theta.columns[i][0]))
    plt.xlabel('2θ')
    plt.ylabel('Intensity')

    # store the information about the peak in a new dataframe 
plt.legend()
    
# %%



# %%
