#%%
from functions import *
#%%
sample = "mittma_00015_BR"
folder = r"O:\Nlab\Public\DCH-plasma\phosphosulfides_students\Students\Giulia\01_Characterization\XRD\raw data"

# %% ##############################################
##############################################

#grid = measurement_grid(5,10,32,72,-16,-40) # mittma_00012_first50
#grid = measurement_grid(5,5,32,32,-16,-36) # for one sample FR (mittma 00015)
grid = measurement_grid(5,5,32,32,-16,4) # for one sample BR (mittma 00015)

#filename = "XRD_0003_5points.txt"
#filename = "XRD-mittma_00012_first50.txt" 
filename = "XRD_mittma_00015_BR_map.txt" 
filename = os.path.join(folder, filename)
 # for saving the pickle and the plots

initial_data, coords = read_XRD(filename, grid, n = 0, separator = "\t")
plot_grid(coords,grid)
#%% 
# ---------------------- if you want to work only on a limited number of points ----------------------
which =initial_data.keys()[0:8]
data= initial_data[which]
display(data)
#%% 
# ----------------------- translate data ---------------------
data, coords = translate_data(initial_data, x=0, y=-20)
MI_to_grid(data) #prints new coordinates
#%%
# ----------------------- preliminary finidng peaks ---------------------

dataRangeMin = 0
dataRangeMax = len(data)
filterstrength = 10 # higher number = more noise is removed by the filter
peakprominence = 280 # higher number = only peaks with prominence above this value are considered
peakwidth = 8

peaks, dataCorrected = initial_peaks(data, dataRangeMin, dataRangeMax, filterstrength,
                                         peakprominence, peakwidth,
                                         withplots = True, plotscale = 'log')


#%% 
# ---------------- remove the background ----------------
data_out = XRD_background(data,peaks, cut_range=1, order=4, withplots= True, Si_cut=True)

#%%
# ----------------------- save clean data for future use ---------------------

name = sample + "_clean.pkl"

with (open(os.path.join(folder,"pickles", name), "wb")) as openfile:
    pickle.dump(data_out, openfile)

#%%
# ----------------------- load clean data, if previously processed  ---------------------

name = sample + "_clean.pkl"
with open(os.path.join(folder,"pickles", name), "rb") as openfile:
    data_out = pickle.load(openfile)



#%% ##################################################################################
################################# ANALYSIS SECTION ###################################
######################################################################################

# fit one peak with two pseudovoigts, to plot trends for that one peak

# ------------------------- first peak -------------------------
df_first_peak = fit_this_peak(data_out, 28.27, 0.6, withplots = True, printinfo = False)


list_to_plot = ["Center", "Amplitude", "FWHM", "Fraction"]
plots_path = r"O:\Nlab\Public\DCH-plasma\phosphosulfides_students\Students\Giulia\01_Characterization\XRD\plots\mittma_00015"
for item in list_to_plot:
    #savepath = os.path.join(plots_path, sample+" first_peak "+ item +" .png")
  
    new_heatmap(item, data= df_first_peak, title =  sample + " First Peak - " + item, 
                #savepath = savepath
                )
#%% ------------------------- second peak -------------------------

df_second_peak = fit_this_peak(data_out, 29.4, 0.6, withplots = True, printinfo = False)


for item in list_to_plot:
    savepath = os.path.join(plots_path, sample+" second_peak "+ item +" .png")
    new_heatmap(datatype = item, data= df_second_peak, title = sample +" Second Peak - " + item, savepath = savepath)


# %% if you want to look at specific point in the dataset you can call them with plot_data
plt.figure(figsize=(10, 8))
x=[-16,-16]
y=[-16,-8]

plot_data(df_second_peak, 'range 2θ', 'Fit', x,y, plotscale = 'linear')
plot_data(df_second_peak, 'range 2θ', 'range Intensity',x,y, legend =False, scatter_plot=False,  plotscale = 'linear' )
#plt.savefig(os.path.join(plots_path, "mittma_00015_FR_second_peak_weid points.png"))


#%% 
# load reflections table and match peaks or calculate shifts
ref_path = r"O:\Nlab\Public\DCH-plasma\phosphosulfides_students\Students\Giulia\01_Characterization\XRD\ref_database\reflections"
with open(os.path.join(ref_path, "reflections.pkl"), 'rb') as f:
    ref_peaks_df = pickle.load(f)

ref_peaks_df["Cu3PS4-ICSD"]


#%% 
# ------------------- calculate shift from reference -------------------
ref_peak_pos = 28.228228

data= df_first_peak.copy()
dat_center = data.iloc[:,data.columns.get_level_values(1)=='Center']
dat_center = dat_center.dropna()
shift = dat_center.values - ref_peak_pos

data = math_on_columns(data, 'Center', type2= ref_peak_pos, operation = "-")
data.rename(columns={'Center - 28.228228':'Center - Reference'}, inplace = True)
#%%
new_heatmap("Center - Reference", data= data, title = "First Peak - Shift from reference", savepath = os.path.join(plots_path, sample+" first_peak_shift.png"))

#%%
plot_data(df_first_peak, 'range 2θ', 'Fit', plotscale = 'linear')
plt.vlines(ref_peak_pos, 0, 10000, colors='k', linestyles='--', label='Reference ICSD', alpha=0.5)






















#%%

#%% fit one peak with two pseudovoigts, simple version
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


# Example usage: ( first cut data x_range, y_range)
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