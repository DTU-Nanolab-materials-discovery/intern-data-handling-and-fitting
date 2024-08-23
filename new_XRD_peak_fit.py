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



#%% make a model for ka1. ka2 peak fitting

from lmfit.models import PseudoVoigtModel

def peak_model_xrd(num, i, peaks, col_counts, col_theta, params):
    '''Constructs a model for every peak based on peaks output from initial_peaks.
       This version uses the sum of two PseudoVoigt models with specific constraints.'''
    
    pref1 = f"f{num}_1_"
    pref2 = f"f{num}_2_"
    
    # Peak positions and intensities
    ypeak = peaks[col_counts[i]][peaks.index[num]].astype(float)
    xpeak = peaks[col_theta[i]][num].astype(float)
    
    # Calculate the constrained x-position for the second peak
    xpeak2 = 2 * np.arcsin((0.154439 / 0.1540562) * np.sin(xpeak / 2))
    
    # Initialize two PseudoVoigt models
    model1 = PseudoVoigtModel(prefix=pref1)
    model2 = PseudoVoigtModel(prefix=pref2)
    
    # Set common parameters
    A = ypeak
    M = 0.5
    #w = 0.5
    
    # Update parameters for the first model
    params.update(model1.make_params(
        center=dict(value=xpeak, min=xpeak * 0.9, max=xpeak * 1.1),
        amplitude=dict(value='A', min=0.2 * ypeak, max=1.2 * ypeak),
        #sigma=dict(value='w', min=0.1, max=10.0),
        fraction=dict(value='M', min=0, max=1)
    ))
    
    # Update parameters for the second model with constraints
    params.update(model2.make_params(
        center=dict(value=xpeak2, min=xpeak2 * 0.9, max=xpeak2 * 1.1),
        amplitude=dict(expr=f'{pref1}amplitude / 2'),  # Constrain amplitude to half of the first peak
        #sigma=dict(value=w, min=0.1, max=10.0),
        fraction=dict(value=M, min=0, max=1)
    ))
    
    # Combine the two models
    combined_model = model1 + model2
    
    return combined_model

# %% define the peak to fit
data_to_fit_x = data_out[('-16.0,-4.0',              '2θ (°)')]
data_to_fit_y = data_out[('-16.0,-4.0',              'Corrected Intensity')]

cut_range=1 
peak_pos = peaks[('-16.0,-4.0',         '2θ (°)')][0]
idx = np.where((data_to_fit_x >= peak_pos-cut_range) & (data_to_fit_x<=  peak_pos+cut_range))[0]
x_range = data_to_fit_x[idx]
y_range = data_to_fit_y[idx]
plt.plot(x_range, y_range)
# %% try t fit

params ={}

# Assuming you have defined the necessary variables for the function
num = 0  # Index of the peak you want to fit
i = 0  # Index for columns, if you have multiple datasets
col_counts = ['counts']  # Example column name for counts in the DataFrame
col_theta = ['theta']  # Example column name for theta in the DataFrame

# You should have your peaks DataFrame ready (here, I'm assuming peaks is defined)
# Example peaks DataFrame
the_peaks = pd.DataFrame({
    'counts': [peaks['-16.0,-4.0',         'Intensity, cps'][0]],  # Example counts for each peak
    'theta': [peak_pos]  # Example 2-theta positions for each peak
})

# Generate the combined model using your function
combined_model = peak_model_xrd(num, i, the_peaks, col_counts, col_theta, params)


# %%
from lmfit import Parameters, minimize

# Convert params to lmfit Parameters object
lmfit_params = Parameters()
for param_name, param_info in params.items():
    lmfit_params.add(param_name, **param_info)

# Fit the model to your data
result = combined_model.fit(y_range, params=lmfit_params, x=x_range)

# Print the fitting result
print(result.fit_report())
