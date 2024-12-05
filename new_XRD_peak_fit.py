#%%
from functions import *
#%%
sample = "mittma_00019_FR"
folder = r"Z:\P110143-phosphosulfides-Andrea\Data\Samples\mittma_0019_Cu\XRD"

# %% ##############################################
##############################################

#grid = measurement_grid(5,10,32,72,-16,-40) # mittma_00012_first50 (2 samples)
#grid = measurement_grid(5,5,32,32,-16,-36) # for one sample FR (mittma 00015)
#grid = measurement_grid(5,5,32,32,-16,4) # for one sample BR (mittma 00015)
#grid = measurement_grid(5,5,30,30,-15,-15) #5points
#grid = measurement_grid(5,5,34,34,-17,3) # BR_map 3mm margin
#grid = measurement_grid(5,5,34,34,-17,-37) # FR_map 3mm margin
grid = measurement_grid(5,9,40,80,-20,-40) #  2 slow scans ( 2 center points)

#filename = "XRD_0003_5points.txt"
#filename = "XRD-mittma_00012_first50.txt" 
filename = "mittma_0019_R_slowscans.txt" 
#filename = "mittma_0017_GIXRD_5points.txt"
filename = os.path.join(folder, filename)
 # for saving the pickle and the plots

initial_data, coords = read_XRD(filename, grid, n = 0, separator = "\t")
plot_grid(coords,grid)
#%%
datatype_y= 'Intensity, cps'
datatype_x='2θ (°)'
plot_data(initial_data, datatype_x, datatype_y ,x=0 , y=20 , plotscale = 'log')
#%% 
# ---------------------- if you want to work only on a limited number of points ----------------------
which =initial_data.keys()[0:8]
data= initial_data[which]
display(data)
#%% 
# ----------------------- translate data ---------------------
data, coords = translate_data(initial_data, x=0, y=20)
MI_to_grid(data) #prints new coordinates
#%%
# ----------------------- preliminary finidng peaks ---------------------
#data= initial_data.copy() # un-comment if you want to work on original data
dataRangeMin = 0
dataRangeMax = len(data)
filterstrength = 15 # higher number = more noise is removed by the filter
peakprominence = 60 # higher number = only peaks with prominence above this value are considered
peakwidth = 15 # higher number = wider peaks are considered

peaks, dataCorrected = initial_peaks(data, dataRangeMin, dataRangeMax, filterstrength,
                                         peakprominence, peakwidth,
                                         withplots = True, plotscale = 'linear')


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
with open(os.path.join(folder, name), "rb") as openfile:
    data_out = pickle.load(openfile)


#%%
# ----------------------- plot all points ---------------------
x= [-17,-8.5,0,8.5,17]
y1 =[-17,-17,-17,-17,-17]
y2= [-8.5,-8.5,-8.5,-8.5,-8.5]
y3 = [0,0,0,0,0]
y4 = [8.5,8.5,8.5,8.5,8.5]
y5 = [17,17,17,17,17]
y= [y1,y2,y3,y4,y5]
plot_path = r"O:\Nlab\Public\DCH-plasma\phosphosulfides_students\Students\Giulia\01_Characterization\XRD\plots\mittma_0019"
for pos in y:
    #savepath = os.path.join(plot_path, sample+"_"+str(pos[0])+".png")
    plot_XRD_shift(data_out, '2θ (°)', 'Corrected Intensity',  shift=200, x=x, y=pos, title=sample+" y = "+str(pos[0]), savepath= False)
#plot_XRD_shift(data_out, '2θ (°)', 'Corrected Intensity',  shift=200, x=x, y=y1, title=sample+" y = "+str(y1[0]), savepath= False)
#%% ##################################################################################
################################# ANALYSIS SECTION ###################################
######################################################################################

# fit one peak with two pseudovoigts, to plot trends for that one peak

# ------------------------- first peak -------------------------
peak_position = 27.81
df_first_peak = fit_this_peak(data_out, peak_position, 0.6, withplots = True, printinfo = False)

#%%
list_to_plot = ["Center", "Amplitude", "FWHM", "Fraction"]
plots_path = r"Z:\P110143-phosphosulfides-Andrea\Data\Analysis\guidal\XRD\mittma_0019"
for item in list_to_plot:
    savepath = os.path.join(plots_path, f'{sample}_{peak_position}_{item}.png')
  
    new_heatmap(item, data= df_first_peak, exclude=["-8.5,-17.0"], title =  sample + f' peak at 2θ= {peak_position}° - ' + item, 
                savepath = savepath
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
#------------------- plot specific data for specific comparisons -------------------

name_GI = "mittma_00017_GI_clean.pkl"
with open(os.path.join(folder,"pickles", name_GI), "rb") as openfile:
    data_GI = pickle.load(openfile)

name_map = "mittma_00017_clean.pkl"
with open(os.path.join(folder,"pickles", name_map), "rb") as openfile:
    data_map = pickle.load(openfile)


plot_data(data_GI, '2θ (°)', 'Corrected Intensity', plotscale = 'linear', title = "Grazing incidence, w= 1 $^\circ$")
plt.show()
plot_data(data_map, '2θ (°)', 'Corrected Intensity', plotscale = 'linear', title = "w/2θ , offset = 5$^\circ$")

#%%
x= [[-15],[-15],[0],[15],[15]]
y= [[-15],[-15],[0],[15],[15]]

for i in range(5):
    print(x[i],y[i])
    plot_data(data_GI, '2θ (°)', 'Corrected Intensity', x[i],y[i], plotscale = 'linear', title = "Grazing incidence, w= 1 $^\circ$")
    plt.show()
    plot_data(data_map, '2θ (°)', 'Corrected Intensity', x[i],y[i], plotscale = 'linear', title = "w/2θ , offset = 5$^\circ$")
    plt.show()

#%%
x= [-15,-15,0,15,15]
y= [-15,-15,0,15,15]
plot_XRD_shift(data_GI,'2θ (°)', 'Corrected Intensity',  shift=120, x=x, y=y, title="mittma_0017 GIXRD, w= 1 $^\circ$", savepath= False)

#%%
plot_XRD_shift(data_map,'2θ (°)', 'Corrected Intensity',  shift=500, x=x, y=y, title="mittma_0017 w/2θ, offset = 5$^\circ$", savepath= False)

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


#%%
############################ WORK IN PROGRESS ########################################

# plot all points or some points with a vertical shift for each point optionally with a reference and its lines

def plot_XRD_shift(data,datatype_x, datatype_y,  shift,x,y, ref_label = "Reference", title=None, savepath= False, reference = None, ref_lines = False ): #x, y = list of points to plot]
    'plot data with a vertical shift for each point optionally with a reference and its lines'
    x_data = []
    y_data = []
    labels = []
    colors = plt.cm.jet(np.linspace(0, 1, len(x)))
    
    for i in range(len(x)):
        x_data.append(get_data(data, datatype_x, x[i], y[i], False,False))
        y_data.append(get_data(data, datatype_y, x[i], y[i], False,False))
        if x[0] == "all" and y[0] == "all":
            labels = data.columns.get_level_values(0).unique().values
        else:
            grid = MI_to_grid(data)
            xcoord, ycoord = closest_coord(grid, x[i], y[i])
            labels.append('{:.1f},{:.1f}'.format(xcoord, ycoord))

        plt.plot(x_data[i], y_data[i]+ shift*i, color =colors[i], label = labels[i])

    if reference is not None:
        plt.plot(reference["2theta"], reference["I"]*0.5+ shift*(-2), label= ref_label)
    if ref_lines == True:
        plt.vlines(reference["Peak 2theta"], shift*(-2), plt.ylim()[1], colors='k', linestyles='--', alpha=0.3)

    plt.xlabel(datatype_x)
    plt.ylabel(datatype_y)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if savepath:
        path = os.path.join('plots', title + 'shift.png')
        plt.savefig(path, dpi=120, bbox_inches='tight')
        
    plt.show()

#%%

x= [-17,-8.5,0,8.5,17]
y1 =[-17,-17,-17,-17,-17]
y2= [-8.5,-8.5,-8.5,-8.5,-8.5]
y3 = [0,0,0,0,0]
y4 = [8.5,8.5,8.5,8.5,8.5]
y5 = [17,17,17,17,17]
y= [y1,y2,y3,y4,y5]
plot_path = r"O:\Nlab\Public\DCH-plasma\phosphosulfides_students\Students\Giulia\01_Characterization\XRD\plots\mittma_0019"
material = "Cu7S4"
#for pos in y:
fig = plt.figure(figsize=(12, 6))
    #savepath = os.path.join(plot_path, sample+"_"+str(pos[0])+".png")
plot_XRD_shift(data_out, '2θ (°)', 'Corrected Intensity',  shift=400, x=x, y=y5, title=sample+" y = "+str(pos[0]),   savepath= False)

#%%

ref_path = r"Z:\P110143-phosphosulfides-Andrea\Data\Analysis\guidal\XRD\ref_database\reflections"

with open(os.path.join(ref_path, "reflections.pkl"), 'rb') as f:
    ref_peaks_df = pickle.load(f)
# %%
x= MI_to_grid(data_out)["x"].values[0::5]
y1 =MI_to_grid(data_out)["y"].values[0::5]
reference = ref_peaks_df["Cu2P7"]

plt.figure(figsize = (12,20))
plot_XRD_shift(data_out, '2θ (°)', 'Corrected Intensity',  shift=400, x=x, y=y1, title=sample+" y = "+str(y1[0]), reference = reference, savepath= False)
# %%
# --------------- interactive plot ------------------
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def rgba_to_hex(rgba):
    """Convert an RGBA tuple to a hex color string."""
    r, g, b, a = [int(c * 255) for c in rgba]
    return f'#{r:02x}{g:02x}{b:02x}'

def plot_XRD_shift_plotly(data, datatype_x, datatype_y, shift, x, y, ref_peaks_df, ref_label="Reference", title=None):
    # Create subplots with reduced vertical spacing
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        row_heights=[0.8, 0.2],  # Proportion of height for each plot
        vertical_spacing=0.02    # Adjust this to reduce space between plots
    )

    x_data = []
    y_data = []
    #colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    colormap = plt.get_cmap('turbo')  # You can choose any matplotlib colormap
    colors = [rgba_to_hex(colormap(i / len(x))) for i in range(len(x))]  # Convert colors to hex

    # Store all y-data to find the global maximum
    all_y_data = []

    # Loop through and plot the XRD spectra with a vertical shift in the top plot
    for i in range(len(x)):
        x_data = get_data(data, datatype_x, x[i], y[i], False, False)
        y_data = get_data(data, datatype_y, x[i], y[i], False, False)
        shifted_y_data = y_data + shift * i
        
        all_y_data.extend(shifted_y_data)  # Collect y-data with shift for max computation
        
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=shifted_y_data,
                mode='lines',
                line=dict(color=colors[i]),
                name=f'{x[i]}, {y[i]}'
            ),
            row=1, col=1
        )

    # Compute the global maximum y-value, considering shifts
    global_max_y = max(all_y_data)

    # Create traces for each reference material (hidden initially)
    ref_traces = []
    buttons = []

    for ref_material, ref_df in ref_peaks_df.items():
        # Reference spectrum plotted in the bottom plot
        ref_trace = go.Scatter(
            x=ref_df["2theta"],
            y=ref_df["I"] + shift * (-1),  # Shift reference spectrum down
            mode='lines',
            name=f'{ref_material} Reference',
            visible=False
        )
        
        # Create vertical peak lines for top plot (raw data plot)
        peak_lines = go.Scatter(
            x=[value for peak in ref_df["Peak 2theta"] for value in [peak, peak, None]],  # x: peak, peak, None to break the line
            y=[-100, global_max_y * 1.1, None] * len(ref_df["Peak 2theta"]),  # y: 0 -> global_max_y for each line, with None to break lines
            mode='lines',
            line=dict(color='grey', dash='dot'),
            showlegend=False,
            visible=False
        )

        # Append traces for each reference spectrum and its peaks
        ref_traces.append(ref_trace)
        ref_traces.append(peak_lines)
        
        # Create a button for each reference
        buttons.append(dict(
            label=ref_material,
            method='update',
            args=[{'visible': [True] * len(x) + [False] * len(ref_traces)},  # Show all raw spectra, hide refs by default
                  {'title': f'{title} - {ref_material} Reference'}]
        ))

    # Add reference traces to figure (initially hidden)
    for trace in ref_traces:
        # Ensure trace.name is not None before checking 'Reference' in name
        fig.add_trace(trace, row=2 if trace.name and 'Reference' in trace.name else 1, col=1)

    # Update buttons to control the visibility of one reference at a time
    for i, button in enumerate(buttons):
        # Make the selected reference spectrum visible in the bottom plot and its peaks visible in the top plot
        button['args'][0]['visible'][len(x):] = [False] * len(ref_traces)  # Hide all refs initially
        button['args'][0]['visible'][len(x) + 2 * i:len(x) + 2 * i + 2] = [True, True]  # Show selected ref and peaks

    # Add the dropdown menu to switch between reference spectra
    fig.update_layout(
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 1.05,
            'xanchor': 'left',
            'y': 1.1,
            'yanchor': 'top'
        }],
        template='plotly_white',  # Choose a template (e.g., 'plotly_dark')
        title=title,
        height=500,  # Adjust the height of the figure (e.g., 700)
        width=900,   # Adjust the width of the figure (e.g., 900)
        legend=dict(x=1.05, y=1),
        xaxis2_title=datatype_x,
        yaxis_title=datatype_y
    )


    return fig
# Example usage
# plot_XRD_shift_plotly(data, '2theta', 'Intensity', 200, [0, 1], [0, 1], ref_peaks_df, title="XRD Spectra")

# %%
plot_XRD_shift_plotly(data_out, '2θ (°)', 'Corrected Intensity', 200, [-17, -8.5], [17,17], ref_peaks_df, title="XRD Spectra")

# %%
x = [-17, -8.5, 0, 8.5, 17,
-17, -8.5, 0, 8.5, 17,
-17, -8.5, 0, 8.5, 17,
-17, -8.5, 0, 8.5, 17,
-17, -8.5, 0, 8.5, 17]

y = [-17] * 5 + [-8.5] * 5 + [0] * 5 + [8.5] * 5 + [17] * 5
# %%
fig= plot_XRD_shift_plotly(data_out, '2θ (°)', 'Corrected Intensity', 300, x,y, ref_peaks_df, title="XRD Spectra")
fig.show()
# %%
savepath= r"Z:\P110143-phosphosulfides-Andrea\Data\Analysis\guidal\XRD"
savepath = os.path.join(savepath, sample+"_all_interactive")
#fig.write_image(savepath+".png", scale=2) #scale sets the resolution here
        
fig.write_html(savepath+".html")


# %%
options = ['option1', 'option2', 'option3']
print("Choose an option:")
for i, option in enumerate(options, 1):
    print(f"{i}. {option}")

choice = input("Enter the number of your choice: ")
try:
    choice_index = int(choice) - 1
    if 0 <= choice_index < len(options):
        print(f"You chose: {options[choice_index]}")
    else:
        print("Invalid choice. Please select a valid number.")
except ValueError:
    print("Invalid input. Please enter a number.")

# %%
def get_user_input():
    positions = input("Which points show the same phase? ")
    phase = input("Which phase is it? ")
    
    try:
        positions = int(age)
        print(f"Hello, {name}. You are {age} years old.")
    except ValueError:
        print("Invalid age. Please enter a number.")

get_user_input()

# %%
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
from scipy.interpolate import griddata
from scipy.signal import find_peaks
import seaborn as sns
from lmfit.models import PseudoVoigtModel, SplineModel,LinearModel, GaussianModel
from lmfit import Parameters
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

from functions import select_points, get_data, extract_coordinates

folder= r"Z:\P110143-phosphosulfides-Andrea\Data\Samples\mittma_0023_Cu\XRD"
name = "mittma_0023_FR_clean.pkl"
with open(os.path.join(folder, name), "rb") as openfile:
    data_out = pickle.load(openfile) 

#  load refence spectra

ref_path = r"Z:\P110143-phosphosulfides-Andrea\Data\Analysis\guidal\XRD\ref_database\reflections"

with open(os.path.join(ref_path, "reflections.pkl"), 'rb') as f:
    ref_peaks_df = pickle.load(f)
#%%
def rgba_to_hex(rgba):
    """Convert an RGBA tuple to a hex color string."""
    r, g, b, a = [int(c * 255) for c in rgba]
    return f'#{r:02x}{g:02x}{b:02x}'

def interactive_XRD_shift(data, datatype_x, datatype_y, shift, x, y, ref_peaks_df, ref_label="Reference", title=None, colors='rows'):
    'interactive shifted plot for assigning phases to XRD data, specify if you want different colors per each row or a rainbow colormap'
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        row_heights=[0.8, 0.2],  # Proportion of height for each plot
        vertical_spacing=0.02    # Adjust this to reduce space between plots
    )
    
    if colors == 'rows':
        # Define a color palette with as many colors as there are unique values in y
        coords_colors = pd.DataFrame({'X': x, 'Y': y})
        unique_y_values = coords_colors['Y'].unique()
        
        color_palette = px.colors.qualitative.Plotly[:len(unique_y_values)]
        
        unique_x_values = coords_colors['X'].unique()
        color_dict = {}
        for i, color in enumerate(color_palette):
            # Generate lighter hues of the color for each x value
            base_color = mcolors.to_rgb(color)
            lighter_hues = [mcolors.to_hex((base_color[0] + (1 - base_color[0]) * (j / len(unique_x_values)),
                                            base_color[1] + (1 - base_color[1]) * (j / len(unique_x_values)),
                                            base_color[2] + (1 - base_color[2]) * (j / len(unique_x_values))))
                            for j in range(len(unique_x_values))]
            color_dict[unique_y_values[i]] = lighter_hues
        coords_colors['Color'] = coords_colors.apply(lambda row: color_dict[row['Y']][list(unique_x_values).index(row['X'])], axis=1)
        colors = coords_colors['Color'].values

    elif colors == 'rainbow':
        colormap = plt.get_cmap('turbo')  # You can choose any matplotlib colormap
        colors = [rgba_to_hex(colormap(i / len(x))) for i in range(len(x))]  # Convert colors to hex
    
    x_data = []
    y_data = []
    # Store all y-data to find the global maximum
    all_y_data = []
    # Loop through and plot the XRD spectra with a vertical shift in the top plot
    for i in range(len(x)):
        x_data = get_data(data, datatype_x, x[i], y[i], False, False)
        y_data = get_data(data, datatype_y, x[i], y[i], False, False)
        shifted_y_data = y_data - shift * i
        
        all_y_data.extend(shifted_y_data)  # Collect y-data with shift for max computation
        
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=shifted_y_data,
                mode='lines',
                line=dict(color=colors[i]),
                name=f'{x[i]}, {y[i]}'
            ),
            row=1, col=1
        )

    # Compute the global maximum y-value, considering shifts
    global_min_y = min(all_y_data)

    # Create traces for each reference material (hidden initially)
    ref_traces = []
    buttons = []

    for ref_material, ref_df in ref_peaks_df.items():
        # Reference spectrum plotted in the bottom plot
        ref_trace = go.Scatter(
            x=ref_df["2theta"],
            y=ref_df["I"],
            mode='lines',
            name=f'{ref_material} Reference',
            visible=False
        )
        
        # Create vertical peak lines for top plot (raw data plot)
        peak_lines = go.Scatter(
            x=[value for peak in ref_df["Peak 2theta"] for value in [peak, peak, None]],  # x: peak, peak, None to break the line
            y=[global_min_y, 1000 * 1.1, None] * len(ref_df["Peak 2theta"]),  # y: 0 -> global_max_y for each line, with None to break lines
            mode='lines',
            line=dict(color='grey', dash='dot'),
            showlegend=False,
            visible=False
        )

        # Append traces for each reference spectrum and its peaks
        ref_traces.append(ref_trace)
        ref_traces.append(peak_lines)
        
        # Create a button for each reference
        buttons.append(dict(
            label=ref_material,
            method='update',
            args=[{'visible': [True] * len(x) + [False] * len(ref_traces)},  # Show all raw spectra, hide refs by default
                  {'title': f'{title} - {ref_material} Reference'}]
        ))

    # Add reference traces to figure (initially hidden)
    for trace in ref_traces:
        # Ensure trace.name is not None before checking 'Reference' in name
        fig.add_trace(trace, row=2 if trace.name and 'Reference' in trace.name else 1, col=1)

    # Update buttons to control the visibility of one reference at a time
    for i, button in enumerate(buttons):
        # Make the selected reference spectrum visible in the bottom plot and its peaks visible in the top plot
        button['args'][0]['visible'][len(x):] = [False] * len(ref_traces)  # Hide all refs initially
        button['args'][0]['visible'][len(x) + 2 * i:len(x) + 2 * i + 2] = [True, True]  # Show selected ref and peaks

    # Add the dropdown menu to switch between reference spectra
    fig.update_layout(
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 1.05,
            'xanchor': 'left',
            'y': 1.1,
            'yanchor': 'top'
        }],
        template='plotly_white',  # Choose a template (e.g., 'plotly_dark')
        title=title,
        height=600,  # Adjust the height of the figure (e.g., 700)
        width=900,   # Adjust the width of the figure (e.g., 900)
        legend=dict(x=1.05, y=1),
        xaxis2_title=datatype_x,
        yaxis_title=datatype_y
    )

    return fig

# %%
#  usage

x,y = select_points(data_out, x_max=10)
print(x, y) 
fig= interactive_XRD_shift(data_out, '2θ (°)', 'Corrected Intensity', 400, x,y, ref_peaks_df, title= ' color test')
fig.show()
# %%
x,y = extract_coordinates(data_out)
# x,y = select_points(data_out, x_max=10)
coords_colors = pd.DataFrame({'X': x, 'Y': y})

# %%
import plotly.express as px
import matplotlib.colors as mcolors

coords_colors = pd.DataFrame({'X': x, 'Y': y})
unique_y_values = coords_colors['Y'].unique()
# Define a color palette with as many colors as there are unique values in y
color_palette = px.colors.qualitative.Plotly[:len(unique_y_values)]
# Count the number of unique values in x
unique_x_values = coords_colors['X'].unique()
# Create a dictionary to store the colors for each unique value in y
color_dict = {}
for i, color in enumerate(color_palette):
    # Generate lighter hues of the color
    base_color = mcolors.to_rgb(color)
    lighter_hues = [mcolors.to_hex((base_color[0] + (1 - base_color[0]) * (j / len(unique_x_values)),
                                    base_color[1] + (1 - base_color[1]) * (j / len(unique_x_values)),
                                    base_color[2] + (1 - base_color[2]) * (j / len(unique_x_values))))
                    for j in range(len(unique_x_values))]
    color_dict[unique_y_values[i]] = lighter_hues
coords_colors['Color'] = coords_colors.apply(lambda row: color_dict[row['Y']][list(unique_x_values).index(row['X'])], axis=1)


# %%
fig= interactive_XRD_shift(data_out, '2θ (°)', 'Corrected Intensity', 400, x,y, ref_peaks_df, title= ' color test, same color= same y', colors='rainbow')
fig.show()
# %%
