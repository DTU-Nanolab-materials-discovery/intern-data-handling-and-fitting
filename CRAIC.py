#%%
from functions import *
#%%
folder = r"Z:\P110143-phosphosulfides-Andrea\Students\Giulia\01_Characterization\CRAIC\mittma_0018_Cu"
#%%
#define which file is the background you want to remover from the reflection data
file_background = "Nothing_after_restart.msp"

background = read_CRAIC(os.path.join(folder, file_background), print_header=False)
print(background)

plt.plot(background["Wavelength"], background["Intensity"])


# %% what are these files? Unsure if they are needed
dark_scan= read_CRAIC(os.path.join(folder, "darkscan.msp"))
plt.plot(dark_scan["Wavelength"], dark_scan["Intensity"])

refscan = read_CRAIC(os.path.join(folder, "refscan.msp"))
plt.plot(refscan["Wavelength"], refscan["Intensity"])


# %% ----- real data ----- READ ONE FILE AT A TIME 

sample_after_restart = read_CRAIC(os.path.join(folder, "sample_after_resrart.msp"))
plt.plot(sample_after_restart["Wavelength"], sample_after_restart["Intensity"], label="Sample")
plt.plot(background["Wavelength"], background["Intensity"], label="Background")

#%%
clean_intensity = sample_after_restart["Intensity"] - background["Intensity"]
plt.plot(sample_after_restart["Wavelength"], clean_intensity, label="Cleaned")
plt.plot(sample_after_restart["Wavelength"], sample_after_restart["Intensity"], label="Sample")
plt.plot(background["Wavelength"], background["Intensity"], label="Background")
plt.legend()
# %% --------------------------------
# READ AND PLOT MAP FILES

background = "Nothing_after_restart.msp"
reflection_name = "sample_after_resrart1"
transmission_name = "transmission_sample_map"

# make a grid, make sure that it is done properly (**** check in craic that it moves always in snake from bottom left, horizontal to bottom righ, then up.. ****)
# cooridinates are not saved, but files in a map are numbered with this order
x = np.linspace(1, 3, 3)
y= np.linspace(1, 3, 3)

grid = snake_grid(x, y)
print(grid)

#plots R, T and absorption coefficient in all points. returns a dataframe with the data and the coordinates
data= CRAIC_map(folder, background, reflection_name, transmission_name, 
                grid, thickness= 10**(-7), # in m (100 nm = 10**(-7) m)
                 unit= "eV",
                 savepath=None # specify savepath if you want to save it
                 ) 




# %%
plot_data(data, datatype_x= "Wavelength",datatype_y="T", x=[1,2,3], y=[1,1,1])
# %%
display(data)
# %%
