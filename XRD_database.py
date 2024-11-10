#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.signal import find_peaks
import pickle

#%%

# Directory containing the text files
directory = r"Z:\P110143-phosphosulfides-Andrea\Data\Analysis\guidal\XRD\ref_database\reflections"

# List to store the DataFrames
dataframes = {}

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        filepath = os.path.join(directory, filename)
        
        # Read the text file into a DataFrame
        df = pd.read_csv(filepath,sep='\s+', engine='python',
                         header=0, 
                         names = ["h", "k","l","d (Å)","F(real)","F(imag)","|F|","2theta","I", "M","ID(λ)", "Phase"]
                         )
        
        # Get the file name without the extension
        file_key = os.path.splitext(filename)[0]
        
        # Store the DataFrame in the list with the file name as the key
        dataframes[file_key] = df

# Print the DataFrames to verify
for key, df in dataframes.items():
    print(f"DataFrame from file: {key}")
    print(df)
    print()


# dataframes.keys()
# 'Cu2PS3', 'Cu3PS4', 'Cu4P2S7', 'Cu7PS6', 'CuPS', 'CuPS3'
"print dataframes in vertical lines"

for key, df in dataframes.items():
    plt.figure(figsize=(10, 2))
    plt.vlines(x=df['2theta'], ymin=0, ymax=df['I'], label=key)
    
    plt.xlabel('2θ')
    plt.ylabel('I')
    plt.title('Vertical Lines from DataFrames')
    plt.legend()
    plt.show()

#%%

# Define a function to plot with Gaussian broadening for a specific dataframe
def plot_with_gaussian_broadening(df, label, broadening=0.1):
    plt.figure(figsize=(10, 2))
    savepath = os.path.join(directory, 'plots_with_broadening', f'XRD_ref-{label}.png')
    # Define the range for the x-axis
    x_range = np.linspace(20,80, 1000)
    y_broad = np.zeros_like(x_range)
    
    for _, row in df.iterrows():
        y_broad += row['I'] * norm.pdf(x_range, row['2theta'], broadening)
    
    new_df = pd.DataFrame({'2theta': x_range, 'I': y_broad})
    plt.plot(x_range, y_broad, label=label)
    #plt.xlim(20,80)
    plt.xlabel('2θ')
    plt.ylabel('Intensity')
    plt.title(f'Calculated reflections for {label} - Gaussian broadening={broadening}')
    plt.legend()
    plt.savefig(savepath, dpi=300)
    plt.show()
    return new_df

# Example usage with the specific dataframe
#plot_with_gaussian_broadening(dataframes["Cu2PS3"], label="Cu2PS3", broadening=0.1)

# # Loop through all dataframes and plot with Gaussian broadening
new_dataframes = {}

for key, df in dataframes.items():
    new_df= plot_with_gaussian_broadening(df, label=key, broadening=0.1)
    #plt.vlines(x=df['2theta'], ymin=0, ymax=df['I'], label=key)
    new_dataframes[key] = new_df
    



# %% find peaks and build a database table for all materials, store all info in a dictionary
peaks_df = {}
for key, df in new_dataframes.items():
    peaks, _ = find_peaks(df["I"], height=30, prominence=10, width=0.1)
    #sort and only use 8 highest peaks
    peaks = peaks[np.argsort(df.iloc[peaks]["I"])[::-1][:9]]
    #store the peaks in a dataframe
    peaks_df[key] = df.iloc[peaks].reset_index(drop=True)
    plt.plot(df["2theta"], df["I"], label=key)
    plt.plot(df["2theta"].iloc[peaks], df["I"].iloc[peaks], "x")
    plt.legend()
    plt.show()
    display(peaks_df[key])
    
# rename columns for better formatting
for key, df in peaks_df.items():
    df.rename(columns={"2theta": "Peak 2theta", "I": "Peak intensity"}, inplace=True)
    display(df)
# save dataframe to dictionary
ref_df= {}
for key,df in new_dataframes.items():
    new_df = pd.concat([peaks_df[key], df], axis=1)
    ref_df[key] = new_df
    #display(new_df)

# %% IF MAKING A NEW DATABASE: save the peaks_df dictionary to a csv file
for key, df in ref_df.items():
    df.to_csv(os.path.join(directory, f"ref_{key}.csv"), index=False)

# %%
# load properly the dict of dataframes from csv files in the dictionary

ref_peaks_df = {} 
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        key = filename.split("_")[1].split(".")[0]
        ref_peaks_df[key] = pd.read_csv(os.path.join(directory, filename))

display(ref_peaks_df)

# %% save the whole dictionary to a pickle file for later use
path = os.path.join(directory, "reflections.pkl")
with open (path, "wb") as file:
    pickle.dump(ref_peaks_df, file)

# %%

plt.plot(ref_peaks_df["Cu2S"]["2theta"], ref_peaks_df["Cu2S"]["I"], label="Cu2S")
plt.plot(ref_peaks_df["Cu2S - ICSD 23596"]["2theta"], ref_peaks_df["Cu2S - ICSD 23596"]["I"]*3, label="Cu2S - ICSD 23596")
plt.legend()
# %%
plt.plot(ref_peaks_df["Cu7S4"]["2theta"], ref_peaks_df["Cu7S4"]["I"],label= "Cu7S4")
plt.plot(ref_peaks_df["Cu7S4 - ICSD 16011"]["2theta"], ref_peaks_df["Cu7S4 - ICSD 16011"]["I"]*3,label= "Cu7S4 - ICSD 16011")
plt.legend()
# %%
