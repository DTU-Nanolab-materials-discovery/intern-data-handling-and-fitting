#%%
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import os


folder= r"Z:\P110143-phosphosulfides-Andrea\Data\Samples\mittma_0011_Cu\PL mapper"
deposition = 'mittma_0011_Cu'
sample = 'mittma_0011_br'
square = 'BR'
plots_path= folder+'\\plots_'+square

os.makedirs(plots_path, exist_ok=True)


# %% plot of all PL spectra in the _full.dat file


# Step 1: Open the .dat file in read mode
file_path = folder + '\\' + sample+'-full.dat'

with open(file_path, 'r') as file:
    # Step 2: Read the contents of the file
    lines = file.readlines()

# Step 3: Initialize containers for metadata and data sections
metadata_lines = []
data_lines = []
data_section = False

# Step 4: Parse the contents and split at 'DATA:'
for line in lines:
    stripped_line = line.strip()
    if stripped_line == 'DATA:':
        data_section = True
        continue
    if data_section:
        data_lines.append(stripped_line)
    else:
        metadata_lines.append(stripped_line)

# Step 5: Create DataFrames
metadata_df = pd.DataFrame(metadata_lines, columns=['Metadata'])
data_df = pd.DataFrame(data_lines, columns=['Data'])

# Reprocess the data DataFrame
data_dict = {'Wavelength in nm': []}
current_header = 'Wavelength in nm'
data_dict[current_header] = []

for line in data_lines:
    if line.startswith('POS:'):
        # Extract the header from the line
        current_header = line.split('POS:')[1].strip()
        data_dict[current_header] = []
    else:
        data_dict[current_header].append(line)

# Convert the data dictionary to a DataFrame
data_spectral = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_dict.items()]))

# Remove columns that only contain NaN values
data_spectral.dropna(axis=1, how='all', inplace=True)

# Convert columns to numeric values
data_spectral = data_spectral.apply(pd.to_numeric, errors='coerce')

# Plot the data using Plotly
fig = px.line(data_spectral, x='Wavelength in nm', y=data_spectral.columns[1:], title=f'PL spectra in {deposition} {square}')

fig.write_html(plots_path +'\\'+ sample + '_full_spectra.html')
fig.write_image(plots_path +'\\'+ sample + '_full_spectra.png')
fig.show()





# %%

# ------------------------ data treatment ------------------------

good_data = data_spectral.copy()
bad_data = pd.DataFrame()
bad_data['Wavelength in nm'] = data_spectral['Wavelength in nm']
for column in data_spectral.columns[1:]:
    good_data[column] = good_data[column] - good_data[column].min()

    idx = np.where(good_data[column] == good_data[column].max())[0][0]
    wl_max = good_data.iloc[idx, 0]

    if good_data[column].max() - good_data[column].min() <= 0.06:
        bad_data[column] = data_spectral[column]
        good_data.drop(column, axis=1, inplace=True)
        
    elif wl_max < 700:
        bad_data[column] = data_spectral[column]
        good_data.drop(column, axis=1, inplace=True)

fig = px.line(good_data, x='Wavelength in nm', y=good_data.columns[1:], title=f'Good PL spectra in {deposition} {square}')

# fig = px.line(bad_data, x='Wavelength in nm', y=bad_data.columns[1:], title='Wavelength vs POS Columns - data to not use')
fig.write_html(plots_path +'\\'+ sample + '_good_spectra.html')
fig.write_image(plots_path +'\\'+ sample + '_good_spectra.png')
fig.show()

print(len(good_data.columns), len(bad_data.columns), len(data_spectral.columns))   



#%%

# %%
# %% heatmaps of the overview quantities

file_path = folder + '\\' + sample + '.dat'

# Initialize lists to hold the metadata and data
metadata = []
data = []

with open(file_path, 'r') as file:
    # Step 2: Read the contents of the file line by line
    data_section = False
    for line in file:
        stripped_line = line.strip()
        if not stripped_line:
            data_section = True
            continue
        if data_section:
            # Split the line into columns based on ';' delimiter
            columns = stripped_line.split(';')
            # print(columns)
            data.append(columns)
        else:
            # Split the line into columns based on ':' delimiter
            columns = stripped_line.split(':')
            metadata.append(columns)

# Convert the lists into DataFrames
metadata_df = pd.DataFrame(metadata)
data_df = pd.DataFrame(data, columns=data[0])
units = data_df.iloc[1]
# Ensure the first row is used as the header and skip the second row
data_df = data_df.drop([0,1]).reset_index(drop=True)
data_df = data_df.drop('', axis=1)

# Print the DataFrames
print("Metadata DataFrame:")
print(metadata_df)

print("\nData DataFrame:")
display(data_df)

# Function to plot scatter plots
def plot_scatter_plots(data_df):
    # Remove all spaces and tabs from the column names
    data_df.columns = data_df.columns.str.replace(' ', '').str.replace('\t', '')

    # Convert columns to numeric values where possible
    data_df = data_df.apply(pd.to_numeric, errors='ignore')

    # Sort the DataFrame by 'X' and 'Y' columns
    data_df = data_df.sort_values(by=['X', 'Y'], key=lambda col: col.astype(float))


    # Iterate over each column except 'X' and 'Y'
    for column in data_df.columns:
        if column not in ['X', 'Y']:
            fig = px.scatter(data_df, x='X', y='Y', color=column, size=None, title=f'Scatter Plot of {column} in {deposition} {square}')
            fig.update_traces(marker=dict(size=20))  # Set a constant size for all markers
            fig.update_layout(
                autosize=False,
                width=500,
                height=400,
                margin=dict(l=50, r=50, b=50, t=50, pad=4),
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )
            fig.write_image(plots_path +'\\'+ sample + '_'+column+'.png')
            fig.show()
            # Save the figure as an HTML file
            # pio.write_html(fig, file=f'scatter_plot_{column}.html', auto_open=False)

# Plot the scatter plots
plot_scatter_plots(data_df)

#%%

# find which points do or do not have pl
XX,YY=[], []
for column in good_data.columns[1:]: 
    # print(column)
    XX.append(float(column.split(', ')[0]))
    YY.append(float(column.split(', ')[1]))
# print(x,y)
xx,yy=[],[] 
for column in bad_data.columns[1:]:
    # print(column)
    xx.append(float(column.split(', ')[0]))
    yy.append(float(column.split(', ')[1]))

#%%

# Create the main scatter plot with good data points
fig = go.Figure()


fig.add_trace(go.Scatter(
    x=data_df[data_df.columns[0]].values.astype(float),
    y=data_df[data_df.columns[1]].values.astype(float),
    mode='markers',
    marker=dict(
        size=20,
        color=data_df['Int.Signal'].values.astype(float),
        colorscale='Plasma',  # Add a colorscale for better visualization
        colorbar=dict(title='Integrated signal'),
        symbol='circle'
    ),
    name='Peak Int',
    showlegend=False
))

# Add good data points in green
fig.add_trace(go.Scatter(
    x=XX,
    y=YY,
    mode='markers',
    marker=dict(
        size=20,
        color='green',
        symbol='circle-open'
    ),
    name='Good Data',
    showlegend=False
))

# Add bad data points in red
fig.add_trace(go.Scatter(
    x=xx,
    y=yy,
    mode='markers',
    marker=dict(
        size=10,
        color='white',
        symbol='x'
    ),
    name='Bad Data',
    showlegend=False
))


# Update layout for better visualization
fig.update_layout(
    title=f'Good and Bad Data Points in {deposition} {square}',
    xaxis_title='X',
    yaxis_title='Y',
    autosize=False,
    width=500,
    height=400,
    margin=dict(l=50, r=50, b=50, t=50, pad=4),
    xaxis=dict(scaleanchor="y", scaleratio=1),
    yaxis=dict(scaleanchor="x", scaleratio=1)
)

# Show the figure
fig.write_image(plots_path +'\\'+ sample + '_good_signal.png')
fig.show()

# %%
