# %%
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from functions import get_data, select_points, add_info, MI_to_grid, extract_coordinates

# Example data as a MultiIndex DataFrame
data = {
    ('Point1', 'Cu %'): [30],
    ('Point1', 'P %'): [40],
    ('Point1', 'S %'): [30],
    ('Point1', 'Intensity'): ['A'],
    ('Point1', 'Phase'): ['a'],
    ('Point2', 'Cu %'): [50],
    ('Point2', 'P %'): [20],
    ('Point2', 'S %'): [30],
    ('Point2', 'Intensity'): ['B'],
    ('Point2', 'Phase'): ['b'],
    ('Point3', 'Cu %'): [10],
    ('Point3', 'P %'): [50],
    ('Point3', 'S %'): [40],
    ('Point3', 'Intensity'): ['C'],
    ('Point3', 'Phase'): ['c'],
}

# Creating a MultiIndex DataFrame
index = pd.MultiIndex.from_tuples(data.keys(), names=['Coordinates', 'Element'])
df = pd.DataFrame(list(data.values()), index=index).transpose()


import pandas as pd
import plotly.graph_objects as go

def ternary_discrete_attempt(df, el1, el2, el3, intensity_label, shape_label, title):
    """
    Create a ternary plot with discrete colors for string intensities and different marker shapes for phases.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    el1 (str): Label for the first element.
    el2 (str): Label for the second element.
    el3 (str): Label for the third element.
    intensity_label (str): Label for the intensity values.
    shape_label (str): Label for the phase values.
    title (str): Title of the plot.
    """

    # Extract element percentages, intensity, and phase
    A= 'Layer 1 Cu Atomic %'
    B= 'Layer 1 P Atomic %'
    C= 'Layer 1 S Atomic %'

    A_percent = get_data(df, A).loc[0].values.flatten()
    B_percent = get_data(df, B).loc[0].values.flatten()
    C_percent = get_data(df, C).loc[0].values.flatten()
    intensity = get_data(df, intensity_label).loc[0]
    phase = get_data(df, shape_label).loc[0]
    X,Y= extract_coordinates(df)

    # Create a color mapping for unique intensity values

    unique_intensities = list(set(intensity))
    color_map = {val: i for i, val in enumerate(unique_intensities)}
    colors = [color_map[val] for val in intensity]

    # Create a marker shape mapping for unique phase values
    unique_phases = list(set(phase))
    marker_shapes = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right']
    shape_map = {val: marker_shapes[i % len(marker_shapes)] for i, val in enumerate(unique_phases)}
    shapes = [shape_map[val] for val in phase]

    custom_data = list(zip(X, Y, intensity))
    print(custom_data)

    # Create the ternary plot with custom hover text, colored markers, and different shapes
    fig = go.Figure(go.Scatterternary({
        'mode': 'markers',
        'a': A_percent,  # el1 percentages
        'b': B_percent,  # el2 percentages
        'c': C_percent,  # el3 percentages
        'marker': {
            'size': 8,
            'color': colors,  # Use mapped colors for marker color
            'symbol': shapes,  # Use mapped shapes for marker shape
            'colorscale': 'Turbo',  # Choose a colorscale
            'colorbar': {
                'title': intensity_label,
                'tickvals': list(color_map.values()),  # Set tick values to the mapped color indices
                'ticktext': unique_intensities  # Set tick text to the unique intensity values
            },
            'line': {'width': 0}
        },
        # 'text': df.columns.get_level_values(0).unique(),  # Labels for the points
        # 'hovertemplate': f'{el1}: %{{a:.1f}}%<br>{el2}: %{{b:.1f}}%<br>{el3}: %{{c:.1f}}%<br>Coordinates: %{{text}}<br>{intensity_label}: %{{marker.color}}%',  # Custom hover text format
        'customdata': custom_data, # Add combined text for hover
        'hovertemplate': f'{el1}: %{{a:.1f}}%<br>{el2}: %{{b:.1f}}%<br>{el3}: %{{c:.1f}}%'
                        f'<br>Coordinates: (%{{customdata[0]}}, %{{customdata[1]}})<br>{intensity_label}: %{{customdata[2]}}',  # Custom hover text format
        'name': 'Data Points',
        'showlegend': False
    }))

    # Add dummy traces for the legend
    for phase_name, shape in shape_map.items():
        fig.add_trace(go.Scatterternary({
            'mode': 'markers',
            'a': [None],  # Dummy data
            'b': [None],  # Dummy data
            'c': [None],  # Dummy data
            'marker': {
                'size': 8,
                'symbol': shape,
                'color': 'black'
            },
            'name': phase_name,
            'showlegend': True
        }))

    # Update layout
    fig.update_layout({
        'ternary': {
            'sum': 100,
            'aaxis': {'title': f'{el1} %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
            'baxis': {'title': f'{el2} %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
            'caxis': {'title': f'{el3} %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'}
        },
        'title': title,
        'legend': {'x': -0.1, 'y': 1}  # Position the legend on the left
    })

    # Show the plot
    fig.show()

#%%

def ternary_discrete(df, el1, el2, el3, intensity_label, phase_label, title):
    """
    Create a ternary plot with discrete colors for string intensities and different marker shapes for phases.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    el1 (str): Label for the first element.
    el2 (str): Label for the second element.
    el3 (str): Label for the third element.
    intensity_label (str): Label for the intensity values.
    phase_label (str): Label for the phase values.
    title (str): Title of the plot.
    """
    # Extract element percentages, intensity, and phase
    el1_percent = df.xs(f'{el1} %', level='Element', axis=1).values.flatten()
    el2_percent = df.xs(f'{el2} %', level='Element', axis=1).values.flatten()
    el3_percent = df.xs(f'{el3} %', level='Element', axis=1).values.flatten()
    intensity = df.xs(intensity_label, level='Element', axis=1).values.flatten()
    phase = df.xs(phase_label, level='Element', axis=1).values.flatten()

    # Create a color mapping for unique intensity values
    unique_intensities = list(set(intensity))
    color_map = {val: i for i, val in enumerate(unique_intensities)}
    colors = [color_map[val] for val in intensity]

    # Create a marker shape mapping for unique phase values
    unique_phases = list(set(phase))
    marker_shapes = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right']
    shape_map = {val: marker_shapes[i % len(marker_shapes)] for i, val in enumerate(unique_phases)}
    shapes = [shape_map[val] for val in phase]

    # Create the ternary plot with custom hover text, colored markers, and different shapes
    fig = go.Figure(go.Scatterternary({
        'mode': 'markers',
        'a': el1_percent,  # el1 percentages
        'b': el2_percent,  # el2 percentages
        'c': el3_percent,  # el3 percentages
        'marker': {
            'size': 8,
            'color': colors,  # Use mapped colors for marker color
            'symbol': shapes,  # Use mapped shapes for marker shape
            'colorscale': 'Viridis',  # Choose a colorscale
            'colorbar': {
                'title': intensity_label,
                'tickvals': list(color_map.values()),  # Set tick values to the mapped color indices
                'ticktext': unique_intensities  # Set tick text to the unique intensity values
            },
            'line': {'width': 0}
        },
        'text': df.columns.get_level_values(0).unique(),  # Labels for the points
        'hovertemplate': f'{el1}: %{{a:.1f}}%<br>{el2}: %{{b:.1f}}%<br>{el3}: %{{c:.1f}}%<br>{intensity_label}: %{{text}}<br>{phase_label}: %{{marker.symbol}}',  # Custom hover text format
        'name': 'Data Points',
        'showlegend': False  # Do not show this trace in the legend
    }))

    # Add dummy traces for the legend
    for phase_name, shape in shape_map.items():
        fig.add_trace(go.Scatterternary({
            'mode': 'markers',
            'a': [None],  # Dummy data
            'b': [None],  # Dummy data
            'c': [None],  # Dummy data
            'marker': {
                'size': 8,
                'symbol': shape,
                'color': 'black'
            },
            'name': phase_name,
            'showlegend': True
        }))

    # Update layout
    fig.update_layout({
        'ternary': {
            'sum': 100,
            'aaxis': {'title': f'{el1} %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
            'baxis': {'title': f'{el2} %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
            'caxis': {'title': f'{el3} %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'}
        },
        'title': title,
        'legend': {'x': -0.1, 'y': 1}  # Position the legend on the left
    })

    # Show the plot
    fig.show()

# %%
#-------------- with copilot data and function -------------

ternary_discrete(df, 'Cu', 'P', 'S', 'Intensity','Phase', 'Ternary Phase Diagram for Cu-P-S with Intensity Coloring')
# %%
# -------------- mock data ----------------
import pickle
path=r"Z:\P110143-phosphosulfides-Andrea\Data\Analysis\guidal\XRD\mittma_0019\mittma_00019_FRmock_phases_coords.pkl"

with open(path, 'rb') as f:
    mock_df = pickle.load(f)
# %%
ternary_discrete_attempt(mock_df, 'Cu', 'P', 'S', 'Phase','Sample ID', 'Ternary Phase Diagram for Cu-P-S with Intensity Coloring')



# %%


intensity = mock_df.xs('Sample ID', level='Data type', axis=1).values.flatten()
phase = mock_df.xs('Phase', level='Data type', axis=1).values.flatten()
print(len(intensity), len(phase))
# Drop NaN values from intensity and phase
intensity = intensity[~pd.isna(intensity)]
phase = phase[~pd.isna(phase)]
print(len(intensity), len(phase))
print(intensity)
print(phase)
# %%
var = get_data(mock_df, 'Sample ID').loc[0]
intensities =[]
for i in var:
    try: # check if it is a number
         intens = np.round(i, 2)
         if intens % 1 == 0:  # Check if the number is an integer
            intens = f'sample-{int(intens)}'  # Convert to integer to remove the .0
         #print('rounded', intens)
    except:  # if it is already a string, use the string
        if i == '-':
            intens = 'missing EDS'
        else: 
            intens = i
        #print('missing')
    intensities.append(intens)

print(intensities)

# %%
get_data(mock_df, 'Phase')
# %%
# -------------- real data ----------------

path=r"Z:\P110143-phosphosulfides-Andrea\Data\Analysis\guidal\XRD\mittma_0019"
data_BR = os.path.join(path, "mittma_0019_BR_phases_coords.pkl")
data_FR = os.path.join(path, "mittma_0019_FR_phases_coords.pkl")

with open(data_BR, 'rb') as f:
    BR_df = pickle.load(f)

with open(data_FR, 'rb') as f:
    FR_df = pickle.load(f)

df_19_R = pd.concat([BR_df, FR_df], axis=1)
display(BR_df)
#%%
labels = [
    'Layer 1 Cu Atomic %',
    'Layer 1 P Atomic %',
    'Layer 1 S Atomic %',
    'Layer 1 Thickness (nm)'
]
# Create a boolean mask for the columns that match the labels
cols = BR_df.columns.get_level_values(1).isin(labels)
EDS_BR_only = BR_df.loc[:, cols]

# create the dictionary of sample id to insert in the df
num_unique_columns = len(EDS_BR.columns.get_level_values(0).unique())
sample_id_dict = {'Sample ID': [19] * num_unique_columns}

EDS_BR = add_info(EDS_BR_only, sample_id_dict)

#add also fake phases 

phases_dict  = {'Phase': ['amorphous','sulphide', 'phosphide', 'unknown', 'unknown' ]*5} 
BR_mock_df = add_info(EDS_BR, phases_dict)
display(BR_mock_df)
#%%
ternary_discrete_attempt(BR_mock_df, 'Cu', 'P', 'S', 'Sample ID','Phase', 'Ternary Phase Diagram for Cu-P-S with Intensity Coloring')

# %%
ternary_discrete_attempt(df_19_R, 'Cu', 'P', 'S',  'Cu7S4','Sample ID','Ternary Phase Diagram for Cu-P-S with Intensity Coloring')
# %%

get_data(FR_df,  x=-17, y=-17)
# %%
get_data(BR_df, x=-17, y=-17)
# %%


#--------------------------------- other functions that I tried, they do not work ---------------------------------
def ternary_plot_v01(df, el1, el2, el3, datatype, title, savepath=None):
    """Make a ternary plot of the data in df, with el1, el2, el3 as the corners, and colorscale based on datatype."""

    A = f'Layer 1 {el1} Atomic %'
    B = f'Layer 1 {el2} Atomic %'
    C = f'Layer 1 {el3} Atomic %'

    A_percent = df.xs(A, level='Data type', axis=1).values.flatten()
    B_percent = df.xs(B, level='Data type', axis=1).values.flatten()
    C_percent = df.xs(C, level='Data type', axis=1).values.flatten()
    intensity = df.xs(datatype, level='Data type', axis=1).values.flatten()
    coordinates = MI_to_grid(df).values  # df.columns.get_level_values(0)[::7].values.flatten() also works, but gives a lot of decimals

    fig = go.Figure()

    if pd.api.types.is_numeric_dtype(intensity):
        marker = {
            'symbol': 100,
            'size': 8,
            'color': intensity,  # Use intensity for marker color
            'colorscale': 'Turbo',  # Choose a colorscale
            'colorbar': {'title': datatype},  # Add a colorbar
            'line': {'width': 2}
        }
    else:
        unique_intensities = list(set(intensity))
        color_map = {val: i for i, val in enumerate(unique_intensities)}
        colors = [color_map[val] for val in intensity]
        marker = {
            'symbol': 100,
            'size': 8,
            'color': colors,  # Use mapped colors for marker color
            'colorscale': 'Viridis',  # Choose a discrete colorscale
            'colorbar': {'title': datatype, 'tickvals': list(color_map.values()), 'ticktext': unique_intensities},  # Add a colorbar with discrete values
            'line': {'width': 2}
        }

    fig.add_trace(go.Scatterternary({
        'mode': 'markers',
        'a': A_percent,  # el1 percentages
        'b': B_percent,  # el2 percentages
        'c': C_percent,  # el3 percentages
        'marker': marker,
        'text': coordinates,
        'hovertemplate': f'{el1}: %{{a:.1f}}%<br>{el2}: %{{b:.1f}}%<br>{el3}: %{{c:.1f}}%<br>{datatype}: %{{marker.color:.1f}}<br>Coordinates:%{{text:str}}',  # Custom hover text format
        'showlegend': False
    }))

    # Update layout
    fig.update_layout({
        'ternary': {
            'sum': 100,
            'aaxis': {'title': f'{el1} %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
            'baxis': {'title': f'{el2} %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
            'caxis': {'title': f'{el3} %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'}
        },
        'title': title},
        width=800,
        height=600,
    )
    if savepath:
        if savepath.endswith(".png"):
            fig.write_image(savepath, scale=2)
        if savepath.endswith(".html"):
            fig.write_html(savepath)

    # Show the plot
    fig.show()

# %%
# version 1 ( version0 in functions.py)
def ternary_plot_v1(dfs, el1, el2, el3, datatype, title, savepath=None):
    """
    Make a ternary plot of the data in dfs, with el1, el2, el3 as the corners, and colorscale based on datatype.
    dfs is a dictionary where keys are labels and values are dataframes.
    """
    fig = go.Figure()
    markers = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right']
    
    for i, (label, df) in enumerate(dfs.items()):
        A = f'Layer 1 {el1} Atomic %'
        B = f'Layer 1 {el2} Atomic %'
        C = f'Layer 1 {el3} Atomic %'

        A_percent = df.xs(A, level='Data type', axis=1).values.flatten()
        B_percent = df.xs(B, level='Data type', axis=1).values.flatten()
        C_percent = df.xs(C, level='Data type', axis=1).values.flatten()
        try:
            intensity = df.xs(datatype, level='Data type', axis=1).values.flatten()
        except KeyError:
            intensity = 5
            markers[i] = 100
            
        X,Y = extract_coordinates(df)

        fig.add_trace(go.Scatterternary({
            'mode': 'markers',
            'a': A_percent,   # el1 percentages
            'b': B_percent,   # el2 percentages
            'c': C_percent,   # el3 percentages
            'marker': {
                'symbol': markers[i % len(markers)],  # Use different marker for each dataframe
                'size': 8,
                'color': intensity,  # Use intensity for marker color
                'colorscale': 'Turbo',  # Choose a colorscale
                'colorbar': {'title': datatype},  # Add a colorbar
                'line': None  # Remove the outside border
            },
            'text': [f'{x}, {y}' for x, y in zip(X, Y)],  # Add coordinates as text
            #'text': f'{X}, {Y}',  # Add coordinates as text
            'hovertemplate': f'{el1}: %{{a:.1f}}%<br>{el2}: %{{b:.1f}}%<br>{el3}: %{{c:.1f}}%<br>{datatype}: %{{marker.color:.1f}}<br>Coordinates:%{{text:str}}',  # Custom hover text format
            'name': label,
            'showlegend': True
        }))

    # Update layout
    fig.update_layout({
        'ternary': {
            'sum': 100,
            'aaxis': {'title': f'{el1} %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
            'baxis': {'title': f'{el2} %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
            'caxis': {'title': f'{el3} %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'}
        },
        'title': title,
        'legend': {'title': 'Dataframes', 'x': 0.1, 'y': 1}  # Move legend to the left side
    },
        width=800,
        height=600, 
    )       
    if savepath:
        if savepath.endswith(".png"):
            fig.write_image(savepath, scale=2)
        if savepath.endswith(".html"):
            fig.write_html(savepath)

    # Show the plot
    fig.show()

# %%
# version 2
def ternary_plot_v2(dfs, el1, el2, el3, datatype, title, savepath=None):
    """
    Make a ternary plot of the data in dfs, with el1, el2, el3 as the corners, and colorscale based on datatype.
    dfs is a dictionary where keys are labels and values are dataframes.
    """
    fig = go.Figure()
    markers = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right']
    
    for i, (label, df) in enumerate(dfs.items()):
        A = f'Layer 1 {el1} Atomic %'
        B = f'Layer 1 {el2} Atomic %'
        C = f'Layer 1 {el3} Atomic %'

        A_percent = df.xs(A, level='Data type', axis=1).values.flatten()
        B_percent = df.xs(B, level='Data type', axis=1).values.flatten()
        C_percent = df.xs(C, level='Data type', axis=1).values.flatten()
        try:
            intensity = df.xs(datatype, level='Data type', axis=1).values.flatten()
        except KeyError:
            intensity = -1
            markers[i] = 100
            
        X,Y = extract_coordinates(df)

        fig.add_trace(go.Scatterternary({
            'mode': 'markers',
            'a': A_percent,   # el1 percentages
            'b': B_percent,   # el2 percentages
            'c': C_percent,   # el3 percentages
            'marker': {
                'symbol': markers[i % len(markers)],  # Use different marker for each dataframe
                'size': 8,
                'color': intensity,  # Use intensity for marker color
                'colorscale': 'Turbo',  # Choose a colorscale
                'colorbar': {'title': datatype} if i == 0 else None,  # Add a colorbar only for the first trace
                'showscale': i == 0,  # Show the colorbar only for the first trace
                'line': None  # Remove the outside border
            },
            'text': [f'{x}, {y}' for x, y in zip(X, Y)],  # Add coordinates as text
            'hovertemplate': f'{el1}: %{{a:.1f}}%<br>{el2}: %{{b:.1f}}%<br>{el3}: %{{c:.1f}}%<br>{datatype}: %{{marker.color:.1f}}<br>Coordinates:%{{text:str}}',  # Custom hover text format
            'name': label,
            'showlegend': True
        }))

    # Update layout
    fig.update_layout({
        'ternary': {
            'sum': 100,
            'aaxis': {'title': f'{el1} %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
            'baxis': {'title': f'{el2} %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
            'caxis': {'title': f'{el3} %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'}
        },
        'title': title,
        'legend': {'title': 'Dataframes', 'x': 0.1, 'y': 1}  # Move legend to the left side
    },
        width=800,
        height=600, 
    )       
    if savepath:
        if savepath.endswith(".png"):
            fig.write_image(savepath, scale=2)
        if savepath.endswith(".html"):
            fig.write_html(savepath)

    # Show the plot
    fig.show()