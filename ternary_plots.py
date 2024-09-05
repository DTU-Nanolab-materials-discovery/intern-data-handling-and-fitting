# %%
import pandas as pd
import plotly.graph_objects as go

# Example data as a MultiIndex DataFrame
data = {
    ('Point1', 'Cu %'): [30],
    ('Point1', 'P %'): [40],
    ('Point1', 'S %'): [30],
    ('Point1', 'Intensity'): [50],
    ('Point2', 'Cu %'): [50],
    ('Point2', 'P %'): [20],
    ('Point2', 'S %'): [30],
    ('Point2', 'Intensity'): [70],
    ('Point3', 'Cu %'): [10],
    ('Point3', 'P %'): [50],
    ('Point3', 'S %'): [40],
    ('Point3', 'Intensity'): [90]
}

# Creating a MultiIndex DataFrame
index = pd.MultiIndex.from_tuples(data.keys(), names=['Point', 'Element'])
df = pd.DataFrame(list(data.values()), index=index).transpose()

# Extract Cu, P, S percentages and intensity
cu_percent = df.xs('Cu %', level='Element', axis=1).values.flatten()
p_percent = df.xs('P %', level='Element', axis=1).values.flatten()
s_percent = df.xs('S %', level='Element', axis=1).values.flatten()
intensity = df.xs('Intensity', level='Element', axis=1).values.flatten()

# Create the ternary plot with custom hover text and colored markers
fig = go.Figure(go.Scatterternary({
    'mode': 'markers',
    'a': cu_percent,  # Cu percentages
    'b': p_percent,   # P percentages
    'c': s_percent,   # S percentages
    'marker': {
        #'symbol': ,
        'size': 8,
        'color': intensity,  # Use intensity for marker color
        'colorscale': 'Viridis',  # Choose a colorscale
        'colorbar': {'title': 'Intensity'},  # Add a colorbar
        'line': {'width': 0}
    },
    'text': df.columns.get_level_values(0).unique(),  # Labels for the points
    'hovertemplate': 'Cu: %{a:.1f}%<br>P: %{b:.1f}%<br>S: %{c:.1f}%<br>Intensity: %{marker.color:.1f}',  # Custom hover text format
    'name': 'Data Points'
}))

# Update layout
fig.update_layout({
    'ternary': {
        'sum': 100,
        'aaxis': {'title': 'Cu %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
        'baxis': {'title': 'P %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
        'caxis': {'title': 'S %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'}
    },
    'title': 'Ternary Phase Diagram for Cu-P-S with Intensity Coloring'
})

# Show the plot
fig.show()
#%%
fig.write_image(os.path.join(savepath, "ternary_plot.png"), scale=2)

# %%

#----------------------- Ternary plot with real data -----------------------#
import pickle



EDS_path = r"O:\Nlab\Public\DCH-plasma\phosphosulfides_students\Students\Giulia\01_Characterization\layerprobe\anait_pickles"
#names = [12, 14,15,16,17,19         ]
author = "anait"
names = [1,2]
dfs={}
for name in names:
    df = []
    
    #filename = f"mittma_000{name}_EDS.pkl"
    filename = f"{author}_000{name}_EDS.pkl"
    EDS_pkl = os.path.join(EDS_path, filename)
    
    with open(EDS_pkl, 'rb') as f:
       df= pickle.load(f)
    dfs[f"{author}_000{name}"] = df
    #dfs[f"mittma_000{name}"] = df

#%%
EDS_df = pd.DataFrame()
for key in dfs:
    print(key)
    df = dfs[key]
    EDS_df = combine_data((EDS_df, df))


# %%
#colors = ["red", "blue", "green", "orange", "purple", "black"]

#for idx, key in enumerate(dfs.keys()):
    #print(key)
    #df = dfs[key]


df= EDS_df

A= 'Layer 1 Ba Atomic %'
B= 'Layer 1 Zr Atomic %'
C= 'Layer 1 S Atomic %'
datatype = "Sample ID"
title = "All "+ f"{author}" +" samples so far"
#df[df.columns.get_level_values(1)=="Sample ID"] = df[df.columns.get_level_values(1)=="Sample ID"].astype(str)
A_percent = df.xs(A,  level='Data type', axis=1).values.flatten()
B_percent = df.xs(B, level='Data type', axis=1).values.flatten()
C_percent = df.xs(C, level='Data type', axis=1).values.flatten()
intensity = df.xs(datatype, level='Data type', axis=1).values.flatten()
coordinates = df.columns.get_level_values(0).unique()
#color = colors[idx]

fig = go.Figure(go.Scatterternary({
    'mode': 'markers',
    'a': A_percent,  # Cu percentages
    'b': B_percent,   # P percentages
    'c': C_percent,   # S percentages
    'marker': {
        'symbol': 100 ,
        'size': 8,
        'color': intensity,  # Use intensity for marker color
        'colorscale': 'Turbo',  # Choose a colorscale
        'colorbar': {'title': datatype},  # Add a colorbar
        'line': {'width': 2}
    },
    #'text': df.columns.get_level_values(0).unique(),  # Labels for the points
    'hovertemplate': 'Cu: %{a:.1f}%<br>P: %{b:.1f}%<br>S: %{c:.1f}%<br> Datatype: %{marker.color:.1f}',  # Custom hover text format
    'name': 'Data Points'
}))

# Update layout
fig.update_layout({
    'ternary': {
        'sum': 100,
        'aaxis': {'title': 'Ba %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
        'baxis': {'title': 'Zr %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'},
        'caxis': {'title': 'S %', 'min': 0, 'linewidth': 2, 'ticks': 'outside'}
    },
    'title': title
})

# Show the plot
fig.show()
# %%
fig.write_image(os.path.join(EDS_path, title+"_ternary.png"), scale=2)
# %%
