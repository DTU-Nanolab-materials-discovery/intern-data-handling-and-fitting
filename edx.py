#%%
from functions import *

#%%
#%%

def new_heatmap(datatype, data=None, filepath = None, savepath=None, title=datatype):
    "plot heatmaps with interpolated background, like in Nomad"

    if filepath is not None:
        raw_data = pd.read_excel(filepath, header=0)
        x = raw_data["X (mm)"].values
        y = raw_data["Y (mm)"].values
        z = raw_data[datatype].values

    if data is not None: 
        xy = MI_to_grid(data).drop_duplicates(ignore_index=True)
        x = xy["x"].values
        y = xy["y"].values
        z = data.iloc[:, data.columns.get_level_values(1)==datatype].values.flatten()

    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='linear')

    scatter = go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=15,
                color=z,  # Set color to thickness values
                colorscale='Viridis',  # Choose a colorscale
                #colorbar=dict(title='Thickness (nm)'),  # Add a colorbar
                showscale=False,  # Hide the colorbar for the scatter plot
                line=dict(
                    width=2,  # Set the width of the border
                    color='DarkSlateGrey'  # Set the color of the border
            ) ),
        )
    if datatype == "Layer 1 Thickness (nm)":
        cbar_title = "Thickness (nm)"
    elif datatype == "Layer 1 P Atomic %":
        cbar_title = "P Atomic %"
    elif datatype == "Layer 1 S Atomic %":
        cbar_title = "S Atomic %"
    elif datatype == "Layer 1 Cu Atomic %":
        cbar_title = "Cu Atomic %"

    heatmap = go.Heatmap(
    x=xi[0],
    y=yi[:, 0],
    z=zi,
    colorscale='Viridis',
    colorbar=dict(title=cbar_title),
    #zmin = 10, zmax = 60
    )

    fig = go.Figure(data=[heatmap, scatter])

    fig.update_layout(title=title,
    xaxis_title='X Position (mm)',
    yaxis_title='Y Position (mm)',
    template='plotly_white',
    autosize =False,
    width = 600,
    height = 500)

    if savepath.endswith(".png"):
        fig.write_image(savepath, scale=2)
    
    if savepath.endswith(".html"):
        fig.write_html(savepath)
    
    fig.show()

#%% TEST 
filepath = os.path.join(folder, "mittma_00017_BL_coords.xlsx")
new_heatmap("Layer 1 P Atomic %", filepath=filepath, title = "00017_BL", savepath = os.path.join(folder, "test.html"))

#%% try to plot with all quadrants

folder = r"O:\Nlab\Public\DCH-plasma\phosphosulfides_students\Students\Giulia\01_Characterization\layerprobe\eugbe-0001"
sample = "eugbe_0001"

pos = ["BR", "FR",
        "FL", 
        #"BL"
        ]
#%% IF NEEDED: translate the excel files

for i in range(0,len(pos)):
    file = sample+"_"+pos[i]+".xlsx"
    filepath = os.path.join(folder, file)
    new_path = os.path.join(folder, sample+"_"+pos[i]+"_translated.xlsx")
    lp_translate_excel(filepath, new_path)

#%%IF NEEDED: add coordinated to the excel files
for i in range(0,len(names)): 
    filepath= os.path.join(folderpath, names[i]+".xlsx")
    new_path = os.path.join(folderpath, names[i]+ "_coords"+".xlsx")

    if names[i] == "mittma_00017_BR":

        X,Y, grid_input, areax, areay = EDS_coordinates(4, 5, 30000, 70000, filepath, new_path )
        print(names[i], areax, X)

    if names[i] == "mittma_00017_FR":
        X,Y, grid_input, areax, areay = EDS_coordinates(4, 5, 30000, 70000, filepath, new_path )
        print(names[i], areax, X)

    if names[i] == "mittma_00017_FL":
        X,Y, grid_input, areax, areay = EDS_coordinates(3, 5, 30000, 80000, filepath, new_path )
        print(names[i], areax, X)
    
    if names[i] == "mittma_00017_BL":
        X,Y, grid_input, areax, areay = EDS_coordinates(3, 5, 30000, 80000, filepath, new_path )
        print(names[i], areax, X)
 

#%%
data_BR = pd.DataFrame()
data_FR = pd.DataFrame()
data_FL = pd.DataFrame()
#data_BL = pd.DataFrame()
datas= [data_BR, data_FR,
         data_FL,
          # data_BL
         ]

for i in range(0,len(pos)):
    file = sample+"_"+pos[i]+"_translated.xlsx"
    filepath = os.path.join(folder, file)
    grid = measurement_grid(100,100,30,30,-15,-15)
    datas[i], coords = read_layerprobe(filepath, grid)
    plt.figure()
    plot_grid(coords, grid)

data__BR, coords_BR = translate_data(datas[0], 20,20)
data__FR, coords_FR = translate_data(datas[1], 20,-20)
data__FL, coords_FL = translate_data(datas[2], -20,-20)
#data__BL, coords_BL = translate_data(datas[3], -20,20)

data = combine_data((data__BR, data__FR,
                      data__FL, 
                      #data__BL
                     ))

# %%
datatypes=["Layer 1 Thickness (nm)", "Layer 1 P Atomic %", "Layer 1 S Atomic %", "Layer 1 Zr Atomic %"] 

for datatype in datatypes:
    savepath = os.path.join(folder, f"{sample} {datatype}.png")
    new_heatmap(datatype, data=data, savepath=savepath, title = f"{sample} {datatype}")
# %%
filepath = r"O:\Nlab\Public\DCH-plasma\phosphosulfides_students\Students\Giulia\01_Characterization\layerprobe\eugbe_0002_Zr\eugbe_0002_FR_res2048_K_line.xlsx"
new_heatmap("Layer 1 Thickness (nm)", filepath= filepath)
# %%
