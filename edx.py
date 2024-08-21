#%%
from functions import *

#%%
#%% TEST 
folder = r"O:\Nlab\Public\DCH-plasma\phosphosulfides_students\Students\Giulia\01_Characterization\layerprobe\mittma_00017\excels"
filepath = os.path.join(folder, "mittma_00017_BL_coords.xlsx")
new_heatmap("Layer 1 P Atomic %", filepath=filepath, title = "00017_BL", 
#savepath = os.path.join(folder, "test.html")
)
#%%


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
    new_heatmap(datatype, data=data, title = f"{sample} {datatype}",
     #savepath=savepath,
     )
# %% hot to get data info per each point (works also on interpolatd data)

# get_data(data, type= all, x= all, y= all)
get_data(data,x= -9.2,y=-5.3, type = "Layer 1 P Atomic %")

# %%
