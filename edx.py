#%%
from functions import *

#%%
#%% TEST 
folder = r"O:\Nlab\Public\DCH-plasma\phosphosulfides_students\Students\Giulia\01_Characterization\layerprobe\anait_0001_BaS_Zr"
filepath = os.path.join(folder, "anait_0001_BR.xlsx")
new_heatmap("Layer 1 Ba Atomic %", filepath=filepath, title = "anait_0001_BR", 
#savepath = os.path.join(folder, "test.html")
)
#%%


#%% try to plot with all quadrants

folderpath = r"O:\Nlab\Public\DCH-plasma\phosphosulfides_students\Students\Giulia\01_Characterization\layerprobe\mittma_0019\excels"
sample = "mittma_0019"
sample_n =19
#sample = "anait_0001"
#name= sample
#%%
pos = ["BR",
        "FR",
        "FL", 
        "BL"
        ]
names = [sample+"_"+pos[i] for i in range(0,len(pos))]
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

    if names[i] == sample +"_BR":

        X,Y, grid_input, areax, areay = EDS_coordinates(4, 6, 30000, 70000, filepath, new_path, edge = 8 )
        print(names[i], areax, X)

    if names[i] == sample +"_FR":
        X,Y, grid_input, areax, areay = EDS_coordinates(4, 6, 30000, 60000, filepath, new_path )
        print(names[i], areax, X)

    if names[i] == sample +"_FL":
        X,Y, grid_input, areax, areay = EDS_coordinates(5, 6, 30000, 50000, filepath, new_path )
        print(names[i], areax, X)
    
    if names[i] == sample +"_BL":
        X,Y, grid_input, areax, areay = EDS_coordinates(4, 6, 30000, 70000, filepath, new_path, edge= 10 )
        print(names[i], areax, X)
    

#%% load the data from the excel files

data_BR = pd.DataFrame()
data_FR = pd.DataFrame()
data_FL = pd.DataFrame()
data_BL = pd.DataFrame()
datas= [data_BR, 
        data_FR,
        data_FL,
        data_BL,
         ]

for i in range(0,len(pos)):
    file = sample+"_"+pos[i]+"_coords.xlsx" #change to coords or translated if needed
    filepath = os.path.join(folderpath, file)
    grid = measurement_grid(100,100,36,36,-18,-18)
    datas[i], coords = read_layerprobe(filepath, grid, sheetname="Sheet1")
    plt.figure()
    plot_grid(coords, grid)

data__BR, coords_BR = translate_data(datas[0], 20,20)
data__FR, coords_FR = translate_data(datas[1], 20,-20)
data__FL, coords_FL = translate_data(datas[2], -20,-20)
data__BL, coords_BL = translate_data(datas[1], -20,20)

data = combine_data((data__BR,
                      data__FR,
                      data__FL, 
                      data__BL
                     ))
data0= data.copy()
#%% add a column with sample ID to the data
headerlength = len(data.columns.get_level_values(1).unique())
k=0
data = data0.copy()
for i in range(0, len(data.columns.get_level_values(0).unique())):
    print(data.columns.get_level_values(0).unique()[i])
    data.insert(headerlength*(i+1)+k, "{}".format(data.columns.get_level_values(0).unique()[i]), sample_n, allow_duplicates=True)
    data.rename(columns={'': 'Sample ID'}, inplace = True)
    k=k+1
new_data = data.copy()
print(data.columns.get_level_values(1))
# %% 
datatypes=["Layer 1 Thickness (nm)", "Layer 1 Ba Atomic %", "Layer 1 S Atomic %", "Layer 1 Zr Atomic %"] 

for datatype in datatypes:
    savepath = os.path.join(folderpath, f"{sample} {datatype}.png")
    new_heatmap(datatype, data=data, title = f"{sample} {datatype}",
    savepath=savepath,
     )
#%% ------------------export pickle to use elsewhere ---------------------

pickel_path = r"Z:\P110143-phosphosulfides-Andrea\Students\Giulia\01_Characterization\layerprobe\mittma_pickles"
with open(os.path.join(pickel_path, sample+"_EDS.pkl"), 'wb') as f:
    pickle.dump(new_data, f)



# %% how to get data info per each point (works also on interpolatd data)

# get_data(data, type= all, x= all, y= all)
get_data(data,x= 32,y=34, type = "Layer 1 S Atomic %")


# %%
def stats(data_all, type):
    data = get_data(data_all, type = type)
    data = data.sort_values(by = 0, axis=1 )
    min_= data.iloc[0,0]
    max_ = data.iloc[0,-1]
    mean_ = data.mean(axis=1)[0]

    data = pd.DataFrame([min_, max_, mean_], index = ["min","max", "mean"])
    return data

S = get_data(data, type = "Layer 1 S Atomic %")
S=S.sort_values(by = 0, axis=1 )

S_min= S.iloc[0,0]
S_max = S.iloc[0,-1]
S_mean = S.mean(axis=1)[0]

S_data = pd.DataFrame([ S_min, S_max, S_mean,], index = ["min","max", "mean"])
# %%
Cu_data = stats(data, "Layer 1 Cu Atomic %")
P_data = stats(data, "Layer 1 P Atomic %")
S_data = stats(data, "Layer 1 S Atomic %")

results = pd.concat([Cu_data, P_data, S_data], axis=1)
results.columns = ["Cu", "P", "S"]
# %%

ratio0 = results.loc["mean", "Cu"]/results.loc["mean", "P"]
ratio2 = results.loc["mean", "S"]/results.loc["mean", "P"]

# %%



# %%

filepath= os.path.join(folderpath, name+".xlsx")
new_path = os.path.join(folderpath, name+ "_coords"+".xlsx")
X,Y, grid_input, areax, areay = EDS_coordinates(4, 6, 30000, 70000, filepath, new_path )
print(name, areax, X)
# %%
print(name, areay, Y)
# %%
filepath= os.path.join(folderpath, "anait_0001_BL.xlsx")
new_path = os.path.join(folderpath, "anait_0001_BL_coords.xlsx")
X,Y, grid_input, areax, areay = EDS_coordinates(4, 6, 30000, 70000, filepath, new_path, rotate="90" )
# %%
