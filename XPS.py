#%%
from functions import *
import re
#%%
folder = r"Z:\P110143-phosphosulfides-Andrea\Data\Samples\mittma_0019_Cu\XPS"
filename= "mittma_0019_core.txt"
sample = "mittma_0019_core"
#filename = "mittma_0019_core_Cu2p3"
filename = os.path.join(folder, filename)
# %%
grid= measurement_grid(9,9,32,32,-16,-16) 

data,coords = read_XPS(filename, grid) # does not work, wrong file or wrong function?
# %%
plot_grid(data,coords)
# %%
display(data)
# %%
test = get_data(data,x= -16,y=0)
# %%
#------------- try to fix the read_XPS function ----------------
def read_XPS_simple(filename, grid):
    '''"Read data and coordinates from an XPS datafile. The file should be an csv (.txt) file. The data is constrained to a custom grid which must be provided via the "measurement_grid" function."
    Usage: data, coords = read_XPS(filename, grid)"'''
    # read the file
    file = pd.read_csv(filename, encoding = 'ANSI', engine='python', sep='delimiter', header = None, skiprows = 29)
    file.drop(file.iloc[4::7].index, inplace=True)
    file.reset_index(drop = True)

    # the file has a really weird format so we need to do a lot of work to extract data
    # get amount of peaks
    peaknumb = []
    for i in range(0, len(file), 6):
        peaknumb.append(int(file.iloc[i][0].split()[8].replace(";","")))
    n = max(peaknumb) + 1

    # remove useless rows
    file.drop(file.iloc[0::6].index, inplace=True)
    file.reset_index(drop = True)

    # get data from remaining rows
    full_peaklist = []
    peaklist = []
    coordlist = []
    datalist = []
    for i in range(0, len(file), 5):
        # load peak type and coordinates and fix formatting
        peaktype = ' '.join(file.iloc[i][0].split()[5:len(file.iloc[i][0].split())]).replace("VALUE='","").replace("';","")
        xcoord = float(file.iloc[i+1][0].split()[5].replace("VALUE=","").replace(";",""))
        ycoord = float(file.iloc[i+2][0].split()[5].replace("VALUE=","").replace(";",""))
        coords = [xcoord, ycoord]
        # load data
        data = file.iloc[i+3][0].split()[2::]
        data.append(file.iloc[i+4][0].split()[2::][0])
        # fix data formatting
        data = [j.replace(",","") for j in data]
        data = [round(float(j),3) for j in data]

        full_peaklist.append(peaktype)
        peaklist.append(peaktype.split()[0])
        coordlist.append(coords)
        datalist.append(data)

    # create data dataframe
    dataframe = pd.DataFrame(datalist, columns = ['Intensity (counts)','Atomic %','Area (counts*eV)','FWHM (eV)','Peak BE (eV)'])
    # modify some values
    # convert cps to counts (machine does 25 cps)
    dataframe['Intensity (counts)'] = dataframe['Intensity (counts)']/25
    # convert KE to BE (KE of machine X-rays is 1486.68 eV)
    dataframe['Peak BE (eV)'] = 1486.68 - dataframe['Peak BE (eV)']
    # reorder columns to be similar to Avantage
    columnorder = ['Peak BE (eV)','Intensity (counts)','FWHM (eV)','Area (counts*eV)','Atomic %']
    dataframe = dataframe.reindex(columnorder, axis=1)
    
    # create coordinate dataframe
    coords = pd.DataFrame(coordlist, columns=['x', 'y'])
    # remove duplicate coordinates
    coords = coords.drop_duplicates(ignore_index = True)
    # adjust range to center coords on 0,0 instead of upper left corner
    coords['x'] = coords['x'] - max(coords['x'])/2
    coords['y'] = coords['y'] - max(coords['y'])/2
    # convert coords from µm to mm
    coords = coords/1000
    # flip y coordinate because Avantage is mental
    coords['y'] = coords['y'].values[::-1]

    # create peak dataframe
    peaks = pd.DataFrame(peaklist, columns = ['Peak'])
    # add peak dataframe to front of data dataframe
    dataframe = pd.concat([peaks, dataframe], axis = 1)

    # add column with summed atomic %
    # --------------insert here ---------------
    
    # align data to grid
    coordgrid = coords_to_grid(coords, grid)
    coord_header = grid_to_MIheader(coordgrid)

    # construct dataframe with multiindexing for coordinates
    header = pd.MultiIndex.from_product([coord_header, dataframe.columns],names=['Coordinate','Data type'])
    # reorder dataframe stacking to fit coordinate attachment
    n2 = n
    stackedframe = np.hstack([dataframe.values[0:n2],(dataframe.values[n2:2*n2])])
    for i in range(2*n2, len(dataframe), n2):
        stackedframe = np.hstack([stackedframe, (dataframe.values[i:i+n2])])
    data = pd.DataFrame(stackedframe, columns=header)


    return data, coords


#%%
def XPS_calculate_elements(data):
    col_peak = data.columns.get_level_values(1)=="Peak"
    data_peak = data.loc[:,col_peak]

    np.unique(data_peak.values.flatten())
    names=[]
    for peak in np.unique(data_peak.values.flatten()):
        for peak in np.unique(data_peak.values.flatten()):
            name = re.findall('[A-Za-z]+|\d+', peak)
            names = np.append(names, name[0])
    element_list = np.unique(names)
    print('element_list:', element_list)

    new_data = data.copy()
    headerlength = len(data.columns.get_level_values(1).unique())
    coords= data.columns.get_level_values(0).unique()
    k=0
    for i in range(0, len(coords)):
        
        coord= coords[i]    
        dataframe = data[coord]
        
        for el in element_list:
            tot_element_percent = np.array([])

            for peak in np.unique(data_peak.values.flatten()):
                name = re.findall('[A-Za-z]+|\d+', peak)
                peak_el = name[0]

                if peak_el == el:
                    element_percent = dataframe.loc[dataframe['Peak'] == peak]['Atomic %']
                    tot_element_percent = np.append(tot_element_percent, element_percent)

            new_df= pd.DataFrame([sum(tot_element_percent)], columns=[(coords, f'{el}%')])

            new_data.insert(headerlength*(i+1)+k, "{}".format(data.columns.get_level_values(0).unique()[i]), new_df, allow_duplicates=True)
            new_data.rename(columns={'':  f'{el}%'}, inplace = True)

        k=k+len(element_list)
        
    new_frame = new_data.copy()

    return new_frame, element_list


# %%

test = get_data(new_frame, x= -16, y=0)
display(test)
# %%
for el in element_list:
    print(el)
    savepath = os.path.join(folder, 'plots', f'{sample}_{el}.png')
    new_heatmap(datatype= f'{el}%', data=new_frame, title = sample+' '+ el+'%', savepath= savepath)
# %%
pickle_path = os.path.join(folder, f'{sample}_XPS.pkl')
with open (pickle_path, 'wb') as f:
    pickle.dump(new_frame, f)
# %%
df = math_on_columns(new_frame, 'P%', 'S%', '/')
df = math_on_columns(df, 'P%', 'S%', '+')
df = math_on_columns(df, 'Cu%', 'S%', '/')
df = math_on_columns(df, 'Cu%', 'P% + S%', '/')
# %%
savepath = os.path.join(folder, 'plots', f'{sample}_CuS_ratio.png')
new_heatmap(datatype= 'Cu% / S%', data=df, title = sample+' Cu% / S%', savepath= savepath)
# %% rotate the df by 90 degrees clockwise : x=y, y=-x

def rotate_coordinates(data_df, how ='clockwise'):
    'Rotate the coordinates of the data by 90 degrees clockwise, counterclockwise or 180 degrees'
    MI_rotated=[]
    initial_coordinates = MI_to_grid(data)

    if how == 'clockwise':
        xx = initial_coordinates['y']
        yy = - initial_coordinates['x']

    if how == 'counterclockwise':
        xx = - initial_coordinates['y']
        yy = initial_coordinates['x']

    if how == '180':
        xx = - initial_coordinates['x']
        yy = - initial_coordinates['y']

    for i in range(len(xx)):
        MI_rotated = np.append(MI_rotated,('{},{}'.format(xx[i], yy[i])))
    rotated_columns = pd.MultiIndex.from_tuples([(str(coord), col) for coord, col in zip(MI_rotated, data.columns.get_level_values(1))])
    data_rotated = data.copy()
    data_rotated.columns = rotated_columns
    return data_rotated
# %%
df_rotated = rotate_coordinates(df, how='clockwise')

# %%
list_to_plot = ['P% / S%', 'Cu% / P% + S%']
names = ['PSratio', 'C_Anion_ratio']
for i in range(len(list_to_plot)):
    savepath = os.path.join(folder, 'plots', f'{sample}_rotated_{name[i]}.png')
    new_heatmap(datatype= datatype[i], data=df_rotated, title = sample+' rotated '+ datatype, savepath= savepath)
# %%
new_heatmap(datatype= 'P% / S%', data=df_rotated, title = sample+' P / S ratio', savepath = os.path.join(folder, 'plots', f'{sample}_rotated_PSratio.png'))
new_heatmap(datatype= 'Cu% / P% + S%', data=df_rotated, title = sample+' Cu / anions ratio', savepath = os.path.join(folder, 'plots', f'{sample}_rotated_Cu_Anion_ratio.png'))
# %%
for el in element_list:
    savepath = os.path.join(folder, 'plots', f'{sample}_rotated_{el}.png')
    new_heatmap(datatype= f'{el}%', data=df_rotated, title = sample+' '+ el+'%', savepath= savepath)
# %%
