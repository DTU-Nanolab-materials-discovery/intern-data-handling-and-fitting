#%%
from functions import *
#%% with mock data
folder = r"C:\Users\s222531\OneDrive - Danmarks Tekniske Universitet\MASTER_PROJECT_PHOSPHOSULFIDES\CODE_DataAnalysis\old_code"
#gridname = measurement_grid(ncolumns, nrows, gridlength, gridheight, startx, starty)
grid = measurement_grid(4,11,37.5, 40,-18.75,-20)
filename_thick = os.path.join(folder, "ellipsometry_thickness.txt")
filename_nk = os.path.join(folder, "ellipsometry_n_k.txt")
# NB: the ellipsometry data are in cm, while the others in mm, but the read_ellipsometry function already converts cm to mm, you just have to remember to *10 when defininf the grid (grid should be in mm!)
#%% let´s try with my data

grid = measurement_grid()
filename = ""

#%%
ellips_thick, coords = read_ellipsometry_thickness(filename_thick, grid, n = 0, separator = "\t")
ellips_nk, coords = read_ellipsometry_nk(filename_nk, grid, n = 0, separator = "\t")
# %%
#plot_grid(coords,grid)
#print(grid)

print(ellips_nk)
# %%
heatmap(ellips_thick,"Z (nm)", "Z (nm)")

#%%
x = [-6.25, -6.25,-6.25]
y= [-20,0,20]

plot_data(ellips_nk, "Energy (eV)", "n", scatter_plot = True, legend = False)
plt.show()

plot_data(ellips_nk, "Energy (eV)", "k", legend = False)

# %%
new_heatmap(datatype = "Z (nm)", data=ellips_thick, title= "attempt")
# %%
