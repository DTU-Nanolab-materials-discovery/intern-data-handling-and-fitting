#%%

def open_csv_as_multiindex(csv_path, replace_nan=False):
    """
    Reopen a CSV file as a MultiIndex DataFrame.
    Args:
        csv_path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The MultiIndex DataFrame.
    """
    # Read the CSV file with the appropriate header levels
    df = pd.read_csv(csv_path, header=[0, 1, 2, 3], na_filter=False)
    # The columns are already MultiIndex, so no need to split them again
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    # set the index names to ['sample','event','source','parameter']
    df.columns.names = ['sample', 'event', 'source', 'parameter']
    if replace_nan:
        # Replace the string 'nan' with an empty string in MultiIndex column names
        df.columns = df.columns.set_levels(
            [level.str.replace('nan', '') for level in df.columns.levels]
        )
    return df

#method to get a parameter from a MultiIndex DataFrame
#path being a list of strings pointing to the parameter
def get_df_param(df, path: list):
    for key in path:
        df = df.xs(key, axis=1, level=1)
    # set the data row index to path[-1]
    df.index = pd.Index(df.index, name=path[-1])
    return df

all_params_path = r'Z:\P110143-phosphosulfides-Andrea\Data\Samples\all_params.csv'
all_params_df = open_csv_as_multiindex(all_params_path)
#%%
get_df_param(all_params_df,['deposition','general','material_space'])
# %%
# get all the Mittma samples
mittma_samples = []
samples= all_params_df.columns.get_level_values(0).unique()
for sample in samples:
    if sample.endswith('Cu'): 
        mittma_samples.append(sample)

print(mittma_samples)
# %%
# print vertically the avg_capman_pressure for all Mittma samples
pressure=get_df_param(all_params_df[mittma_samples], ['deposition','general','avg_capman_pressure'])
pressure.transpose()
# %%
