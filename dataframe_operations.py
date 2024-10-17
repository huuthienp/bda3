import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def explore_structure(df, return_dict=False):
    """
    Explore and summarize the structure of a DataFrame.

    Args:
    df (pandas.DataFrame): The DataFrame to explore.
    return_dict (bool): If True, return a dictionary of DataFrames grouped by data type.

    Returns:
    dict: A dictionary of DataFrames grouped by data type if return_dict is True, else None.
    """
    hr = '-' * 50
    df_dict = {}
    
    # Print the shape of the DataFrame
    print('Shape:', df.shape)
    
    # Get value counts of data types
    dtypes_vc = df.dtypes.value_counts()
    print('Data types')
    print(dtypes_vc.to_frame())
    print(hr)
    
    # Iterate through each data type
    for dt in dtypes_vc.index.astype('str'):
        cols = df.select_dtypes(include=[dt]).columns
        
        # If return_dict is True, store columns of this data type in the dictionary
        if return_dict:
            df_dict[dt] = df[cols]
        
        # For float data types, print summary statistics
        if 'float' in dt:
            # Print the data type in uppercase
            print(dt.upper())
            df_stats = summary_statistics(df[cols])
            try_display(df_stats.round(4))
            print(hr)
        else:
            # For non-float data types, print unique value counts and plot
            print(dt.upper())

            for c in cols:
                # Get the column index
                i = df.columns.get_loc(c)

                # Count unique values in the column
                len_uv = len(df[c].unique())

                # Calculate value counts for the column
                vc = df[c].value_counts()

                print()
                print(f'\'{c}\' (column {i+1}) has {len_uv} unique values.')

                # Check if the column might be an ID column
                if len_uv == df.shape[0]:
                    print(f'\'{c}\' might be an ID column.')

                # Check if all unique values have the same frequency
                elif vc.std() == 0:
                    print(f'Each unique value in \'{c}\' appears {vc.mean():.0f} times.')

                # If neither condition is met, plot the distribution of value counts
                else:
                    kdeplot(vc, c)
            print(hr)
    
    # Return the dictionary if return_dict is True
    if return_dict:
        return df_dict


def try_display(df):
    try:
        display(df)
    except:
        print(df)


def summary_statistics(df, stats=['min', 'max', 'mean', 'median', 'std']):
    """
    Calculate summary statistics for a DataFrame.

    Args:
    df (pandas.DataFrame): The DataFrame to summarize.
    stats (list): List of statistics to calculate.

    Returns:
    pandas.DataFrame: A DataFrame containing the calculated statistics.
    """
    return df.agg(stats)


def kdeplot(series, title):
    """
    Create a kernel density estimation plot for a given series.

    Args:
    series (pandas.Series): The series to plot.
    title (str): The title for the plot.
    """
    sns.set_theme(style='darkgrid')
    ax = sns.kdeplot(series, color='green')
    ax.set_title(title)



def scale_dataframe(df, scaler, dtypes=['float64']):
    # Identify float64 columns
    dtype_columns = df.select_dtypes(include=dtypes).columns
    dtype_column_indices = [df.columns.get_loc(col) for col in dtype_columns]

    # Extract float64 columns
    df_dtypes = df.iloc[:, dtype_column_indices]

    # Fit a new scaler to transform the data
    scaled_df = scaler.fit_transform(df_dtypes)

    # Return the scaled dataframe and the scaler
    return scaled_df
