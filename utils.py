import pandas as pd
import numpy as np

columns = [
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-type",
    "num-of-cylinders",
    "engine-size",
    "fuel-system",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price",
]

numerics = [
    "normalized-losses",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-size",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price",
]


def clean_df(df: pd.DataFrame):
    """Cleans the DataFrame as suggested in 'machine learning task.pdf'
    
    - Replaces '?' with NaNs
    - Removes 'symboling'
    - Ignores rows with missing values in target variable
    - Transforms misclassified 'objects' to numeric values

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame created by reading 'imports-85.data'

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame to work with
    """

    tmp, len_before = df.copy(), len(df)

    # Following Task instructions
    tmp = tmp.replace("?", np.nan)
    tmp.drop("symboling", axis="columns", inplace=True)
    tmp = tmp[tmp["normalized-losses"].notna()]

    # Correcting data types (also see: 'Attribute Information in `data/imports-85.names`')
    for k, v in tmp.dtypes.items():
        if k in numerics and v == "object":
            tmp[k] = tmp[k].astype(float)

    print(
        f"{len(tmp)} rows ({round(100*(len(tmp)/len_before),2)}%) left after preprocessing"
    )

    assert len(tmp.notna()) == len(tmp), "DataFrame still contains missing values"

    return tmp
