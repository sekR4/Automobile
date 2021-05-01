import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

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


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the DataFrame as suggested in 'machine learning task.pdf'

    - Replaces '?' with NaNs
    - Removes 'symboling'
    - Ignores rows with missing values in target variable
    - Transforms misclassified 'objects' to numeric values
    - Replaces missing numeric values with their mean

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

    # Replace missing values for continuous variables (base model)
    for k, v in tmp.isna().sum().items():
        if v > 0 and k in numerics:
            tmp[k].fillna(value=tmp[k].mean(), inplace=True)

    print(
        f"{len(tmp)} rows ({round(100*(len(tmp)/len_before),2)}%) left after preprocessing"
    )

    assert (
        tmp[numerics].isna().sum().sum() == 0
    ), "DataFrame still contains missing values"

    return tmp


def summary(model, X: pd.DataFrame, y: pd.Series):
    """R-like summary of a linear regression model

    Inspired by
    https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression

    Parameters
    ----------
    model : LinearRegression()
        A fitted regression model from sklearn e.g. LinearRegression().fit(X, y)
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    """

    lm, features = model, list(X.columns)
    features.insert(0, "(Intercept)")
    params, y_hat = np.append(lm.intercept_, lm.coef_), lm.predict(X)

    newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(
        pd.DataFrame(X.reset_index(drop=True))
    )
    MSE = (sum((y - y_hat) ** 2)) / (len(newX) - len(newX.columns))

    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

    sd_b = np.round(sd_b, 3)
    ts_b = np.round(ts_b, 3)
    p_values = np.round(p_values, 3)
    params = np.round(params, 4)

    result = pd.DataFrame()

    result[" "] = features
    (
        result["Coefficients"],
        result["Std. Error"],
        result["t value"],
        result["Pr(>|t|)"],
    ) = [params, sd_b, ts_b, p_values]

    # NOTE: Let's assume there are no transformations on y (e.g. log).
    # Otherwise the formula printed is wrong.
    print("Call:")
    print(f"lm(formula = {y.name} ~ {' + '.join(features[1:])})", "\n")
    pd.set_option("display.max_rows", None)
    print(result, "\n")
    # print(result[result["Pr(>|t|)"].notna()].sort_values(by="Pr(>|t|)"), "\n")
    pd.set_option("display.max_rows", 10)
    print("R squared: ", round(model.score(X, y), 4), "\n")


def create_batches_of_columns(df: pd.DataFrame, batch_size: int = 3) -> list():
    """Splits a list of columns into smaller batches for further processing.

    Parameters
    ----------
    df : pd.DataFrame

    batch_size: int, optional
        Size of your batch, by default 3

    Returns
    -------
    List
        List of batches (lists) e.g.: [ [col1,col2], [col3, col4] ]
    """

    cols = [c for c in df.columns if c in numerics]
    chunks, batches = (len(cols) - 1) // batch_size + 1, []

    for i in range(chunks):

        batch = cols[i * batch_size : (i + 1) * batch_size]

        if "normalized-losses" not in batch:
            batch.insert(0, "normalized-losses")

        batches.append(batch)

    return batches


def plot_correlation(df: pd.DataFrame, save=True, dpi=300):
    """Shows a correlation matrix

    Parameters
    ----------
    df : pd.DataFrame
        Contains training data
    save : bool, optional
        Saves figure at `img/`, by default False
    dpi : int, optional
        Figure resolution, by default 150
    """
    # Compute the correlation matrix
    corr_all = df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr_all, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(7, 7))
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_all, mask=mask, square=True, linewidths=0.5, ax=ax, cmap="BuPu")
    if save:
        plt.savefig("img/correlation_heatmap.png", dpi=dpi)
    # print(corr_all)
    plt.show()
