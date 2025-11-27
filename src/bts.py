import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import kagglehub
import numpy as np
from pandas.plotting import autocorrelation_plot
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


def preproc(csv_path, verbose=True): 
    """Process the household power consumption data"""

    # Load CSV
    df = pd.read_csv(csv_path)

    # Build datetime, mixed data format (super annoying)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, format='mixed')

    # Drop rows where datetime failed
    df = df.dropna(subset=['datetime'])

    # Set index
    df = df.set_index("datetime").sort_index()

    # Remove unused columns
    df = df.drop(columns=["Date", "Time"], errors="ignore")
    df = df.drop(columns=["index"], errors="ignore")

    # Convert all measurement columns to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Missing data %
    pct_missing = df.isna().sum().sum() / df.size * 100
    if verbose == True:
        print(f"Percentage of missing data: {pct_missing:.3f}%")

    # Remove all remaining missing rows (best choice for electricity data)
    df = df.dropna()

    # Remove spikes / extreme outliers
    df = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)

    return df


def resample(df, time="none"):
    """function that resamples the data to a coarser res"""
    # Resample if requested, and fill missing values
    if time == 'hourly':
        df = df.resample("H").mean().interpolate().bfill().ffill()
    elif time == 'daily':
        df = df.resample("D").mean().interpolate().bfill().ffill()
    elif time == 'monthly':
        df = df.resample("M").mean().interpolate().bfill().ffill()

    return df


def plot_df(df, x, y, title="", xlabel='Date', ylabel='Number of Passengers', dpi=100):
    """function to plot a line graph to explore data trend"""
    plt.figure(figsize=(15,4), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(15,4), dpi= dpi)
    plt.fill_between(x.values, y1=y.values, y2=-y.values, alpha=0.5, linewidth=2, color='seagreen')
    plt.ylim(-y.max(), y.max())
    plt.title(f'{title} (Two Side View)')
    plt.hlines(y=0, xmin=np.min(df.index), xmax=np.max(df.index), linewidth=.5)
    plt.show()


def normalisation(df, method):
    """
    Perform normalisation on df using the specified method.
    Options:
        - 'log'     : log10 transform with epsilon
        - 'zscore'  : StandardScaler (mean 0, std 1)
        - 'minmax'  : Min-max scaling to [0, 1]
        - 'none'    : return original df
    """

    df = df.copy()

    # normalisation strategies
    if method == "log":
        epsilon = 1e-12
        df = np.log10(df + epsilon)
    
    elif method == "zscore":
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    elif method == "minmax":
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    elif method == "none":
        return df

    else:
        raise ValueError("Invalid method. Choose 'log', 'zscore', 'minmax', or 'none'.")

    df = df
    return df
    

def stationarity(df): 
    """Determine if stationarity exists in the data using dickey-fuller test"""
    
    if df is None:
        raise ValueError("Run preproc() before stationarity testing.")

    results = []

    # iterate through columns and perform dickey fuller test for stationarity
    for col in df.columns:
        series = df[col].dropna()
        # stationary test
        res = adfuller(series)

        adf_stat = res[0]
        p_value = res[1]
        crit_vals = res[4]
        
        # results
        results.append({
            "Variable": col,
            "ADF Statistic": adf_stat,
            "p-value": p_value,
            "1% Critical": crit_vals["1%"],
            "5% Critical": crit_vals["5%"],
            "10% Critical": crit_vals["10%"],
            "Stationary": "Yes" if p_value < 0.05 else "No"
        })
    
    res = pd.DataFrame(results)
    return res


def seasonal(df, variable='Global_active_power', model='additive', period=None):
    """
    Decompose the selected variable into trend, seasonal, and residual components.
    Automatically handles frequency-based periods and avoids modifying internal DF.
    """

    if variable not in df.columns:
        raise ValueError(f"'{variable}' is not a column. Available: {list(df.columns)}")

    if model not in ['additive', 'multiplicative']:
        raise ValueError("model must be 'additive' or 'multiplicative'")

    # Extract series
    series = df[variable].copy()

    # Multiplicative requires positive values
    if model == 'multiplicative' and (series <= 0).any():
        raise ValueError(
            f"Variable '{variable}' contains zeros/negatives; multiplicative model is invalid. "
            "Use additive instead."
        )

    # Decompose
    result = seasonal_decompose(series, model=model, period=period)

    return result


def plot_seasonal_decompose(result, model='additive', title="Seasonal Decomposition", figsize=(14, 10)):
    """function to plot the seasonal decompostion in a more aesthetic manner"""

    plt.rcParams.update({'figure.figsize': figsize})
    plt.rcParams['figure.dpi'] = 300
    result.plot().suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


def feature_eng(df, var='Global_active_power', rollingw=False, lags=False, time='hourly'):
    """
    Adds lag features and rolling window statistics based on frequency:
    time : {'min', 'hourly', 'daily'}
    """

    timelist = {
        'min':   [1, 5, 30, 60, 720, 1440],       # minute data
        'hourly':[1, 12, 24, 36, 168],            # hourly data
        'daily': [1, 3, 7, 30]                    # daily data
    }

    # Select correct windows/lags
    selected = timelist[time]

    # LAG FEATURES
    if lags:
        for t in selected:
            df[f"{var}_lag_{t}"] = df[var].shift(t)

    # ROLLING WINDOW FEATURES
    if rollingw:
        for t in selected:
            df[f"{var}_window_{t}_mean"] = df[var].rolling(t).mean()
            df[f"{var}_window_{t}_std"]  = df[var].rolling(t).std()

    return df

            


        