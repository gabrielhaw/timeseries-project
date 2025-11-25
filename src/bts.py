import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import kagglehub
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

class EnergyConsumption: 
    """energy consumption class containing various functions"""

    def __init__(self, csv_path, time='none', verbose=False):
        self.csv_path = csv_path
        self.df = None
        self.time = time 
        self.original_df = None  
        self.verbose = verbose

    def preproc(self): 
        """Process the household power consumption data"""

        # Load CSV
        df = pd.read_csv(self.csv_path)

        # Build datetime
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')

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
        if self.verbose == True:
            print(f"Percentage of missing data: {pct_missing:.3f}%")

        # Remove all remaining missing rows (best choice for electricity data)
        df = df.dropna()

        # Remove spikes / extreme outliers
        df = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)

        # Resample if requested, and fill missing values
        if self.time == 'daily':
            df = df.resample("D").mean().interpolate().bfill().ffill()
        elif self.time == 'hourly':
            df = df.resample("H").mean().interpolate().bfill().ffill()

        # Save cleaned df to object
        self.df = df
        self.original_df = df
        
    
    def datavis(self, plot="menu"):
        """
        Visualise data distributions to assess normalization/outliers.
        """

        if self.df is None:
            raise ValueError("Run preproc() before visualization.")

        explanations = {
            "pairplot": "Pairplot → see relationships between features (sampled).",
            "hist":     "Histogram → check skewness + heavy tails.",
            "kde":      "KDE → smooth distribution shape.",
            "box":      "Boxplot → detect outliers + variable spread.",
            "qq":       "QQ Plot → shows if data is normally distributed.",
            "line":     "Line plot → view how each variable changes over time.",
            "menu":     "Options: pairplot, hist, kde, box, qq, line"
        }

        # Show menu
        if plot == "menu":
            print("\n=== Visualisation Menu ===")
            for k, v in explanations.items():
                print(f"{k:10} - {v}")
            return

        # Print explanation for selected plot
        print(f"\n{explanations.get(plot, 'Unknown plot type.')}\n")

        # plotting 

        if plot == 'pairplot':
            sns.pairplot(self.df.sample(min(len(self.df), 2000)))
            plt.show()

        elif plot == 'hist':
            self.df.hist(bins=40, figsize=(12,8))
            plt.show()

        elif plot == 'kde':
            self.df.plot(kind='density', subplots=True, layout=(3,3), figsize=(14,10))
            plt.show()

        elif plot == 'box':
            plt.figure(figsize=(14,6))
            self.df.boxplot(rot=90)
            plt.show()

        elif plot == 'qq':
            for col in self.df.columns:
                plt.figure(figsize=(5,5))
                stats.probplot(self.df[col].dropna(), dist="norm", plot=plt)
                plt.title(col)
                plt.show()

        elif plot == 'line':
            num_vars = len(self.df.columns)
            plt.figure(figsize=(14, 3 * num_vars))

            for i, col in enumerate(self.df.columns, 1):
                plt.subplot(num_vars, 1, i)
                plt.plot(self.df.index, self.df[col], color='blue')
                plt.title(col)
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.tight_layout()

            plt.show()


        else:
            print("Invalid plot type. Run datavis('menu') to see options.")


    def normalisation(self, method):
        """
        Perform normalisation on self.df using the specified method.
        Options:
            - 'log'     : log10 transform with epsilon
            - 'zscore'  : StandardScaler (mean 0, std 1)
            - 'minmax'  : Min-max scaling to [0, 1]
            - 'none'    : return original df
        """

        df = self.df.copy()

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

        self.df = df
        

    def reset(self):
        """Restore the original untransformed DataFrame."""
        self.df = self.original_df.copy()
        

    def stationarity(self): 
        """Determine if stationarity exists in the data using dickey-fuller test"""
        
        if self.df is None:
            raise ValueError("Run preproc() before stationarity testing.")

        results = []

        # iterate through columns and perform dickey fuller test for stationarity
        for col in self.df.columns:
            series = self.df[col].dropna()
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
        

        dfres = pd.DataFrame(results)
        return dfres


    def seasonal(self, variable='Global_active_power', model='additive', period=7):
        """
        Decompose the selected variable into trend, seasonal, and residual components.
        Useful for understanding additive vs multiplicative behavior.
        """

        # Validate variable
        if variable not in self.df.columns:
            raise ValueError(f"'{variable}' is not a column. Available: {list(self.df.columns)}")

        if model not in ['additive', 'multiplicative']:
            raise ValueError("model must be 'additive' or 'multiplicative'")

        # Extract the series safely
        series = self.original_df[variable].copy()

        # Decompose
        result = seasonal_decompose(series, model=model, period=period)
        result.plot()
        plt.suptitle(f"{variable} decomposition ({model})", fontsize=14)
        plt.tight_layout()
        plt.show()

        return result
