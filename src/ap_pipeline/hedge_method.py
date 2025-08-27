import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np


def run_cross_sectional_factor_regression(df_engle, df_clim, clim_col='1dF', concern_col='WSJ.AR1', start_year: int = 2010, end_year = 2017):
    # Load and prepare data
    df_engle['MONTH'] = pd.to_datetime(df_engle['MONTH'])
    df_clim['MONTH'] = pd.to_datetime(df_clim['MONTH'])

    # Delete tickers for which no return data is available
    tickers_to_delete = ['AMTM-US', 'BRK.A-US', 'DPZ-US', 'GEV-US', 'KVUE-US', 'MTCH-US', 'OTIS-US', 'SOLV-US', 'SW-US', 'VLTO-US']

    # Merge and filter
    df_merged = df_engle.merge(df_clim, on=['MONTH', 'ISSUER_TICKER'], how='left')
    
    df_merged = df_merged[df_merged.MONTH.dt.year > start_year - 1]
    df_merged = df_merged[df_merged.MONTH.dt.year < end_year + 1]

    df_merged = df_merged.loc[~df_merged.ISSUER_TICKER.isin(tickers_to_delete)]

    cols = ['MONTH', 'ISSUER_TICKER', clim_col, 'Z_MKT', 'Z_HML', 'Z_SIZE', 'Stock_Return']
    df_merged[cols] = df_merged[cols].fillna(0)

    Z_clim_rs, Z_mkt_rs, Z_size_rs, Z_hml_rs, CC = [], [], [], [], []
    months = df_merged.MONTH.unique()

    df_merged = df_merged.sort_values(["ISSUER_TICKER", "MONTH"]).reset_index(drop=True)

    for idx, month in enumerate(months):
        if idx == 0:
            continue

        # Select data at time {t} for returns and CC, time {t-1} for the Z-matrix
        subset_t = df_merged[df_merged.MONTH == month].drop_duplicates(subset=['ISSUER_TICKER', 'MONTH']).copy()
        subset_t_prev = df_merged[df_merged.MONTH == months[idx - 1]].drop_duplicates(subset=['ISSUER_TICKER', 'MONTH']).copy()

        # Select only the stocks at {t-1} where return data is available at {t}
        subset_t_prev = subset_t_prev.loc[subset_t_prev.ISSUER_TICKER.isin(subset_t.ISSUER_TICKER.unique())]

        tickers1 = subset_t.ISSUER_TICKER.unique()
        tickers2 = subset_t_prev.ISSUER_TICKER.unique()

        diff1 = list(set(tickers1) - set(tickers2))
        diff2 = list(set(tickers2) - set(tickers1))
        diff = diff1 + diff2
        
        if len(diff) > 0:
            subset_t = subset_t.loc[~subset_t.ISSUER_TICKER.isin(diff)]
            subset_t_prev = subset_t_prev.loc[~subset_t_prev.ISSUER_TICKER.isin(diff)]

        CC_t = subset_t[concern_col].dropna().values[0]
        CC.append(CC_t)
        
        r_t = subset_t['Stock_Return'].values
        Z_clim = subset_t_prev[clim_col].values

        Z_mkt = subset_t_prev['Z_MKT'].values
        Z_size = subset_t_prev['Z_SIZE'].values
        Z_hml = subset_t_prev['Z_HML'].values

        Z_clim_rs.append(Z_clim.T @ r_t)

        Z_mkt_rs.append(Z_mkt.T @ r_t)
        Z_size_rs.append(Z_size.T @ r_t)
        Z_hml_rs.append(Z_hml.T @ r_t)

    x1 = pd.Series(Z_clim_rs)
    x2 = pd.Series(CC)

    # Run regression
    X = pd.DataFrame({
        'Z_clim_r': Z_clim_rs,
        'Z_size_r': Z_size_rs,
        'Z_hml_r': Z_hml_rs,
        'Z_mkt_r': Z_mkt_rs
    })
    X = sm.add_constant(X)

    y = np.array(CC)

    # 📈 Plot Z_clim_r vs. CC
    plt.figure(figsize=(8, 5))
    plt.scatter(X['Z_clim_r'], y, alpha=0.7)
    m, b = np.polyfit(X['Z_clim_r'], y, 1)
    plt.plot(X['Z_clim_r'], m * X['Z_clim_r'] + b, color='red', label='Fitted line')
    plt.xlabel('Z_clim_r (climate-weighted return)')
    plt.ylabel('CC (climate concern)')
    plt.title('Z_clim_r vs. Climate Concern')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Fit and return model
    model = sm.OLS(y, X).fit()
    print(model.summary())

    return model
