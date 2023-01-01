import numpy as np
import pandas as pd
import pandas_ta as ta
import scipy.signal as sig 
import logging 
import warnings
from stockprocessing import *
from groves import trend_lines as tl

logger = logging.getLogger('__name__')

def calc_slope(x:np.array)-> float:
    slope = np.polyfit(range(len(x)), x, 1)[0]
    return slope

def normalize_data(x:np.array)->np.array:
    min = x.min()
    max = x.max()
    
    # normalization part
    norm = (x - min) / (max - min)
    
    return norm

def dollar_value(self, df:pd.DataFrame):
        '''
        Calculates the Dollar Value for a OHLCV dataframe
        '''
        return (df[HIGH]+df[LOW])/2 * df[VOLUME]

def find_extrema(x:pd.DataFrame, col_name:str=CLOSE, use_cwt:bool=False, width=3, distance=3, prominence=0 )->pd.DataFrame:
    """Uses SciPy to find the peaks in the signal and calulate their widths and prominence

    Args:
        x (pandas.DataFrame): The is the dataframe containing the timeseries
        col_name (str, optional): The is the name of the colum to search in the dataframe. Defaults to CLOSE.
        use_cwt (bool, optional): Should convolution be used? Defaults to False.
        width (int, optional): the minium width of a peak. Defaults to 3.
        distance (int, optional): Minimum distance between peaks. Defaults to 3.
        prominence (int, optional): Minimum prominence. Defaults to 0.

    Returns:
        pandas.DataFrame: This dataframe has the same index as 'x' and contains columnns:
            'peaks' - 1 = peak detected, 0 = not a peak
            'halfwidth' - the half height width of the peak
            'fullwidth' - the full height width of the peak
            'prominence' - the promience at the peak
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="some peaks have.*")
        if use_cwt:
            peaks = sig.find_peaks_cwt(x[col_name], width)
        else:
            peaks = sig.find_peaks(x[col_name], distance=distance, 
                            prominence=prominence, width=width)[0]

        pw_half = sig.peak_widths(x[col_name], peaks, rel_height=0.5)[0]
        pw_full = sig.peak_widths(x[col_name], peaks, rel_height=1)[0]
        prom = sig.peak_prominences(x[col_name], peaks)[0]

    logger.info(f'[{x.index[0]}:{x.index[-1]} {col_name}] - Mean(std):  {round(pw_full.mean(), 3)}({round(pw_full.std(), 3)}) Width and {round(prom.mean(), 3)}({round(prom.std(), 3)}) Prominence')
    
    # Create empty arrays for peak data 
    v = np.zeros(x.shape[0])
    pr = np.zeros(x.shape[0])
    ph = np.zeros(x.shape[0])
    pf = np.zeros(x.shape[0])
    #j = 0
    # Get an array of index values so we can log rejections by date
    dt = x.index.values
    # Populate the arrays with the peak data where the data is placed at the index of the peak
    for t in zip(peaks, pw_half, pw_full, prom):
        idx = t[0]
        hw = int(t[1]/2)
        fw = int(t[2]/2)
        # Exclude peaks less than width and prominence thresholds
        if t[2] > width and t[3] > prominence:
            v[idx] = 1
            ph[idx] = t[1]
            pf[idx] = t[2]
            pr[idx] = t[3] 
        else:
            if t[3] != 0:
                logger.info(f'Rejected peak {dt[idx]} with {t[2]} Width and {t[3]} Prominence')
    # Create a new dataframe with the same index as 'x' to hold all the peak data.
    df = pd.DataFrame(index=x.index)
    df['peaks'] = v
    df['halfwidth'] = ph
    df['fullwidth'] = pf
    df['prominence'] = pr 
    
    return df

def append_ohlcv_percent_change_and_growth(df:pd.DataFrame)-> pd.DataFrame:
    # Call the percent change column using the same OHLCV naming
    close_pct_chg = "{}_pct_chg".format(CLOSE)
    open_pct_chg = "{}_pct_chg".format(OPEN)
    low_pct_chg = "{}_pct_chg".format(LOW)
    high_pct_chg = "{}_pct_chg".format(HIGH)
    volume_pct_chg = "{}_pct_chg".format(VOLUME)

    df[close_pct_chg] = df[CLOSE].pct_change(fill_method='ffill')
    df[open_pct_chg] = df[OPEN].pct_change(fill_method='ffill')
    df[low_pct_chg] = df[LOW].pct_change(fill_method='ffill')
    df[high_pct_chg] = df[HIGH].pct_change(fill_method='ffill')
    df[volume_pct_chg] = df[VOLUME].pct_change(fill_method='ffill')

    df['growth'] = df[close_pct_chg].cumsum()
    return df 

def append_simple_moving_averages(df:pd.DataFrame, col_name:str=CLOSE, windows=[10, 20, 50, 100, 200])->pd.DataFrame:
    for w in windows:
        newCol = f'{col_name}_{w}sma'
        df[newCol] = df[col_name].rolling(window=w).mean()
    
    return df

def apply_range_indicators(df:pd.DataFrame, window=20)->pd.DataFrame:
    # Calculate the Average Daily Range (%) of the last 'window' bars
    
    df['adr_pct'] = 100*(((df[HIGH]/df[LOW]).rolling(window=window).mean())-1)

    return df

def append_in_percentage_range(df:pd.DataFrame, percentage=0.02):
    #column name for if the high and low are withing the percentage
    in_range = f"in_{percentage*100}%_range"
    # Number of consecutive bars the high and low are in range
    consol = f"bars_in_{percentage*100}%_range"

    # Calculate the percent change if not already done
    hldiff = f'{HIGH}_{LOW}_pct_diff'
    if hldiff not in df.columns:
        df[hldiff] = abs( (df[HIGH]-df[LOW])/((df[HIGH]+df[LOW])/2) )

    # Mark 1 if in range
    df[consol]  = df[hldiff].apply(lambda x: 1 if x <= percentage else 0)

    # Calculate the number of bars in the range
    ir_list = []
    # Prior number of bars
    x0 = 0.0
    for i, row in df.iterrows():
        # Calculate the next value from prior value (adds 1 if not 0 otherwise resets to 0)
        x0 =  x0*row[consol] + row[consol]
        ir_list.append(x0)
    df[in_range] = ir_list 

    return df 

def _gen_x(df):
    return np.arange(len(df))

def _find_gradient_resistance(df:pd.DataFrame):
    m_res, c_res = tl.find_grad_intercept(
        case = 'resistance', 
        x = _gen_x(df), 
        y = tl.heat_eqn_smooth(df['high'].values.copy()),
    )
    return f'{m_res}|{c_res}'

def _find_gradient_support(df:pd.DataFrame):
    m_supp, c_supp = tl.find_grad_intercept(
        case = 'support', 
        x = _gen_x(df), 
        y = tl.heat_eqn_smooth(df['low'].values.copy()),
    )
    return f'{m_supp}|{c_supp}'

def append_moving_trend_lines(df:pd.DataFrame, window=7)->pd.DataFrame:
    df['r_slope_intercept'] = df.rolling(window=window).apply(_find_gradient_resistance)
    df['resistance_slope'], df['resistance_intercept'] = df['r_slope_intercept'].split("|")

    df['s_slope_intercept'] = df.rolling(window=window).apply(_find_gradient_resistance)
    df['support_slope'], df['support_intercept'] = df['s_slope_intercept'].split("|")

    df.drop(['r_slope_intercept', 's_slope_intercept'],inplace=True)
    return df
