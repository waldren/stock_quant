import numpy as np
import pandas as pd
import pandas_ta as ta
import scipy.signal as sig 
import logging 
import warnings

def calc_slope(x:np.array)-> float:
    slope = np.polyfit(range(len(x)), x, 1)[0]
    return slope

def normalize_data(x:np.array)->np.array:
    min = x.min()
    max = x.max()
    
    # normalization part
    norm = (x - min) / (max - min)
    
    return norm

def find_extrema(self, x:pd.Dataframe, col_name='close', use_cwt=False, width=3, distance=3, prominence=0, ):
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

    self.logger.debug(f'[{x.index[0]}:{x.index[-1]} {col_name}] - Mean(std):  {round(pw_full.mean(), 3)}({round(pw_full.std(), 3)}) Width and {round(prom.mean(), 3)}({round(prom.std(), 3)}) Prominence')
    
    v = np.zeros(x.shape[0])
    pr = np.zeros(x.shape[0])
    ph = np.zeros(x.shape[0])
    pf = np.zeros(x.shape[0])
    j = 0
    dt = x.index.values
    for t in zip(peaks, pw_half, pw_full, prom):
        idx = t[0]
        hw = int(t[1]/2)
        fw = int(t[2]/2)
        # Exclude peaks less than width and prominence thresholds
        if t[2] > self.pw_threshold and t[3] > self.prom_threshold:
            v[idx] = 1
            ph[idx] = t[1]
            pf[idx] = t[2]
            pr[idx] = t[3] 
        else:
            if t[3] != 0:
                logging.warning(f'Rejected peak {dt[idx]} with {t[2]} Width and {t[3]} Prominence')
    df = pd.DataFrame(index=x.index)
    df['peaks'] = v
    df['halfwidth'] = ph
    df['fullwidth'] = pf
    df['prominence'] = pr 
    
    return df
