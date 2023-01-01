import numpy as np
import pandas as pd
import logging 
import yaml

from stockprocessing import analyers as az
from groves import trend_lines as tl

class BreakoutScanner():
    
    def __init__(self, config_path:str='./config.yaml'):
        with open(config_path) as f:
            self.CONFIG = yaml.safe_load(f)
        
        self.OPEN = self.CONFIG['OHLCV']['OPEN']
        self.HIGH = self.CONFIG['OHLCV']['HIGH']
        self.LOW = self.CONFIG['OHLCV']['LOW']
        self.CLOSE = self.CONFIG['OHLCV']['CLOSE']
        self.VOLUME = self.CONFIG['OHLCV']['VOLUME']

        self.peaks = None
        self.df = None

    def scan(self, df):
        self.df = df
        self._preprocess()
        self._evaluate_peaks()

    def _evaluate_peaks(self, consolidation_windows:np.array=[5, 10, 15]):
        """The core method that identifies the setup. Look at each peak (where there is ample prior
             history) and then see if there is consolidation prior.
             #TODO look for volume spike and for prior move

        Args:
            consolidation_windows (np.array, optional): _description_. Defaults to [5, 10, 15].
        """

        for idx, row in self.peaks.iterrows():
            # get the location (i.e., the i-th row) that matches the peak
            pk_loc = self.df.index.get_loc(idx)
            # only analyze peaks with enough history
            if pk_loc > (row['fullwidth'] + np.amax(consolidation_windows)):
                pk_start_loc = int(pk_loc - row['fullwidth'])
                pk_end_loc = int(pk_loc + row['fullwidth'])
                for cw in consolidation_windows:
                    cw_start_idx = self.df.index[pk_start_loc-cw]
                    cw_end_idx = self.df.index[pk_start_loc]
                    cw_df = self.df.loc[cw_start_idx:cw_end_idx].copy()
                    # get the trends
                    
                    print(f'cw_df = {cw_df.index[0]} - {cw_df.index[-1]}')

    def _preprocess(self):
        # Find Peaks
        self.peaks = az.find_extrema(self.df, self.CLOSE)
        self.df = az.append_simple_moving_averages(self.df, self.CLOSE)
        self.df = az.apply_range_indicators(self.df)
        self.df = az.append_in_percentage_range(self.df, percentage=0.05)

    def _gen_x(self, df):
        return np.arange(len(df))
   
    def _get_trend_lines(self, df_trend):
        # Apply the smoothing algorithm and get the gradient/intercept terms
        m_res, c_res = tl.find_grad_intercept(
            case = 'resistance', 
                x = self._gen_x(df_trend), 
                y = tl.heat_eqn_smooth(df_trend[self.HIGH].values.copy()),
            )
        m_supp, c_supp = tl.find_grad_intercept(
                case = 'support', 
                x = self._gen_x(df_trend), 
                y = tl.heat_eqn_smooth(df_trend[self.LOW].values.copy()),
            )
        return {"m_res": m_res, "c_res": c_res, "m_supp":m_supp, "c_supp": c_supp}
