import numpy as np
import pandas as pd
import pandas_ta as ta
import scipy.signal as sig 

from .baseprocessors import BaseProcessor
from stockprocessing import grove_functions as gf

class BreakoutProcessor(BaseProcessor):
    def __init__(self, symbol: str = "", date_col: str = 'dt', open_col: str = 'open', high_col: str = 'high'
                , low_col: str = 'low', close_col: str = 'close', volume_col: str = 'volume'):
        super().__init__(symbol, date_col, open_col, high_col, low_col, close_col, volume_col)

        # Parameters
        self.prv_brkout_thr = 2 #percent  1 = 100%
        self.pullback_pct   = -0.2

        # Call the percent change column using the same OHLCV naming
        self.close_pct_chg_col = "{}_pct_chg".format(self.close)
        self.open_pct_chg_col = "{}_pct_chg".format(self.open)
        self.low_pct_chg_col = "{}_pct_chg".format(self.low)
        self.high_pct_chg_col = "{}_pct_chg".format(self.high)

    def apply_moav(self, df):
        if df is None:
            print("apply_moavs: dataframe is None")
            return df 
        # Create the rolling averages 
        df['10sma'] = df[self.close].rolling(window=10).mean()
        df['20sma'] = df[self.close].rolling(window=20).mean()
        df['50sma'] = df[self.close].rolling(window=50).mean()
        df['100sma'] = df[self.close].rolling(window=100).mean()
        df['200sma'] = df[self.close].rolling(window=200).mean() 
        # Find the highest of the moving averages to help determine if price is above all   
        df['max_mav'] = df[["10sma", "20sma", "50sma", "100sma"]].max(axis=1)
        df['above_20sma'] = df[self.close] > df['20sma']
        df['20sma_above_50sma'] = df['20sma'] > df['50sma']
        
        return df

    ### Datafrom APPLY Functions ###
    def _dollar_value(self, df:pd.DataFrame):
        '''
        Calculates the Dollar Value for a OHLCV dataframe
        '''
        return (df[self.high]+df[self.low])/2 * df[self.volume]
    
    def _mark_bigmoves(self, df):
        if df['peaks'] == 1 and df['growth']>= self.prv_brkout_thr:
            return 1
        elif df['troughs'] == 1 and abs(df['growth']) >= self.prv_brkout_thr:
            return -1
        else:
            return 0
    ################################

    def find_extrema(self, df):
        if df is None:
            print("apply_extrema: df is None")
            return df 
        close = df[self.close]
        df = self._find_peaks(close, 'peaks', df)
        close = close.mul(-1)
        df = self._find_peaks(close, 'troughs', df)
        return df
    def _find_peaks(self, close, col_name, df):
        peaks, dic = sig.find_peaks(close)
        v = np.zeros(df.shape[0])
        h = np.zeros(df.shape[0])
        j = 0
        for i in peaks:
            v[i] = 1
        df[col_name] = v
        
        return df

    def apply_volume_indicators(self, df:pd.DataFrame)->pd.DataFrame:
        # Calculate the Average Volume for last 20 days
        df['vol_avg'] = df[self.volume].rolling(window=20).mean()
        # Calculate the Dollar Value at the day
        df['dol_val'] = df.apply(self._dollar_value, axis=1)
        return df
    
    def apply_range_indicators(self, df:pd.DataFrame)->pd.DataFrame:
        # Calculate the Average Daily Range (%) of the last 20 days
        df['adr_pct'] = 100*(((df[self.high]/df[self.low]).rolling(window=20).mean())-1)
        return df
    
    def apply_growth_consolidation(self, df:pd.DataFrame)->pd.DataFrame:
        #get the close price
        prices = df[self.close]
        df.loc[:, 'consolidating'] = gf.find_consolidation(prices.values)
        df.loc[:, 'trend_filter'] = gf.trend_filter(prices)
        df.loc[:, 'filtered'] = np.where(
            df['consolidating'] + df['trend_filter'] == 2,
            True,
            False,
        ) 
        return df 
    
    def apply_adx_indicator(self, df:pd.DataFrame)->pd.DataFrame:
        a = ta.adx(df[self.high], df[self.low], df[self.close], length = 14)
        a.rename(columns={'ADX_14': 'adx_14', 'DMP_14': 'dmp_14', 'DMN_14': 'dmn_14'}, inplace=True)
        df = df.join(a)
        return df

    def process(self, symbol:str, df:pd.DataFrame)->pd.DataFrame:
        if df is None:
            print("Cannot process: df is None")
            return df 
        if df.empty:
            print("Cannot process: df is empty")
            return df 
        self.symbol = symbol
        df = self.apply_moav(df)
        df = self.apply_volume_indicators(df)
        df = self.apply_range_indicators(df)
        df = self.find_extrema(df)
        df = self.apply_adx_indicator(df)
        df = self.apply_growth_consolidation(df)
        return df 
    
    def process_check(self, df:pd.DataFrame) -> int:
        #grab a column name from each function
        indicators = set(["10sma", "vol_avg", "adr_pct", "peaks", "adx_14", "consolidating"])
        try:
            if df.empty:
                return -1
            elif indicators.issubset(df.columns):
                return 1
            else:
                return 0
        except Exception as e:
            print (f"Exception in precheck")
            print (e)
            return -2