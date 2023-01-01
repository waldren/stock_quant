import numpy as np
import pandas as pd
import pandas_ta as ta
import scipy.signal as sig 
import logging 
import warnings

from .baseprocessors import BaseProcessor
from groves import grove_functions as gf

class BreakoutProcessor(BaseProcessor):
    def __init__(self, symbol: str = "", date_col: str = 'dt', open_col: str = 'open', high_col: str = 'high'
                , low_col: str = 'low', close_col: str = 'close', volume_col: str = 'volume'):
        super().__init__(symbol, date_col, open_col, high_col, low_col, close_col, volume_col)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Parameters
        self.prv_brkout_thr = 2 #percent  1 = 100%
        self.pullback_pct   = -0.2
        self.consolidation_pct_chg = 0.02
        self.consolidation_window = 3
        self.lookback_period = 5
        self.lookforward_period = 5
        self.big_move_percentage = 10
        self.use_cwt = False 
        self.peak_min_height = 1 #Used by scipy find_peaks to filter out peaks by height
        self.peak_min_distance = 14 # the minimum distance between peaks 
        self.peak_width = np.arange(2,14) #Used by scipy find_peaks_cwt for the expected peak width
        self.extrema_chunk_size = 90 # window used to look for peaks and troughs
        self.pw_threshold =  1.5 
        self.prom_threshold = 0.9

        # Call the percent change column using the same OHLCV naming
        self.close_pct_chg = "{}_pct_chg".format(self.close)
        self.open_pct_chg = "{}_pct_chg".format(self.open)
        self.low_pct_chg = "{}_pct_chg".format(self.low)
        self.high_pct_chg = "{}_pct_chg".format(self.high)
        self.volume_pct_chg = "{}_pct_chg".format(self.volume)

    def apply_moav(self, df)->pd.DataFrame:
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
    
    def _calc_slope(self, x):
        slope = np.polyfit(range(len(x)), x, 1)[0]
        return slope
    ################################

    def _normalize_data(self, data:pd.Series)->pd.Series:
        min = data.min()
        max = data.max()
        
        # normalization part
        norm = (data - min) / (max - min)
        
        return norm

    def find_extrema(self, df):
        if df is None:
            print("apply_extrema: df is None")
            return df 
        # If we do not break up the dataframe the larger future growth
        # will interfere with peak identification in the more distant past
        #TODO The slicing of the dataframe is arbitrary. Probably need to look 
        #     for a more emperic method to determin break points. OR normalize the data
        frames = []
        chunk_size = self.extrema_chunk_size 
        for start in range(0, df.shape[0], chunk_size):
            df_subset = df.iloc[start:start + chunk_size].copy()
            close = df_subset[self.close]
            df_subset = self._find_peaks(close, 'peaks', df_subset)
            close = close.mul(-1)
            df_subset = self._find_peaks(close, 'troughs', df_subset)
            frames.append(df_subset)
        df = pd.concat(frames)
        
        #TODO try to use .shift(-full width)
        # Convert Prominences to percent moves
        df['peaks_prct_move'] = np.where(df['peaks'] == 1, df[self.close]/(df[self.close]-df['peaks_prominence']) ,0)
        df['troughs_prct_move'] = np.where(df['troughs'] == 1, df[self.close]/(df[self.close]+df['troughs_prominence']) ,0)
        
        return df
         
    def _find_peaks(self, close, col_name, df):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="some peaks have.*")
            if self.use_cwt:
                peaks = sig.find_peaks_cwt(close, self.peak_width)
            else:
                peaks = sig.find_peaks(close, distance=self.peak_min_distance, 
                                prominence=self.prom_threshold, width=self.pw_threshold)[0]

            pw_half = sig.peak_widths(close, peaks, rel_height=0.5)[0]
            pw_full = sig.peak_widths(close, peaks, rel_height=1)[0]
            prom = sig.peak_prominences(close, peaks)[0]

        self.logger.debug(f'[{self.symbol}-{df.index[0]}:{df.index[-1]} {col_name}] - Mean(std):  {round(pw_full.mean(), 3)}({round(pw_full.std(), 3)}) Width and {round(prom.mean(), 3)}({round(prom.std(), 3)}) Prominence')
        
        v = np.zeros(df.shape[0])
        pr = np.zeros(df.shape[0])
        ph = np.zeros(df.shape[0])
        pf = np.zeros(df.shape[0])
        j = 0
        dt = df.index.values
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
        df[col_name] = v
        df[f'{col_name}_halfwidth'] = ph
        df[f'{col_name}_fullwidth'] = pf
        df[f'{col_name}_prominence'] = pr 
        
        return df


    def in_consolidation(self, df:pd.DataFrame, percentage=0.02):
        in_range = "in_consolidation_range"
        consol = "in_consolidation"

        hldiff = 'high_low_pct_diff'
        if hldiff not in df.columns:
            df[hldiff] = abs( (df[self.high]-df[self.low])/((df[self.high]+df[self.low])/2) )

        df[consol]  = df[hldiff].apply(lambda x: 1 if x <= percentage else 0)


        ir_list = []
        x0 = 0.0
        for i, row in df.iterrows():
            x0 =  x0*row[consol] + row[consol]
            ir_list.append(x0)
        df[in_range] = ir_list 
        return df 

    def calculate_percent_changes(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Calculates the percent change for the OHLCV columns and adds "{column}.pct_change"
        to the data frame

        Args:
            df (pd.DataFrame): Dataframe that contains the OHLCV columns

        Returns:
            pd.DataFrame: The original dataframe with the additional "pct_change" columns
        """
        df[self.close_pct_chg] = df[self.close].pct_change(fill_method='ffill')
        df[self.open_pct_chg] = df[self.open].pct_change(fill_method='ffill')
        df[self.low_pct_chg] = df[self.low].pct_change(fill_method='ffill')
        df[self.high_pct_chg] = df[self.high].pct_change(fill_method='ffill')
        df[self.volume_pct_chg] = df[self.volume].pct_change(fill_method='ffill')
        return df 

    def calculate_growth(self, df)->pd.DataFrame:
        df['growth'] = df[self.close_pct_chg].cumsum()
        df = self.in_consolidation(df, percentage=self.consolidation_pct_chg)
        df['slope_5'] = df[self.close].rolling(5,min_periods=5).apply(self._calc_slope,raw=False)        
        df['slope_20']= df[self.close].rolling(20,min_periods=20).apply(self._calc_slope,raw=False)  
        #df = self._mark_bigmoves(df, self.big_move_percentage)
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
        df.loc[:, 'consolidating'] = gf.find_consolidation(prices.values, days_to_smooth= 20,
                       perc_change_days = 5, perc_change_thresh=self.consolidation_pct_chg, check_days= 5)
        df.loc[:, 'dg_trend_filter'] = gf.trend_filter(prices)
        df.loc[:, 'dg_filtered'] = np.where(
            df['consolidating'] + df['dg_trend_filter'] == 2,
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
        df = self.calculate_percent_changes(df)
        df = self.apply_moav(df)
        df = self.apply_volume_indicators(df)
        df = self.apply_range_indicators(df)
        df = self.find_extrema(df)
        df = self.calculate_growth(df)
        df = self.apply_adx_indicator(df)
        df = self.apply_growth_consolidation(df)
        return df 
    
    def process_check(self, df:pd.DataFrame) -> int:
        #grab a column name from each function
        indicators = set([self.close_pct_chg, "10sma", "vol_avg", "adr_pct", 
                            "peaks", "adx_14", "dg_consolidating"])
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
