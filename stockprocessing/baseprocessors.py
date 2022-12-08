from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import os
import pickle
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime 
from datetime import timedelta

class BaseProcessor(ABC):
    def __init__(self, symbol:str="", date_col:str='dt', open_col:str='open', high_col:str='high',
                    low_col:str='low', close_col:str='close', volume_col:str='volume'):
        self.symbol = symbol
        self.date = date_col
        self.open = open_col
        self.high = high_col
        self.low = low_col
        self.close = close_col
        self.volume = volume_col

    # Yahoo Finance returns a datafrome with a datetime column not with the datetime as the index
    # processors assume the index of the dataframe is the datetime data.
    def ensure_datetime_index(self, df):
        if isinstance(df.index, pd.DatetimeIndex):
            return df
        elif isinstance(df[self.date], pd.DatatimeIndex):
                df['old_index'] = df.index
                df.set_index(self.date)
                return df
        else:
            raise Exception("The dataframe's index or date_col must be of type DatetimeIndex")

    def create_shaded_col(self, df, filter_col, high, low)->np.array:
        return np.where(
                df[filter_col].values,
                high,
                low,
                )

    @abstractmethod
    def process(self, symbol, df) ->pd.DataFrame:
        pass
    '''
    @abstractmethod
    def get_indicators(self, indicator:str) ->set(str):
        return set()
    '''

class SqueezeProcessor(BaseProcessor):
    '''
    Processes a stock by creating the neccessary indicators to determine if the stock has been consolidating 
        and if it has now broken out or broken down (or still consolidating)
        * currently assumption is that the history is daily
        data_folder = This is the folder to store the output data
        indices = This is a list of the column names of any stock indices close values.
        drop_nan = This enables the dropping of rows in the result where columns have NAN values (such as the first rows if a moving average is used)
        squeeze_threshold = this is the percentage that the closes must be within to be considered consolidating
        breakout_duration = the number of bar to calculate the percentage of close change to determine if a breakout/breakdown occurred
        breakout_threshold = the close change percentage that the close must be >= to mark it as a breakout
        hx_length = the number of bars before the breakout/breakdown that should be included in the samples generated
        future_length the number of bars after the breakout/breakdown (including the breakout bar) that should be included in the samples generated
    '''
    def __init__(self, drop_nan=True,
                squeeze_threshold=0.025, breakout_duration=5, breakout_threshold=0.05, hx_length=40, future_length=5,
                moav_period=20, moav_long=100, moav_short=10, keltner_multiplier=1.5, **kwargs):

        self.drop_nan = drop_nan
        self.moav_period = moav_period
        self.keltner_multiplier = keltner_multiplier
        self.ma_name = "{}sma".format(self.moav_period)
        self.max_name = "{}max".format(self.moav_period)
        self.lo_name = "{}low".format(self.moav_period)
        self.moav_short = moav_short
        self.moav_long  = moav_long
        self.ma_s_name = "{}sma".format(self.moav_short)
        self.ma_l_name = "{}sma".format(self.moav_long)
        self.squeeze_threshold = 1 - squeeze_threshold
        self.breakout_duration = breakout_duration
        self.breakout_threshold = breakout_threshold
        self.bar_duration = timedelta(days=1)
        self.hx_length = hx_length
        self.future_length = future_length
        #self.indicators = super().get_indicators.update({self.ma_name, self.max_name, self.lo_name, self.ma_s_name, 
        #                    self.ma_l_name, })
        super().__init__(**kwargs)
    '''
    def get_indicators(self, indicator: str) -> set(str):
        return self.indicators
    '''

    def _in_squeeze_bands(self, df):
        return df['lower_band'] > df['lower_keltner'] and df['upper_band'] < df['upper_keltner']
    def _donchian(self, df):
        return (df[self.max_name] + df[self.lo_name]) / 2
    def _mom_hist(self, df):
        return df[self.close] - ((df['donchian']+ df[self.ma_name])/2)
    def _over_ma(self, df):
        return (df[self.close] > df[self.ma_s_name] and df[self.close] > df[self.ma_l_name])
    
    def process(self, symbol, df):
        self.symbol=symbol
        if df is None:
            return None
        if df.empty:
            return None
        
        df[self.ma_name] = df[self.close].rolling(window=self.moav_period).mean()
        df['stddev'] = df[self.close].rolling(window=self.moav_period).std()
        df[self.max_name] = df[self.close].rolling(window=self.moav_period).max()
        df[self.lo_name] = df[self.close].rolling(window=self.moav_period).min()
        
        df[self.ma_s_name] = df[self.close].rolling(window=self.moav_short).mean()
        df[self.ma_l_name] = df[self.close].rolling(window=self.moav_long).mean()
        
        
        df['donchian'] =  df.apply(self._donchian, axis=1)
        df['mom_hist'] = df.apply(self._mom_hist, axis=1)
        df['lower_band'] = df[self.ma_name] - (2 * df['stddev'])
        df['upper_band'] = df[self.ma_name] + (2 * df['stddev'])
        df['TR'] = abs(df[self.high] - df[self.low])
        df['ATR'] = df['TR'].rolling(window=self.moav_period).mean()
        df['lower_keltner'] = df[self.ma_name] - (df['ATR'] * self.keltner_multiplier)
        df['upper_keltner'] = df[self.ma_name] + (df['ATR'] * self.keltner_multiplier)
        df['close_pct_change'] = df[self.close].pct_change(fill_method='ffill')
        df['growth'] = df['close_pct_change'].cumsum()
        
        df['has_momentum'] = df.apply(self._over_ma, axis=1)
        df['TTM_squeeze'] = df[self.close].rolling(window=5).min() > (df[self.close].rolling(window=5).max() * self.squeeze_threshold)
        
        if self.drop_nan:
            df = df.dropna(axis=0)
        return df 

    def chart(self, symbol, df):
        print(self.date)

        df['shaded'] = self.create_shaded_col(df, 'TTM_squeeze', df[self.high].max(), 0)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace( go.Candlestick(x=df.index, open=df[self.open], high=df[self.high], low=df[self.low], close=df[self.close]), row=1, col=1)
        fig.add_trace( go.Scatter(
            x=df.index, y=df['upper_band'], name='Upper Bollinger Band', line={'color': 'red'}
            ), row=1, col=1)
        fig.add_trace( go.Scatter(x=df.index, y=df['lower_band'], name='Lower Bollinger Band', line={'color': 'red'}), row=1, col=1)
        fig.add_trace( go.Scatter(x=df.index, y=df['upper_keltner'], name='Upper Keltner Channel', line={'color': 'blue'}), row=1, col=1)
        fig.add_trace( go.Scatter(x=df.index, y=df['lower_keltner'], name='Lower Keltner Channel', line={'color': 'blue'}), row=1, col=1)
        #fig.add_trace( go.Bar(name='Momentum', x=df.index, y=df['mom_hist'] ), row=2, col=1)
        fig.add_trace( go.Candlestick(x=df.index, open=df[self.open], high=df[self.high], low=df[self.low], close=df[self.close]), row=2, col=1)
        fig.add_trace( go.Scatter(x = df.index, y = df['shaded'],
                fill = 'tonexty', fillcolor = 'rgba(255, 0, 0, 0.2)',
                mode = 'lines', line = {'width': 0, 'shape': 'hvh'},
                showlegend = False,), row=2, col=1
                )
        fig.layout.xaxis.type = 'category'
        fig.layout.xaxis.rangeslider.visible = False
        fig.update_layout(height=800, width=1500, title_text="Squeeze Scan for {}".format(symbol))
        fig.show()

class TestProcessor(BaseProcessor):
    '''
    To test out new signal generation
    '''
    def __init__(self, data_folder='./data', lookback_period=5, 
            lookforward_period=5, big_move_percentage=10, **kwargs):
        if (not os.path.exists(data_folder)):
            os.mkdir(data_folder)
        self.data_folder = data_folder
        self.lookback_period = lookback_period
        self.lookforward_period = lookforward_period
        self.big_move_perc = big_move_percentage
        # self.indicators = super().get_indicators # .update({ })
        super().__init__(**kwargs)
    '''
    def get_indicators(self, indicator: str) -> set(str):
        return self.indicators
    '''
    def process(self, symbol, df) ->pd.DataFrame:
        self.symbol=symbol
        if df is None:
            return None
        if df.empty:
            return None
        
        df['close_pct_change'] = df[self.close].pct_change(fill_method='ffill')
        df['growth'] = df['close_pct_change'].cumsum()
        df = self.in_consolidation(df, self.lookback_period)
        df = self.near_future_growth(df)
        df = self.near_past_growth(df)
        df = self.mark_big_moves(df, self.big_move_perc)
        return df
    
    def in_consolidation(self, df, percentage=2, window=15):
        max_prior = "max_prior_{}".format(window)
        min_prior = "min_prior_{}".format(window)
        df[max_prior] = df[self.close].rolling(window).max()
        df[min_prior] = df[self.close].rolling(window).min()

        threshold = 1 - (percentage / 100)
        df['in_consolidation']  = df[min_prior] > (df[max_prior] * threshold)
        return df 

    def near_future_growth(self, df, window=7):
        df['near_future_growth'] = df['close_pct_change'].shift(window).rolling(window).sum()
        return df
    
    def near_past_growth(self, df, window=7):
        df['near_past_growth'] = df['close_pct_change'].shift(-1).rolling(window).sum()
        return df

    def mark_big_moves(self, df, percentage=10):
        threshold = percentage / 100
        df['big_move'] =  abs(df['near_future_growth'] - df['near_past_growth']) > threshold
        return df

    def chart(self, symbol, df):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace( go.Candlestick(x=df.index, open=df[self.open], high=df[self.high], low=df[self.low], close=df[self.close]), row=1, col=1)
        fig.add_trace( go.Scatter(
            x=df.index, y=df['near_past_growth']*100, name='Past Growth', line={'color': 'blue'}
            ), row=2, col=1)
        fig.add_trace( go.Scatter(x=df.index, y=df['near_future_growth']*100, name='Future Growth', line={'color': 'red'}), row=2, col=1)
        fig.add_trace( go.Scatter(name='Big Moves', x=df.index, y=df['big_move']*100 ), row=2, col=1)
        fig.layout.xaxis.type = 'category'
        fig.layout.xaxis.rangeslider.visible = False
        fig.update_layout(height=800, width=1500, title_text="Squeeze Scan for {}".format(symbol))
        fig.show()