import pandas as pd
import os
import pickle
import datetime
from datetime import timedelta
import time 
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from scipy.stats import linregress as lr
import logging 

from stockprocessing.setupprocessors import BreakoutProcessor
 
def create_shaded_col(df, filter_col, high, low)->np.array:
    return np.where(
            df[filter_col],
            high,
            low,
            )

def chart(symbol:str, df:pd.DataFrame):

        #df.loc[:, 'breakout_region'] = create_shaded_col(df, 'dg_filtered', df['high'].max(), df['low'].min())
    
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace( go.Candlestick(x=df.index, name=symbol, open=df['open'], high=df['high'], 
                        low=df['low'], close=df['close']), row=1, col=1)
              
        #fig.add_trace( go.Line(name="ADX_14", x=df.index, y=df['adx_14']), row=2, col=1)
        
        if 'breakout_region' in df.columns:
            fig.add_trace( go.Scatter(x = df.index, y = df['breakout_region'],
                fill = 'tonexty', fillcolor = 'rgba(0, 255, 0, 0.2)',
                mode = 'lines', line = {'width': 0, 'shape': 'hvh'},
                showlegend = False,), row=1, col=1
                )
        
        if 'shaded' in df.columns:
            fig.add_trace( go.Scatter(x = df.index, y = df['shaded'],
                fill = 'tonexty', fillcolor = 'rgba(255, 0, 0, 0.2)',
                mode = 'lines', line = {'width': 0.5, 'shape': 'hvh'},
                showlegend = False,), row=1, col=1
                )
        if 'shaded2' in df.columns:
            fig.add_trace( go.Scatter(x = df.index, y = df['shaded2'],
                fill = 'tonexty', fillcolor = 'rgba(0, 0, 255, 0.2)',
                mode = 'lines', line = {'width': 0.5, 'shape': 'hvh'},
                showlegend = False,), row=1, col=1
                )
        #fig.add_trace(go.Line(name="% Up Move", x=df.index, y=df['peaks_prct_move']), row=2, col=1)
        #fig.add_trace(go.Line(name="% Down Move", x=df.index, y=df['troughs_prct_move']), row=2, col=1)

        if True:
            # Show volume
            fig.add_trace(go.Bar(x = df.index, y= df['volume_pct_chg']), row=2, col=1)

        if True:
            # Mark the key peaks
            mark_peak_widths(fig, df)

        #fig.layout.xaxis.type = 'category'
        #fig.layout.xaxis.rangeslider.visible = False
        fig.update_layout(
            xaxis = {'title': 'Date'},
            yaxis = {'range': [df['low'].min(), df['high'].max()], 'title': 'Price ($)'},
            title = f'{symbol} - Breakout Candidates',
            width = 1500,
            height = 800
        )
        
        fig.update_xaxes(
            rangebreaks = [{'bounds': ['sat', 'mon']}],
            rangeslider_visible = False,
        )
        fig.show()

def mark_peak_widths(fig:Figure, df:pd.DataFrame)-> pd.DataFrame:
    lookback_period = 3
    df_peaks = df.query("peaks == 1 and peaks_prct_move > 1.1")
    for index, row in df_peaks.iterrows():
        pk_start = index- timedelta(row['peaks_fullwidth'])
        fig.add_vrect(x0=pk_start, x1=index+ timedelta(row['peaks_fullwidth']), line_width=1, fillcolor="green", opacity=0.1)
        
        step_size = lookback_period
        max_steps = 5

        consolidate_end = pk_start -timedelta(1)
        df_test = df.loc[:consolidate_end.strftime('%Y-%m-%d')]
        max_consolidation = find_consolidation(df_test['high'].to_numpy(), df_test['low'].to_numpy(), step_size=step_size, max_steps=max_steps)
        if max_consolidation > 0:
            fig.add_vrect(x0=consolidate_end-timedelta(max_consolidation), x1=consolidate_end, 
                        line_width=1, fillcolor="yellow", opacity=0.2 )
        
        # Check for Volume Spike
        df_test = df.loc[pk_start:index]
        vol_spike = where_volume_spike(df_test['volume_pct_chg'].to_numpy())
        if vol_spike >= 0:
            vol_idx = df.index[vol_spike]
            fig.add_vrect(x0=vol_idx, x1=vol_idx, line_width=1, fillcolor="blue", opacity=0.2 )

def fit_to_line(y:np.array):
    x = np.arange(1,len(y)+1)
    return lr(x, y=y)

def has_lower_highs_and_higher_lows(high:np.array, low:np.array):
    hi_res = fit_to_line(high)
    logger.debug(f"High Slope is {hi_res.slope}")
    if hi_res.slope < 0:
        lo_res = fit_to_line(low)
        logger.debug(f"Low Slope is {lo_res.slope}")
        if lo_res.slope > 0:
            return True
        else:
            return False
    else:
        return False

def find_consolidation(high:np.array, low:np.array, step_size=5, max_steps=5)-> int:
    """
    Check for consolidation via lower highs and higher lows in steps backward.

    Args:
        high (np.array): array of high prices
        low (np.array):  array of low prices
        step_size (int, optional): Number of trading days to look back per step. Defaults to 5.
        max_steps (int, optional): Maximum number of steps to take back. Defaults to 5.

    Returns:
        int: Number of trading days (in multiple of step_size) that there is consolidation. 0 = no consolidation
    """
    current_end = -1
    max_consolidation = 0
    stop_after = -step_size*max_steps
    for x in range (1,max_steps):
        current_start = current_end - step_size
        if current_start < stop_after:
            logger.debug(f'{max_consolidation} bars of consolidation in {x} steps')
            return max_consolidation
        #Check current range for consolidation
        if has_lower_highs_and_higher_lows(high=high[current_start:current_end], low=low[current_start:current_end]):
            max_consolidation = -current_start
            current_end = current_start
        else:
            logger.debug(f'{max_consolidation} bars of consolidation in {x} steps')
            return max_consolidation
    logger.debug(f'{max_consolidation} bars of consolidation in {x} steps')
    return max_consolidation


def where_volume_spike(volume_prct_change:np.array)-> int:
    for i, v in enumerate(volume_prct_change):
        if v > 0.3:
            return i
    return -1
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(filename='./app.log', filemode='w', 
                    format='%(name)s - %(levelname)s - %(message)s')
    logger.info(f'Run time is {datetime.datetime.now()}')

    symbol = 'ETSY'
    logger.info(f'Analyzing ticker: {symbol}')
    dir = './data/stock_history'

    filename = f'{symbol}.pickle'  #utils.get_random_file(dir)

    history = None 
    with open(f"{dir}/{filename}", 'rb') as handle:
        history = pickle.load(handle)

    bp = BreakoutProcessor()
    stime = time.perf_counter()
    history = bp.process(symbol, history)
    etime = time.perf_counter()
    logger.info(f'Breakoutprocessor took {etime-stime} secs.')

    # Create a new columen for ADX below 14
    #history['adx_below'] = np.zeros(len(history))
    #history.loc[history['adx_14'] < 14, 'adx_below'] = 1

    history.loc[:, 'shaded'] = create_shaded_col(history, 'peaks', history['high'].max(), history['low'].min())
    #history.loc[:, 'shaded2'] = create_shaded_col(history, 'troughs', history['high'].max(), history['low'].min())

    trunc_hx = history.loc['2018-01-01':'2021-12-31']
    #tmp = trunc_hx.select_dtypes(include=[np.number])
    #trunc_hx.loc[:, tmp.columns] = np.round(tmp, 2)

    #trunc_hx.to_csv('./tsla_2018-2021.csv')
    chart(symbol, trunc_hx)

