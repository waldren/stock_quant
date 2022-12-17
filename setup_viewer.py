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
import logging 

from stockprocessing.setupprocessors import BreakoutProcessor
 
def create_shaded_col(df, filter_col, high, low, filterValue=None)->np.array:
    if filterValue is None:
        return np.where(df[filter_col], high, low, )
    else:
        return np.where(df[filter_col] > filterValue, high, low, )

def test_chart(symbol:str, df:pd.DataFrame, row2_columns = ['in_consolidation', 'consolidating']):

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace( go.Candlestick(x=df.index, name=symbol, open=df['open'], high=df['high'], 
                        low=df['low'], close=df['close']), row=1, col=1)

        for c in row2_columns:
            fig.add_trace( go.Line(name=c, x=df.index, y=df[c]), row=2, col=1)

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

        if False:
            # Show volume
            fig.add_trace(go.Bar(x = df.index, y= df['volume_pct_chg']), row=2, col=1)

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


def chart(symbol:str, df:pd.DataFrame):

        #df.loc[:, 'breakout_region'] = create_shaded_col(df, 'dg_filtered', df['high'].max(), df['low'].min())
    
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace( go.Candlestick(x=df.index, name=symbol, open=df['open'], high=df['high'], 
                        low=df['low'], close=df['close']), row=1, col=1)
              
        #fig.add_trace( go.Line(name="ADX_14", x=df.index, y=df['adx_14']), row=2, col=1)
        '''
        if 'breakout_region' in df.columns:
            fig.add_trace( go.Scatter(x = df.index, y = df['breakout_region'],
                fill = 'tonexty', fillcolor = 'rgba(0, 255, 0, 0.2)',
                mode = 'lines', line = {'width': 0, 'shape': 'hvh'},
                showlegend = False,), row=1, col=1
                )
        '''
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

        if False:
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

   

    test_chart(symbol, history, row2_columns=['slope_5', 'slope_20'])

