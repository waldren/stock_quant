import pandas as pd
import os
import pickle
import datetime
from datetime import timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

from stockprocessing.setupprocessors import BreakoutProcessor
 
def create_shaded_col(df, filter_col, high, low)->np.array:
    return np.where(
            df[filter_col],
            high,
            low,
            )

def chart(symbol:str, df:pd.DataFrame):

        df.loc[:, 'breakout_region'] = create_shaded_col(df, 'filtered', df['high'].max(), df['low'].min())
    
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace( go.Candlestick(x=df.index, name=symbol, open=df['open'], high=df['high'], 
                        low=df['low'], close=df['close']), row=1, col=1)
        fig.add_trace( go.Line(name="ADX_14", x=df.index, y=df['adx_14']), row=2, col=1)
        fig.add_trace( go.Scatter(x = df.index, y = df['breakout_region'],
            fill = 'tonexty', fillcolor = 'rgba(0, 255, 0, 0.2)',
            mode = 'lines', line = {'width': 0, 'shape': 'hvh'},
            showlegend = False,), row=1, col=1
            )
        if 'shaded' in df.columns:
            fig.add_trace( go.Scatter(x = df.index, y = df['shaded'],
                fill = 'tonexty', fillcolor = 'rgba(255, 0, 0, 0.2)',
                mode = 'lines', line = {'width': 0, 'shape': 'hvh'},
                showlegend = False,), row=1, col=1
                )
        
        #fig.layout.xaxis.type = 'category'
        #fig.layout.xaxis.rangeslider.visible = False
        fig.update_layout(
            xaxis = {'title': 'Date'},
            yaxis = {'range': [df['low'].min(), df['high'].max()], 'title': 'Price ($)'},
            title = f'{symbol} - Breakout Candidates',
            width = 1500,
            height = 1200
        )
        
        fig.update_xaxes(
            rangebreaks = [{'bounds': ['sat', 'mon']}],
            rangeslider_visible = False,
        )
        fig.show()
'''
def getBreakoutSetupDates(df:pd.DataFrame):
    df_f = df.query("filtered == True")
    f1d = df_f.index[-1]
    f2d = df_f.index[-2]
    ftd = f1d - f2d
    sDate = f1d - ftd
    eDate = f1d
    return sDate, eDate
'''

if __name__ == "__main__":
    dir = './data/stock_history'
    filename = 'TSLA.pickle'  #utils.get_random_file(dir)
    symbol = filename.split('.')[0]
    print("Showing file: {}".format(filename))

    with open(f"{dir}/{filename}", 'rb') as handle:
        history = pickle.load(handle)

    bp = BreakoutProcessor()
    history = bp.process(symbol, history)

    # Create a new columen for ADX below 14
    history['adx_below'] = np.zeros(len(history))
    history.loc[history['adx_14'] < 14, 'adx_below'] = 1

    history.loc[:, 'shaded'] = create_shaded_col(history, 'adx_below', history['high'].max(), history['low'].min())

    print("Data:")
    print(history.tail(10))
    chart(symbol, history)
    
