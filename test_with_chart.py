# %% [markdown]
# ## Test with Charting ##

# %%
import numpy as np
import pandas as pd
import pickle

import datetime
from datetime import timedelta
import time 

from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# %%
def create_shaded_col(df, filter_col, high, low, filterValue=None)->np.array:
    if filterValue is None:
        return np.where(df[filter_col], high, low, )
    else:
        return np.where(df[filter_col] > filterValue, high, low, )

def test_chart(symbol:str, df:pd.DataFrame, row2_columns = None):

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace( go.Candlestick(x=df.index, name=symbol, open=df['open'], high=df['high'], 
                        low=df['low'], close=df['close']), row=1, col=1)

        if row2_columns is not None:
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

# %%
symbol = 'ETSY'
dir = './data/stock_history'

filename = f'{symbol}.pickle'  #utils.get_random_file(dir)

with open(f"{dir}/{filename}", 'rb') as handle:
    df = pickle.load(handle)

# %%
from stockprocessing import analyers as az
from stockprocessing import OPEN, HIGH, LOW, CLOSE, VOLUME
df['close_norm'] = az.normalize_data(df[CLOSE])

peaks = az.find_extrema(df)
peaks_vol = az.find_extrema(df, col_name=VOLUME, width=1, prominence=2)
df['peaks'] = peaks['peaks']
df['peaks_volume'] = peaks_vol['peaks']
df['volume_prom']= peaks_vol['prominence']

# %%
df.loc[:, 'shaded'] = create_shaded_col(df, 'peaks', df['high'].max(), df['low'].min())
df.loc[:, 'shaded2'] = create_shaded_col(df, 'peaks_volume', df['high'].max(), df['low'].min())

# %%
test_chart(symbol, df, row2_columns=['volume', 'volume_prom'])


