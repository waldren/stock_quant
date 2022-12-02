import pandas as pd
import os
import pickle
import datetime
from datetime import timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from stockprocessing.setupprocessors import BreakoutProcessor
 

def chart(symbol:str, df:pd.DataFrame):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace( go.Candlestick(x=df.index, name=symbol, open=df['open'], high=df['high'], 
                        low=df['low'], close=df['close']), row=1, col=1)
        fig.add_trace( go.Line(name="ADX_14", x=df.index, y=df['adx_14']), row=2, col=1)
        fig.add_trace(go.Scatter(name='Y=14', x = [df.index[0].value, df.index[-1].value], y = [14, 14],
                        mode = "lines", marker = dict(color = 'rgba(80, 26, 80, 1)')),row=2, col=1)
        fig.add_trace( go.Bar(name='consolidating', x=df.index, y=df['consolidating'] ), row=3, col=1)
        fig.add_trace( go.Bar(name='trend_filter', x=df.index, y=df['trend_filter'] ), row=3, col=1)
        #fig.layout.xaxis.type = 'category'
        fig.layout.xaxis.rangeslider.visible = False
        fig.show()

def getBreakoutSetupDates(df:pd.DataFrame):
    df_f = df.query("filtered == True")
    f1d = df_f.index[-1]
    f2d = df_f.index[-2]
    ftd = f1d - f2d
    sDate = f1d - ftd
    eDate = f1d
    return sDate, eDate

if __name__ == "__main__":
    dir = './data/stock_history'
    filename = 'TSLA.pickle'  #utils.get_random_file(dir)
    symbol = filename.split('.')[0]
    print("Showing file: {}".format(filename))
    with open(f"{dir}/{filename}", 'rb') as handle:
        history = pickle.load(handle)
    bp = BreakoutProcessor()
    history = bp.process(symbol, history)
    print("Data:")
    print(history.tail(10))
    chart(symbol, history)
    
