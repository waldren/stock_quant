# %%
from groves import trend_lines as tl
import pickle
import pandas as pd
import numpy as np
# Imports for plotting the result
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

#%%
def getstock(symbol)->pd.DataFrame:
    dir = './data/stock_history'

    filename = f'{symbol}.pickle'  #utils.get_random_file(dir)
 
    with open(f"{dir}/{filename}", 'rb') as handle:
        history = pickle.load(handle)
    return history

def gen_x(df):
    return np.arange(len(df))


#%%
from stockprocessing import analyers as az
df = az.append_moving_trend_lines(df, window=7)









#%%
df = getstock('VLO')
df = df['2022-03-15':'2022-06-30']

# Get another df of the dates where we draw the support/resistance lines
df_trend = df['2022-05-03':'2022-05-25']

# Apply the smoothing algorithm and get the gradient/intercept terms
m_res, c_res = tl.find_grad_intercept(
    case = 'resistance', 
    x = gen_x(df_trend), 
    y = tl.heat_eqn_smooth(df_trend['high'].values.copy()),
)
m_supp, c_supp = tl.find_grad_intercept(
    case = 'support', 
    x = gen_x(df_trend), 
    y = tl.heat_eqn_smooth(df_trend['low'].values.copy()),
)

#%%
# Get the plotly figure
layout = go.Layout(
    title = 'VLO Stock Price Chart',
    xaxis = {'title': 'Date'},
    yaxis = {'title': 'Price'},
    legend = {'x': 0, 'y': 1.075, 'orientation': 'h'},
    width = 900,
    height = 700,
) 

fig = go.Figure(
    layout=layout,
    data=[
        go.Candlestick(
            x = df.index,
            open = df['open'], 
            high = df['high'],
            low = df['low'],
            close = df['close'],
            showlegend = False,
        ),
        go.Line(
            x = df_trend.index, 
            y = m_res*gen_x(df_trend) + c_res, 
            showlegend = False, 
            line = {'color': 'rgba(89, 105, 208, 1)'}, 
            mode = 'lines',
        ),
        go.Line(
            x = df_trend.index, 
            y = m_supp*gen_x(df_trend) + c_supp, 
            showlegend = False, 
            line = {'color': 'rgba(89, 105, 208, 1)'}, 
            mode = 'lines',
        ),
    ]
)

fig.update_xaxes(
    rangeslider_visible = False,
    rangebreaks = [{'bounds': ['sat', 'mon']}]
)
fig.show()