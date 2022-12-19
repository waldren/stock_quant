# From Danny Groves - https://wire.insiderfinance.io/algorithmically-drawing-stock-market-trend-lines-ecb5ef6d112a

import numpy as np
import yfinance as yf
from typing import Tuple
from scipy.optimize import minimize, LinearConstraint

# Imports for plotting the result
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'


def heat_eqn_smooth(prices: np.array,
                    t_end: float = 3.0) -> np.array:
    '''
    Smoothen out a time series using a simple explicit finite difference method.
    The scheme uses a first-order method in time, and a second-order centred
    difference approximation in space. The scheme is only numerically stable
    if the time-step 0<=k<=1.

    Parameters
    ----------
    prices : np.array
        The price to smoothen
    t_end : float
        The time at which to terminate the smootheing (i.e. t = 3)
        
    Returns
    -------
    P : np.array
        The smoothened time-series
    '''
    
    k = 0.1 # Time spacing
    
    P = prices # Set up the initial condition
    
    t = 0
    while t < t_end:
        
        # Scheme on the interior nodes
        P[1:-1] = k*(P[2:] + P[:-2]) + (1-2*k)*P[1:-1]
        
        # Implementing the boundary conditions
        P[0] = 2*k*P[1] + (1-2*k)*P[0]
        P[-1] = prices[-1]

        t += k

    return P


def find_grad_intercept(case: str,
                        x: np.array, 
                        y: np.array) -> Tuple[float, float]:
    '''
    Get the gradient and intercept parameters for the support/resistance line

    Parameters
    ----------
    case : str
        Either 'support' or 'resistance'
    x : np.array
        The day number for each price in y
    y : np.array
        The stock prices

    Returns
    -------
    Tuple[float, float]
        The gradient and intercept parameters
    '''
    
    pos = np.argmax(y) if case == 'resistance' else np.argmin(y)
        
    # Form the points for the objective function
    X = x-x[pos]
    Y = y-y[pos]
    
    if case == 'resistance':
        const = LinearConstraint(
            X.reshape(-1, 1),
            Y,
            np.full(X.shape, np.inf),
        )
    else:
        const = LinearConstraint(
            X.reshape(-1, 1),
            np.full(X.shape, -np.inf),
            Y,
        )
    
    # Min the objective function with a zero starting point for the gradient
    ans = minimize(
        fun = lambda m: np.sum((m*X-Y)**2),
        x0 = [0],
        jac = lambda m: np.sum(2*X*(m*X-Y)),
        method = 'SLSQP',
        constraints = (const),
    )
    
    # Return the gradient (m) and the intercept (c)
    return ans.x[0], y[pos]-ans.x[0]*x[pos] 


if __name__ == '__main__':

    df = yf.download('VLO').reset_index()
    
    # Filter the dates to plot only the region of interest
    df = df[(df['Date'] > '2022-03-15') & (df['Date'] < '2022-06-30')]
    df = df.reset_index(drop=True)

    # Get another df of the dates where we draw the support/resistance lines
    df_trend = df[(df['Date'] > '2022-05-03') & (df['Date'] < '2022-05-25')]

    # Apply the smoothing algorithm and get the gradient/intercept terms
    m_res, c_res = find_grad_intercept(
        case = 'resistance', 
        x = df_trend.index.values, 
        y = heat_eqn_smooth(df_trend['High'].values.copy()),
    )
    m_supp, c_supp = find_grad_intercept(
        case = 'support', 
        x = df_trend.index.values, 
        y = heat_eqn_smooth(df_trend['Low'].values.copy()),
    )
    
    # Get the plotly figure
    layout = go.Layout(
        title = 'VLO Stock Price Chart',
        xaxis = {'title': 'Date'},
        yaxis = {'title': 'Price'},
        legend = {'x': 0, 'y': 1.075, 'orientation': 'h'},
        width = 700,
        height = 600,
    ) 

    fig = go.Figure(
        layout=layout,
        data=[
            go.Candlestick(
                x = df['Date'],
                open = df['Open'], 
                high = df['High'],
                low = df['Low'],
                close = df['Close'],
                showlegend = False,
            ),
            go.Line(
                x = df_trend['Date'], 
                y = m_res*df_trend.index + c_res, 
                showlegend = False, 
                line = {'color': 'rgba(89, 105, 208, 1)'}, 
                mode = 'lines',
            ),
            go.Line(
                x = df_trend['Date'], 
                y = m_supp*df_trend.index + c_supp, 
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