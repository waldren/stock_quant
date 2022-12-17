import pandas as pd
import numpy as np
import pickle
import time
import yaml
import datetime
from scipy.stats import linregress as lr

from stockprocessing.baseprocessors import SqueezeProcessor, TestProcessor

def testSqueeze(df):
    prc = SqueezeProcessor()
    st = time.perf_counter()
    df = prc.process("AAPL", df)
    et = time.perf_counter()
    print (f"Completed in {et-st}")
    #print(df.columns)
    df_recent = df.tail(365)
    prc.chart("AAPL", df_recent)

def test(df):
    prc = TestProcessor()
    st = time.perf_counter()
    df = prc.process("AAPL", df)
    et = time.perf_counter()
    print (f"Completed in {et-st}")
    #print(df.columns)
    df_recent = df.tail(365)
    prc.chart("AAPL", df_recent)

def get_config():
    config = None
    with open('./config.yaml', "r") as f:
        config = yaml.safe_load(f)
    return config

def test_td():
    from brokermanager.td_client import Client
    c = get_config()
    td = Client(c)
    df = td.get_price_daily_history('TSLA')
    print (df.head())

def test_date_range():
    from brokermanager.td_client import Client
    c = get_config()
    td = Client(c)
    sd = datetime.datetime(1971, 1, 1)
    ed = datetime.datetime(1985, 12, 31)
    df = td.get_price_daily_history_by_range('DE', sd, ed)
    print (df.head())

def test_chunk_dataframe(df):
    chunk_size = 7 # int(df.shape[0] / 4)
    frames = []
    for start in range(0, df.shape[0], chunk_size):
        df_subset = df.iloc[start:start + chunk_size]
        frames.append(process_data(df_subset, start))
    return pd.concat(frames)


def process_data(df, i)->pd.DataFrame:
    df['TEST'] = df['TEST'].apply(lambda x: x*-1)
    return df

def calc_slope(x):
    slope = np.polyfit(range(len(x)), x, 1)[0]
    return slope

def fit_to_line(y:np.array):
    x = np.arange(1,len(y)+1)
    return lr(x, y=y).slope

def test_rolling_slope(df):

    df['slope_5'] = df['close'].rolling(5,min_periods=5).apply(calc_slope,raw=False)        
    df['slope_20']= df['close'].rolling(20,min_periods=20).apply(calc_slope,raw=False)  
    return df

def getstock(symbol)->pd.DataFrame:
    dir = './data/stock_history'

    filename = f'{symbol}.pickle'  #utils.get_random_file(dir)
 
    with open(f"{dir}/{filename}", 'rb') as handle:
        history = pickle.load(handle)
    return history

def truncate(df:pd.DataFrame, sdate, edate, keep_cols)->pd.DataFrame:
    return df.loc[sdate:edate][keep_cols]

if __name__ == '__main__':
    df = getstock('ETSY')
    print(f'Initial length is {len(df)}')
    keep_cols = ['close', 'slope_5', 'slope_20']

    df = test_rolling_slope(df)
    df = truncate(df, '2018-01-01', '2021-12-31', keep_cols)
    df.to_csv('./test_rolling.csv')
