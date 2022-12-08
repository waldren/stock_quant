import pandas as pd
import numpy as np
import time
import yaml
import datetime
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


if __name__ == '__main__':
    df = pd.DataFrame([1,2,3,4,5,6,7,8,9,10,11,12], columns=['TEST'])
    print(f'Initial length is {len(df)}')
    df = test_chunk_dataframe(df)
    print(f'Post length is {len(df)}')
    print(df)