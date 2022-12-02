import pandas as pd
import time
import yaml
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

if __name__ == '__main__':
    df = pd.read_pickle("./data/stock_history/AAPL.pickle")
    test_td()