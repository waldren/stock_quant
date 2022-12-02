import pandas as pd
import time
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

if __name__ == '__main__':
    df = pd.read_pickle("./data/stock_history/AAPL.pickle")
    test(df)