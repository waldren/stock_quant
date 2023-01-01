from util import datamanagement as dm
from stockprocessing import setupscanners as ss


def _test():
    bos = ss.BreakoutScanner()
    df = dm.get_stock_history('VLO', './data/stock_history')

    bos.scan(df)

if __name__ == '__main__':
    _test()
