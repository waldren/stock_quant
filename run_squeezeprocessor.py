import pandas as pd
import os
import pickle
import numpy as np

from stockprocessing.baseprocessors import SqueezeProcessor



if __name__ == "__main__":
    dir = './data/stock_history'
    filename = 'TSLA.pickle'  #utils.get_random_file(dir)
    symbol = filename.split('.')[0]
    print("Showing file: {}".format(filename))

    history = None 
    with open(f"{dir}/{filename}", 'rb') as handle:
        history = pickle.load(handle)
    sp = SqueezeProcessor()
    history = sp.process(symbol, history)
    df = history.tail(365)
    sp.chart("TSLA", df)