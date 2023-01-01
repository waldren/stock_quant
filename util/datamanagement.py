import pickle
import pandas as pd

def get_stock_history(symbol, data_dir)->pd.DataFrame:

    filename = f'{symbol}.pickle'  #utils.get_random_file(dir)
 
    with open(f"{data_dir}/{filename}", 'rb') as handle:
        history = pickle.load(handle)
    return history