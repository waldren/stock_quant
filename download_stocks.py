import pandas as pd
import yaml
from brokermanager.td_client import Client 

def getStockList() -> list:
    # TODO change list to set
    txt_file = open("./data/russel2000.txt", "r")
    file_content = txt_file.read()
    content_list = file_content.split(",")
    txt_file.close()
    txt_file = open("./data/sp500.csv", "r")
    splist = txt_file.read().splitlines()
    for s in splist:
        content_list.append(s.split(",")[0])
    content_list.remove("Symbol")
    return content_list

dataFolder = './data/stock_history'
def save_history(s: str, df: pd.DataFrame):
    filename = f'{dataFolder}/{s}.pickle'
    df.to_pickle(filename)

def removed_saved(stocks: list) -> list:
    print(f'Full list of stocks contain {len(stocks)} stocks')
    
    saved_list = os.listdir(dataFolder)
    for f in saved_list:
        f = f.split('.')[0]
        if f in stocks:
            stocks.remove(f)
    print(f'{len(stocks)} remain to be saved')

    return stocks


def download_list():
    import time
    import os
    
    td = Client()
    
    stock_list = removed_saved(getStockList())
    
    
    i = 0
    for s in stock_list:
        i += 1
        if i%10 == 0:
            print("Sleeping for 10s...")
            time.sleep(10)
        if i%100 == 0:
            print("Sleeping for additional 50s...")
            time.sleep(50)
        print (f'Getting Hx for {s}')
        save_history(s, td.get_price_daily_history(s))

    print ('************')
    print (f'Processed {i} stocks')
    print ('************')

def download_single( td, symbol):
    print (f'Getting Hx for {symbol}')
    save_history(symbol, td.get_price_daily_history(symbol))

if __name__ == '__main__':
    config = None
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)

    td = Client(config)
    download_single(td, 'LAC')
