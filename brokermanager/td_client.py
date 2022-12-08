from tda import auth, client
import json
import datetime
from datetime import timedelta
import pandas as pd 
import os
import sys
import atexit 

class Client:
    def __init__(self, config, manual=False):
        self.config = config
        self.manual = manual
        self.TOKEN_PATH = config['BROKER']['TD']['TOKEN_PATH']
        self.API_KEY = config['BROKER']['TD']['API_KEY']
        self.REDIRECT_URL = config['BROKER']['TD']['REDIRECT_URL']
        self.td = self.authenticate_tda()
        
    def _make_webdriver(self):
        # Import selenium here because it's slow to import
        from selenium import webdriver

        driver = webdriver.Chrome()
        atexit.register(lambda: driver.quit())
        return driver
    
    # authenticate to TD Ameritrade API
    def authenticate_tda(self):
        return auth.easy_client(
                    self.API_KEY,
                    self.REDIRECT_URL,
                    self.TOKEN_PATH,
                    self._make_webdriver)

    def get_freq_type_str(self, frequency_type):
        if frequency_type == client.Client.PriceHistory.FrequencyType.DAILY:
            return 'daily'
        if frequency_type == client.Client.PriceHistory.FrequencyType.MINUTE:
            return 'minute'
        if frequency_type == client.Client.PriceHistory.FrequencyType.WEEKLY:
            return 'weekly'
        if frequency_type == client.Client.PriceHistory.FrequencyType.MONTHLY:
            return 'monthly'
        return 'other'
    
    def get_year_period(self, years):
        if years == 1:
            return client.Client.PriceHistory.Period.ONE_YEAR
        elif years == 2:
            return client.Client.PriceHistory.Period.TWO_YEARS
        elif years == 3:
            return client.Client.PriceHistory.Period.THREE_YEARS
        elif years == 5:
            return client.Client.PriceHistory.Period.FIFTEEN_YEARS
        elif years == 10:
            return client.Client.PriceHistory.Period.TEN_YEARS
        elif years == 15:
            return client.Client.PriceHistory.Period.FIFTEEN_YEARS
        elif years == 20:
            return client.Client.PriceHistory.Period.TWO_YEARS
        else:
            print("year must be an integer in (1, 2, 3, 5, 10, 15, 20")
    def convert_symbol_price_hx_todataframe(self, res):
        if res['empty'] == True:
            print("****Empty result****")
            return pd.DataFrame()
        
        rows_list = []
        #df = pd.DataFrame(columns = ['datetime', 'open', 'high', 'low', 'close','volume']) 
        for row in res['candles']:
            rows_list.append(row)
        df = pd.DataFrame(rows_list)[['datetime', 'open', 'high', 'low', 'close','volume']]
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
        df.set_index('datetime', inplace=True, drop=True)
        return df

    def get_price_history(self, **kwargs):
        # call the TD API and grab the json
        r = self.td.get_price_history(**kwargs)
        return self._process_api_result(r)

    def _process_api_result(self, r):
        j = r.json()
        # Make sure there are candles to know that it is not an error
        if 'candles' not in j:
            print("**** No Candles in Result****")
            print(r.json())
            return pd.DataFrame()
        # If there are candles, make sure the result is not empty
        if j['empty'] == True:
            print("****Empty result****")
            return pd.DataFrame()
        # get the dataframe from the candles and save the dataframe as a CSV file
        df = self.convert_symbol_price_hx_todataframe(j)
        return df 
    
    def get_fundamentals(self, symbol):
        r = self.td.search_instruments(symbol, client.Client.Instrument.Projection.FUNDAMENTAL)
        j = r.json()
        try:
            fund = j[symbol]['fundamental']
            fund['datetime'] = datetime.datetime.now()
        except:
            try:
                # We have have sent too many API calls, stop the program
                if j['error'] == "Individual App\'s transactions per seconds restriction reached. Please contact us with further questions":
                    sys.exit(1)
            except:
                # Must have been another error, print the JSON and then set fund to None
                print("ERROR=============")
                print(j) 
                print("==================")
                fund = None
        return fund
    '''
    Function to get a standard intraday candles for every 5 minutes for the supplied period. Period type 
    is `DAY`
    '''
    def get_price_intraday_history(self, symbol, period=client.Client.PriceHistory.Period.THREE_MONTHS):
        
        period_type=client.Client.PriceHistory.PeriodType.DAY
        frequency_type=client.Client.PriceHistory.FrequencyType.MINUTE
        frequency=client.Client.PriceHistory.Frequency.EVERY_FIVE_MINUTES
        return self.get_price_history(symbol=symbol,period_type=period_type,period=period,frequency_type=frequency_type,frequency=frequency)
    '''
    Function to get a standand daily candles for the supplied period. Period type 
    is `YEAR`
    '''
    def get_price_daily_history(self, symbol, period=client.Client.PriceHistory.Period.TWENTY_YEARS):
        period_type=client.Client.PriceHistory.PeriodType.YEAR
        frequency_type=client.Client.PriceHistory.FrequencyType.DAILY
        frequency=client.Client.PriceHistory.Frequency.DAILY
        return self.get_price_history(symbol=symbol,period_type=period_type,period=period,frequency_type=frequency_type,frequency=frequency)
    def get_standard_5min_price_history(self, start_datetime, symbol):
        frequency_type=client.Client.PriceHistory.FrequencyType.MINUTE
        frequency=client.Client.PriceHistory.Frequency.EVERY_FIVE_MINUTES
        
        end_datetime = datetime.datetime.now()
        return self.get_price_history(symbol=symbol,end_datetime=end_datetime, start_datetime=start_datetime,frequency_type=frequency_type,frequency=frequency)
    
    '''
    Function to get a standand daily candles for the supplied period. Period type 
    is `YEAR`
    '''
    def get_price_daily_history_by_range(self, symbol, startDate:datetime, endDate:datetime):
        r = self.td.get_price_history_every_day(symbol=symbol, start_datetime=startDate, end_datetime=endDate)
        return self._process_api_result(r)
   

    def _unix_time_millis(self, dt):
        return (dt - datetime.datetime.utcfromtimestamp(0)).total_seconds() * 1000.0