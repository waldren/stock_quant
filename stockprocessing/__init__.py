import yaml 

with open('./config.yaml') as f:
        CONFIG = yaml.safe_load(f)
        
OPEN = CONFIG['OHLCV']['OPEN']
HIGH = CONFIG['OHLCV']['HIGH']
LOW = CONFIG['OHLCV']['LOW']
CLOSE = CONFIG['OHLCV']['CLOSE']
VOLUME = CONFIG['OHLCV']['VOLUME']