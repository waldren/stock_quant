#!F:\gitrepo\stock_quant\.stockquant\Scripts\python.exe
from tda.scripts.orders_codegen import latest_order_main

if __name__ == '__main__':
    import sys
    sys.exit(latest_order_main(sys.argv[1:]))
