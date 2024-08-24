#------------------------------------Import Libraries------------------------------------------------------------------------------ 

import streamlit as st
import pandas as pd
import numpy as np
import requests as re
import yfinance as yf
import sklearn
import sqlite3
from pytickersymbols import PyTickerSymbols

#------------------------------------Import Libraries------------------------------------------------------------------------------

#------------------------------------stock data------------------------------------------------------------------------------------

stock_data = PyTickerSymbols()
indices = stock_data.get_all_indices()
countries = stock_data.get_all_countries()
industries = stock_data.get_all_industries()
stocks = stock_data.get_all_stocks()
print(stocks[0])
