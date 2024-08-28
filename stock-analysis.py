#------------------------------------Import Libraries------------------------------------------------------------------------------ 

import streamlit as st
import pandas as pd
import numpy as np
import requests as re
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume
import yfinance as yf
import sklearn
import sqlite3
from pytickersymbols import PyTickerSymbols
from get_all_tickers import get_tickers as gt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
matplotlib.use('TkAgg')
import plotly.graph_objects as go
import mplfinance as mpf
import ta

#------------------------------------Import Libraries------------------------------------------------------------------------------

#------------------------------------stock data------------------------------------------------------------------------------------
stocks = pd.DataFrame(pd.read_csv('tickers.csv'))
symbols = stocks['Symbol']

class Market:
    
    time_period = '1y' # you can chose the time period from this list : ['1d','5d','1mo', '3mo', '6mo', '1y', '2y', '5y', '10y']
    interval = '1h' #you can chose the time intervals from this list : ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo']
    locator_dic = {'1d':[mdates.HourLocator(interval=1), mdates.DateFormatter('%H:%M')],
                   '5d':[mdates.DayLocator(),mdates.DateFormatter('%M/%d')],
                   '1mo':[mdates.WeekdayLocator(),mdates.DateFormatter('%m/%d')],
                   '3mo':[mdates.MonthLocator(),mdates.DateFormatter('%m/%d')],
                   '6mo':[mdates.MonthLocator(interval=1),mdates.DateFormatter('%m/%d')],
                   '1y':[mdates.MonthLocator(interval=6),mdates.DateFormatter('%m/%d')],
                   '2y':[mdates.YearLocator(),mdates.DateFormatter('%Y/%m/%d')],
                   '5y':[mdates.YearLocator(),mdates.DateFormatter('%Y/%m/%d')],
                   '10':[mdates.YearLocator(),mdates.DateFormatter('%Y/%m/%d')]
                    }   
                   
    market_symbol = 'ZVRA'
    moving_average = (20,50,100)
    volume = True
    
    def extract_information(self):
        
        self.ticker = yf.Ticker(Market.market_symbol)
        self.market_info = self.ticker.info
        self.market_history = self.ticker.history(period=Market.time_period, interval=Market.interval,actions=True)
        self.financials = self.ticker.financials
        self.dividends = self.ticker.dividends
        self.recommendations = self.ticker.recommendations
        
    def linear_historyprice(self):
        
        plt.figure(figsize=(30, 20))
        plt.plot(self.market_history.index, self.market_history['Close'], color='blue', linewidth=2)  # Set line color and width
        plt.gca().xaxis.set_major_locator(Market.locator_dic[Market.time_period][0])
        plt.gca().xaxis.set_major_formatter(Market.locator_dic[Market.time_period][1])
        plt.ylabel('Close Price', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        plt.title('Price Chart', fontsize=18)
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        
    def candlestick_historyprice(self):
        
        mc = mpf.make_marketcolors(up='g', down='r', wick='i', edge='i', volume='in')
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', y_on_right=False, facecolor='white')


        mpf.plot(self.market_history, type='candle', mav=Market.moving_average, volume=Market.volume,
                show_nontrading=True, title='Candle Stick Chart', ylabel='Price', xlabel='Date',
                style=s, figsize=(40, 30), tight_layout=True)


    def macd_historyprice(self):        
               
        macd_object = ta.trend.MACD(self.market_history['Close'])
        self.market_history.loc[:, 'MACD'] = macd_object.macd()
        self.market_history.loc[:, 'MACD_Signal'] = macd_object.macd_signal()
        self.market_history.loc[:, 'MACD_Diff'] = macd_object.macd_diff()

        plt.figure(figsize=(14, 10))
        plt.subplot(2, 1, 1)
        plt.plot(self.market_history['Close'], label='Close Price', color='purple')
        plt.title('Tesla Stock Price and MACD', fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(self.market_history['MACD'], label='MACD Line', color='blue')
        plt.plot(self.market_history['MACD_Signal'], label='Signal Line', color='red')
        plt.bar(self.market_history.index, self.market_history['MACD_Diff'], label='Histogram', color=(self.market_history['MACD_Diff'] > 0).map({True: 'green', False: 'red'}), alpha=0.5)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def rsi_historyprice(self):
        
        self.market_history['RSI'] = ta.momentum.RSIIndicator(self.market_history['Close']).rsi() 
        
        fig, ax1 = plt.subplots(figsize=(30, 15))       
        ax1.plot(self.market_history.index, self.market_history['Close'], label='Closing Price', color='b')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Closing Price', color='b')
        ax1.tick_params(axis='y', labelcolor='b')        
        ax2 = ax1.twinx()
        ax2.plot(self.market_history.index, self.market_history['RSI'], label='RSI', color='g')
        ax2.set_ylabel('RSI', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
        ax2.axhline(y=30, color='orange', linestyle='--', label='Oversold (30)')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')       
        plt.title('Relative Strength Index (RSI) and Closing Price')
        plt.xticks(rotation=90, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        
    def bollinger_historyprice(self):
        
        self.market_history['SMA'] = self.market_history['Close'].rolling(window=20).mean()
        self.market_history['SD'] = self.market_history['Close'].rolling(window=20).std()
        self.market_history['UB'] = self.market_history['SMA'] + (2 * self.market_history['SD'])
        self.market_history['LB'] = self.market_history['SMA'] - (2 * self.market_history['SD'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.market_history.index, y=self.market_history['Close'], mode='lines', name='Price', line=dict(color='black', width=1)))
        fig.add_trace(go.Scatter(x=self.market_history.index, y=self.market_history['UB'], mode='lines', name='Upper Bollinger Band', line=dict(color='red', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=self.market_history.index, y=self.market_history['LB'], fill='tonexty', mode='lines', name='Lower Bollinger Band', line=dict(color='green', width=1, dash='dash'), fillcolor='rgba(0, 255, 0, 0.1)'))
        fig.add_trace(go.Scatter(x=self.market_history.index, y=self.market_history['SMA'], mode='lines', name='Middle Bollinger Band', line=dict(color='blue', width=1, dash='dot')))
        
        fig.update_layout(
            title='Stock Price with Bollinger Bands',
            xaxis_title='Date',
            yaxis_title='Price',
            showlegend=True,
            template='plotly_white'
        )
        
        fig.show()
        
        
class Thechnical_features():
    
    def __init__(self,market_history):
        
        self.market_history = market_history
    
    def MACD(self):
        
        macd_object = ta.trend.MACD(self.market_history['Close'])
        self.market_history.loc[:, 'MACD'] = macd_object.macd()
        self.market_history.loc[:, 'MACD_Signal'] = macd_object.macd_signal()
        self.market_history.loc[:, 'MACD_Diff'] = macd_object.macd_diff()
    
    def RSI(self):
        
        self.market_history['RSI'] = ta.momentum.RSIIndicator(self.market_history['Close']).rsi() 
    
    def bollinger(self):
        
        self.market_history['SMA'] = self.market_history['Close'].rolling(window=20).mean()
        self.market_history['SD'] = self.market_history['Close'].rolling(window=20).std()
        self.market_history['UB'] = self.market_history['SMA'] + (2 * self.market_history['SD'])
        self.market_history['LB'] = self.market_history['SMA'] - (2 * self.market_history['SD'])
    
    def EMA(self,period):
            
        self.period = period
        self.ema = self.market_history['Close'].ewm(span = self.period).mean()
        self.market_history[f'ema{self.period}'] = self.ema
    
    def OBV(self):
        
        self.market_history['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=self.market_history['Close'],volume=self.market_history['Volume']).on_balance_volume()
    
    def stochastic(self):
        
        st = ta.momentum.StochasticOscillator(
            close=self.market_history['Close'],
            high=self.market_history['High'],
            low=self.market_history['Low'],
            window=12,
            smooth_window=3
        )
        self.market_history['%K'] = st.stoch()
        self.market_history['%D'] = st.stoch_signal()
    
    def ATR(self):
        
        self.market_history['ATR'] = ta.volatility.AverageTrueRange(
            high=self.market_history['High'],
            low=self.market_history['Low'],
            close=self.market_history['Close'],
            window=14).average_true_range()
    
    def fibonacci(self):
        
        high = self.market_history['High'].max()
        low = self.market_history['Low'].min()
        self.market_history['Fibonacci_23.6'] = low + 0.236 * (high - low)
        self.market_history['Fibonacci_38.2'] = low + 0.382 * (high - low)
        self.market_history['Fibonacci_50.0'] = low + 0.5 * (high - low)
        self.market_history['Fibonacci_61.8'] = low + 0.618 * (high - low)
        self.market_history['Fibonacci_100'] = high
            


ZVRA1 = Market()
ZVRA1.extract_information()
#ZVRA1.linear_historyprice()     
#ZVRA1.candlestick_historyprice()   
#ZVRA1.macd_historyprice()
#ZVRA1.rsi_historyprice()
#ZVRA1.bollinger_historyprice()
ZVRA2 = Thechnical_features(ZVRA1.market_history)
ZVRA2.EMA(12)
ZVRA2.OBV()
ZVRA2.stochastic()
ZVRA2.ATR()
ZVRA2.fibonacci()
print(ZVRA1.market_history.columns)