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
import finplot as fplt

#------------------------------------Import Libraries------------------------------------------------------------------------------

#------------------------------------stock data------------------------------------------------------------------------------------

stocks = pd.DataFrame(pd.read_csv('tickers.csv'))
symbols = stocks['Symbol']

class Data:
    
    time_period = '1y' # you can chose the time period from this list : ['1d','5d','1mo', '3mo', '6mo', '1y', '2y', '5y', '10y']
    interval = '1d' #you can chose the time intervals from this list : ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo']
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
        
        self.ticker = yf.Ticker(Data.market_symbol)
        self.market_info = self.ticker.info
        self.market_history = self.ticker.history(period=Data.time_period, interval=Data.interval,actions=True)
        self.financials = self.ticker.financials
        self.dividends = self.ticker.dividends
        self.recommendations = self.ticker.recommendations

#------------------------------------------FEATURES--------------------------------------------------------------------------------------

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
    
#-------------------------------------PLOTTING----------------------------------------------------------------------------------------------

class Plot :
    
    def __init__(self,market_history):
        
        self.market_history = market_history
        
    
    def linear_plot(self):
        
        plt.figure(figsize=(30, 20))
        plt.plot(self.market_history.index, self.market_history['Close'], color='blue', linewidth=2)  # Set line color and width
        plt.gca().xaxis.set_major_locator(Data.locator_dic[Data.time_period][0])
        plt.gca().xaxis.set_major_formatter(Data.locator_dic[Data.time_period][1])
        plt.ylabel('Close Price', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        plt.title('Price Chart', fontsize=18)
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        
    def candlestick_plot(self):
        
        mc = mpf.make_marketcolors(up='g', down='r', wick='i', edge='i', volume='in')
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', y_on_right=False, facecolor='white')


        addplot = [mpf.make_addplot(self.market_history['Close'].rolling(window=ma).mean(), label=f'{ma}-day MA') for ma in Data.moving_average]


        mpf.plot(self.market_history, type='candle', volume=Data.volume, addplot=addplot,
                show_nontrading=True, title='Candle Stick Chart', ylabel='Price', xlabel='Date',
                style=s, figsize=(40, 30), tight_layout=True)

        plt.legend(loc='best')




    def macd_plot(self):
        mc = mpf.make_marketcolors(up='g', down='r', wick='i', edge='i', volume='in')
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', y_on_right=False, facecolor='white')    
        
        colors = ['green' if val > 0 else 'red' for val in self.market_history['MACD_Diff']]
        
        addplot = [
            mpf.make_addplot(self.market_history['MACD'], panel=1, color='blue', ylabel='MACD', label='MACD'),
            mpf.make_addplot(self.market_history['MACD_Signal'], panel=1, color='red', label='MACD Signal'),
            mpf.make_addplot(self.market_history['MACD_Diff'], panel=1, type='bar', color=colors, alpha=0.5, label='MACD Histogram')
        ]
        
        fig, axlist = mpf.plot(self.market_history, type='candle', style=s, addplot=addplot, title='Candlestick Chart with MACD', ylabel='Price', returnfig=True)
        
        # Add legends
        for ax in axlist:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles=handles, labels=labels, loc='best')
                
        plt.show()        
        

        


    def rsi_plot(self):

        mc = mpf.make_marketcolors(up='g', down='r', wick='i', edge='i', volume='in')
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', y_on_right=False, facecolor='white')
        
        addplot = [mpf.make_addplot(self.market_history['RSI'], panel=1, color='blue', ylabel='RSI'),
                   mpf.make_addplot([70]*len(self.market_history), panel=1, color = 'red'),
                   mpf.make_addplot([30]*len(self.market_history), panel=1, color= 'red')]
        
        mpf.plot(self.market_history, type='candle',
                show_nontrading=True, title='Candle Stick Chart with RSI', ylabel='Price', xlabel='Date',
                style=s, figsize=(40, 30), tight_layout=True,addplot=addplot)      
        
        plt.legend(['RSI'],loc='best')
        
    def bollinger_plot(self):
        
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
        
    def obv_plot(self):
        
        mc = mpf.make_marketcolors(up='g', down='r', wick='i', edge='i', volume='in')
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', y_on_right=False, facecolor='white')


        addplot = [mpf.make_addplot(self.market_history['OBV'], panel=1, color='blue', ylabel='OBV')]

        mpf.plot(self.market_history, type='candle',
                show_nontrading=True, title='Candle Stick Chart with OBV', ylabel='Price', xlabel='Date',
                style=s, figsize=(40, 30), tight_layout=True,addplot=addplot)  


        plt.legend(['OBV'], loc='best')
        
    def stochastic_plot(self):

        mc = mpf.make_marketcolors(up='g', down='r', wick='i', edge='i', volume='in')
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', y_on_right=False, facecolor='white')

        addplot = [
                mpf.make_addplot(self.market_history['%K'], panel=1, color='blue', ylabel='Stochastic', label='%K'),
                mpf.make_addplot(self.market_history['%D'], panel=1, color='red', label='%D')
            ]

        fig, axes = mpf.plot(self.market_history, type='candle', style=s, addplot=addplot, 
                                title='Candle Stick Chart with Stochastic', ylabel='Price', xlabel='Date', 
                                figsize=(40, 30), tight_layout=True, returnfig=True)

        axes[2].legend(loc='best')
        plt.show()

    def ATR_plot(self):
        
        mc = mpf.make_marketcolors(up='g', down='r', wick='i', edge='i', volume='in')
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', y_on_right=False, facecolor='white')

        addplot = [mpf.make_addplot(self.market_history['ATR'], panel=1, color='blue', ylabel='ATR')]

        mpf.plot(self.market_history, type='candle',
                show_nontrading=True, title='Candle Stick Chart', ylabel='Price', xlabel='Date',
                style=s, figsize=(40, 30), tight_layout=True,addplot=addplot)  
        
        
    def fibonaci_plot(self):
        
        mc = mpf.make_marketcolors(up='g', down='r', wick='i', edge='i', volume='in')
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', y_on_right=False, facecolor='white')
    
        
        fib_levels = [
            (self.market_history['Fibonacci_23.6'].iloc[0], 'Fibonacci 23.6%','blue'),
            (self.market_history['Fibonacci_38.2'].iloc[0], 'Fibonacci 38.2%','red'),
            (self.market_history['Fibonacci_50.0'].iloc[0], 'Fibonacci 50.0%','yellow'),
            (self.market_history['Fibonacci_61.8'].iloc[0], 'Fibonacci 61.8%','black'),
            (self.market_history['Fibonacci_100'].iloc[0], 'Fibonacci 100%','green')
        ]

        ap = [mpf.make_addplot([fib[0]] * len(self.market_history), color=fib[2], label=fib[1]) for fib in fib_levels]
        mpf.plot(self.market_history, type='candle',addplot=ap,style=s, figsize=(40, 30), tight_layout=True, title='Candlestick Chart with Fibonacci Levels',show_nontrading=True, ylabel='Price', volume=True)
        plt.legend(loc='best')
        
data = Data()
data.extract_information()

features = Thechnical_features(data.market_history)
features.MACD()
features.RSI()
features.bollinger()
features.EMA(12)
features.OBV()
features.stochastic()
features.ATR()
features.fibonacci()
print(data.market_history.columns)

#-------------------------------------------------------------------------------------------------------------------

plotting = Plot(data.market_history)
plotting.linear_plot()     
plotting.candlestick_plot() 
plotting.macd_plot()
plotting.rsi_plot()
plotting.bollinger_plot()
plotting.obv_plot()
plotting.stochastic_plot()
plotting.ATR_plot()
plotting.fibonaci_plot()