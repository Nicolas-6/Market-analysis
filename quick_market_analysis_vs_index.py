# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 18:53:25 2022

@author: 06nic
"""

import yfinance as yf
import matplotlib.pyplot  as plt
import pandas as pd
import numpy as np

import datetime as dt
ticker = str(input('Enter yahoo ticker of your security eg : AAPL: \n'))
ticker_index = str(input('Enter yahoo ticker of your index eg : ^FCHI: \n'))
start_date = "2015-12-31"
end_date = str(dt.date.today()-dt.timedelta(1))

globals()[f"Data_{ticker}"]=yf.download(ticker,start_date, end_date)["Adj Close"]
globals()[f"Data_{ticker}_return"]= globals()[f"Data_{ticker}"].pct_change()
globals()[f"Data_{ticker}_std"] = (globals()[f"Data_{ticker}_return"]).rolling(window=252).std()

globals()[f"Data_{ticker}_rolling_mean"] = (globals()[f"Data_{ticker}_return"]).rolling(window=252).mean()

globals()[f"Data_{ticker}_moving_price_max"] = (globals()[f"Data_{ticker}"]).rolling(window=100).max()
globals()[f"Data_{ticker}_moving_price_min"] = (globals()[f"Data_{ticker}"]).rolling(window=100).min()

globals()[f"Data_{ticker_index}"] = yf.download(ticker_index,start_date, end_date)["Adj Close"]
globals()[f"Data_{ticker_index}_return"]= globals()[f"Data_{ticker_index}"].pct_change()

globals()[f"Data_{ticker_index}_std"] = (globals()[f"Data_{ticker_index}_return"]).rolling(window=252).std()

globals()[f"Data_{ticker_index}_rolling_mean"] = (globals()[f"Data_{ticker_index}_return"]).rolling(window=252).mean()





Data_merge = pd.merge(globals()[f"Data_{ticker}_return"],globals()[f"Data_{ticker_index}_return"], on="Date" )
Data_merge_cov = Data_merge["Adj Close_x"].rolling(252).cov(Data_merge["Adj Close_y"])
 
globals()[f"Data_beta_{ticker}_vs_{ticker_index}"] = Data_merge_cov/(globals()[f"Data_{ticker_index}_std"]**2)

plt.plot(globals()[f"Data_{ticker}"])
plt.plot(globals()[f"Data_{ticker}_moving_price_max"])
plt.plot(globals()[f"Data_{ticker}_moving_price_min"])
plt.legend([f"Data_{ticker}",f"Data_{ticker}_moving_price_max",f"Data_{ticker}_moving_price_min"])
plt.savefig(f'C:/Users/06nic/Desktop/XTB/output/Price_max_min_.png')
plt.clf () 


plt.plot(globals()[f"Data_{ticker_index}_return"])
plt.plot(globals()[f"Data_{ticker}_return"])
plt.legend([f"Data_{ticker_index}_return",f"Data_{ticker}_return"])
plt.savefig(f'C:/Users/06nic/Desktop/XTB/output/perf_{ticker}_vs_{ticker_index}.png')
plt.clf () 

plt.plot(globals()[f"Data_{ticker_index}_rolling_mean"])
plt.plot(globals()[f"Data_{ticker}_rolling_mean"])
plt.legend([f"Data_{ticker_index}_rolling_mean",f"Data_{ticker}_rolling_mean"])
plt.savefig(f'C:/Users/06nic/Desktop/XTB/output/perf_rolling sur 252 jours_{ticker}_vs_{ticker_index}.png')
plt.clf () 

plt.plot(globals()[f"Data_{ticker_index}_std"])
plt.plot(globals()[f"Data_{ticker}_std"])
plt.legend([f"Data_{ticker_index}_std",f"Data_{ticker}_std"])
plt.savefig(f'C:/Users/06nic/Desktop/XTB/output/standard_deviaion_rolling sur 252 jours_{ticker}_vs_{ticker_index}.png')
plt.clf () 


plt.plot(globals()[f"Data_beta_{ticker}_vs_{ticker_index}"])
plt.legend([f"Data_beta_{ticker}_vs_{ticker_index}"])
plt.savefig(f'C:/Users/06nic/Desktop/XTB/output/Beta_rolling sur 252 jours_{ticker}_vs_{ticker_index}.png')
plt.clf () 