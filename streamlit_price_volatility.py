# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 13:40:58 2022

@author: 06nic
"""

import streamlit as st
import requests
import datetime as dt
import locale
import pandas as pd
import yfinance as yf
import seaborn as se
import numpy as np
from PIL import Image
image = Image.open(r"C:\Users\06nic\Desktop\XTB\Python\image_logo\finance_st2.jpg")


#from streamlit_functions import *

def find_ticker(isin="ISIN"):
        url = 'https://query1.finance.yahoo.com/v1/finance/search'

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.109 Safari/537.36',
        }
    
        params = dict(
            q=isin,
            quotesCount=1,
            newsCount=0,
            listsCount=0,
            quotesQueryId='tss_match_phrase_query'
        )
    
        resp = requests.get(url=url, headers=headers, params=params)
        data = resp.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            return data['quotes'][0]['symbol']
        else:
            return None
        
def histo_data(ticker,start,end):
    data_close = yf.download(ticker, start,end)['Close']
    return data_close
def get_business_detail(ticker):
    data = yf.Ticker(ticker)
    data = data.info['longBusinessSummary']
    return data

locale._setlocale(locale.LC_ALL,"FR")
dt_today = dt.date.today().strftime("Le %d %B %Y")

st.title(dt_today)
st.image(image, width = 900)
st.title("Module d'analyse de titres")

isin = st.text_input(label="Insérer une référence à un ticker yahoo", max_chars=12)

# Functions to move to function file.py


if len(isin)==0:
    st.write("You should entrer a reference to a ticker")
else:       
    ticker_yahoo = find_ticker(isin)
    if ticker_yahoo==None:
        st.write("We cannot find a ticker on yahoo for your ISIN:", isin)
    else:
        st.write(isin," ticker on yahoo is ",ticker_yahoo)
        try:
            pass
            #st.write(get_business_detail(ticker_yahoo))
        except:
            pass
        start_date = st.date_input( "Start Date", value=pd.to_datetime("2018-12-31", format="%Y-%m-%d")).strftime("%Y-%m-%d")
        end_date = st.date_input("End Date", value=pd.to_datetime("today", format="%Y-%m-%d")).strftime("%Y-%m-%d")
        data = histo_data(ticker_yahoo,start_date,end_date)
        data_return = data.pct_change()
        data_vol = data_return.rolling(window=7).std()*np.sqrt(252)
        data_vol = pd.DataFrame({"Annualized 7 days volatility":data_vol})
        data_vol['Annualized 30 days']= data_return.rolling(window=30).std()*np.sqrt(252)
        data_vol['Annualized 90 days']= data_return.rolling(window=90).std()*np.sqrt(252)
        data_vol['Annualized 252 days']= data_return.rolling(window=252).std()*np.sqrt(252)
        data = pd.DataFrame({str(ticker_yahoo):data})
        
        data['Moyenne mobile 7 jours']= data[str(ticker_yahoo)].rolling(window=7).mean()
        data['Moyenne mobile 30 jours']= data[str(ticker_yahoo)].rolling(window=30).mean()
        data['Moyenne mobile 90 jours']= data[str(ticker_yahoo)].rolling(window=90).mean()
        data['Moyenne mobile 252 jours']= data[str(ticker_yahoo)].rolling(window=252).mean()
        st.title(f"Suivi de l'évolution du prix de {ticker_yahoo}")
        st.line_chart(data)
        st.title(f"Suivi de la volatilité  de {ticker_yahoo}")
        st.line_chart(data_vol)
        st.title("Les dernieres informations")
        st.dataframe(pd.DataFrame(yf.Ticker(ticker_yahoo).news)[['title','link']])
        st.dataframe(yf.Ticker(ticker_yahoo).recommendations)
        st.dataframe(yf.Ticker(ticker_yahoo).balance_sheet)
        st.dataframe(yf.Ticker(ticker_yahoo).earnings)
        
        