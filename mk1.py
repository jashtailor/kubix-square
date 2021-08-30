import pandas as pd
import numpy as np

import nltk 
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from datetime import date

from coinbase.wallet.client import Client
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import simplejson as json

import itertools

import yfinance as yf

import tweepy as tw

from newsapi import NewsApiClient

import streamlit as st


st.set_page_config(layout="wide")


st.write("""
# Malkovich """)


# COINBASE API
# Before we take data from Twitter we need to know the top 10 cryptocurrencies based on market capitalization for which we use the Coinbase API
url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
  'start':'1',
  'limit':'5000',
  'convert':'USD'
}
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': '975df79a-7413-495e-8f06-67564fb215eb',
}

session = Session()
session.headers.update(headers)

try:
  response = session.get(url, params=parameters)
  data = json.loads(response.text)
  #print(data)
except (ConnectionError, Timeout, TooManyRedirects) as e:
  print(e)

technicalities = []
symbol = []
search_words = []
for i in range(0,10):
    technicalities.append(data['data'][i]['quote']['USD'])
    symbol.append(data['data'][i]['symbol'])
    search_words.append(data['data'][i]['name'])

def func(name):
    today = date.today()
    tickerSymbol = name+'-USD'
    tickerData = yf.Ticker(tickerSymbol)
    df = tickerData.history(period='1d', start='2000-01-01', end=today)
    df.reset_index(inplace=True)
     
    fig = go.Figure(data=go.Ohlc(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']))
    fig.show()    
    
def twitter(name):
  # TWITTER API
  consumer_key= 'uLPC3KfMtGFcEeq4CxEOohZeg'
  consumer_secret= 'tywsJRvcr2zz5ICg7bkadbSIIjhGFmAlOLjJECjPqMfaRuwc1T'
  access_token= '1300465599823314944-VkC6tWnEUrbxTZ1wYpWIxbc8LQCPNL'
  access_token_secret= 'DDiF0cmidxoQlT2rgEUCGkP4E2DI8PBwz6WMS5QL51zOG'
  auth = tw.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_token_secret)
  api = tw.API(auth, wait_on_rate_limit=True)

  date_since = '2021-04-01'
  tweet_text = []
  date_time = []
  location = []

  tweets = tw.Cursor(api.search,
              q=name,
              lang="en",
              since=date_since).items(100)
  for tweet in tweets:
    str1 = tweet.text
    str2 = tweet.created_at
    str3 = tweet.user.location
    tweet_text.append(str1)
    date_time.append(str2)
    location.append(str3)

  df_twitter = pd.DataFrame()
  df_twitter['Tweets'] = tweet_text
  df_twitter['Created at'] = date_time
  df_twitter['Location'] = location
  
  return df_twitter


def news_api(name):
  # NEWS API    
  # Init
  newsapi = NewsApiClient(api_key='56885df3e9f04b6a9762a4b1a33f9f1e')

  # /v2/everything
  all_articles = newsapi.get_everything(q=name,
                                      sources='axios, bloomberg, business-insider, crypto-coins-news, engadget, financial-post, google-news, hacker-news, mashable, next-big-futre, recode, reuters, techcrunch-cn, techradar, wired, the-wall-street-journal, bbc-news, fortune',
                                      domains='bbc.co.uk,techcrunch.com',
                                      from_param='2021-18-20',
                                      to='2021-08-30',
                                      language='en',
                                      sort_by='relevancy',
                                      page=1)   
  dict_0 = all_articles['articles']
  df_0 = pd.DataFrame(dict_0)
  return df_0
    

col1,col2 = st.beta_columns(2) 
col3,col4 = st.beta_columns(2)

choice = st.sidebar.selectbox("Menu", search_words)

if choice == search_words[0]:
  df_t = twitter(search_words[0])
  df_n = news_api(search_api[0])
  col4.write(df_n['title'])
  func(symbol[0])
