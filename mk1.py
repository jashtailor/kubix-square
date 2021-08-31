import yfinance as yf

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier

from datetime import date

from coinbase.wallet.client import Client
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import simplejson as json

import itertools
from zipfile import ZipFile

import pickle

import tweepy as tw

from newsapi import NewsApiClient

import streamlit as st


st.write("""
# Malkovich """)



# ------------------------------------------------------------------------------------------------------------------------------------------------
# PREPROCESSING
stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

def further_processing(processed_docs):
    tokenizer = Tokenizer(oov_token='<00V>')
    tokenizer.fit_on_texts(processed_docs)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(processed_docs)
    padded = pad_sequences(sequences, padding='post', maxlen=18)
    
    return padded
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
# IMPORTING THE MODELS    
def models(padded_1):
    
  file_name = "RFC.zip"
  with ZipFile(file_name, 'r') as zip1:
    zip1.extractall()
  with open('RFC', 'rb') as f:
    RFC = pickle.load(f)  
    
  with open('GBC', 'rb') as f:
    GBC = pickle.load(f)

  with open('LGBM', 'rb') as f:
    LGBM = pickle.load(f)    

  with open('AdaB', 'rb') as f:
    AdaB = pickle.load(f) 
  
  
   

  def just_for_prediction(model, tweets):
    prediction = model.predict(tweets)
    
    return prediction
  
  lst1 = just_for_prediction(RFC,padded_1)
  lst2 = just_for_prediction(LGBM, padded_1)
  lst3 = just_for_prediction(AdaB, padded_1)
  lst4 = just_for_prediction(GBC, padded_1)
  lst5 = range(len(lst2))

  lst0 = []
  negative = 0
  neutral = 0 
  positive = 0
  for (a,b,c,d) in zip(lst1,lst2,lst3,lst4):
      lst6 = list([a,b,c,d])
      # print(lst6)
      if lst6.count(-1)>lst6.count(1) and lst6.count(-1)>lst6.count(0):
          str1 = 'Negative'
          negative = negative + 1 
      elif lst6.count(0)>lst6.count(1) and  lst6.count(0)>lst6.count(-1):
          str1 = 'Neutral'
          neutral = neutral + 1
      else:
          str1 = 'Positive'
          positive = positive + 1
      lst0.append(str1)

  lst7 = ['Negative', 'Neutral', 'Positive']
  lst8 = [negative, neutral, positive]
  df3 = pd.DataFrame({'Sentiment': lst7, 'Count of Sentiment':lst8})
  
  return df3
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
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

technicalities = ['None']
symbol = ['NaN']
search_words = ['None']
for i in range(0,10):
    technicalities.append(data['data'][i]['quote']['USD'])
    symbol.append(data['data'][i]['symbol'])
    search_words.append(data['data'][i]['name'])
# ------------------------------------------------------------------------------------------------------------------------------------------------
 
# ------------------------------------------------------------------------------------------------------------------------------------------------   
# TWITTER API
def twitter(name):
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
  
  processed_docs = df_twitter['Tweets'].map(preprocess)
  padded = further_processing(processed_docs)
  df_SA = models(padded)
  
  return df_SA
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
# NEWS API   
def news_api(name): 
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
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
# OHLC GRAPH
def func(name):
    today = date.today()
    tickerSymbol = name+'-USD'
    tickerData = yf.Ticker(tickerSymbol)
    df = tickerData.history(period='1d', start='2000-01-01', end=today)
    df.reset_index(inplace=True)
    
    return df
# ------------------------------------------------------------------------------------------------------------------------------------------------

# FRONTEND
choice = st.sidebar.selectbox("Menu", search_words)

# ------------------------------------------------------------------------------------------------------------------------------------------------
if choice == search_words[0]:
    st.write("""
    This website runs entirely on your local machine, we do not have a backend or a database, all we ask from you is your patience as the time it takes to load this website up entirely depends on the speed on your computer and your internet.
             """)
# ------------------------------------------------------------------------------------------------------------------------------------------------
   
# ------------------------------------------------------------------------------------------------------------------------------------------------
if choice == search_words[1]:
  df_t = twitter(search_words[1])
  df_n = news_api(search_words[1])
  ohlc = func(symbol[1])
  fig = go.Figure(data=go.Ohlc(x=ohlc['Date'],
                    open=ohlc['Open'],
                    high=ohlc['High'],
                    low=ohlc['Low'],
                    close=ohlc['Close']))
  st.write(choice+' in USD')
  st.plotly_chart(fig) 
  st.write('Public Sentiment on '+choice)
  fig_SA = px.bar(df_t, x='Sentiment', y='Count of Sentiment')
  st.plotly_chart(fig_SA)
  st.write('News regarding '+choice)
  for i in range(len(df_n['title'])):
    link = '[' + df_n['title'][i] + ']' + '(' + df_n['url'][i] + ')'
    st.write(link, unsafe_allow_html=True)
# ------------------------------------------------------------------------------------------------------------------------------------------------
    
  
# ------------------------------------------------------------------------------------------------------------------------------------------------  
if choice == search_words[2]:
  df_t = twitter(search_words[2])
  df_n = news_api(search_words[2])
  ohlc = func(symbol[2])
  fig = go.Figure(data=go.Ohlc(x=ohlc['Date'],
                    open=ohlc['Open'],
                    high=ohlc['High'],
                    low=ohlc['Low'],
                    close=ohlc['Close']))
  st.write(choice+' in USD')
  st.plotly_chart(fig)   
  st.write('Public Sentiment on '+choice)
  fig_SA = px.bar(df_t, x='Sentiment', y='Count of Sentiment')
  st.plotly_chart(fig_SA)   
  st.write('News regarding '+choice)
  for i in range(len(df_n['title'])):
    link = '[' + df_n['title'][i] + ']' + '(' + df_n['url'][i] + ')'
    st.write(link, unsafe_allow_html=True)
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
if choice == search_words[3]:
  df_t = twitter(search_words[3])
  df_n = news_api(search_words[3])
  ohlc = func(symbol[3])
  fig = go.Figure(data=go.Ohlc(x=ohlc['Date'],
                    open=ohlc['Open'],
                    high=ohlc['High'],
                    low=ohlc['Low'],
                    close=ohlc['Close']))
  st.write(choice+' in USD')
  st.plotly_chart(fig)   
  st.write('Public Sentiment on '+choice)
  fig_SA = px.bar(df_t, x='Sentiment', y='Count of Sentiment')
  st.plotly_chart(fig_SA)   
  st.write('News regarding '+choice)
  for i in range(len(df_n['title'])):
    link = '[' + df_n['title'][i] + ']' + '(' + df_n['url'][i] + ')'
    st.write(link, unsafe_allow_html=True)  
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
if choice == search_words[4]:
  df_t = twitter(search_words[4])
  df_n = news_api(search_words[4])
  ohlc = func(symbol[4])
  fig = go.Figure(data=go.Ohlc(x=ohlc['Date'],
                    open=ohlc['Open'],
                    high=ohlc['High'],
                    low=ohlc['Low'],
                    close=ohlc['Close']))
  st.write(choice+' in USD')
  st.plotly_chart(fig)   
  st.write('Public Sentiment on '+choice)
  fig_SA = px.bar(df_t, x='Sentiment', y='Count of Sentiment')
  st.plotly_chart(fig_SA)   
  st.write('News regarding '+choice)
  for i in range(len(df_n['title'])):
    link = '[' + df_n['title'][i] + ']' + '(' + df_n['url'][i] + ')'
    st.write(link, unsafe_allow_html=True) 
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
if choice == search_words[5]:
  df_t = twitter(search_words[5])
  df_n = news_api(search_words[5])
  ohlc = func(symbol[5])
  fig = go.Figure(data=go.Ohlc(x=ohlc['Date'],
                    open=ohlc['Open'],
                    high=ohlc['High'],
                    low=ohlc['Low'],
                    close=ohlc['Close']))
  st.write(choice+' in USD')
  st.plotly_chart(fig)   
  st.write('Public Sentiment on '+choice)
  fig_SA = px.bar(df_t, x='Sentiment', y='Count of Sentiment')
  st.plotly_chart(fig_SA) 
  st.write('News regarding '+choice)
  for i in range(len(df_n['title'])):
    link = '[' + df_n['title'][i] + ']' + '(' + df_n['url'][i] + ')'
    st.write(link, unsafe_allow_html=True)    
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
if choice == search_words[6]:
  df_t = twitter(search_words[6])
  df_n = news_api(search_words[6])
  ohlc = func(symbol[6])
  fig = go.Figure(data=go.Ohlc(x=ohlc['Date'],
                    open=ohlc['Open'],
                    high=ohlc['High'],
                    low=ohlc['Low'],
                    close=ohlc['Close']))
  st.write(choice+' in USD')
  st.plotly_chart(fig)   
  st.write('Public Sentiment on '+choice)
  fig_SA = px.bar(df_t, x='Sentiment', y='Count of Sentiment')
  st.plotly_chart(fig_SA)   
  st.write('News regarding '+choice)
  for i in range(len(df_n['title'])):
    link = '[' + df_n['title'][i] + ']' + '(' + df_n['url'][i] + ')'
    st.write(link, unsafe_allow_html=True)
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
if choice == search_words[7]:
  df_t = twitter(search_words[7])
  df_n = news_api(search_words[7])
  ohlc = func(symbol[7])
  fig = go.Figure(data=go.Ohlc(x=ohlc['Date'],
                    open=ohlc['Open'],
                    high=ohlc['High'],
                    low=ohlc['Low'],
                    close=ohlc['Close']))
  st.write(choice+' in USD')
  st.plotly_chart(fig)   
  st.write('Public Sentiment on '+choice)
  fig_SA = px.bar(df_t, x='Sentiment', y='Count of Sentiment')
  st.plotly_chart(fig_SA)   
  st.write('News regarding '+choice)
  for i in range(len(df_n['title'])):
    link = '[' + df_n['title'][i] + ']' + '(' + df_n['url'][i] + ')'
    st.write(link, unsafe_allow_html=True)
# ------------------------------------------------------------------------------------------------------------------------------------------------
