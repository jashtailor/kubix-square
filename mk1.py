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

import datetime 
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

technicals = []
max_supply = []
circulating_supply = []
symbol = []
search_words = []
for i in range(0,10):
    technicals.append(data['data'][i]['quote']['USD'])
    max_supply.append(data['data'][i]['max_supply'])
    circulating_supply.append(data['data'][0]['circulating_supply'])
    symbol.append(data['data'][i]['symbol'])
    search_words.append(data['data'][i]['name'])

df_crypto = pd.DataFrame()
df_crypto['Cryptocurrency'] = search_words
df_crypto.dropna(inplace=True)
df_crypto['Symbol'] = symbol

price = [] 
volume_24h = []
percent_change_1h = []
percent_change_24h = []
percent_change_7d = []
percent_change_30d = []
percent_change_60d = []
percent_change_90d = []
market_cap = []
market_cap_dominance = []
fully_diluted_market_cap = []
last_updated = []

for i in technicals:
    count = 0
    for j in i:
        k = i.get(j)
        count = count + 1
        if count == 1:
            price.append(k)
        if count == 2:
            volume_24h.append(k)
        if count == 3:
            percent_change_1h.append(k)
        if count == 4:
            percent_change_24h.append(k)
        if count == 5:
            percent_change_7d.append(k)
        if count == 6:
            percent_change_30d.append(k)
        if count == 7:
            percent_change_60d.append(k)
        if count == 8:
            percent_change_90d.append(k)
        if count == 9:
            market_cap.append(k)
        if count == 10:
            market_cap_dominance.append(k)
        if count == 11:
            fully_diluted_market_cap.append(k)
        if count == 12:
            last_updated.append(k)

df_crypto['Price in USD'] = price
df_crypto['Volume in 24h'] = volume_24h
df_crypto['% change in 1h'] = percent_change_1h
df_crypto['% change in 24h'] = percent_change_24h
df_crypto['% change in 7d'] = percent_change_7d
df_crypto['% change in 30d'] = percent_change_30d
df_crypto['% change in 60d'] = percent_change_60d
df_crypto['% change in 90d'] = percent_change_90d
df_crypto['Market Cap'] = market_cap
df_crypto['Fully diluted Market Cap'] = fully_diluted_market_cap
df_crypto['Last updated'] = last_updated
df_crypto['Circulating Supply'] = circulating_supply
df_crypto['Max Supply'] = max_supply
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
    
  today = date.today()
  d = datetime.timedelta(days = 5)
  a = today - d
    
  # /v2/everything
  all_articles = newsapi.get_everything(q=name,
                                      sources='axios, bloomberg, business-insider, crypto-coins-news, engadget, financial-post, google-news, hacker-news, mashable, next-big-future, recode, reuters, techcrunch-cn, techradar, wired, the-wall-street-journal, bbc-news, fortune, abc-news, al-jazeera-english,ars-technica, australian-financial-review, breitbart-news, associated-press, buzzfeed, cbs-news, fox-news, independent, medical-news-today, national-review, nbc-news, new-scientist, news24, news-com-au, newsweek, new-york-magazine, politico, polygon, reddit-r-all, techcrunch, the-american-conservative, the-hindu, the-huffington-post, the-lad-bible, the-next-web, the-times-of-india, the-washington-post, the-washington-times, time, vice-news',
                                      domains='bbc.co.uk,techcrunch.com',
                                      from_param=a,
                                      to=today,
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
side_bar = ['None'] + search_words
choice = st.sidebar.selectbox("Menu", side_bar)

# ------------------------------------------------------------------------------------------------------------------------------------------------
if choice == side_bar[0]:
    st.write("""
    This website runs entirely on your local machine, we do not have a backend or a database, all we ask from you is your patience as the time it takes to load this website up entirely depends on the speed on your computer and your internet.
             """)
# ------------------------------------------------------------------------------------------------------------------------------------------------
   
# ------------------------------------------------------------------------------------------------------------------------------------------------
if choice == side_bar[1]:
  df_t = twitter(side_bar[1])
  df_n = news_api(side_bar[1])
  ohlc = func(symbol[0])
  fig = go.Figure(data=go.Ohlc(x=ohlc['Date'],
                    open=ohlc['Open'],
                    high=ohlc['High'],
                    low=ohlc['Low'],
                    close=ohlc['Close']))
  st.write(choice+' in USD')
  st.plotly_chart(fig) 
  st.write(df_crypto[df_crypto['Cryptocurrency']==side_bar[1]])
  st.write('Public Sentiment on '+choice)
  fig_SA = px.bar(df_t, x='Sentiment', y='Count of Sentiment')
  st.plotly_chart(fig_SA)
  st.write('News regarding '+choice)
  for i in range(len(df_n['title'])):
    link = '[' + df_n['title'][i] + ']' + '(' + df_n['url'][i] + ')'
    st.write(link, unsafe_allow_html=True)
# ------------------------------------------------------------------------------------------------------------------------------------------------
    
  
# ------------------------------------------------------------------------------------------------------------------------------------------------  
if choice == side_bar[2]:
  df_t = twitter(side_bar[2])
  df_n = news_api(side_bar[2])
  ohlc = func(symbol[1])
  fig = go.Figure(data=go.Ohlc(x=ohlc['Date'],
                    open=ohlc['Open'],
                    high=ohlc['High'],
                    low=ohlc['Low'],
                    close=ohlc['Close']))
  st.write(choice+' in USD')
  st.plotly_chart(fig)   
  st.write(df_crypto[df_crypto['Cryptocurrency']==side_bar[2]])
  st.write('Public Sentiment on '+choice)
  fig_SA = px.bar(df_t, x='Sentiment', y='Count of Sentiment')
  st.plotly_chart(fig_SA)   
  st.write('News regarding '+choice)
  for i in range(len(df_n['title'])):
    link = '[' + df_n['title'][i] + ']' + '(' + df_n['url'][i] + ')'
    st.write(link, unsafe_allow_html=True)
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
if choice == side_bar[3]:
  df_t = twitter(side_bar[3])
  df_n = news_api(side_bar[3])
  ohlc = func(symbol[2])
  fig = go.Figure(data=go.Ohlc(x=ohlc['Date'],
                    open=ohlc['Open'],
                    high=ohlc['High'],
                    low=ohlc['Low'],
                    close=ohlc['Close']))
  st.write(choice+' in USD')
  st.plotly_chart(fig)   
  st.write(df_crypto[df_crypto['Cryptocurrency']==side_bar[3]])
  st.write('Public Sentiment on '+choice)
  fig_SA = px.bar(df_t, x='Sentiment', y='Count of Sentiment')
  st.plotly_chart(fig_SA)   
  st.write('News regarding '+choice)
  for i in range(len(df_n['title'])):
    link = '[' + df_n['title'][i] + ']' + '(' + df_n['url'][i] + ')'
    st.write(link, unsafe_allow_html=True)  
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
if choice == side_bar[4]:
  df_t = twitter(side_bar[4])
  df_n = news_api(side_bar[4])
  ohlc = func(symbol[3])
  fig = go.Figure(data=go.Ohlc(x=ohlc['Date'],
                    open=ohlc['Open'],
                    high=ohlc['High'],
                    low=ohlc['Low'],
                    close=ohlc['Close']))
  st.write(choice+' in USD')
  st.plotly_chart(fig)
  st.write(df_crypto[df_crypto['Cryptocurrency']==side_bar[4]])
  st.write('Public Sentiment on '+choice)
  fig_SA = px.bar(df_t, x='Sentiment', y='Count of Sentiment')
  st.plotly_chart(fig_SA)   
  st.write('News regarding '+choice)
  for i in range(len(df_n['title'])):
    link = '[' + df_n['title'][i] + ']' + '(' + df_n['url'][i] + ')'
    st.write(link, unsafe_allow_html=True) 
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
if choice == side_bar[5]:
  df_t = twitter(side_bar[5])
  df_n = news_api(side_bar[5])
  ohlc = func(symbol[4])
  fig = go.Figure(data=go.Ohlc(x=ohlc['Date'],
                    open=ohlc['Open'],
                    high=ohlc['High'],
                    low=ohlc['Low'],
                    close=ohlc['Close']))
  st.write(choice+' in USD')
  st.plotly_chart(fig) 
  st.write(df_crypto[df_crypto['Cryptocurrency']==side_bar[5]])
  st.write('Public Sentiment on '+choice)
  fig_SA = px.bar(df_t, x='Sentiment', y='Count of Sentiment')
  st.plotly_chart(fig_SA) 
  st.write('News regarding '+choice)
  for i in range(len(df_n['title'])):
    link = '[' + df_n['title'][i] + ']' + '(' + df_n['url'][i] + ')'
    st.write(link, unsafe_allow_html=True)    
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
if choice == side_bar[6]:
  df_t = twitter(side_bar[6])
  df_n = news_api(side_bar[6])
  ohlc = func(symbol[5])
  fig = go.Figure(data=go.Ohlc(x=ohlc['Date'],
                    open=ohlc['Open'],
                    high=ohlc['High'],
                    low=ohlc['Low'],
                    close=ohlc['Close']))
  st.write(choice+' in USD')
  st.plotly_chart(fig) 
  st.write(df_crypto[df_crypto['Cryptocurrency']==side_bar[6]])
  st.write('Public Sentiment on '+choice)
  fig_SA = px.bar(df_t, x='Sentiment', y='Count of Sentiment')
  st.plotly_chart(fig_SA)   
  st.write('News regarding '+choice)
  for i in range(len(df_n['title'])):
    link = '[' + df_n['title'][i] + ']' + '(' + df_n['url'][i] + ')'
    st.write(link, unsafe_allow_html=True)
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
if choice == side_bar[7]:
  df_t = twitter(side_bar[7])
  df_n = news_api(side_bar[7])
  ohlc = func(symbol[6])
  fig = go.Figure(data=go.Ohlc(x=ohlc['Date'],
                    open=ohlc['Open'],
                    high=ohlc['High'],
                    low=ohlc['Low'],
                    close=ohlc['Close']))
  st.write(choice+' in USD')
  st.plotly_chart(fig)   
  st.write(df_crypto[df_crypto['Cryptocurrency']==side_bar[7]])
  st.write('Public Sentiment on '+choice)
  fig_SA = px.bar(df_t, x='Sentiment', y='Count of Sentiment')
  st.plotly_chart(fig_SA)   
  st.write('News regarding '+choice)
  for i in range(len(df_n['title'])):
    link = '[' + df_n['title'][i] + ']' + '(' + df_n['url'][i] + ')'
    st.write(link, unsafe_allow_html=True)
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
if choice == side_bar[8]:
  df_t = twitter(side_bar[8])
  df_n = news_api(side_bar[8])
  ohlc = func(symbol[7])
  fig = go.Figure(data=go.Ohlc(x=ohlc['Date'],
                    open=ohlc['Open'],
                    high=ohlc['High'],
                    low=ohlc['Low'],
                    close=ohlc['Close']))
  st.write(choice+' in USD')
  st.plotly_chart(fig)   
  st.write(df_crypto[df_crypto['Cryptocurrency']==side_bar[8]])
  st.write('Public Sentiment on '+choice)
  fig_SA = px.bar(df_t, x='Sentiment', y='Count of Sentiment')
  st.plotly_chart(fig_SA)   
  st.write('News regarding '+choice)
  for i in range(len(df_n['title'])):
    link = '[' + df_n['title'][i] + ']' + '(' + df_n['url'][i] + ')'
    st.write(link, unsafe_allow_html=True)
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
if choice == side_bar[9]:
  df_t = twitter(side_bar[9])
  df_n = news_api(side_bar[9])
  ohlc = func(symbol[8])
  fig = go.Figure(data=go.Ohlc(x=ohlc['Date'],
                    open=ohlc['Open'],
                    high=ohlc['High'],
                    low=ohlc['Low'],
                    close=ohlc['Close']))
  st.write(choice+' in USD')
  st.plotly_chart(fig)   
  st.write(df_crypto[df_crypto['Cryptocurrency']==side_bar[9]])
  st.write('Public Sentiment on '+choice)
  fig_SA = px.bar(df_t, x='Sentiment', y='Count of Sentiment')
  st.plotly_chart(fig_SA)   
  st.write('News regarding '+choice)
  for i in range(len(df_n['title'])):
    link = '[' + df_n['title'][i] + ']' + '(' + df_n['url'][i] + ')'
    st.write(link, unsafe_allow_html=True)
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
if choice == side_bar[10]:
  df_t = twitter(side_bar[10])
  df_n = news_api(side_bar[10])
  ohlc = func(symbol[9])
  fig = go.Figure(data=go.Ohlc(x=ohlc['Date'],
                    open=ohlc['Open'],
                    high=ohlc['High'],
                    low=ohlc['Low'],
                    close=ohlc['Close']))
  st.write(choice+' in USD')
  st.plotly_chart(fig)   
  st.write(df_crypto[df_crypto['Cryptocurrency']==side_bar[10]])
  st.write('Public Sentiment on '+choice)
  fig_SA = px.bar(df_t, x='Sentiment', y='Count of Sentiment')
  st.plotly_chart(fig_SA)   
  st.write('News regarding '+choice)
  for i in range(len(df_n['title'])):
    link = '[' + df_n['title'][i] + ']' + '(' + df_n['url'][i] + ')'
    st.write(link, unsafe_allow_html=True)
# ------------------------------------------------------------------------------------------------------------------------------------------------
