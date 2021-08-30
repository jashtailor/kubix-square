import tweepy as tw

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
from newsapi import NewsApiClient

import streamlit as st

st.beta_set_page_config(layout="wide")

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

# Init
newsapi = NewsApiClient(api_key='56885df3e9f04b6a9762a4b1a33f9f1e')

# /v2/everything
all_articles = newsapi.get_everything(q='Bitcoin',
                                      sources='axios, bloomberg, business-insider, crypto-coins-news, engadget, financial-post, google-news, hacker-news, mashable, next-big-futre, recode, reuters, techcrunch-cn, techradar, wired, the-wall-street-journal, bbc-news, fortune',
                                      domains='bbc.co.uk,techcrunch.com',
                                      from_param='2021-18-20',
                                      to='2021-08-30',
                                      language='en',
                                      sort_by='relevancy',
                                      page=1)    
    
st.write("""
# Malkovich """)



st.title("Let's create a table!")
for i in range(1, 10):
    cols = st.beta_columns(4)
    cols[0].write(f'{i}')
    cols[1].write(f'{i * i}')
    cols[2].write(f'{i * i * i}')
    cols[3].write('x' * i)
