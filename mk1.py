import tweepy as tw
import pandas as pd
import numpy as np
import nltk 
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import date
from coinbase.wallet.client import Client
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import simplejson as json
import streamlit as st



# c1, c2, c3, c4 = st.beta_columns((2, 1, 1, 1))

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

option = st.selectbox(
     'Please select your preferred cryptocurrency',
     ('None', 'All', search_words[0], search_words[1], search_words[2], search_words[3], search_words[4], search_words[5], search_words[6], search_words[7], search_words[8], search_words[9]))
if option == 'All':
  time_series()
elif option == search_words[0]:
  sentiment_analysis()
 
st.set_page_config(layout="wide")
    
def my_widget(key):
    st.subheader('Hello there!')    
   

# This works in the main area
clicked = my_widget("first")



st.write(search_words)
    
col1, col2 = st.beta_columns(2)
col1.write('I am base col1!')
col2.write('I am base col2!')

# And within an expander
my_expander = st.beta_expander("Expand", expanded=True)
with my_expander:
    clicked = my_widget("second")




