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
import streamlit as st

st.set_page_config(layout="wide")

c1, c2, c3, c4 = st.columns((2, 1, 1, 1))

def my_widget(key):
    st.subheader('Hello there!')
    clicked = st.button("Click me " + key)

# This works in the main area
clicked = my_widget("first")

# And within an expander
my_expander = st.beta_expander("Expand", expanded=True)
with my_expander:
    clicked = my_widget("second")

# AND in st.sidebar!
with st.sidebar:
    clicked = my_widget("third")

