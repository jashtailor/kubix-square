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

# Use the full page instead of a narrow central column
st.beta_set_page_config(layout="wide")

# Space out the maps so the first one is 2x the size of the other three
c1, c2, c3, c4 = st.beta_columns((2, 1, 1, 1))

