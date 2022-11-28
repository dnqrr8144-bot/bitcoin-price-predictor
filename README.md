# bitcoin-price-predictor
This repository will make an attempt to predict Bitcoin prices using long short-term memory

Libraries needed include:
pandas, numpy, math, matplotlib.pyplot

For evaluation we need:
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.preprocessing import MinMaxScaler

To install sklearn on Windows use the console and type: "pip install -U scikit-learn" For Mac users type: "pip install -U numpy scipy scikit-learn"

For building the model:
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

To install TensorFlow you need Python 3.7 or better. For the CPU version, install through the terminal by typing:
"python -m pip install --upgrade pip
pip install tensorflow"

For plotting:
import matplotlib.pyplot as plt
from itertools import cycle
import plotly.express as px