# AppleStock_prediction_LSTM
# Stock Prediction using Hybrid Reinforcement Learning Model (LSTM and Q-learning)

## Introduction

This repository contains a Python implementation of a hybrid reinforcement learning model for stock price prediction, combining Long Short-Term Memory networks (LSTM) with Q-learning.

## Requirements

To run this project, you will need the following packages:

- `keras`
- `tensorflow`
- `pandas`
- `matplotlib`
- `numpy`
- `scikit-learn`

You can install the required packages using the following command:

```bash
pip install keras tensorflow pandas matplotlib numpy scikit-learn
````
## Data

The model uses historical stock price data in CSV format. The CSV file should have a column named "Close" which contains the closing prices of the stock.

## Usage

1. Load the historical stock price data from a CSV file.

```python
import pandas as pd
df = pd.read_csv("path/to/your/data.csv")
```
## Preprocess the data.
```python
df1 = df.reset_index()['Close']
```
## Scale the data.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
```
## Train the Hybrid Reinforcement Learning Model

To train the hybrid reinforcement learning model, follow these steps:

1. Import the necessary libraries and modules.

```python
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque
```
2. Create the agent.
```python
class Agent:
```
