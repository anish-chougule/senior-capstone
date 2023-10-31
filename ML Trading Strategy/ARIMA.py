import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

import time

# Start timer
start_time = time.time()

df = yf.download(tickers="GOOG", period="1d", interval="1m")

df = df[["Open", "High", "Low", "Close", "Volume"]]
X = df.index.values
y = df['Low'].values
offset = 6          # Number of candles in a day

X_train = X[:-offset]
y_train = y[:-offset]
X_test  = X[-offset:]
y_test  = y[-offset:]

model = ARIMA(y, order=(5,0,1)).fit()

test_pred = model.forecast(steps=offset)

end_time = time.time()

elapsed_time = end_time - start_time

print("RMSE of Prophet is: ", np.sqrt(np.mean(np.array(test_pred-y_test)**2)))
print("MAE of Prophet is: ", mean_absolute_error(y_test, test_pred))
print("Elapsed time: ", elapsed_time)