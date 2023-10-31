import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import yfinance as yf
import numpy as np
from sklearn.metrics import mean_absolute_error

import time

# Start timer
start_time = time.time()

data = yf.download(tickers="GOOG", interval="1h", period="730d")
df = pd.DataFrame([], columns=['y', 'ds'])
df['y'] = data['Close']
df['ds'] = data.index
df.dropna(inplace=True)

train = df[:-6]
test = data[-6:]

demo = Prophet()
demo.fit(train)

test_actual = pd.DataFrame([], columns=['y', 'ds'])
test_actual['y'] = test['Close']
test_actual['ds'] = test.index.astype(dtype='str')


# Make predictions for the future dates
forecast = demo.predict(test_actual['ds'].to_frame())
end_time = time.time()

test_pred = forecast.set_index(forecast['ds'])

print("RMSE of Prophet is: ", np.sqrt(np.mean(np.array(test_pred['yhat']-test_actual['y'])**2)))
print("MAE of Prophet is: ", mean_absolute_error(test_actual['y'], test_pred['yhat']))

elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)