import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import time

import warnings
warnings.filterwarnings("ignore")

df = yf.download(tickers="TSLA", period="7d", interval="1m")[["Open", "High", "Low", "Close"]]

arima_start = time.time()

X = df.index.values
y = df['Close'].values# The split point is the 10% of the dataframe length
offset = 30          # Number of candles in a day

X_train = X[:-offset]
y_train = y[:-offset]
X_test  = X[-offset:]
y_test  = y[-offset:]

model = ARIMA(y, order=(5,0,1)).fit()
forecast = model.forecast(steps=offset)[0]
arima_pred = model.forecast(steps=offset)

arima_end = time.time()

arima_pred = pd.DataFrame(arima_pred, index=X_test, columns=['Close'])
y_test = pd.DataFrame(y_test, index=X_test, columns=['Close'])

prophet_start = time.time()

data = df.copy()
data['ds'] = data.index
data.rename(columns={'Close':'y'}, inplace=True)

train = data[:-offset]
test = data[-offset:]

demo = Prophet(daily_seasonality=True)
demo.add_regressor('High', standardize=False)
demo.add_regressor('Low', standardize=False)
demo.add_regressor('Open', standardize=False)
demo.fit(train)

forecast = demo.predict(test[['ds', 'High', 'Low', 'Open']])
prophet_pred = forecast.set_index(forecast['ds'])

prophet_end = time.time()

print("RMSE of ARIMA is: ", np.sqrt(np.mean(np.array(arima_pred-y_test)**2)))
print("MAE of ARIMA is: ", mean_absolute_error(y_test, arima_pred))
print("ARIMA Runtime:", (arima_end-arima_start))
print("RMSE of Prophet is: ", np.sqrt(np.mean(np.array(prophet_pred['yhat']-test['y'])**2)))
print("MAE of Prophet is: ", mean_absolute_error(test['y'], prophet_pred['yhat']))
print("Prophet Runtime: ", (prophet_end-prophet_start))

plt.plot(prophet_pred['yhat'], label="PROPHET Prediction Line")

plt.plot(arima_pred, label="ARIMA Prediction Line")
plt.plot(test['y'], label="Actual prices")

plt.legend()
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.show()
