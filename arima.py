import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Function to fetch stock data
def fetch_stock_data(ticker_symbol):
    data = data = yf.Ticker(ticker_symbol).history(start ='2024-12-13', end='2024-12-14', interval="1m")
    return data

# Function to fit ARIMA model and predict the next day's stock price
def predict_stock_action(ticker_symbol):
    # Fetch historical data
    data = fetch_stock_data(ticker_symbol)  # Change dates to yesterday and today
    
    # Check if the data is available
    if data.empty:
        raise ValueError("No data available for the specified date range.")
    
    # Prepare data for modeling
    data['Price'] = data['Close']
    data.dropna(inplace=True)

    # Fit the ARIMA model
    model = ARIMA(data['Price'], order=(5, 1, 0))
    model_fit = model.fit()
    
    # Forecast the next day's price
    forecast = model_fit.forecast(steps=1)
    predicted_price = forecast.iloc[0]
    
    # Get the last price
    last_price = data['Price'].iloc[-1]

    # Determine buy/sell action
    if predicted_price > last_price:
        action = "Buy"
        
    else:
        action = "Sell"

    return last_price, predicted_price, action


last , predict, review = predict_stock_action('rpower.NS')
print(last, predict, review)

