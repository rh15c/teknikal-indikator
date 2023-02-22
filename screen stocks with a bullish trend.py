# To screen stocks with a bullish trend, you can use technical indicators such as moving averages and relative strength index (RSI) to identify stocks that are trending upwards. Here's an example code in Python using the yfinance library to retrieve stock data and calculate the moving averages and RSI:

import yfinance as yf

# Define the stock symbol
symbol = 'AAPL'

# Retrieve the stock data
stock = yf.Ticker(symbol)
df = stock.history(period="max")

# Calculate the moving averages
ma_20 = df['Close'].rolling(window=20).mean()
ma_50 = df['Close'].rolling(window=50).mean()

# Calculate the RSI
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

# Check if the stock has a bullish trend
if df['Close'][-1] > ma_20[-1] and ma_20[-1] > ma_50[-1] and rsi[-1] > 50:
    print(f"{symbol} has a bullish trend")
else:
    print(f"{symbol} does not have a bullish trend")


#In this code, we first retrieve the historical stock data for the symbol 'AAPL' using the yfinance library. We then calculate the 20-day and 50-day moving averages of the stock's closing price, as well as the 14-day RSI. Finally, we check if the stock's closing price is above the 20-day moving average, which is above the 50-day moving average, and the RSI is above 50. If all of these conditions are met, we print that the stock has a bullish trend.