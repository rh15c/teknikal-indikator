# download data from yahoo finance
import yfinance as yf

saham = yf.Ticker("SBMA")
saham_hist = saham.history(period="max")

#install pandas
import os
import pandas as pd

DATA_PATH = "saham_data.json"

if os.path.exists(DATA_PATH):
# baca dari file jika sudah download data
	with open(DATA_PATH) as C:
		saham_hist = pd.read_json(DATA_PATH)
else:
	saham = yf.Ticker("SBMA")
	saham_hist = saham.history(period="max")
	
	# simpan file to json in case di perlukan nanti, jadi ga perlu download ulang
	saham_hist.to_json(DATA_PATH)
	
# Exploring the data
saham_hist.head(5)

# visualkan harga saham 
saham_hist.plot.line(y="Close", use_index=True)

# Ensure we know the actual closing price
data = saham_hist[["Close"]]
data = data.rename(columns = {'Close':'Actual_Close'})

# Setup our target.  This identifies if the price went up or down
data["Target"] = saham_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

data.head()

# Shift stock prices forward one day, so we're predicting tomorrow's stock prices from today's prices.
saham_prev = saham_hist.copy()
saham_prev = saham_prev.shift(1)

saham_prev.head()

# Combining our data
# Create our training data
predictors = ["Close", "Volume", "Open", "High", "Low"]
data = data.join(saham_prev[predictors]).iloc[1:]
data.head()

# Creating a machine learning model
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create a random forest classification model.  Set min_samples_split high to ensure we don't overfit.
model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)

# create a train and test set
train = data.iloc[:-100]
test = data.iloc[-100:]

model.fit(train[predictors], train["Target"])

# Measuring error
from sklearn.metrics import precision_score

# Evaluate error of predictions
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
precision_score(test["Target"], preds)

combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
combined.plot()

# Backtesting
i = 1000
step = 750

train = data.iloc[0:i].copy()
test = data.iloc[i:(i+step)].copy()
model.fit(train[predictors], train["Target"])
preds = model.predict(test[predictors])

# Predicting probabilities
preds = model.predict_proba(test[predictors])[:,1]
preds = pd.Series(preds, index=test.index)
preds[preds > .6] = 1
preds[preds<=.6] = 0

preds.head()

# Pulling it into a loop

predictions = []
# Loop over the dataset in increments
for i in range(1000, data.shape[0], step):
    # Split into train and test sets
    train = data.iloc[0:i].copy()
    test = data.iloc[i:(i+step)].copy()

    # Fit the random forest model
    model.fit(train[predictors], train["Target"])

    # Make predictions
    preds = model.predict_proba(test[predictors])[:,1]
    preds = pd.Series(preds, index=test.index)
    preds[preds > .6] = 1
    preds[preds<=.6] = 0

    # Combine predictions and test values
    combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)

    predictions.append(combined)
	
	predictions[0].head()
	
# Creating a backtesting function
def backtest(data, model, predictors, start=1000, step=750):
    predictions = []
    # Loop over the dataset in increments
    for i in range(start, data.shape[0], step):
        # Split into train and test sets
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()

        # Fit the random forest model
        model.fit(train[predictors], train["Target"])

        # Make predictions
        preds = model.predict_proba(test[predictors])[:,1]
        preds = pd.Series(preds, index=test.index)
        preds[preds > .6] = 1
        preds[preds<=.6] = 0

        # Combine predictions and test values
        combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)

        predictions.append(combined)

    return pd.concat(predictions)
	
# Running the function
predictions = backtest(data, model, predictors)
predictions["Prediction"].value_counts()
predictions["Target"].value_counts()
precision_score(predictions["Target"], predictions["Predictions"])

# Improving accuracy
weekly_mean = data.rolling(7).mean()["Close"]
quarterly_mean = data.rolling(90).mean()["Close"]
annual_mean = data.rolling(365).mean()["Close"]

weekly_trend = data.shift(1).rolling(7).sum()["Target"]

data["weekly_mean"] = weekly_mean / data["Close"]
data["quarterly_mean"] = quarterly_mean / data["Close"]
data["annual_mean"] = annual_mean / data["Close"]

data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]

data["weekly_trend"] = weekly_trend

data["open_close_ratio"] = data["Open"] / data["Close"]
data["high_close_ratio"] = data["High"] / data["Close"]
data["low_close_ratio"] = data["Low"] / data["Close"]

full_predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "open_close_ratio", "high_low_ratio"]

predictions = backtest(data.iloc[365:], model, full_predictors)

precision_score(predictions["Target"], predictions["Predictions"])

# Evaluating our predictions
# Show how many trades we would make

predictions["Predictions"].value_counts()

# Look at trades we would have made in the last 100 days

predictions.iloc[-100:].plot()


