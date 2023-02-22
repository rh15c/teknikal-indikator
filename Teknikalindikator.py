#Code untuk menggabungkan beberapa indikator teknikal, yaitu Moving Average, Support and Resistance, Relative Strength Index, dan Moving Average Convergence Divergence,

# Import library yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Mengambil data saham
df = pd.read_csv('data_saham.csv')

# Moving Average
ma_20 = df['Close'].rolling(window=20).mean()
ma_50 = df['Close'].rolling(window=50).mean()

# Support and Resistance
pivot_point = (df['High'].iloc[-1] + df['Low'].iloc[-1] + df['Close'].iloc[-1]) / 3
support_1 = (2 * pivot_point) - df['High'].iloc[-1]
resistance_1 = (2 * pivot_point) - df['Low'].iloc[-1]

# Relative Strength Index (RSI)
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

# Moving Average Convergence Divergence (MACD)
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
macd = ema_12 - ema_26
signal = macd.ewm(span=9, adjust=False).mean()

# Membuat plot data saham dan indikator teknikal
fig, axs = plt.subplots(2, 1, figsize=(15, 10))
axs[0].plot(df['Close'], label='Harga Saham')
axs[0].plot(ma_50, label='MA 50')
axs[0].plot(ma_200, label='MA 200')
axs[0].axhline(y=support_1, color='g', linestyle='--', label='Support 1')
axs[0].axhline(y=resistance_1, color='r', linestyle='--', label='Resistance 1')
axs[0].legend()
axs[1].plot(rsi, label='RSI')
axs[1].plot(macd, label='MACD')
axs[1].plot(signal, label='Signal')
axs[1].legend()
plt.show()
