# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:36:58 2025

@author: 06nic
"""

import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt

# Define tickers for the assets
assets_tickers = ["RTY=F", "VLUE", "MTUM", "USMV"]

# Define macroeconomic tickers
macro_tickers = {
    'rates': '^IRX',  # 13-week T-bill rate
    'equity_volatility': '^VIX',  # VIX index for equity volatility
}

# Download asset data (prices and adjusted close)
data_assets = yf.download(assets_tickers, start="2015-01-01", end="2025-01-01")
data_assets = data_assets.resample("Q").ffill()

# Download macroeconomic data
data_rates = yf.download(macro_tickers['rates'], start="2015-01-01", end="2025-01-01", interval='1mo')
data_volatility = yf.download(macro_tickers['equity_volatility'], start="2015-01-01", end="2025-01-01", interval='1mo')

# Calculate rolling 3-month averages (90-day) for macroeconomic variables
data_rates['Rolling_Rates'] = data_rates['Adj Close'].rolling(window=3).mean().shift(1)
data_volatility['Rolling_Volatility'] = data_volatility['Adj Close'].rolling(window=3).mean().shift(1)

# Download inflation (CPI) data from FRED (using pandas_datareader)
data_inflation = web.DataReader('CPIAUCSL', 'fred', start="2015-01-01", end="2025-01-01")
data_inflation = data_inflation.resample('M').last()  # Ensure it's monthly data
data_inflation['Rolling_Inflation'] = data_inflation['CPIAUCSL'].rolling(window=3).mean().shift(1)

# Calculate quarterly returns for the assets
data_assets['RTY=F_return'] = data_assets['Adj Close']['RTY=F'].pct_change()  # 3-month return for RTY=F
data_assets['VLUE_return'] = data_assets['Adj Close']['VLUE'].pct_change() # 3-month return for VLUE
data_assets['MTUM_return'] = data_assets['Adj Close']['MTUM'].pct_change()  # 3-month return for MTUM
data_assets['USMV_return'] = data_assets['Adj Close']['USMV'].pct_change()  # 3-month return for USMV

data_assets = data_assets[['RTY=F_return', 'VLUE_return', 'MTUM_return', 'USMV_return']]
# Drop rows with NaN values after calculating returns
data_assets.dropna(inplace=True)

# Resample macroeconomic data to quarterly (using last value of each quarter)
data_inflation_qtr = data_inflation['Rolling_Inflation'].resample('Q').last()  # Get last value of each quarter
data_rates_qtr = data_rates['Rolling_Rates'].resample('Q').last()  # Get last value of each quarter
data_volatility_qtr = data_volatility['Rolling_Volatility'].resample('Q').last()  # Get last value of each quarter

# Merge data on dates (indexes)
data_rates_qtr = data_rates_qtr.to_frame(name='Rates')
data_volatility_qtr = data_volatility_qtr.to_frame(name='Volatility')
data_inflation_qtr = data_inflation_qtr.to_frame(name='Inflation')

# Merging the datasets on the index (date)
data_combined = data_assets \
    .merge(data_inflation_qtr, left_index=True, right_index=True, how='inner') \
    .merge(data_rates_qtr, left_index=True, right_index=True, how='inner') \
    .merge(data_volatility_qtr, left_index=True, right_index=True, how='inner')
data_combined.columns = ['RTY=F_return', 'VLUE_return', 'MTUM_return', 'USMV_return','Inflation', 'Rates', 'Volatility']
# Drop rows with any NaN values (although it should not happen with inner join)
data_combined.dropna(inplace=True)

# Define features (macroeconomic factors) and target (asset returns)
X = data_combined[['Inflation', 'Rates', 'Volatility']]
y = data_combined[['RTY=F_return', 'VLUE_return', 'MTUM_return', 'USMV_return']]

# Split the data into training (80%) and testing (20%) sets
train_size = int(0.8 * len(data_combined))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Train the model (RandomForestRegressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Portfolio rebalancing: for each quarter, select the asset with the highest predicted return
portfolio = []
for i in range(len(y_pred)):
    predicted_returns = y_pred[i]
    selected_asset = y.columns[np.argmax(predicted_returns)]
    portfolio.append(selected_asset)

# Create a DataFrame for the portfolio decisions
portfolio_decisions = pd.DataFrame(portfolio, columns=['Selected Asset'], index=y_test.index)

# Calculate the returns for the selected portfolio
selected_returns = y_test.lookup(y_test.index, portfolio_decisions['Selected Asset'])

# Download S&P 500 data for comparison
sp500_data = yf.download('^GSPC', start="2015-01-01", end="2025-01-01", interval="1mo")
sp500__return = sp500_data['Adj Close'].resample('Q').ffill().pct_change()
sp500__return = sp500__return[y_test.index]


# Calculate the cumulative returns
portfolio_cumulative_return = (1 + selected_returns).cumprod()
sp500_cumulative_return = (1 + sp500__return).cumprod()

plt.figure(figsize=(10,6))
bar_width = 0.35
index = np.arange(len(selected_returns))

# Plotting the bars for both the strategy and S&P 500
plt.bar(index, selected_returns, bar_width, label="Strategy Portfolio")
plt.bar(index + bar_width, sp500__return, bar_width, label="S&P 500", linestyle='--')
plt.legend(["Factor Rotation Strategy", "S&P 500"])
plt.xlabel('Quarter')
plt.ylabel('Quarterly Return')
plt.title('Quarterly Return Comparison: Strategy Portfolio vs S&P 500')
plt.xticks(index + bar_width / 2, y_test.index.strftime('%Y-%m'))
plt.legend()
plt.show()

# Evaluate performance (e.g., Sharpe ratio, volatility, etc.)
mean_return = selected_returns.mean()
volatility = selected_returns.std()
sharpe_ratio = mean_return / volatility

# Display results
print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Cumulative Return: {portfolio_cumulative_return[-1]}")
