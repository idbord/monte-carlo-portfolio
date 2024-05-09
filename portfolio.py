import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from yfinance import download

# Importing Data and returning the mean and covariance of the stock
def get_data(tickers: list[str], start: datetime, end: datetime):
    data = download(tickers, start, end)['Adj Close'] # Accounts for stock dividends and stock splits
    stockData = data.pct_change(fill_method=None)
    return stockData.mean(), stockData.cov()

stocks = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'META', 'KO', 'JPM', 'CSCO', 'BA']

endDate = datetime.now()
startDate = endDate - timedelta(days = 365)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)


weights = np.array([1 / len(stocks)] * len(stocks)) # Even distribution of weights

# If you would like to randomly generate weights, uncomment the following lines and comment the line above
# weights = np.random.random(len(meanReturns))
# weights /= np.sum(weights)

simulations = 500
timeInDays = 365

meanReturnsTransposed = np.full(shape=(timeInDays, len(stocks)), fill_value=meanReturns).T

portfolioSims = np.full(shape=(timeInDays, simulations), fill_value=0.0)

initialInvestment = 1_000

for i in range(simulations):
    Z = np.random.normal(size=(timeInDays, len(stocks))) 
    L = np.linalg.cholesky(covMatrix) # Cholesky Decomposition
    dailyReturns = meanReturnsTransposed + np.inner(L, Z)
    portfolioSims[:, i] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initialInvestment

y_avg = np.mean(portfolioSims[-1, :])

plt.plot(portfolioSims, color='b', alpha=0.075, linewidth=1.5)
plt.ylabel('Portfolio Value')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation of a Stock Portfolio')
plt.ticklabel_format(axis='y', style='plain')
plt.axhline(y_avg, color='r', linestyle='--')
plt.show()
