# Monte Carlo Simulation for Stock Portfolio Analysis

This Python code performs a Monte Carlo simulation to analyze the potential performance of a stock portfolio over a specified time period. It uses historical stock data to estimate the expected returns and covariance matrix, which are then used to generate simulated returns for the portfolio.

## Why I wrote this code
I wrote this code to get an understanding of how Monte Carlo simulations can be used to analyze the performance of a stock portfolio. By simulating the daily returns of a portfolio based on historical data, we can assess the range of possible outcomes and evaluate the associated risks. This code provides a practical example of how to implement a Monte Carlo simulation for stock portfolio analysis in Python.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- yfinance

## Installation

1. Clone the repository:

    ```bash
    git clone git@github.com:idbord/monte-carlo-portfolio.git
    ```

2. Navigate to the project directory:

    ```bash
    cd monte-carlo-portfolio
    ```

3. Install a virtual environment:

    ```bash
    python3 -m venv .venv
    ```

4. Activate the virtual environment:

    Linux:
    ```bash
    source .venv/bin/activate
    ```

    Windows:
    ```bash
    source .venv/Scripts/activate
    ```

5. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Import the required libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from yfinance import download
```

2. Define the list of stock tickers you want to include in the portfolio:
```python
stocks = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'META', 'KO', 'JPM', 'CSCO', 'BA']
```

3. Set the start and end dates for the historical data:
```python
start_date = datetime.now() - timedelta(days=365)
end_date = datetime.now()
```

4. Call the `get_data` function to retrieve the mean returns and covariance matrix for the given stocks and date range:
```python
meanReturns, covMatrix = get_data(stocks, start_date, end_date)

```
5. Randomly initialize portfolio weights, ensuring they sum to 1:
```python
weights = np.array([1 / len(stocks)] * len(stocks))
```

Note that if we want to randomly distribute the weights, we can use the following code:
```python
weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)
```

6. Set the number of Monte Carlo simulations and the time horizon (in days):

```python
simulations = 100
timeInDays = 365
```

7. Run the Monte Carlo simulations:
```python
meanReturnsTransposed = np.full(shape=(timeInDays, len(stocks)), fill_value=meanReturns).T

portfolioSims = np.full(shape=(timeInDays, simulations), fill_value=0.0)

initialInvestment = 1_000

for i in range(simulations):
    Z = np.random.normal(size=(timeInDays, len(stocks)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanReturnsTransposed + np.inner(L, Z)
    portfolioSims[:, i] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initialInvestment
```

8. Plot the results:
```python
y_avg = np.mean(portfolioSims[-1, :])

plt.plot(portfolioSims, color='b', alpha=0.075, linewidth=1.5)
plt.ylabel('Portfolio Value')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation of a Stock Portfolio')
plt.ticklabel_format(axis='y', style='plain')
plt.axhline(y_avg, color='r', linestyle='--')
plt.show()
```

## Description
The `get_data` function retrieves the adjusted closing prices for the specified stock tickers and date range using the `yfinance` library. It then calculates the daily percentage returns and returns the mean returns and covariance matrix.

The code then generates random portfolio weights and initializes arrays to store the simulated daily returns and portfolio values for the specified number of Monte Carlo simulations and time horizon.
For each simulation, the code generates random normal numbers (Z), calculates the daily returns using the Cholesky decomposition of the covariance matrix, and computes the cumulative portfolio value over time. The simulated portfolio values are stored in the `portfolioSims` array.

Finally, the code plots the simulated portfolio values over time using Matplotlib.
The Monte Carlo simulation is a widely used technique for analyzing the potential performance of investment portfolios under various scenarios. It helps to understand the range of possible outcomes and assess the associated risks.