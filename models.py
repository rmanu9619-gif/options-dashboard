import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def forecast_volatility(ticker, periods=10):
    """
    Fetch historical stock data and forecast future volatility using ARIMA (statsmodels version)
    """
    # Fetch 1 year of daily stock prices
    data = yf.download(ticker, period="1y")
    
    # Daily returns
    returns = data['Close'].pct_change().dropna()
    
    # Rolling volatility (21-day window)
    volatility = returns.rolling(window=21).std().dropna()
    
    # Fit ARIMA(1,0,1) model — simple version
    model = ARIMA(volatility, order=(1,0,1))
    model_fit = model.fit()
    
    # Forecast next 'periods' days
    forecast = model_fit.forecast(steps=periods)
    
    return forecast.values


def monte_carlo_option(S, K, T, r, sigma_forecast, option_type="call", simulations=10000):
    """
    Simulate option prices using Monte Carlo.
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate
    sigma_forecast: list or array of predicted volatilities (from ARIMA)
    option_type: "call" or "put"
    simulations: number of simulated price paths
    """
    import numpy as np

    # Average forecasted volatility
    sigma_avg = np.mean(sigma_forecast)

    # Simulate end-of-period stock prices
    Z = np.random.standard_normal(simulations)
    ST = S * np.exp((r - 0.5 * sigma_avg**2) * T + sigma_avg * np.sqrt(T) * Z)

    # Calculate payoff for each simulation
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    # Discount back to present value
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price



if __name__ == "__main__":
    # Test Black-Scholes
    price = black_scholes(S=100, K=105, T=1, r=0.05, sigma=0.2, option_type="call")
    print("Option Price (Black-Scholes):", price)

    # Test ARIMA volatility forecast
    forecast = forecast_volatility("AAPL", periods=5)
    print("Forecasted volatility (next 5 days):", forecast)

    # Test Monte Carlo option price
    mc_price = monte_carlo_option(S=100, K=105, T=1, r=0.05, sigma_forecast=forecast, option_type="call")
    print("Option Price (Monte Carlo):", mc_price)
