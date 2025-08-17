# pages/info.py
import streamlit as st

st.title("ℹ️ Options Dashboard Info")

# -----------------------------
# What are Options
# -----------------------------
st.header("What Are Options?")
st.write("""
Options are financial contracts that give the buyer the right, but not the obligation, 
to buy or sell an underlying asset (like a stock) at a specified price (the strike price) 
before or at a certain date (the expiration date).  

There are two main types: 
- **Calls:** Right to buy  
- **Puts:** Right to sell
""")

st.subheader("Example")
st.write("""
Suppose you buy a call option on AAPL with a strike price of \$200, expiring in one month, 
and you pay a premium of $5.  

- If AAPL rises to \$220 before expiration, you can exercise your option to buy at $200, 
  making a profit (minus the premium).  
- If AAPL stays below \$200, you let the option expire and only lose the $5 premium.
""")

# -----------------------------
# About the Engine
# -----------------------------
st.header("What is the Stochastic ARIMA Monte Carlo Volatility Engine?")
st.write("""
This dashboard provides advanced option pricing and volatility analysis using:
- **ARIMA time series modeling**
- **Monte Carlo simulation**
- **Black-Scholes model**  

It is designed for traders, analysts, and students who want to explore the dynamics of 
option pricing and implied volatility in a modern, interactive way.
""")

# -----------------------------
# How It Works
# -----------------------------
st.header("How It Works: Stochastic ARIMA Monte Carlo Volatility Engine for Black-Scholes Pricing")

st.subheader("1. Data Ingestion & ETL")
st.write("""
- Fetches two years of daily close prices and live option chains (bid, ask, strike, expiry).  
- Enriches data with latest 3-month Treasury yield.  
- All data pulls run in parallel, cached for speed.
""")

st.subheader("2. Volatility Forecasting with ARIMA")
st.write("""
- Converts price series into log-returns and fits ARIMA(1,1,1) to capture autocorrelation and drift.  
- Forecasts tomorrow's variance, annualized to produce baseline volatility σ₁.  
- Uses auto_arima for optimal hyperparameters.
""")

st.subheader("3. Stochastic Scenario Generation via Monte Carlo")
st.write("""
- Simulates 1,000 volatility draws from a normal distribution around σ₁.  
- Computes 10th, 50th, 90th percentile bands (σ_low, σ_mid, σ_high) for low–mid–high volatility envelope.
""")

st.subheader("4. Risk-Neutral Option Valuation")
st.write("""
- Applies Black-Scholes formula using median volatility σ_mid.  
- Produces theoretical prices for calls and puts in milliseconds.
""")

st.subheader("5. Mispricing Signal & Traffic-Light Alerts")
st.write("""
- Edge = (market_mid – fair) / market_mid.  
- Contracts >10% rich/cheap labeled OVER/UNDER, rest labeled FAIR.  
- Color-coded traffic light shows portfolio-level risk.
""")

st.subheader("6. API Orchestration & Interactive Dashboard")
st.write("""
- All steps run inside a serverless API route.  
- Front-end components fetch JSON payloads, render volatility cards, status alerts, and options tables.  
- Users can tweak ticker, expiry, and simulation parameters in real time.
""")

# -----------------------------
# How to Use
# -----------------------------
st.header("How to Use")
st.write("""
- **Ticker Symbol:** Enter stock ticker (e.g., AAPL).  
- **Expiration Date:** Select available expiry.  
- **Number of Simulations:** Adjust Monte Carlo paths.  
- **Confidence Interval:** Set confidence level.  
- Click **Run Simulation** to generate volatility scenarios and option edge analysis.
""")

# -----------------------------
# Key Vocabulary
# -----------------------------
st.header("Key Vocabulary")
st.write("""
- **ARIMA:** Statistical model for analyzing/forecasting time series.  
- **Monte Carlo Simulation:** Uses random sampling to estimate complex outcomes.  
- **Black-Scholes Model:** Mathematical model for pricing European options.  
- **Volatility:** Measure of price variation over time.  
- **Option Edge:** Difference between model fair value and market price (%).
""")

# -----------------------------
# Tips
# -----------------------------
st.header("Tips")
st.write("""
- Compare volatility scenarios to identify potentially mispriced options.  
- Hover over info icons and table headers for explanations.  
""")

# -----------------------------
# DISLCLAIMER
# -----------------------------

st.header("**⚠️DISCLAIMER⚠️**")
st.write("""
- This tool is for educational/research purposes only—always do your own due diligence before trading.
""")
