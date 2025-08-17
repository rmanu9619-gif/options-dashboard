import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import scipy.stats as si
from datetime import datetime, timedelta

# -----------------------------
# Black-Scholes Option Pricing
# -----------------------------
def black_scholes(S, K, T, r, sigma, option_type="call"):
    r = 0.0391  # 3.91% risk-free rate
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = (S * si.norm.cdf(d1, 0.0, 1.0) -
                 K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    else:
        price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) -
                 S * si.norm.cdf(-d1, 0.0, 1.0))
    return price

# -----------------------------
# Monte Carlo Simulation
# -----------------------------
def monte_carlo_sim(S, T, r, sigma, n_sims=1000):
    np.random.seed(42)
    Z = np.random.standard_normal(n_sims)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return np.percentile(ST, [5, 50, 95])

# -----------------------------
# Streamlit App
# -----------------------------
st.title("📈 Options Pricing Dashboard")



# -----------------------------
# Stock Ticker Input
# -----------------------------
tickers_df = pd.read_csv("tickers.csv")
unique_tickers = tickers_df['Symbol'].drop_duplicates()
all_tickers = unique_tickers.tolist()
ticker = st.selectbox(
    "Select Stock Ticker", 
    options=all_tickers, 
    index=all_tickers.index("AAPL") if "AAPL" in all_tickers else 0
)

# -----------------------------
# Expiration Date
# -----------------------------
one_week = datetime.today() + timedelta(days=7)
days_ahead = 4 - one_week.weekday()  # Friday = 4
if days_ahead < 0:
    days_ahead += 7
closest_friday = one_week + timedelta(days=days_ahead)

stock = yf.Ticker(ticker)
available_dates = stock.options

if not available_dates:
    st.error("No option expirations available for this stock.")
else:
    closest_date = min(
        available_dates, 
        key=lambda x: abs(datetime.strptime(x, "%Y-%m-%d").date() - closest_friday.date())
    )
    exp_date = datetime.strptime(closest_date, "%Y-%m-%d").date()
    exp_date = st.date_input("Expiration Date", exp_date)
    st.info(f"Using closest available expiration: {exp_date}")

# -----------------------------
# Monte Carlo sliders
# -----------------------------
n_sims = st.slider("Number of Monte Carlo Simulations", 1000, 50000, 5000, step=100)
conf_level = st.slider("Confidence Level (%)", 80, 99, 95)

# -----------------------------
# Run Simulation Button
# -----------------------------
if st.button("▶️ Run Simulation"):

    S = stock.history(period="1d")["Close"].iloc[-1]  # latest stock price
    r = 0.05  # risk-free rate

    st.subheader(f"Current Stock Price: ${S:.2f}")

    # -----------------------------
    # Styling for Edge (%)
    # -----------------------------
    def style_options(df):
        return (
            df.style
            .applymap(
                lambda x: f"color: {'green' if x > 5 else 'red' if x < -5 else 'black'}" 
                if isinstance(x, (int, float)) else '', 
                subset=['Edge (%)']
            )
            .format({
                'Edge (%)': lambda x: (('+' if x > 0 else '') + f"{x:.2f}".rstrip('0').rstrip('.') + '%') if isinstance(x, (int,float)) else x,
                'Market Price': lambda x: f"${x:.2f}".rstrip('0').rstrip('.'),
                'Theoretical Price': lambda x: f"${x:.2f}".rstrip('0').rstrip('.'),
                'Strike': lambda x: f"${x:.2f}".rstrip('0').rstrip('.')
            })
        )

    try:
        opt_chain = stock.option_chain(exp_date.strftime("%Y-%m-%d"))
        calls = opt_chain.calls
        puts = opt_chain.puts

        # Filter by price bounds
        lower_bound_call = S * 0.3
        upper_bound_call = S * 1
        calls = calls[(calls['strike'] >= lower_bound_call) & (calls['strike'] <= upper_bound_call)]

        lower_bound_put = S * 0.9
        upper_bound_put = S * 1.8
        puts = puts[(puts['strike'] >= lower_bound_put) & (puts['strike'] <= upper_bound_put)]

        # -----------------------------
        # CALL OPTIONS
        # -----------------------------
        sigma_call = calls["impliedVolatility"].iloc[0]
        results_calls = []
        for _, row in calls.iterrows():
            K = row["strike"]
            T = (exp_date - datetime.today().date()).days / 365
            theo = black_scholes(S, K, T, r, sigma_call, "call")
            market = row["lastPrice"]
            edge = ((theo - market) / theo) * 100

            if market < theo * 0.9:
                signal = "UNDER"
            elif theo * 0.9 <= market <= theo * 1.1:
                signal = "FAIR"
            else:
                signal = "OVER"

            results_calls.append({
                "Type": "Call",
                "Strike": K,
                "Market Price": market,
                "Theoretical Price": round(theo, 2),
                "Edge (%)": round(edge, 2),
                "Signal": signal
            })

        cnames_df = pd.read_csv("tickers.csv").drop_duplicates(subset="Symbol")
        company_name = cnames_df.loc[cnames_df['Symbol'] == ticker, 'Name'].values[0]

        df_calls = pd.DataFrame(results_calls)
        st.title(f"Options Chain for {company_name} ({ticker})")
        st.subheader("📊 Call Option Contracts")
        st.dataframe(style_options(df_calls))

        # -----------------------------
        # PUT OPTIONS
        # -----------------------------
        sigma_put = puts["impliedVolatility"].iloc[0]
        results_puts = []
        for _, row in puts.iterrows():
            K = row["strike"]
            T = (exp_date - datetime.today().date()).days / 365
            theo = black_scholes(S, K, T, r, sigma_put, "put")
            market = row["lastPrice"]
            edge = ((theo - market) / theo) * 100

            if market < theo * 0.9:
                signal = "UNDER"
            elif theo * 0.9 <= market <= theo * 1.1:
                signal = "FAIR"
            else:
                signal = "OVER"

            results_puts.append({
                "Type": "Put",
                "Strike": K,
                "Market Price": market,
                "Theoretical Price": round(theo, 2),
                "Edge (%)": round(edge, 2),
                "Signal": signal
            })

        df_puts = pd.DataFrame(results_puts)
        st.subheader("📊 Put Option Contracts")
        st.dataframe(style_options(df_puts))

    except Exception as e:
        st.error(f"Error fetching option chain: {e}")
