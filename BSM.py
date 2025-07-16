# Black Scholes Model Option Pricer
# Evan Gaul - 5/15/25
import streamlit as st
from math import log, sqrt,exp
from scipy.stats import norm
import numpy as np
import plotly.graph_objs as go

def bsm_option_pricer(S, K, T, Sigma, r, option_type = 'call'):
    '''
    Args:
        S: Current Stock Price
        K: Strike Price
        T: Time to expiration
        Sigma: Volatility
        r: Risk-Free Interest rate
        option_type: Call or Put
    Returns:
        A tuple: (Call Price, Put Price)
    '''
    # Check for bad inputs
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or r < 0:
            raise ValueError("Inputs must be positive (except r, which can be zero).")


        d1 = (log(S/K) + (r + 0.5 * Sigma**2) * T) / (Sigma * sqrt(T))
        d2 = d1 - (Sigma * sqrt(T))

        # Call Option Price
        C = S*norm.cdf(d1) - K*(exp(-r * T))*norm.cdf(d2)

        # Put Option Price
        P = K*(exp(-r * T)) * norm.cdf(-d2) - S * norm.cdf(-d1)

        # Delta
        delta = norm.cdf(d1) if option_type.lower() == "call" else norm.cdf(d1) - 1
        return C, P, delta
    except ZeroDivisionError:
        return None
    except ValueError as e:
        st.error(f"Error: {str(e)}")
        return None

def plot_sensitivity(S, K, T, sigma, r, option_type='call'):
    """
    Args:
        S: Current Stock Price
        K: Strike Price
        T: Time to expiration
        Sigma: Volatility
        r: Risk-Free Interest rate
        option_type: Call or Put
    Returns:
        Plotly figure for option price vs. stock pricer
    """

    stock_prices = np.linspace(max(0.1, S * 0.5), S * 1.5, 100)  # Range around S
    prices = []
    for s in stock_prices:
        result = bsm_option_pricer(s, K, T, sigma, r, option_type)
        price = result[0] if option_type.lower() == "call" else result[1]
        prices.append(price if price is not None else 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_prices, y=prices, mode="lines", name=f"{option_type.title()} Price"))
    fig.update_layout(
        title=f"{option_type.title()} Price vs. Stock Price",
        xaxis_title="Stock Price ($)",
        yaxis_title=f"{option_type.title()} Price ($)",
        template="plotly_dark"
    )
    return fig

st.set_page_config(page_title="Black-Scholes Option Pricer", layout="wide")


S = st.sidebar.number_input("Stock Price (S)", min_value=0.01, value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (T, in years)", min_value=0.01, value=1.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05)
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, value=0.2)
option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"], help="Select call or put option")


st.title("Black-Scholes Option Pricer")
st.markdown("""
    This will calculate the price of a European option using the Black-Scholes model.
    Enter and adjust the parameters in the sidebar and click 'Calculate' to see results, the option price, delta, and a plot of the sensitivity.
""")

if st.button("Calculate Option Price"):
    result = bsm_option_pricer(S, K, T, sigma, r, option_type)
    if result:
        call_price, put_price, delta = result
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Results")
            st.success(f"**Call Price**: ${call_price:.2f}")
            st.success(f"**Put Price**: ${put_price:.2f}")
            st.success(f"**Delta**: ${delta:.3f}")
        with col2:
            st.subheader("Sensitivity Plot")
            fig = plot_sensitivity(S, K, T, sigma, r, option_type)
            st.plotly_chart(fig, use_container_width=True)

with st.expander("About the Black-Scholes Model"):
    st.markdown("""
        The Black-Scholes model is a mathematical model for pricing European options. It assumes:
        - The stock price follows a geometric Brownian motion.
        - No dividends are paid
        - Markets are efficient, with constant volatility and risk-free rate
        - No arbitrage oppurtunities
        - Lognormal distribution of returns

        **Formulas**:
        - Call: C = S * N(d1) - K * e^(-rT) * N(d2)
        - Put: P = K e^(-rT) * N(-d2) - S * N(-d1)
        - Where:
          - d1 = (ln(S/K) + (r + σ^2/2) * T) / σ * sqrt(T)
          - d2 = d1 - σ * sqrt(T)
          - N = Cumulative Normal Distribution
    """)
