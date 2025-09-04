import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime
import plotly.express as px

# Page config
st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

# Helper functions
@st.cache_data(ttl=3600)
def get_fx_rates():
    url = "https://api.exchangerate-api.com/v4/latest/GBP"
    data = requests.get(url).json()
    return data.get('rates', {})

@st.cache_data(ttl=3600)
def get_historical(ticker):
    """
    Fetch the maximum available price history for 'ticker'.
    Returns a Series of daily closing prices.
    """
    hist = yf.Ticker(ticker).history(period="max")["Close"]
    return hist

# Initialize portfolio in session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['asset', 'value', 'currency'])

# Sidebar: Import/Export & Add Holdings
with st.sidebar:
    st.header("Holdings Management")
    uploaded = st.file_uploader("Import CSV", type='csv')
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.portfolio = df[['asset', 'value', 'currency']]
    if st.button('Export CSV'):
        st.download_button(
            'Download CSV',
            data=st.session_state.portfolio.to_csv(index=False),
            file_name='holdings.csv'
        )
    st.subheader("Add Holding")
    new_asset = st.text_input('Asset Name')
    new_value = st.number_input('Current Value', min_value=0.0, format="%.2f")
    new_currency = st.selectbox('Currency', ['GBP', 'USD', 'EUR', 'INR'])
    if st.button('Add'):
        st.session_state.portfolio = st.session_state.portfolio.append(
            {'asset': new_asset, 'value': new_value, 'currency': new_currency},
            ignore_index=True
        )

st.title("Portfolio Dashboard")

# Tabs
tabs = st.tabs([
    "Overview", "Risk Metrics", "Risk Analysis",
    "Holdings", "Currency Analysis", "Monte Carlo"
])

# Overview Tab
with tabs[0]:
    st.header("Portfolio Overview")
    df = st.session_state.portfolio.copy()
    fx = get_fx_rates()
    df['value_gbp'] = df.apply(
        lambda r: r['value'] * (fx.get(r['currency'], 1) if r['currency'] != 'GBP' else 1),
        axis=1
    )
    total = df['value_gbp'].sum()
    st.metric("Total Portfolio Value (GBP)", f"£{total:,.2f}")
    fig = px.bar(df, x='asset', y='value_gbp', title='Value by Asset (GBP)')
    st.plotly_chart(fig, use_container_width=True)
    st.table(df[['asset', 'value', 'currency', 'value_gbp']].reset_index(drop=True))

# Risk Metrics Tab
with tabs[1]:
    st.header("Risk Metrics")
    df = st.session_state.portfolio.copy()
    fx = get_fx_rates()
    df['value_gbp'] = df.apply(
        lambda r: r['value'] * (fx.get(r['currency'], 1) if r['currency'] != 'GBP' else 1),
        axis=1
    )
    metrics = []
    for _, r in df.iterrows():
        hist = get_historical(r['asset'])
        if not hist.empty:
            ret = hist.pct_change().dropna()
            vol = ret.std() * np.sqrt(252)
            ann = ret.mean() * 252
            metrics.append({
                'Asset': r['asset'],
                'Annual Volatility': vol,
                'Annual Return': ann
            })
    st.table(pd.DataFrame(metrics).set_index('Asset'))

# Risk Analysis Tab
with tabs[2]:
    st.header("Risk Analysis")
    if metrics:
        risk_df = pd.DataFrame(metrics)
        fig = px.scatter(
            risk_df,
            x='Annual Volatility',
            y='Annual Return',
            text='Asset',
            size='Annual Return',
            title='Risk-Return Profile'
        )
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No risk data available. Enter holdings and ensure proxies are recognized.")

# Holdings Tab
with tabs[3]:
    st.header("Individual Holdings")
    edited = st.experimental_data_editor(st.session_state.portfolio, num_rows="dynamic")
    st.session_state.portfolio = edited

# Currency Analysis Tab
with tabs[4]:
    st.header("Currency Analysis")
    df = st.session_state.portfolio.copy()
    fx = get_fx_rates()
    df['value_gbp'] = df.apply(
        lambda r: r['value'] * (fx.get(r['currency'], 1) if r['currency'] != 'GBP' else 1),
        axis=1
    )
    exposure = df.groupby('currency')['value_gbp'].sum().reset_index()
    fig = px.pie(
        exposure,
        values='value_gbp',
        names='currency',
        title='Currency Exposure (GBP)'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.table(exposure)

# Monte Carlo Tab
with tabs[5]:
    st.header("Monte Carlo Simulation")
    df = st.session_state.portfolio.copy()
    fx = get_fx_rates()
    df['value_gbp'] = df.apply(
        lambda r: r['value'] * (fx.get(r['currency'], 1) if r['currency'] != 'GBP' else 1),
        axis=1
    )
    total = df['value_gbp'].sum()
    years = st.slider("Projection Years", 1, 10, 5)
    sims = st.slider("Number of Simulations", 100, 10000, 1000)
    mus, sig2 = [], []
    for _, r in df.iterrows():
        hist = get_historical(r['asset'])
        if not hist.empty:
            ret = hist.pct_change().dropna()
            mus.append(ret.mean() * 252)
            sig2.append((ret.std() * np.sqrt(252))**2)
    port_mu = sum(np.array(mus) * df['value_gbp'] / total)
    port_sigma = np.sqrt(sum(np.array(sig2) * (df['value_gbp'] / total)**2))
    results = []
    for _ in range(sims):
        vals = np.random.normal(port_mu, port_sigma, years)
        results.append(total * np.prod(1 + vals))
    results = np.array(results)
    df_res = pd.Series(results)
    fig = px.histogram(df_res, nbins=50, title='Simulated Portfolio Value Distribution')
    st.plotly_chart(fig, use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Value", f"£{df_res.mean():,.0f}")
    col2.metric("Median Value", f"£{np.median(df_res):,.0f}")
    col3.metric("5th Percentile", f"£{np.percentile(df_res,5):,.0f}")

# Footer
st.markdown("---")
st.write("Data Sources: Manual inputs, Yahoo Finance (full history), ExchangeRate-API (FX rates)")
