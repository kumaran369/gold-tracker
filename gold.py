import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Public Google Sheet CSV URL
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSkhCx53_fngzpmxn_1h-I3Cr_JwzObE96h_cYgv652wz7yDfyDkV_P7kiXhrirnDwABdmBxM3ZjrO1/pub?gid=0&single=true&output=csv"

# Cached function to load data
@st.cache_data(ttl=600)  # Cache expires every 10 minutes
def load_data():
    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def improved_signal(df):
    df['5DMA'] = df['22K Price'].rolling(window=5).mean()
    df['RSI'] = calculate_rsi(df['22K Price'])
    df['MACD'], df['Signal_Line'], df['MACD_Histogram'] = calculate_macd(df['22K Price'])
    df['Price Change'] = df['22K Price'].diff()
    df['Signal'] = 'Hold'

    buy_condition = (
        (df['22K Price'] < df['5DMA']) &
        (df['MACD_Histogram'] > 0) &
        (df['MACD_Histogram'].shift(1) < 0) &
        (df['RSI'] < 45) &
        (df['22K Price'] == df['22K Price'].rolling(window=7).min())
    )

    avoid_condition = (
        (df['Price Change'] > 20) &
        (df['MACD'] < df['MACD'].shift(1)) &
        (df['RSI'] > 65)
    )

    df.loc[buy_condition, 'Signal'] = 'Buy'
    df.loc[avoid_condition, 'Signal'] = 'Avoid'

    # Ensure at least one Buy per week
    df['Week'] = df['Date'].dt.isocalendar().week
    for week in df['Week'].unique():
        week_df = df[df['Week'] == week]
        if 'Buy' not in week_df['Signal'].values:
            min_idx = week_df['22K Price'].idxmin()
            df.at[min_idx, 'Signal'] = 'Buy'

    return df

def plot_price(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['22K Price'], mode='lines+markers', name='22K Price'))
    buy_signals = df[df['Signal'] == 'Buy']
    fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['22K Price'],
                             mode='markers', name='Buy Signal',
                             marker=dict(color='green', size=10, symbol='triangle-up')))
    fig.update_layout(title="Gold Price & Buy Signals", xaxis_title="Date", yaxis_title="22K Price")
    st.plotly_chart(fig, use_container_width=True)

# Streamlit UI setup
st.set_page_config(page_title="Gold Tracker", layout="wide")
st.title("📈 Gold Investment Signal Tracker")

# Refresh button
refresh = st.button("🔄 Refresh Data")
if refresh:
    st.cache_data.clear()

try:
    df = load_data()
    df = improved_signal(df)
    plot_price(df)
    st.dataframe(df[['Date', '22K Price', 'RSI', 'MACD_Histogram', 'Signal']].iloc[::-1], use_container_width=True)


    num_buys = df[df['Signal'] == 'Buy'].shape[0]
    st.success(f"📌 Total Buy Signals: {num_buys}")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
