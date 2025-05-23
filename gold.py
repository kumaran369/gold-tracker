import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Google Sheet CSV URL
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSkhCx53_fngzpmxn_1h-I3Cr_JwzObE96h_cYgv652wz7yDfyDkV_P7kiXhrirnDwABdmBxM3ZjrO1/pub?gid=0&single=true&output=csv"

@st.cache_data(ttl=600)
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

    df['Week'] = df['Date'].dt.isocalendar().week
    for week in df['Week'].unique():
        week_df = df[df['Week'] == week]
        if 'Buy' not in week_df['Signal'].values:
            min_idx = week_df['22K Price'].idxmin()
            df.at[min_idx, 'Signal'] = 'Buy'

    return df

def apply_ml_strategy(df):
    df['RSI'] = calculate_rsi(df['22K Price'])
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['22K Price'])
    df['Price Change'] = df['22K Price'].pct_change()
    df['Future Return'] = df['22K Price'].shift(-1) > df['22K Price']

    df_ml = df.dropna().copy()
    features = ['RSI', 'MACD', 'MACD_Hist', 'Price Change']
    X = df_ml[features]
    y = df_ml['Future Return'].astype(int)

    if len(X) < 10:
        st.warning("⚠️ Not enough data for ML model. Showing rule-based signals only.")
        df['ML_Signal'] = "N/A"
        return df

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    df_ml['ML_Buy_Pred'] = model.predict(X)

    st.subheader("🧠 ML Model Performance")
    st.text(classification_report(y_test, model.predict(X_test)))

    df['ML_Signal'] = df_ml['ML_Buy_Pred']
    df['ML_Signal'] = df['ML_Signal'].fillna(0).map({0: "Don't Buy", 1: "Buy"})
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

st.set_page_config(page_title="Gold Tracker", layout="wide")
st.title("📈 Gold Investment Signal Tracker")

refresh = st.button("🔄 Refresh Data")
if refresh:
    st.cache_data.clear()

try:
    df = load_data()
    df = improved_signal(df)
    df = apply_ml_strategy(df)
    plot_price(df)

    st.dataframe(df[['Date', '22K Price', 'RSI', 'MACD_Histogram', 'Signal', 'ML_Signal']].iloc[::-1], use_container_width=True)

    num_buys = df[df['Signal'] == 'Buy'].shape[0]
    st.success(f"📌 Total Buy Signals: {num_buys}")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
