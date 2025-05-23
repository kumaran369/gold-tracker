import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

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

def calculate_bollinger_bands(prices, window=20):
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    return upper, lower

def calculate_adx(df, period=14):
    high = df['22K Price'].rolling(window=2).max()
    low = df['22K Price'].rolling(window=2).min()
    close = df['22K Price']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

def improved_signal(df):
    df['5DMA'] = df['22K Price'].rolling(window=5).mean()
    df['RSI'] = calculate_rsi(df['22K Price'])
    df['MACD'], df['Signal_Line'], df['MACD_Histogram'] = calculate_macd(df['22K Price'])
    df['UpperBB'], df['LowerBB'] = calculate_bollinger_bands(df['22K Price'])
    df['ADX'] = calculate_adx(df)
    df['Price Change'] = df['22K Price'].diff()
    df['Signal'] = 'Hold'

    buy_condition = (
        (df['22K Price'] < df['5DMA']) &
        (df['MACD_Histogram'] > 0) &
        (df['MACD_Histogram'].shift(1) < 0) &
        (df['RSI'] < 45) &
        (df['22K Price'] <= df['LowerBB']) &
        (df['ADX'] > 20)
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
    fig.add_trace(go.Scatter(x=df['Date'], y=df['UpperBB'], mode='lines', name='Upper BB', line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['LowerBB'], mode='lines', name='Lower BB', line=dict(color='gray', dash='dot')))
    
    buy_signals = df[df['Signal'] == 'Buy']
    fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['22K Price'],
                             mode='markers', name='Buy Signal',
                             marker=dict(color='green', size=10, symbol='triangle-up')))
    fig.update_layout(title="Gold Price with Buy Signals", xaxis_title="Date", yaxis_title="22K Price")
    st.plotly_chart(fig, use_container_width=True)

def plot_macd(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Signal_Line'], mode='lines', name='Signal Line'))
    fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_Histogram'], name='Histogram'))
    fig.update_layout(title="MACD Indicator", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

def plot_rsi(df):
    if df['RSI'].isnull().all():
        st.info("RSI data not available to plot.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI'))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title="RSI Indicator", xaxis_title="Date", yaxis_title="RSI")
    st.plotly_chart(fig, use_container_width=True)

def plot_adx(df):
    if df['ADX'].isnull().all():
        st.info("ADX data not available to plot.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['ADX'], mode='lines', name='ADX'))
    fig.add_hline(y=20, line_dash="dash", line_color="orange", annotation_text="Trend Threshold", annotation_position="bottom right")
    fig.update_layout(title="ADX Indicator", xaxis_title="Date", yaxis_title="ADX")
    st.plotly_chart(fig, use_container_width=True)

def get_csv_download_link(df):
    return df.to_csv(index=False).encode('utf-8')

def get_excel_download_link(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Gold Data')
    buffer.seek(0)
    return buffer

# Streamlit UI setup
st.set_page_config(page_title="Gold Tracker with Technical Tabs", layout="wide")
st.title("📈 Gold Investment Signal Tracker")

if st.button("🔄 Refresh Now"):
    st.cache_data.clear()

try:
    df = load_data()
    df = improved_signal(df)

    tab1, tab2, tab3, tab4 = st.tabs(["Price & Buy Signals", "MACD", "RSI", "ADX"])

    with tab1:
        plot_price(df)
        st.dataframe(df[['Date', '22K Price', 'RSI', 'MACD_Histogram', 'ADX', 'Signal']].iloc[::-1], use_container_width=True)

    with tab2:
        plot_macd(df)

    with tab3:
        plot_rsi(df)

    with tab4:
        plot_adx(df)

    st.success(f"📌 Total Buy Signals: {df[df['Signal'] == 'Buy'].shape[0]}")

    st.markdown("---")
    st.subheader("📥 Download Reports")

    csv_data = get_csv_download_link(df)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="gold_data_signals.csv",
        mime="text/csv"
    )

    excel_data = get_excel_download_link(df)
    st.download_button(
        label="Download Excel",
        data=excel_data,
        file_name="gold_data_signals.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

except Exception as e:
    st.error(f"❌ Error loading or processing data: {e}")
