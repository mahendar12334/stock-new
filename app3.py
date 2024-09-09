#!/usr/bin/env python
# coding: utf-8

# In[12]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import seaborn as sns

# Set up the page configuration
st.set_page_config(page_title="Tesla Stock Portfolio", layout="wide")

# Load data
data = pd.read_csv('Tesla.csv - Tesla.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Define the number of forecast periods
forecast_periods = 10  # Set the forecast period to a fixed value

# Title
st.title('Tesla Stock Analysis & Forecast')

# Sidebar for plot selection
st.sidebar.title("Select Plot")
plot_options = [
    "Tesla Closing Price Over Time",
    "Moving Averages",
    "Daily Returns",
    "Volume Traded",
    "Candlestick Chart with Moving Averages",
    "Cumulative Returns",
    "Bollinger Bands",
    "MACD",
    "ARIMA Forecast",
    "Residual Analysis",
    "GARCH Volatility Model"
]
selected_plot = st.sidebar.radio("Choose a plot to display", plot_options)

# Fit ARIMA model
model_auto = auto_arima(data['Close'], seasonal=False, trace=False)
model_arima = ARIMA(data['Close'], order=model_auto.order)
arima_result = model_arima.fit()

# Generate forecasts
forecast_arima = arima_result.get_forecast(steps=forecast_periods)
forecast_arima_mean = forecast_arima.predicted_mean
forecast_arima_conf_int = forecast_arima.conf_int()

# Extract residuals
residuals = arima_result.resid

# Plotting based on selection
if selected_plot == "Tesla Closing Price Over Time":
    st.subheader("Tesla Closing Price Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.set_title('Tesla Closing Price Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    st.pyplot(fig)

elif selected_plot == "Moving Averages":
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    st.subheader("Moving Averages")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Close'], label='Close Price')
    ax.plot(data['MA50'], label='50-Day MA', color='red')
    ax.plot(data['MA200'], label='200-Day MA', color='green')
    ax.set_title('Tesla Stock Prices with Moving Averages')
    st.pyplot(fig)

elif selected_plot == "Daily Returns":
    data['Daily Return'] = data['Close'].pct_change()
    st.subheader("Daily Returns")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Daily Return'], label='Daily Return')
    ax.set_title('Tesla Daily Returns Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Return')
    st.pyplot(fig)

elif selected_plot == "Volume Traded":
    st.subheader("Volume Traded Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(data.index, data['Volume'], label='Volume Traded')
    ax.set_title('Tesla Volume Traded')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    st.pyplot(fig)

elif selected_plot == "Candlestick Chart with Moving Averages":
    st.subheader("Candlestick Chart with Moving Averages")
    mpf.plot(data[-100:], type='candle', volume=True, mav=(20, 50), style='yahoo', title='Tesla Candlestick Chart')

elif selected_plot == "Cumulative Returns":
    data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
    st.subheader("Cumulative Returns")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Cumulative Return'], label='Cumulative Return', color='teal')
    ax.set_title('Tesla Cumulative Returns Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    st.pyplot(fig)

elif selected_plot == "Bollinger Bands":
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['BB_up'] = data['MA20'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_down'] = data['MA20'] - 2 * data['Close'].rolling(window=20).std()
    st.subheader("Bollinger Bands")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Close'], label='Close Price')
    ax.plot(data['MA20'], label='20-Day MA', color='blue')
    ax.plot(data['BB_up'], label='Upper Bollinger Band', color='red')
    ax.plot(data['BB_down'], label='Lower Bollinger Band', color='green')
    ax.fill_between(data.index, data['BB_down'], data['BB_up'], color='gray', alpha=0.2)
    ax.set_title('Tesla Bollinger Bands')
    st.pyplot(fig)

elif selected_plot == "MACD":
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    st.subheader("MACD (Moving Average Convergence Divergence)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['MACD'], label='MACD', color='blue')
    ax.plot(data['Signal'], label='Signal Line', color='red')
    ax.set_title('Tesla MACD')
    st.pyplot(fig)

elif selected_plot == "ARIMA Forecast":
    st.subheader("ARIMA Forecast")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Historical Data')

    # Generate future dates without 'closed' argument
    future_dates = pd.date_range(start=data.index[-1], periods=forecast_periods + 1)

    ax.plot(future_dates[1:], forecast_arima_mean, color='red', label='ARIMA Forecast')
    ax.fill_between(future_dates[1:], forecast_arima_conf_int.iloc[:, 0], forecast_arima_conf_int.iloc[:, 1], color='red', alpha=0.3)
    ax.set_title('ARIMA Forecast')
    st.pyplot(fig)

elif selected_plot == "Residual Analysis":
    st.subheader("Residual Analysis")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(residuals, kde=True, ax=ax[0])
    ax[0].set_title('Residuals Histogram')
    ax[1].plot(residuals)
    ax[1].set_title('Residuals Over Time')
    st.pyplot(fig)

# GARCH Volatility Model
if selected_plot == "GARCH Volatility Model":
    st.subheader("GARCH Volatility Model")
    
    # Ensure residuals are a pandas Series
    residuals = pd.Series(residuals)
    
    try:
        garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
        garch_result = garch_model.fit(disp='off')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(garch_result.conditional_volatility, color='blue', label='Conditional Volatility')
        ax.set_title('GARCH Model - Conditional Volatility')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred while fitting the GARCH model: {e}")

# Conclusion section
st.subheader("Conclusion")
st.write("""
This analysis provides a comprehensive overview of Tesla stock's historical trends and forecasts using advanced models such as ARIMA and GARCH. 
We analyzed price trends, moving averages, daily returns, and volatility, allowing for deeper insights into Tesla's stock price movements. 
The ARIMA forecast predicts future stock prices, while the GARCH model captures volatility patterns, indicating potential risk levels.
""")


# In[ ]:




