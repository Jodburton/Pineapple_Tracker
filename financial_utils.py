
#Common imports 
import pandas as pd
import yfinance as yf
import talib as ta
import requests

#Getting stock data from yfinance
def get_stock_data(ticker, timeframe, start_date, end_date):
    valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

    if timeframe not in valid_intervals:
        raise ValueError("Invalid timeframe selected. Valid intervals: " + ", ".join(valid_intervals))

    start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date, interval=timeframe)
    return data

#Obtaining financial metrics
def get_financial_metric(data, metric):
    if metric == "RSI":
        return ta.RSI(data['Close'])
    elif metric == "Bollinger Bands":
        upper, middle, lower = ta.BBANDS(data['Close'])
        return upper, middle, lower
    elif metric == "MACD":
        macd_line, signal_line, macd_hist = ta.MACD(data['Close'])
        return macd_line, signal_line, macd_hist
    elif metric == "OBV":
        return ta.OBV(data['Close'], data['Volume'])
    elif metric == "SMA":
        return ta.SMA(data['Close'])
    elif metric == "EMA":
        return ta.EMA(data['Close'])
    else:
        raise ValueError(f"Metric {metric} not implemented.")
        
#Fetching news articles 
api_key = input("Please enter your NewsAPI key: ")  # Replace with your actual API key

#Fetching news articles 
def fetch_articles(ticker, api_key=api_key, language="en", page_size=3):
    base_url = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": api_key,
        "q": ticker,
        "language": language,
        "pageSize": page_size
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    if data['status'] == "ok":
        return data['articles']
    else:
        return []
