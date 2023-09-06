#!/usr/bin/env python
# coding: utf-8

# ## Interactive Financial Dashboard

# #### Imports

# In[ ]:


#Request
import requests

#Datetime
from datetime import datetime

#Yfinance
import yfinance as yf
from yahoo_fin import stock_info

#Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#Dash
import dash
from dash import Dash
from dash import dash_table
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_daq as daq

#Data Manipulation
import pandas as pd
import talib as ta
import numpy as np

#Scipy
from scipy.optimize import minimize

#Layout Imports
from constants import subplot_indicator_descriptions, overlying_indicator_descriptions, pie_chart_colors, max_roi, min_roi, external_stylesheets
from financial_utils import get_stock_data, get_financial_metric, fetch_articles
from layout_components.final_app_layout import app_layout



# # Dash Web Application

# #### Initialize the Dash app

# In[ ]:


# Initialize the Dash app
# ------------------------

app = dash.Dash(__name__, 
                suppress_callback_exceptions=True,
                external_stylesheets=external_stylesheets
               )


# #### App Layout

# In[ ]:


# Create App Layout
# -----------------

app.layout = app_layout


# #### Portfolio Tracker

# In[ ]:


# Pineapple Tracker
# -----------------

@app.callback(
    [Output('assets-table', 'data'), Output('error-message', 'children')],
    [Input('add-asset-button', 'n_clicks'), Input('remove-asset-button', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('shares-input', 'value'),
     State('purchase-date-input', 'value'),
     State('price-bought-input', 'value'),
     State('assets-table', 'data'),
     State('assets-table', 'selected_rows')]
)
def update_table(add_n_clicks, remove_n_clicks, ticker, shares, purchase_date, purchase_price, table_data, selected_rows):
#     """
#     Update the assets table based on user actions.
    
#     Parameters:
#     - add_n_clicks (int): Number of times the 'Add Asset' button was clicked.
#     - remove_n_clicks (int): Number of times the 'Remove Asset' button was clicked.
#     - ticker (str): Ticker symbol of the asset.
#     - shares (float): Number of shares of the asset.
#     - purchase_date (str): Purchase date of the asset.
#     - purchase_price (float): Purchase price of the asset.
#     - table_data (list of dict): Existing data in the assets table.
#     - selected_rows (list of int): Rows selected in the assets table.
    
#     Returns:
#     - list of dict: Updated data for the assets table.
#     - str: Error message, if any.
#     """
    ctx = dash.callback_context

    if not ctx.triggered:
        return dash.no_update, ""
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'add-asset-button':
        # Validate date format
        try:
            purchase_date_dt = datetime.strptime(purchase_date, '%Y-%m-%d').date()
        except ValueError:
            return dash.no_update, "Invalid date format. Please use YYYY-MM-DD."

        # Check if purchase_date is a future date
        today = datetime.today().date()
        if purchase_date_dt > today:
            return dash.no_update, "Invalid date. Please ensure the date is not in the future."

        try:
            # Fetch the current price and ticker information using yfinance
            current_ticker_data = yf.Ticker(ticker)
            current_price = current_ticker_data.history(period="1d")["Close"].iloc[0]
            ticker_info = current_ticker_data.info

            # Check the quoteType for cryptocurrency
            if ticker_info.get('quoteType') == 'CRYPTOCURRENCY':
                sector = 'Crypto'
            else:
                sector = ticker_info.get('sector', 'Unknown')

            # Calculate Initial Value and Current Value
            initial_value = shares * purchase_price
            current_value = shares * current_price

            # Calculate ROI
            roi = ((current_value - initial_value) / initial_value) * 100

            # If ticker already exists in table_data, remove its previous entry
            table_data = [data for data in table_data if data['Ticker'] != ticker]

            # Append new asset data to table data
            new_data = {
                'Ticker': ticker,
                'Shares': shares,
                'Purchase Date': purchase_date,
                'Purchase Price': purchase_price,
                'Initial Value': round(initial_value, 2),
                'Current Value': round(current_value, 2),
                'ROI (%)': round(roi, 2),
                'Sector': sector
            }
            table_data.append(new_data)

            return table_data, ""  # Return empty string for error message if no error

        except Exception as e:
            return dash.no_update, f"Error: {e}. Please check the input information."

    elif button_id == 'remove-asset-button':
        if selected_rows:
            # Remove the selected rows from table_data
            table_data = [data for i, data in enumerate(table_data) if i not in selected_rows]
            return table_data, ""
        else:
            return dash.no_update, "Please select rows to remove."

    else:
        return dash.no_update, ""

    
# Call back for the gauge chart - section 3 visusalisations #1 - ROI Gauge Chart

# Function to compute the overall ROI and total portfolio value
def compute_roi_and_total_value(table_data):
#     """
#     Compute the overall ROI and total portfolio value.
#     :param table_data: List of dictionaries containing asset data
#     :return: Tuple containing total ROI and total current value
#     """
    try:
        # Convert table_data to DataFrame
        df = pd.DataFrame(table_data)

        # If the DataFrame is empty, return default values
        if df.empty:
            return 0, 0
        
        # Calculate the total initial and current values
        total_initial_value = df['Initial Value'].sum()
        total_current_value = df['Current Value'].sum()

        # Calculate the total ROI
        if total_initial_value == 0:  # Avoid division by zero
            total_roi = 0
        else:
            total_roi = ((total_current_value - total_initial_value) / total_initial_value) * 100

        return total_roi, total_current_value

    except Exception as e:
        print(f"Error in compute_roi_and_total_value: {e}")
        return 0, 0


@app.callback(
    [Output('gauge-chart', 'figure'), Output('current-value-kpi', 'children'), Output('initial-investment-kpi', 'children')],
    [Input('assets-table', 'data'), Input('add-asset-button', 'n_clicks')]
)
def update_gauge_and_kpis(table_data, n_clicks):
#     """
#     Update the gauge chart and KPIs based on the current state of the assets table.
    
#     Parameters:
#     - table_data (list of dict): The current data in the assets table.
#     - n_clicks (int): The number of times the 'add-asset-button' has been clicked.
    
#     Returns:
#     - fig (plotly.graph_objs.Figure): The updated gauge chart.
#     - total_current_value (str): The total current value of the portfolio, formatted as currency.
#     - total_initial_investment_str (str): The total initial investment, formatted as currency.
#     """
    # Compute ROI and Total Current Value
    current_roi, total_val = compute_roi_and_total_value(table_data)
    total_current_value = f"${total_val:,.2f}"  # Format as currency with commas

    # Compute Total Initial Investment directly from table_data
    total_initial_investment = sum([asset.get('Initial Value', 0) for asset in table_data])
    total_initial_investment_str = f"${total_initial_investment:,.2f}"

    # Use the current ROI
    global max_roi, min_roi
    if current_roi > max_roi:
        max_roi = current_roi
    if current_roi < min_roi:
        min_roi = current_roi

    # Calculate tick values based on your range
    tick_interval = (max_roi - 0) / 10
    tick_values = list(range(0, max_roi + 1, int(tick_interval)))

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_roi,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {
                'range': [0, max_roi],  # Start from 0
                'tickvals': tick_values,
                'ticktext': [str(int(tick)) for tick in tick_values],
                'tickfont': {'color': 'LightGrey', 'size': 12},
                'tickcolor': 'LightGrey'
            }
    
        },
        title={'text': "ROI", 'font': {'size': 42, 'color': 'LightGrey'}},
        number={'suffix': "%", 'font': {'color': 'LightGrey'}}
    ))

    fig.update_layout(
        paper_bgcolor="black",
        plot_bgcolor="#333",
        margin=dict(t=20, b=40, l=50, r=50),
        autosize=True
    )

    return fig, total_current_value, total_initial_investment_str

# Call back for the pie chart - section 3 visusalisations #2 - Portfolio Pie Chart

# Callback for the pie chart
@app.callback(
    Output('pie-chart', 'figure'),
    [Input('assets-table', 'data'), 
     Input('add-asset-button', 'n_clicks'), 
     Input('value-toggle', 'value')]
)
def update_pie_chart(table_data, n_clicks, value_type):
#     """
#     Update the pie chart based on the selected value type.
#     :param table_data: List of dictionaries containing asset data
#     :param n_clicks: Number of clicks on the add-asset-button
#     :param value_type: Type of value to display ('current', 'initial', 'sector')
#     :return: Updated pie chart figure
#     """
    # Depending on the value_type, extract the relevant values for the pie chart
    if value_type == 'current':
        labels = [asset['Ticker'] for asset in table_data]
        values = [asset['Current Value'] for asset in table_data]
    elif value_type == 'initial':
        labels = [asset['Ticker'] for asset in table_data]
        values = [asset['Initial Value'] for asset in table_data]
    elif value_type == 'sector':
        # Group data by sector and sum the values
        sectors = {}
        for asset in table_data:
            if asset['Sector'] in sectors:
                sectors[asset['Sector']] += asset['Current Value']
            else:
                sectors[asset['Sector']] = asset['Current Value']
        labels = list(sectors.keys())
        values = list(sectors.values())

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        textinfo='percent+label',
        marker=dict(colors=pie_chart_colors),
    )])
    
    fig.update_layout(
        margin=dict(t=20, b=40, l=50, r=50),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            font=dict(size=12, color='lightGrey')
        ),
        paper_bgcolor="black",
        plot_bgcolor="#333",
        font=dict(size=14, color='lightGrey'),
        autosize=True
    )
    
    return fig


# Call back for the Portfolio Perf. Graph - section 3 visusalisations #3 - Portfolio Performance History Chart

@app.callback(
    Output('portfolio-performance-graph', 'figure'),
    Input('timeframe-dropdown', 'value'),
    State('assets-table', 'data')
)
def update_graph(timeframe, table_data):
#     """
#     Update the portfolio performance graph based on the selected timeframe and the current portfolio data.
    
#     Parameters:
#     - timeframe (str): The selected timeframe ('3M', '6M', '1Y').
#     - table_data (list of dict): The current portfolio data.
    
#     Returns:
#     - fig (dict): The updated figure for the portfolio performance graph.
#     """
    # Find the earliest purchase date in the portfolio
    dates = [datetime.strptime(data['Purchase Date'], '%Y-%m-%d').date() for data in table_data]
    start_date = min(dates)

    # Initialize an empty dataframe to hold portfolio values
    portfolio_value_df = pd.DataFrame()

    # For each ticker in the portfolio:
    for data in table_data:
        ticker = data['Ticker']
        shares = data['Shares']
        asset_purchase_date = pd.Timestamp(datetime.strptime(data['Purchase Date'], '%Y-%m-%d'))

        # Fetch weekly closing prices
        df = yf.download(ticker, start=start_date, end=datetime.today().date(), interval='1wk')

        # Before the asset's purchase date, its value is 0
        df[f'{ticker}_Value'] = 0
        # From the asset's purchase date onwards, its value is its closing price times the number of shares
        df.loc[df.index >= asset_purchase_date, f'{ticker}_Value'] = df['Close'] * shares
        
        # Append the asset's weekly value to the portfolio_value_df
        if portfolio_value_df.empty:
            portfolio_value_df = df[[f'{ticker}_Value']]
        else:
            portfolio_value_df = pd.concat([portfolio_value_df, df[f'{ticker}_Value']], axis=1)

    # Calculate total weekly portfolio value
    portfolio_value_df['Total_Value'] = portfolio_value_df.sum(axis=1)

    # Ensure that the index is of type DatetimeIndex
    portfolio_value_df.index = pd.to_datetime(portfolio_value_df.index)

    # Filter the data according to the selected timeframe
    end_date = portfolio_value_df.index[-1]  # Get the last date in the DataFrame
    if timeframe == '3M':
        start_date_filter = end_date - pd.DateOffset(months=3)
    elif timeframe == '6M':
        start_date_filter = end_date - pd.DateOffset(months=6)
    elif timeframe == '1Y':
        start_date_filter = end_date - pd.DateOffset(months=12)
    else:
        start_date_filter = portfolio_value_df.index[0]  # Get the first date

    final_df = portfolio_value_df.loc[start_date_filter:end_date, 'Total_Value']

    # Plotting the graph
    fig = {
            'data': [
                {
                    'x': final_df.index,
                    'y': final_df.values,
                    'type': 'line',
                    'name': 'Portfolio Value',
                    'line': {'color': 'rgba(254, 234, 99, 1)'}
                },
            ],
            'layout': {
                'paper_bgcolor': 'black', 
                'plot_bgcolor': 'black', 
                'xaxis': {
                    'tickfont': {'color': 'lightGrey'},
                    'gridcolor': 'rgb(70, 70, 70)'  
                },
                'yaxis': {
                    'title': 'Portfolio Value ($)',
                    'tickfont': {'color': 'lightGrey'},
                    'gridcolor': 'rgb(70, 70, 70)'  
                }
            }
        }
    
    return fig


# #### Portfolio Analysis

# In[ ]:


# Pineapple Analysis
# -----------------

# Fetch stock data from Yahoo Finance
def fetch_stock_data(ticker, purchase_date):
#     """
#     Fetch stock data for a single ticker.
    
#     Parameters:
#     - ticker (str): Stock ticker symbol.
#     - purchase_date (str): Purchase date of the stock.
    
#     Returns:
#     - pd.Series: Adjusted close prices.
#     """
    try:
        end_date = datetime.today().strftime('%Y-%m-%d')
        stock_data = yf.download(ticker, start=purchase_date, end=end_date)['Adj Close']
        print(f"Fetched data for {ticker} from {purchase_date} to {end_date}: {stock_data.head()}")
        return stock_data
    except Exception as e:
        print(f"Error in fetching data for {ticker}: {e}")
        
# Fetch stock data frames
def fetch_stock_data_frames(tickers, oldest_purchase_date):
#     """
#     Fetch stock data for multiple tickers.
    
#     Parameters:
#     - tickers (list): List of stock ticker symbols.
#     - oldest_purchase_date (str): Oldest purchase date among all stocks.
    
#     Returns:
#     - dict: Dictionary containing stock data for each ticker.
#     """
    stock_data_frames = {}
    for ticker in tickers:
        stock_data_frames[ticker] = fetch_stock_data(ticker, oldest_purchase_date)
    return stock_data_frames

# Process stock data
def process_stock_data(stock_data_frames):
#     """
#     Process stock data to align it for analysis.
    
#     Parameters:
#     - stock_data_frames (dict): Dictionary containing stock data for each ticker.
    
#     Returns:
#     - pd.DataFrame: Processed stock data.
#     """
    stock_data = pd.concat(stock_data_frames.values(), axis=1, keys=stock_data_frames.keys())
    stock_data.ffill(inplace=True)
    stock_data.dropna(inplace=True)
    return stock_data

# Main function
def fetch_and_process_stock_data(tickers, purchase_dates, oldest_purchase_date, risk_free_rate_annual, button_id):
#     """
#     Fetch and process stock data.
    
#     Parameters:
#         tickers (list): List of stock tickers.
#         purchase_dates (list): List of purchase dates for the stocks.
#         oldest_purchase_date (str): Oldest purchase date among the stocks.
#         risk_free_rate_annual (float): Annual risk-free rate.
#         button_id (str): ID of the button clicked.
    
#     Returns:
#         tuple: Processed stock data, expected returns, covariance matrix, and risk-free rate.
#     """
    stock_data_frames = fetch_stock_data_frames(tickers, oldest_purchase_date)
    stock_data = process_stock_data(stock_data_frames)
    expected_returns, covariance_matrix = calculate_expected_returns_and_covariance(stock_data)
    risk_free_rate = risk_free_rate_annual if button_id == 'annualized-button' else risk_free_rate_annual / 252
    return stock_data, expected_returns, covariance_matrix, risk_free_rate

# Fetch the risk-free rate from Yahoo Finance
def fetch_risk_free_rate():
#     """
#     Fetch the risk-free rate from Yahoo Finance.
    
#     Returns:
#         float: Risk-free rate in decimal form.
#     """
    try:
        tbill = yf.Ticker("^IRX")
        tbill_info = tbill.info
        return tbill_info['previousClose'] / 100  # Convert to decimal
    except Exception as e:
        print(f"An error occurred while fetching the risk-free rate: {e}")
        return 0.01

# Calculate expected returns and covariance matrix
def calculate_expected_returns_and_covariance(stock_data, annualized=True):
#     """
#     Calculate expected returns and covariance matrix for the stocks.
    
#     Parameters:
#         stock_data (DataFrame): Stock data.
#         annualized (bool): Whether to annualize the returns and covariance.
    
#     Returns:
#         tuple: Expected returns and covariance matrix.
#     """
    returns = stock_data.pct_change()
    expected_daily_returns = returns.mean()
    daily_covariance_matrix = returns.cov()
    
    if annualized:
        expected_annual_returns = (1 + expected_daily_returns)**252 - 1
        annual_covariance_matrix = daily_covariance_matrix * 252
        return expected_annual_returns, annual_covariance_matrix
    else:
        return expected_daily_returns, daily_covariance_matrix

# Optimize the portfolio to maximize the Sharpe ratio
def optimize_portfolio(expected_returns, covariance_matrix, risk_free_rate=0.01, annualized=True):
#     """
#     Optimize the portfolio to maximize the Sharpe ratio.
    
#     Parameters:
#         expected_returns (Series): Expected returns for each asset.
#         covariance_matrix (DataFrame): Covariance matrix of the assets.
#         risk_free_rate (float): Risk-free rate.
#         annualized (bool): Whether the data is annualized.
    
#     Returns:
#         np.array: Optimal asset weights.
#     """
    num_assets = len(expected_returns)
    initial_weights = [1./num_assets for _ in range(num_assets)]
    
    def objective(weights): 
        weights = np.array(weights)
        return - (np.sum(expected_returns * weights) - risk_free_rate) / np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    solution = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return solution.x

# Calculate optimal asset allocation
def calculate_optimal_weights(expected_returns, covariance_matrix, risk_free_rate):
#     """
#     Calculate the optimal asset weights.
    
#     Parameters:
#         expected_returns (Series): Expected returns for each asset.
#         covariance_matrix (DataFrame): Covariance matrix of the assets.
#         risk_free_rate (float): Risk-free rate.
    
#     Returns:
#         np.array: Optimal asset weights.
#     """
    return optimize_portfolio(expected_returns, covariance_matrix, risk_free_rate)

# Calculate efficient frontier points
def calculate_efficient_frontier_points(expected_returns, covariance_matrix):
#     """
#     Calculate the efficient frontier points.
    
#     Parameters:
#         expected_returns (Series): Expected returns for each asset.
#         covariance_matrix (DataFrame): Covariance matrix of the assets.
    
#     Returns:
#         np.array: Efficient frontier points.
#     """
    num_portfolios = 1000
    results = np.zeros((4, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(expected_returns))
        weights /= np.sum(weights)
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        results[0,i] = portfolio_return
        results[1,i] = portfolio_volatility 
        results[2,i] = results[0,i] / results[1,i]
    return results

# Plot the efficient frontier
def plot_efficient_frontier(expected_returns, optimal_weights, covariance_matrix, risk_free_rate=0.01, annualized=True):
    portfolio_return = np.sum(expected_returns * optimal_weights)
    portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights)))
    results = calculate_efficient_frontier_points(expected_returns, covariance_matrix)
    
    fig = go.Figure()

    # Plotting the user's portfolio point
    fig.add_trace(go.Scatter(x=[portfolio_volatility], y=[portfolio_return], 
                             mode='markers', 
                             name='Your Portfolio',
                             marker=dict(size=14, color='rgba(254, 234, 99, 1)')))

    return_annual = results[0]
    risk_annual = results[1]

    # Plotting efficient frontier
    fig.add_trace(go.Scatter(x=risk_annual, y=return_annual, 
                             mode='markers', 
                             name='Efficient Frontier',
                             marker=dict(color='rgba(255,165,0, 0.5)')))
    # Capital Market Line and tangent point (market portfolio)
    sharpe_max_index = np.argmax(results[2])
    sm_volatility = results[1][sharpe_max_index] 
    sm_return = results[0][sharpe_max_index] 

    fig.add_trace(go.Scatter(x=[0, sm_volatility], y=[risk_free_rate, sm_return], 
                             mode='lines', 
                             name='Capital Market Line',
                             line=dict(color='rgb(70, 130, 180, 1)')))
    fig.add_trace(go.Scatter(x=[sm_volatility], y=[sm_return], 
                             mode='markers', 
                             name='Market Portfolio', 
                             marker=dict(size=10, color='rgba(255, 0, 0, 1)')))
    
    fig.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            xaxis={
                'title': {
                    'text': 'Volatility (Standard Deviation)',
                    'font': {'color': 'lightGrey'}
                },
                'tickfont': {'color': 'lightGrey'},
                'gridcolor': 'rgb(70, 70, 70)',
                'showline': True,
            },
            yaxis={
                'title': {
                    'text': 'Expected Returns',
                    'font': {'color': 'lightGrey'}
                },
                'tickfont': {'color': 'lightGrey'},
                'gridcolor': 'rgb(70, 70, 70)',
                'showline': True,
            },
            autosize=True,
    )

    return fig

#Plot the Optimal Asset Allocation chart
def plot_asset_allocation(optimal_weights, tickers):
    # Create the figure
    fig = go.Figure(data=[go.Pie(
        labels=tickers,
        values=optimal_weights,
        hole=.3,
        textinfo='percent+label',
    )])

    # Update the layout
    fig.update_layout(
        margin=dict(t=40, b=40, l=50, r=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color='lightGrey')
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(size=14, color='lightGrey'),
        autosize=True
    )

    return fig

# Calculate and plot Sharpe Ratios
def plot_sharpe_ratio(expected_returns, covariance_matrix, risk_free_rate, optimal_weights, tickers, user_sharpe_ratios):
    # Calculate Sharpe Ratios for individual assets
    sharpe_ratios = (expected_returns - risk_free_rate) / np.sqrt(np.diag(covariance_matrix))
    
    # Calculate Sharpe Ratio for the portfolio
    portfolio_return = np.dot(optimal_weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights)))
    portfolio_sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Include Portfolio in Sharpe Ratios
    sharpe_ratios['Portfolio'] = portfolio_sharpe_ratio
    
    # Create Sharpe Ratio Bar Chart
    user_sharpe_ratios_rounded = user_sharpe_ratios.round(2)
    
    # Create Sharpe Ratio Bar Chart
    sharpe_ratio_fig = go.Figure(data=[
        go.Bar(x=user_sharpe_ratios_rounded.index, 
               y=user_sharpe_ratios_rounded.values,
               width=0.2,
               marker_color=['rgba(0, 128, 0, 0.7)' if x >= 1 else 'rgba(255, 0, 0, 0.7)' for x in user_sharpe_ratios_rounded.values]
        )
    ])
    
    sharpe_ratio_fig.update_layout(
        xaxis=dict(title='Assets',
                   showgrid=False),
        yaxis=dict(title='Sharpe Ratio',
                   showgrid=False),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(size=14, color='lightGrey'),
        autosize=True,
        annotations=[
            dict(
                x=xi,
                y=yi,
                text=str(yi),
                xanchor='center',
                yanchor='bottom',
                showarrow=False,
            ) for xi, yi in zip(user_sharpe_ratios_rounded.index, user_sharpe_ratios_rounded.values)
        ]
    )
    
    return sharpe_ratio_fig

# Generate all the plots
def generate_plots(expected_returns, optimal_weights, covariance_matrix, risk_free_rate, tickers, user_sharpe_ratios):
    efficient_frontier_fig = plot_efficient_frontier(expected_returns, optimal_weights, covariance_matrix, risk_free_rate)
    asset_allocation_fig = plot_asset_allocation(optimal_weights, tickers)
    sharpe_ratio_fig = plot_sharpe_ratio(expected_returns, covariance_matrix, risk_free_rate, optimal_weights, tickers, user_sharpe_ratios)
    return efficient_frontier_fig, asset_allocation_fig, sharpe_ratio_fig

# Main callback to update plots
@app.callback(
    [Output('efficient-frontier-plot', 'figure'),
     Output('asset-allocation-chart', 'figure'),
     Output('sharpe-ratio-bar-chart', 'figure')],
    [Input('assets-table', 'data'),
     Input('add-asset-button', 'n_clicks'),
     Input('remove-asset-button', 'n_clicks'),
     Input('annualized-button', 'n_clicks'),
     Input('non-annualized-button', 'n_clicks')]
)
def update_plots(table_data, add_clicks, remove_clicks, annualized_clicks, non_annualized_clicks):
    # Determine which button was clicked
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Fetch the annual risk-free rate
    risk_free_rate_annual = fetch_risk_free_rate()
    
    # Extract tickers, purchase dates, and purchase prices from the table
    tickers = [data['Ticker'] for data in table_data]
    purchase_dates = [data['Purchase Date'] for data in table_data]
    purchase_prices = [data['Purchase Price'] for data in table_data]
    oldest_purchase_date = min(purchase_dates)
    
    # Fetch and process stock data
    stock_data, expected_returns, covariance_matrix, risk_free_rate = fetch_and_process_stock_data(tickers, purchase_dates, oldest_purchase_date, risk_free_rate_annual, button_id)

    # Calculate the current prices of the assets
    current_prices = stock_data.iloc[-1]
    
    # Calculate the returns based on purchase prices and current prices
    user_returns = (current_prices - purchase_prices) / purchase_prices
    
    # Calculate Sharpe Ratios for individual assets based on user's portfolio
    user_sharpe_ratios = (user_returns - risk_free_rate) / np.sqrt(np.diag(covariance_matrix))

    # Calculate optimal asset allocation based on expected returns
    optimal_weights = calculate_optimal_weights(expected_returns, covariance_matrix, risk_free_rate)

    # Calculate Sharpe Ratio for the portfolio based on user's portfolio
    portfolio_return = np.dot(optimal_weights, user_returns)
    portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights)))
    portfolio_sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    # Include Portfolio in Sharpe Ratios
    user_sharpe_ratios['Portfolio'] = portfolio_sharpe_ratio
    
    # Generate plots
    efficient_frontier_fig, asset_allocation_fig, sharpe_ratio_fig = generate_plots(expected_returns, optimal_weights, covariance_matrix, risk_free_rate, tickers, user_sharpe_ratios)

    return efficient_frontier_fig, asset_allocation_fig, sharpe_ratio_fig


# #### Interactive Ticker Chart

# In[ ]:


# CandleStick Chart
# -----------------

@app.callback(
    Output('interactive-chart', 'figure'),
    Input('timeframe-selector', 'value'),
    Input('asset-input', 'value'),
    Input('subplot-selector', 'value'),
    Input('overlay-selector', 'value'),
    Input('start-date-picker', 'value'),
    Input('end-date-picker', 'value')
)
def update_chart(selected_timeframe, inputted_asset, selected_subplot, overlay_indicator, start_date, end_date):
#     """
#     Update the main chart based on the selected parameters.
    
#     Parameters:
#         selected_timeframe (str): The selected timeframe.
#         inputted_asset (str): The selected asset.
#         selected_subplot (str): The selected subplot.
#         overlay_indicator (str): The selected overlay indicator.
#         start_date (str): The start date.
#         end_date (str): The end date.
    
#     Returns:
#         fig (Figure): The updated figure.
#     """

    # Get asset data with the selected timeframe and date range
    data = get_stock_data(inputted_asset, selected_timeframe, start_date, end_date)
        
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[inputted_asset, 'Volume', selected_subplot]
    )

    # Candlestick chart as main subplot with custom colors for increasing and decreasing candles
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            increasing_line_color='rgba(0, 128, 0)', 
            decreasing_line_color='rgba(255, 0, 0)', 
        ),
        row=1,
        col=1
    )

    # Add volume subplot
    colors = ['rgb(82, 143, 61)' if row['Open'] - row['Close'] >= 0 else 'rgb(143, 61, 61)' for _, row in data.iterrows()]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], marker_color=colors),
        row=2,
        col=1 
    )
    
    # Underlay indicators in the sub-plot
    if selected_subplot == "RSI":
        rsi = get_financial_metric(data, "RSI")
        fig.add_trace(go.Scatter(x=rsi.index, y=rsi, mode='lines', name='RSI', line=dict(color='MediumPurple', width=2)) 
                      ,row=3, col=1)
        
        # Adding a horizontal line at RSI=70 (overbought) and RSI=30 (oversold)
        fig.add_shape(type="line", x0=rsi.index[0], x1=rsi.index[-1], y0=70, y1=70, line=dict(color="IndianRed", width=1, dash="dot")
                     ,row=3, col=1)
        fig.add_shape(type="line", x0=rsi.index[0], x1=rsi.index[-1], y0=30, y1=30, line=dict(color="SteelBlue", width=1, dash="dot")
                     ,row=3, col=1)

    elif selected_subplot == "MACD":
        macd_line, signal_line, macd_hist = get_financial_metric(data, "MACD")
        fig.add_trace(go.Scatter(x=macd_line.index, y=macd_line, mode='lines', name='MACD Line', line=dict(color='SteelBlue', width=2))
                     ,row=3, col=1)
        fig.add_trace(go.Scatter(x=signal_line.index, y=signal_line, mode='lines', name='Signal Line', line=dict(color='IndianRed', width=2))
                     ,row=3, col=1)
        fig.add_trace(go.Bar(x=macd_hist.index, y=macd_hist, name='MACD Histogram', marker_color='Peru')
                     ,row=3, col=1)

    elif selected_subplot == "OBV":
        obv = get_financial_metric(data, "OBV")
        fig.add_trace(go.Scatter(x=obv.index, y=obv, mode='lines', name='OBV', line=dict(color='SteelBlue', width=3))
                     ,row=3, col=1)

    # Overlay indicators on the main chart
    if overlay_indicator == "Bollinger Bands":
        upper, middle, lower = get_financial_metric(data, "Bollinger Bands")
        fig.add_trace(go.Scatter(x=upper.index, y=upper, mode='lines', name='Upper Band', line=dict(color='SteelBlue', width=2, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=lower.index, y=lower, mode='lines', name='Lower Band', line=dict(color='SteelBlue', width=2, dash="dashdot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=middle.index, y=middle, mode='lines', name='Middle Band', line=dict(color='Khaki', width=1)), row=1, col=1)

    elif overlay_indicator == "SMA":
        sma = get_financial_metric(data, "SMA")
        fig.add_trace(go.Scatter(x=sma.index, y=sma, mode='lines', name='SMA', line=dict(color='Peru', width=2)), row=1, col=1)

    elif overlay_indicator == "EMA":
        ema = get_financial_metric(data, "EMA")
        fig.add_trace(go.Scatter(x=ema.index, y=ema, mode='lines', name='EMA', line=dict(color='OliveDrab', width=2)), row=1, col=1)

    fig.update_layout(
        height=800,
        margin=dict(t=100, b=50),
        xaxis_rangeslider_visible=False,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='gray',
        xaxis=dict(gridcolor='rgb(156, 156, 156, 0.8)', gridwidth=0.01),
        yaxis=dict(gridcolor='rgb(156, 156, 156, 0.8)', gridwidth=0.01),
        yaxis2=dict(gridcolor='rgb(156, 156, 156, 0.8)', gridwidth=0.01),
        xaxis3=dict(gridcolor='rgb(156, 156, 156, 0.8)', gridwidth=0.01),
        yaxis3=dict(gridcolor='rgb(156, 156, 156, 0.8)', gridwidth=0.01)
        
    )

    return fig

# Tool Tip Callback
# -----------------

@app.callback(
    Output('subplot-tooltip', 'children'), 
    Input('subplot-selector', 'value'),    
)
def update_tooltip(selected_subplot):
#         """
#     Update the tooltip for the selected subplot.
    
#     Parameters:
#         selected_subplot (str): The selected subplot.
    
#     Returns:
#         str: The description for the selected subplot.
#     """
    return subplot_indicator_descriptions.get(selected_subplot, "Description not available")

@app.callback(
    Output('overlay-tooltip', 'children'),
    Input('overlay-selector', 'value'),
)
def update_overlay_tooltip(overlay_indicator):
#      """
#     Update the tooltip for the selected overlay indicator.
    
#     Parameters:
#         overlay_indicator (str): The selected overlay indicator.
    
#     Returns:
#         str: The description for the selected overlay indicator.
#     """
    return overlying_indicator_descriptions.get(overlay_indicator, "Description not available")


# Article Definition
# ------------------

ARTICLES_PER_PAGE = 3

def create_news_section(articles, page=1):
#     """
#     Create a section to display news articles with pagination controls.
    
#     Parameters:
#     - articles (list): List of article dictionaries
#     - page (int): Current page number
    
#     Returns:
#     - html.Div: A Div containing the news section
#     """
    start_idx = (page - 1) * ARTICLES_PER_PAGE
    end_idx = start_idx + ARTICLES_PER_PAGE
    paged_articles = articles[start_idx:end_idx]
    
    # Create news items
    news_row_children = [
        dbc.Col([
            html.Div([
                html.Img(src=article['urlToImage'], style={'width': '350px', 'margin': 'auto'}),
                html.H3(article['title'], style={'color': 'lightGrey', 'textAlign': 'center'}),
                html.Div([html.A('Read More', href=article['url'], target='_blank', style={'color': 'lightGrey'})],
                         style={'textAlign': 'center', 'margin-top': 'auto', 'display': 'flex', 'flex-direction': 'column'})
            ], className='news-item', style={'margin-bottom': '20px', 'textAlign': 'center'})
        ], width=4)
        for article in paged_articles
    ]
    news_row = dbc.Row(news_row_children)  # Create the row using all columns

    # Add pagination controls
    total_pages = -(-len(articles) // ARTICLES_PER_PAGE)  # Ceiling division
    pagination_controls = html.Div([
        html.Button("Previous", id="prev-page-button"),
        dcc.Input(id='page-selector-input', type='number', value=page, min=1, max=total_pages, style={'width': '60px', 'textAlign': 'center'}),
        html.Button("Next", id="next-page-button")
    ], style={'textAlign': 'center'})

    return html.Div([news_row, pagination_controls], className='news-section')

@app.callback(
    Output('news-section-container', 'children'),
    [Input('asset-input', 'value'), Input('refresh-button', 'n_clicks')],
    [State('news-section-container', 'children')]
)
def update_news(inputted_asset, n_clicks, current_children):
#     """
#     Update the news section based on the selected asset and refresh button clicks.
    
#     Parameters:
#     - inputted_asset (str): The selected asset
#     - n_clicks (int): Number of times the refresh button has been clicked
#     - current_children (list): Current children of the news section container
    
#     Returns:
#     - html.Div: Updated news section
#     """
    current_page = 1  # default value
    # Extract current page value from current_children if necessary
    articles = fetch_articles(inputted_asset)
    return create_news_section(articles, current_page)

# Page Selector call back and definition
@app.callback(
    Output('page-selector-input', 'value'),
    [Input('next-page-button', 'n_clicks'), Input('prev-page-button', 'n_clicks')],
    [State('page-selector-input', 'value'), State('asset-input', 'value')]  
)
def update_page(next_clicks, prev_clicks, current_page, inputted_asset):
#     """
#     Update the current page number based on next and previous button clicks.
    
#     Parameters:
#     - next_clicks (int): Number of times the next button has been clicked
#     - prev_clicks (int): Number of times the previous button has been clicked
#     - current_page (int): Current page number
#     - inputted_asset (str): The selected asset
    
#     Returns:
#     - int: Updated page number
#     """
    articles = fetch_articles(inputted_asset)
    total_pages = -(-len(articles) // ARTICLES_PER_PAGE)  # Ceiling division

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if "next-page-button" in changed_id:
        return min(current_page + 1, total_pages)
    elif "prev-page-button" in changed_id:
        return max(current_page - 1, 1)
    else:
        return 1




# #### Dash App Theme Setting & Configuration

# In[ ]:


# Set the theme for the Dash app
# --------------

app.theme = "dark" 
dash.callback_context.collapse = True


# #### Dash App Execution

# In[ ]:


# Main Execution
# --------------

if __name__ == '__main__':
    app.run_server(debug=True, port=8054)

