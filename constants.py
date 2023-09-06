import dash_bootstrap_components as dbc

#Tool tip constants
subplot_indicator_descriptions = {
    "RSI": "The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. RSI oscillates between zero and 100. Traditionally, and according to Wilder, RSI is considered overbought when above 70 and oversold when below 30.",
    
    "MACD": "Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. The MACD is calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA. The result of that calculation is the MACD line. A nine-day EMA of the MACD called the 'signal line,' is then plotted on top of the MACD line, which can function as a trigger for buy and sell signals.",
    
    "OBV": "On-Balance Volume (OBV) is a momentum technical indicator that relates volume to price change. The indicator is based on the observation that when prices are rising, volume tends to increase, and when prices are falling, volume tends to decrease. It's used to confirm price trends or detect any divergence that might predict a price change.",
}    

overlying_indicator_descriptions = {    
    "Bollinger Bands": "Bollinger Bands consist of a middle band being an N-period simple moving average (SMA), an upper band at K times an N-period standard deviation above the middle band, and a lower band at K times an N-period standard deviation below the middle band. Bollinger Bands are able to adapt to volatility in the price of a stock. A band squeeze denotes a period of low volatility and is considered by traders to be a potential indicator of future increased volatility and possible trading opportunities.",
    
    "SMA": "The Simple Moving Average (SMA) provides the average price of a stock over a specific time period.",
    
    "EMA": "The Exponential Moving Average (EMA) gives more weight to recent prices, thus reacting more quickly to price changes than SMA."
}

#Pie chart colors
pie_chart_colors = [
    'rgba(99, 254, 145, 1)',  # Muted Green
    'rgba(254, 234, 99, 1)',  # Muted Gold
    'rgba(254, 99, 134, 1)',  # Muted Pink
    'rgba(99, 200, 254, 1)',  # Muted Sky Blue
    'rgba(255, 150, 50, 1)',  # Muted Orange
    'rgba(99, 126, 254, 1)',  # Muted Blue
    'rgba(141, 99, 254, 1)',  # Muted Purple
]

#Define the max_roi and min_roi variables globally.
max_roi = int(100)  # Set initial upper bound.
min_roi = int(-100)  # Set initial lower bound.

#Bootstrap constant
external_stylesheets = [dbc.themes.BOOTSTRAP]