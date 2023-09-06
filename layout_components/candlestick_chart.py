
# Import required libraries
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from datetime import datetime
import pandas as pd

# Section 1: Chart Header
#------------------------

# Header Style
header_style = {'color': 'LightGrey', 'marginBottom': '10px', 'textAlign': 'center', 'fontSize': 20, 'font-family': 'Roboto'}

# Responsive Input Style
responsive_input_style = {'background-color': '#333', 'color': 'rgba(254, 234, 99, 1)', 'width': '100%', 'text-align': 'center'}

# Section 1: Header for Pineapple Chart
interactive_chart_header = html.H1(
    "Pineapple Chart",
    style={
        'color': 'rgba(254, 234, 99, 1)',
        'text-align': 'center',
        'font-family': 'Lobster, cursive',
        'fontSize': 64,
        'marginBottom': '50px',
        'marginTop': '10px'
    },
)

# Section 2: Chart Inputs
#------------------------

end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (pd.to_datetime(end_date) - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
chart_inputs = dbc.Row([
    # Ticker Symbol
    dbc.Col([
        html.Div("Ticker Symbol", style=header_style),
        dcc.Input(
            id='asset-input',
            type='text',
            value='AAPL',
            debounce=True,
            style={**responsive_input_style}
        )
    ], className="d-flex flex-column align-items-center"),
    
    # Candlestick Timeframe
        dbc.Col([
            html.Div("Timeframe", style={'color': 'LightGrey', 
                                                  'marginBottom': '10px', 
                                                  'textAlign': 'center',
                                                  'fontSize': 20,
                                                  'font-family': 'Roboto'}),
            dcc.Dropdown(
                id='timeframe-selector',
                options=[
                    {'label': html.Span(['15m'], style={'color': 'rgba(254, 234, 99, 1)'}), 'value': '15m'},
                    {'label': html.Span(['1h'], style={'color': 'rgba(254, 234, 99, 1)'}), 'value': '1h'},
                    {'label': html.Span(['1d'], style={'color': 'rgba(254, 234, 99, 1)'}), 'value': '1d'},
                    {'label': html.Span(['5d'], style={'color': 'rgba(254, 234, 99, 1)'}), 'value': '5d'},
                ],
                value='1d',
                style={'background-color': '#333', 
                       'color': 'rgba(254, 234, 99, 1)', 
                       'width': '100%', 
                       'text-align': 'center'},
            )
        ], className="d-flex flex-column align-items-center custom-dropdown"), 
    
        # Start Date Picker
        dbc.Col([
                html.Div("Start Date", style=header_style),
                dcc.Input(
                    id='start-date-picker',
                    type='text',
                    value=(pd.to_datetime(end_date) - pd.DateOffset(months=6)).strftime('%Y-%m-%d'),
                    placeholder='YYYY-MM-DD',
                    style={**responsive_input_style}
                ),
            ]),
    
        # End Date Picker
        dbc.Col([
                html.Div("End Date", style=header_style),
                dcc.Input(
                    id='end-date-picker',
                    type='text',
                    value=datetime.today().strftime('%Y-%m-%d'),
                    placeholder='YYYY-MM-DD',
                    style={**responsive_input_style}
                ),
            ]),  
    
            ],className="mb-5", justify="center")

# Section 3: Subplot and Overlay Indicators
#------------------------------------------

subplot_overlay_indicators = dbc.Row([
    # Subplot Indicator
    dbc.Col([
        html.Div("Subplot Indicator", style=header_style),
        dcc.Dropdown(
                id='subplot-selector',
                options=[
                    {'label': html.Span(['RSI'], style={'color': 'rgba(254, 234, 99, 1)'}), 'value': 'RSI'},
                    {'label': html.Span(['MACD'], style={'color': 'rgba(254, 234, 99, 1)'}), 'value': 'MACD'},
                    {'label': html.Span(['OBV'], style={'color': 'rgba(254, 234, 99, 1)'}), 'value': 'OBV'},
                ],
                multi=False,
                style={'color': 'black', 
                       'background-color': '#333', 
                       'width': '220px',
                       'text-align': 'center'},
                placeholder="Select a Subplot Indicator",
                className="custom-dropdown"
            ),
            html.Span("🛈", id="subplot-tooltip-target", style={
                "cursor": "pointer",
                "color": "rgba(254, 234, 99, 1)",
                "fontSize": "20px",
                "marginLeft": '10px'
            }),
            dbc.Tooltip(
                "Tooltip content will be replaced dynamically.",
                target="subplot-tooltip-target",
                id="subplot-tooltip",
                style={"background-color": "black", "color": "white"}
            ),
        ], className="d-flex flex-column align-items-center", width=3),  # Adjust width as needed

        # Overlay Indicator
        dbc.Col([
            html.Div("Overlay Indicator", style={'color': 'LightGrey', 
                                                  'marginBottom': '10px', 
                                                  'textAlign': 'center', 
                                                  'fontSize': 20,
                                                  'font-family': 'Roboto'}),
            dcc.Dropdown(
                id='overlay-selector',
                options=[
                    {'label': html.Span(['Bollinger Bands'], style={'color': 'rgba(254, 234, 99, 1)'}), 'value': 'Bollinger Bands'},
                    {'label': html.Span(['SMA'], style={'color': 'rgba(254, 234, 99, 1)'}), 'value': 'SMA'},
                    {'label': html.Span(['EMA'], style={'color': 'rgba(254, 234, 99, 1)'}), 'value': 'EMA'},
                ],
                multi=False,
                style={'color': 'black', 
                       'background-color': '#333', 
                       'width': '220px',
                       'text-align': 'center'},
                placeholder="Select a Indicator",
                className="custom-dropdown"
            ),
            html.Span("🛈", id="overlay-tooltip-target", style={
                "cursor": "pointer",
                "color": "rgba(254, 234, 99, 1)",
                "fontSize": "20px",
                "marginLeft": '10px'
            }),
            dbc.Tooltip(
                "Tooltip content will be replaced dynamically.",
                target="overlay-tooltip-target",
                id="overlay-tooltip",
                style={"background-color": "black", "color": "white"}
            ),
        ], className="d-flex flex-column align-items-center", width=3)
        ], justify="center")

# Section 4: Chart Layout
#------------------------

chart_layout = html.Div([
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='interactive-chart', style={'marginTop': '20px'}, responsive='auto')
        ], width=12)
    ]),

    dbc.Row([
        # Articles Section
        dbc.Col([
            html.H3("Related Articles", style={'color': 'rgba(254, 234, 99, 1)',
                                               'marginTop': '20px',
                                               'marginBottom': '40px',
                                               'fontSize': 28,
                                               'font-family': 'Lobster, cursive'
                                               }),
            html.Button('Refresh News', id='refresh-button',
                        style={'color': 'LightGrey',
                               'marginTop': '20px',
                               'marginBottom': '20px',
                               'background-color': '#333'}),

            # Container for articles
            html.Div(id='news-section-container', children=[
                dbc.Row([  # Add this row to put articles side by side
                    dbc.Col([
                        html.Div("Article 1 content", style={'text-align': 'center',
                                                            'justify-content': 'center'})
                    ], className= "d-flex flex-column align-items-center", width=4),
                    dbc.Col([
                        html.Div("Article 2 content", style={'text-align': 'center',
                                                            'justify-content': 'center'})
                    ], className= "d-flex flex-column align-items-center", width=4),
                    dbc.Col([
                        html.Div("Article 3 content", style={'text-align': 'center',
                                                            'justify-content': 'center'})
                    ], className= "d-flex flex-column align-items-center", width=4),
                    # Add more columns for more articles
                ])
            ]),
        ], className="d-flex flex-column align-items-center", width=12)
    ]),
])

# Combine all sections to form the complete layout for Pineapple Chart
interactive_candlestick_chart = html.Div([
    interactive_chart_header,
    chart_inputs,
    subplot_overlay_indicators,
    chart_layout
])
