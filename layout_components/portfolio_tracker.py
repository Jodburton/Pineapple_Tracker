
import dash
from dash import dcc
from dash import html
from dash import dash_table
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime

# Define styles
header_style = {
    'color': 'LightGrey', 
    'marginBottom': '10px', 
    'textAlign': 'center',
    'fontSize': 20,
    'font-family': 'Roboto'
}

input_style = {
    'color': 'LightGrey',
    'background-color': '#333',
    'width': '150px',
    'marginBottom': '20px',
    'textAlign': 'center',
    'justifyContent': 'center'
}

button_style = {
    'backgroundColor': '#333',
    'border': '3px solid rgba(254, 234, 99, 1)',
    'color': 'rgba(254, 234, 99, 1)',
    'padding': '0.5rem 1rem',
    'borderRadius': '0.25rem',
    'cursor': 'pointer',
    'textAlign': 'center',
    'marginTop': '20px',
    'marginLeft': '40px',
}

# Define a modified input style
responsive_input_style = {
    'color': 'LightGrey',
    'background-color': '#333',
    'textAlign': 'center',
    'justifyContent': 'center',
    'width': '100%', 
    'marginBottom': '10px'}

# Initialize an empty DataFrame for the assets table
df = pd.DataFrame()

# Create the new container with three row sections
portfolio_tracker = dbc.Container(
    [
        # Section 0: Header
        html.Div([
            html.H1(
                "Pineapple Tracker",
                style={
                    'color': 'rgba(254, 234, 99, 1)',
                    'text-align': 'center',
                    'font-family': 'Lobster, cursive',
                    'fontSize': 64,
                    'marginBottom': '30px',
                    'marginTop': '10px'
                }
            ),
        ], className="d-flex flex-column"),

        # Section 1: Input Section
        dbc.Row([
            # Ticker Symbol Input
            dbc.Col([
                html.Div("Ticker Symbol", style=header_style),
                dcc.Input(
                    id='ticker-input', 
                    type='text', 
                    placeholder='Enter a ticker',
                    style=responsive_input_style
                ),
                html.Div(id='error-message', style={'color': 'red'}),
            ], className="d-flex flex-column align-items-center justify-content-start"),

            # Shares Input
            dbc.Col([
                html.Div("Shares/Coins", style=header_style),
                dcc.Input(
                    id='shares-input',
                    type='number',
                    placeholder='Number of shares',
                    min=0,
                    step=1,
                    style=responsive_input_style
                ),
            ], className="d-flex flex-column align-items-center justify-content-start"),

            # Purchase Price Input
            dbc.Col([
                html.Div("Purchase Price", style=header_style),
                dcc.Input(
                    id='price-bought-input', 
                    type='number', 
                    placeholder='Purchase price',
                    style=responsive_input_style
                ),
            ], className="d-flex flex-column align-items-center justify-content-start"),

            # Purchase Date Picker
            dbc.Col([
                html.Div("Purchase Date", style=header_style),
                dcc.Input(
                    id='purchase-date-input',
                    type='text',
                    placeholder='YYYY-MM-DD',
                    style=responsive_input_style
                ),
            ], className="d-flex flex-column align-items-center justify-content-start"),

            # Add Asset Button
            dbc.Col([
                html.Button('Add', id='add-asset-button', style=button_style)
            ], style={'width': '20%'}, className="d-flex flex-column align-items-center justify-content-start"),
            
            # Remove Asset Button
            dbc.Col([
                html.Button('Remove', id='remove-asset-button', n_clicks=0, style=button_style)
            ], style={'width': '20%'}, className="d-flex flex-column align-items-center justify-content-start"),

        ], className='mb-4', justify="center"),

        # Section 2: Assets DataTable
        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id='assets-table',
                    columns=[
                        {'name': 'Ticker', 'id': 'Ticker'},
                        {'name': 'Shares', 'id': 'Shares'},
                        {'name': 'Purchase Date', 'id': 'Purchase Date'},
                        {'name': 'Purchase Price', 'id': 'Purchase Price'}, 
                        {'name': 'Initial Value', 'id': 'Initial Value'},    
                        {'name': 'Current Value', 'id': 'Current Value'},    
                        {'name': 'ROI (%)', 'id': 'ROI (%)'},                
                    ],
                    editable=True,
                    row_selectable='multi',
                    selected_rows=[],
                    data=df.to_dict('records'),
                    style_table={'height': '300px', 'overflowY': 'auto'},
                    style_cell={'textAlign': 'center', 
                                'backgroundColor': 'rgb(50, 50, 50)', 
                                'color': 'white'},
                    style_header={
                        'backgroundColor': 'rgb(30, 30, 30)',
                        'fontWeight': 'bold'
                    }
                ),
            ], xs=12, sm=12, md=12, lg=12, xl=12),  # Full width on all devices
        ], className='mb-2'),

        # Section 3: Visualizations
        dbc.Row([
            dbc.Col([
                # Row 1: KPI and Headers
                dbc.Row([
                    # Title for Current Portfolio Value
                    html.Div("Total Pineapple Value", style={
                        'fontSize': '30px',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'color': 'rgba(254, 234, 99, 1)',
                        'font-family': 'Lobster, cursive',
                    }),

                    # KPI Section for Current Portfolio Value
                    html.Div(id='current-value-kpi', children='', style={
                        'fontSize': '64px',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'marginBottom': '10px',
                        'color': 'lightGrey'
                    }),

                    # Title for Initial Investment Value
                    html.Div("Total Initial Investment", style={
                        'fontSize': '28x',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'marginBottom': '20px',
                        'color': 'rgba(254, 234, 99, 1)',
                        'font-family': 'Lobster, cursive',
                    }),

                    # KPI Section for Initial Investment Value
                    html.Div(id='initial-investment-kpi', children='', style={
                        'fontSize': '34px',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'marginBottom': '30px',
                        'color': 'lightGrey'
                    }),

                    html.Button(id='dummy-button', style={'display': 'none'}),
                    ]),
                # Row 2: Graph
                dbc.Row([
                    dbc.Col([
                    # ROI Gauge
                    dcc.Graph(id='gauge-chart',  
                              responsive='auto',
                              style={'width': '400px'})
                ]),
                ],)
            ], width=4, className="d-flex flex-column align-items-center justify-content-start"),
            
            dbc.Col([
                # Pie Chart title
                html.Div("PIEnapple Chart", style={
                    'fontSize': '30px',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'color': 'rgba(254, 234, 99, 1)',
                    'font-family': 'Lobster, cursive',
                    'marginBottom': '20px',
                    'width': '100%'
                },
                        ),

                # Pie Chart
                dcc.Graph(id='pie-chart',
                         responsive='auto',
                         style={'width': '450px'}),

                # Toggle for Current Value or Initial Investment
                html.Div([
                    dbc.RadioItems(
                        options=[
                            {'label': 'Current Value', 'value': 'current'},
                            {'label': 'Initial Investment', 'value': 'initial'},
                            {'label': 'Sector Distribution', 'value': 'sector'},
                        ],
                        value='current',  # default selected value
                        id='value-toggle',
                        inline=True,
                        style={
                            'color': 'lightGrey',
                            'background-color': 'black',
                            'border': '1px solid lightGrey',
                            'borderRadius': '5px',
                            'padding': '10px 20px'
                        },
                        labelStyle={
                            'marginRight': '15px',
                            'color': 'lightGrey'
                        }
                    )
                ], style={
                    'fontSize': '18px',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'marginTop': '20px',
                    'color': 'lightGrey',
                    'background-color': '#333',
                    'border': '1px solid lightGrey',
                    'borderRadius': '5px',
                    'padding': '10px',
                }),
            ], width=4, className="d-flex flex-column align-items-center justify-content-start"),

            dbc.Col([
                # Portfolio Performance Graph title
                html.Div("Pineapple Health", 
                            style={
                            'fontSize': '30px',
                            'fontWeight': 'bold',
                            'textAlign': 'center',
                            'color': 'rgba(254, 234, 99, 1)',
                            'font-family': 'Lobster, cursive',
                            'marginBottom': '20px',
                            'width': '100%'},
                       ),

                # Dropdown selection for Timeframe
                dcc.Dropdown(
                    id='timeframe-dropdown',
                    options=[
                        {'label': html.Span(['3 Months'], style={'color': 'rgba(254, 234, 99, 1)'}), 'value': '3M'},
                        {'label': html.Span(['6 Months'], style={'color': 'rgba(254, 234, 99, 1)'}), 'value': '6M'},
                        {'label': html.Span(['1 Year'], style={'color': 'rgba(254, 234, 99, 1)'}), 'value': '1Y'},
                    ],
                    value='All',
                    style={'color': 'rgba(254, 234, 99, 1)', 
                        'background-color': '#333', 
                        'width': '100%',
                        'text-align': 'center'},
                    placeholder="Select a Time Period",
                    className="custom-dropdown"
                            ),
                # Portfolio Performance Graph
                dcc.Graph(
                    id='portfolio-performance-graph',
                    responsive='auto',
                    config={'displayModeBar': True},
                    style={'width': '450px'},
                    figure={
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
                                'gridcolor': 'rgb(70, 70, 70)'}
                                    }
                            }
                            )
                    ],  width=4, className="d-flex flex-column align-items-center justify-content-start"),
        ])
    ],
    fluid=True,
    className="mt-4 d-flex flex-column",
    style={
        'border': '2px solid lightgray',
        'borderRadius': '10px',
        'padding': '20px'
    }
)
