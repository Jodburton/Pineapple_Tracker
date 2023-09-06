
#Common dash imports
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc

#Layout Imports
from layout_components.main_header import main_header
from layout_components.candlestick_chart import interactive_candlestick_chart
from layout_components.portfolio_tracker import portfolio_tracker
from layout_components.portfolio_analysis import portfolio_analysis



app_layout = dbc.Container([  
                #Google Fonts
                html.Link(href="https://fonts.googleapis.com/css2?family=Abril+Fatface&family=Lobster&display=swap", rel="stylesheet"),
                html.Link(href="https://fonts.googleapis.com/css2?family=Abril+Fatface&family=Lobster&display=swap", rel="stylesheet"),
                
                #Main Header Container Code
                main_header,
    
                #Portfolio Tracker Container Code
                portfolio_tracker,
            
                #Portfolio Analysis Container Code
                portfolio_analysis,
                
                #CandleStick Chart Container Code
            dbc.Container([html.Div([
                interactive_candlestick_chart
                   ]) 
                    ],fluid=True,
                        className="mt-4",
                        style={
                            'border': '2px solid lightgray',
                            'borderRadius': '10px',
                            'padding': '20px'}),
                ], style={'backgroundColor': 'black', 
                          'padding': '100px', 
                          'font-family': 'Roboto',
                          'marginTop': '-100px'},
                fluid=True)
