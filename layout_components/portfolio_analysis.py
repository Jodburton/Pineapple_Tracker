
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc

# Define reusable styles
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


# Create the Portfolio Analysis container
portfolio_analysis = dbc.Container([
    # Portfolio Analysis Header
    dbc.Row([
        dbc.Col([
            # Title for the Portfolio Analysis section
            html.H3("Pineapple Analysis", style={
                'color': 'rgba(254, 234, 99, 1)',
                'marginTop': '20px',
                'marginBottom': '20px',
                'fontSize': 64,
                'font-family': 'Lobster, cursive'
            }),
        ], className="d-flex flex-column align-items-center", width=12)
    ]),
    
    # Row for Tropical Frontier and Tropic Allocation Mixer
    dbc.Row([
        dbc.Col([
            # Title for Tropical Frontier of Returns
            html.Div("Tropical Frontier of Returns",
                     style={
                         'fontSize': '30px',
                         'fontWeight': 'bold',
                         'textAlign': 'center',
                         'color': 'rgba(254, 234, 99, 1)',
                         'font-family': 'Lobster, cursive',
                         'marginBottom': '20px',
                         'width': '100%'
                     }),
            # Buttons for Annualized and Non-Annualized View
            html.Div([  
                html.Button('Annualized View',
                            id='annualized-button',
                            n_clicks=0,
                            style={**button_style, 'width': '150px', 'marginRight': '10px'}),
                html.Button('Non-Annualized View',
                            id='non-annualized-button',
                            n_clicks=0,
                            style={**button_style, 'width': '150px'})
            ], className='d-flex justify-content-center')  
        ], width=6, className="d-flex flex-column align-items-center"),
        
        dbc.Col([
            # Title for Tropic Allocation Mixer
            html.Div("Tropic Allocation Mixer",
                     style={
                         'fontSize': '30px',
                         'fontWeight': 'bold',
                         'textAlign': 'center',
                         'color': 'rgba(254, 234, 99, 1)',
                         'font-family': 'Lobster, cursive',
                         'marginBottom': '20px',
                         'width': '100%'
                     }),
        ], width=6, className="d-flex flex-column align-items-center")
    ]),
    
    # Row for Efficient Frontier Plot and Asset Allocation Chart
    dbc.Row([
        dbc.Col([
            # Efficient Frontier Plot
            dcc.Graph(id='efficient-frontier-plot',
                      responsive='auto',
                      config={'displayModeBar': True},
                      figure={
                          'layout': {
                              'paper_bgcolor': 'black',
                              'plot_bgcolor': 'black',
                              'xaxis': {
                                  'tickfont': {'color': 'black'},
                                  'gridcolor': 'black',
                                  'showticklabels': False,
                                  'showline': False,
                                  'zeroline': False
                              },
                              'yaxis': {
                                  'tickfont': {'color': 'black'},
                                  'gridcolor': 'black',
                                  'showticklabels': False,
                                  'showline': False,
                                  'zeroline': False
                              }
                          }
                      }),
        ], width=6, className="d-flex flex-column justify-content-center"),
        
        dbc.Col([
            # Asset Allocation Chart
            dcc.Graph(id='asset-allocation-chart',
                      responsive='auto',
                      config={'displayModeBar': True},
                      figure={
                          'layout': {
                              'paper_bgcolor': 'black',
                              'plot_bgcolor': 'black',
                              'xaxis': {
                                  'tickfont': {'color': 'black'},
                                  'gridcolor': 'black',
                                  'showticklabels': False,
                                  'showline': False,
                                  'zeroline': False
                              },
                              'yaxis': {
                                  'tickfont': {'color': 'black'},
                                  'gridcolor': 'black',
                                  'showticklabels': False,
                                  'showline': False,
                                  'zeroline': False
                              },
                              'showlegend': False
                          }
                      }),
        ], width=6, className="d-flex flex-column justify-content-center"),
    ]),
    
    # Row for Pineapple Sharpness Scale
    dbc.Row([
        # Title for Pineapple Sharpness Scale
        html.Div("Pineapple Sharpness Scale",
                 style={
                     'fontSize': '30px',
                     'fontWeight': 'bold',
                     'textAlign': 'center',
                     'color': 'rgba(254, 234, 99, 1)',
                     'font-family': 'Lobster, cursive',
                     'marginBottom': '20px',
                     'width': '100%'
                 }),
        # Sharpe Ratio Bar Chart
        dcc.Graph(id='sharpe-ratio-bar-chart',
                  responsive='auto',
                  config={'displayModeBar': True},
                  figure={
                      'layout': {
                          'paper_bgcolor': 'black',
                          'plot_bgcolor': 'black',
                          'xaxis': {
                              'tickfont': {'color': 'black'},
                              'gridcolor': 'black',
                              'showticklabels': False,
                              'showline': False,
                              'zeroline': False
                          },
                          'yaxis': {
                              'tickfont': {'color': 'black'},
                              'gridcolor': 'black',
                              'showticklabels': False,
                              'showline': False,
                              'zeroline': False
                          },
                      }
                  }),
    ]),
], fluid=True,
    className="mt-4 d-flex flex-column",
    style={
        'border': '2px solid lightgray',
        'borderRadius': '10px',
        'padding': '20px'
    }
)
