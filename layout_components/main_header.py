
# Import required libraries
import dash_bootstrap_components as dbc
from dash import html

main_header = dbc.Container([
    # H1 Header
    html.H1(
        "Pineapple Express",
        style={
            'color': 'rgba(254, 234, 99, 1)', 
            'text-align': 'center', 
            'font-family': 'Lobster, cursive',
            'fontSize': 82,
            'marginBottom': '50px',
            'marginTop': '10px'
        },
    ),
    
    # H4 Sub-header
    html.H4(
        "Empowering Every Investor Through Analytics",
        style={
            'color': 'LightGrey', 
            'text-align': 'center', 
            'marginBottom': '20px',
            'marginTop': '20px',
            'font-family': 'Roboto'
        }
    )
],
# Container properties
fluid=True,
className="mt-4",
style={
    'border': '6px solid lightgray',
    'borderRadius': '10px',
    'padding': '10px'
}
)
