import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Optional

def plot_iv_surface(grid_s:np.ndarray,grid_t:np.ndarray,grid_z:np.ndarray,title:Optional[str]=None):
    fig=go.Figure(data=[go.Surface(x=grid_s,y=grid_t,z=grid_z,colorscale='Viridis',showscale=True)])
    fig.update_layout(
        title=title or 'Implied Volatility Surface',
        scene=dict(
            x_axis_title='Strike Price',
            y_axis_title='Time to Expiry (Years)',
            z_axis_title='Implied Volatility'
        ),
        autosize=True,
        height=700
    )
    fig.show()
    return fig

def plot_smile(df:pd.DataFrame,expiry:pd.Timestamp,underlying:float=None,title:Optional[str]=None):
    
    sel=df[df['expiry']==pd.to_datetime(expiry)].copy()
    sel=sel.dropna(subset=['iv','strike'])
    sel=sel.sort_values('strike')

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=sel['strike'],y=sel['iv'],mode='markers+lines',name="IV Smile"))

    if underlying is not None:
        fig.add_vline(x=underlying,line_dash='dash',annotation_text="Underlying Price", annotation_position="top left")

    fig.update_layout(title=title or f'IV smile - {expiry}', xaxis_title="Strike Price", yaxis_title="Implied Volatility")
    fig.show()
    return fig