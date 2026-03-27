import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Optional

def get_x_axis(df, use_moneyness: bool):
    if use_moneyness:
        return df["moneyness"], "Log-Moneyness log(K/S)"
    else:
        return df["K"], "Strike (K)"
    
def plot_iv_surface(grid_x: np.ndarray, grid_t: np.ndarray, grid_z: np.ndarray, title: Optional[str] = None):

    fig = go.Figure(data=[
        go.Surface(
            x=grid_x,
            y=grid_t,
            z=grid_z,
            colorscale='Viridis',
            showscale=True
        )
    ])

    fig.update_layout(
        title=title or 'Implied Volatility Surface',
        scene=dict(
            xaxis_title='Log-Moneyness (log(K/S))',   # 🔥 FIXED
            yaxis_title='Time to Expiry (Years)',
            zaxis_title='Implied Volatility'
        ),
        autosize=True,
        height=700
    )
    fig.show()
    return fig


def plot_iv_smile(df, use_moneyness: bool = True, T=None, fig=None, show=True):
    """
    Plot a single IV smile.
    Can be used standalone or as part of multi-smile plotting.
    """
    if fig is None:
        fig = go.Figure()

    x, x_label = get_x_axis(df, use_moneyness)

    # df = df.sort_values(by='K')  

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df['iv_yf'],
            mode='lines+markers',
            name=f"T={round(T, 3)}" if T is not None else "IV Smile"
        )
    )

    fig.update_layout(
        title='IV Smile',
        xaxis_title=x_label,
        yaxis_title="Implied Volatility",
        template="plotly_dark"
    )

    if show:
        fig.show()

    return fig


def plot_iv_smiles(df, use_moneyness: bool = True, show=True):
    """
    Plot IV smiles for all expiries in one figure.
    """
    fig = go.Figure()

    for T, group in df.groupby('T'):
        print(T)
        plot_iv_smile(group, use_moneyness, T, fig, show=False)


    fig.update_layout(
        title='IV Smiles (All Expiries)',
        xaxis_title='Moneyness' if use_moneyness else 'Strike',
        yaxis_title="Implied Volatility",
        template="plotly_dark"
    )

    if show:
        fig.show()

    return fig