import streamlit as st
import plotly.graph_objects as go

from ._markers import mpl_marker_to_plotly


def plot_model_legend(model_to_marker, model_to_color):
    if not model_to_marker:
        return go.Figure()

    unique_models = sorted(model_to_marker.keys())
    fig = go.Figure()

    for m in unique_models:
        color = model_to_color[m]
        marker_symbol = mpl_marker_to_plotly(model_to_marker[m])
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            name=m,
            marker=dict(
                symbol=marker_symbol,
                color=color,
                size=14,
                line=dict(color='black', width=1),
            ),
            showlegend=True,
        ))

    fig.update_layout(
        legend=dict(
            orientation='h',
            y=0.5, x=0.5,
            xanchor='center',
            yanchor='middle',
            title='Models',
            font=dict(size=11),
        ),
        height=70,
        margin=dict(t=0, b=0, l=0, r=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def render_model_legend(model_to_marker, model_to_color):
    fig = plot_model_legend(model_to_marker, model_to_color)
    st.plotly_chart(fig, use_container_width=True)
