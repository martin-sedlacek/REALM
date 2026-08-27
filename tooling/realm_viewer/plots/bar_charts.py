import numpy as np
import streamlit as st
import plotly.graph_objects as go

from ._markers import mpl_marker_to_plotly


def _base_layout(title, ylabel, fontsize):
    d = dict(
        yaxis=dict(title=ylabel, title_font=dict(size=fontsize), gridcolor='lightgray'),
        xaxis=dict(showticklabels=False, showgrid=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(b=150, t=50, l=60, r=20),
        height=420,
    )
    if title:
        d['title'] = dict(text=title, font=dict(size=fontsize + 2))
    return d


def plot_grouped_bars_with_symbols(labels, values, colors, symbols=None, model_names=None, model_colors=None, title=None, ylabel=None, fontsize=12, hline=None, hline_label=None):
    x_positions = np.arange(len(labels))

    group_names = [l.split(' - ')[0] for l in labels]
    unique_groups = []
    unique_indices = []
    for i, g in enumerate(group_names):
        if g not in unique_groups:
            unique_groups.append(g)
            unique_indices.append(i)

    fig = go.Figure(layout=dict(title=dict(text='')))

    # One bar per label, colored individually
    fig.add_trace(go.Bar(
        x=x_positions.tolist(),
        y=values,
        marker_color=colors,
        marker_line_width=0,
        showlegend=False,
        hovertemplate=[
            f'{labels[i]}<br>{ylabel or "Value"}: {values[i]:.3f}<extra></extra>'
            for i in range(len(labels))
        ],
    ))

    # Markers above bars
    if symbols:
        max_val = max(values) if values else 1.0
        offset = max_val * 0.04 if max_val > 0 else 0.05
        for i, (val, sym) in enumerate(zip(values, symbols)):
            if sym:
                fig.add_trace(go.Scatter(
                    x=[int(x_positions[i])],
                    y=[val + offset],
                    mode='markers',
                    marker=dict(symbol=mpl_marker_to_plotly(sym), color='black', size=8),
                    showlegend=False,
                    hoverinfo='skip',
                ))

    shapes = []
    annotations = []

    # Reference line
    if hline is not None:
        shapes.append(dict(
            type='line',
            x0=-0.5, x1=len(labels) - 0.5,
            y0=hline, y1=hline,
            line=dict(color='gray', dash='dash', width=1.5),
            xref='x', yref='y',
        ))
        if hline_label:
            annotations.append(dict(
                x=len(labels) - 0.6, y=hline,
                xref='x', yref='y',
                text=hline_label,
                showarrow=False,
                xanchor='right',
                font=dict(size=max(8, fontsize - 4)),
            ))

    # Group separators and task labels below x-axis
    for i, group_label in enumerate(unique_groups):
        start_idx = unique_indices[i]
        end_idx = unique_indices[i + 1] if i + 1 < len(unique_indices) else len(labels)
        group_size = end_idx - start_idx

        if i > 0:
            shapes.append(dict(
                type='line',
                x0=x_positions[start_idx] - 0.5, x1=x_positions[start_idx] - 0.5,
                y0=0, y1=1,
                line=dict(color='gray', width=0.8, dash='dash'),
                xref='x', yref='paper',
            ))

        center_x = float(start_idx + (group_size - 1) / 2)
        annotations.append(dict(
            x=center_x,
            y=-0.12,
            xref='x', yref='paper',
            text=group_label,
            showarrow=False,
            textangle=-45,
            font=dict(size=max(8, fontsize - 2)),
            xanchor='right',
            yanchor='top',
        ))

    max_val = max(values) if values else 1.0
    if hline is not None:
        max_val = max(max_val, hline)

    layout = _base_layout(title, ylabel, fontsize)
    layout['yaxis']['range'] = [0, max(1.1, max_val + 0.15)]
    layout['xaxis']['range'] = [-0.5, len(labels) - 0.5]
    layout['shapes'] = shapes
    layout['annotations'] = annotations

    fig.update_layout(**layout)
    return fig


def render_metric_bar_chart(df, metric_col, title, ylabel, model_to_marker, model_to_color, color=None, hline=None, hline_label=None):
    if metric_col not in df.columns:
        st.info(f"Column '{metric_col}' not found in data.")
        return

    try:
        metric_df = df[df[metric_col].notna()].copy()
        if metric_df.empty:
            st.info(f"No data available for metric '{metric_col}'.")
            return

        stats = metric_df.groupby(['task', 'model'])[metric_col].mean().reset_index()
        stats = stats.sort_values(['task', 'model'])

        labels = [f"{row['task']} - {row['model']}" for _, row in stats.iterrows()]
        values = stats[metric_col].tolist()

        if model_to_color:
            colors = [model_to_color[row['model']] for _, row in stats.iterrows()]
        else:
            colors = [color if color else 'gold'] * len(labels)

        symbols = [model_to_marker[row['model']] for _, row in stats.iterrows()]
        model_names = stats['model'].tolist()

        fig = plot_grouped_bars_with_symbols(labels, values, colors, symbols, model_names, model_colors=model_to_color, ylabel=ylabel, title=title, hline=hline, hline_label=hline_label)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting {title}: {e}")
