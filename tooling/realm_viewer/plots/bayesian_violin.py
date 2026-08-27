import numpy as np
from scipy.stats import beta
import plotly.graph_objects as go

from ._markers import mpl_marker_to_plotly


def plot_bayesian_violin(labels, successes, failures, colors, symbols=None, model_names=None, model_colors=None, title=None, fontsize=12):
    fig = go.Figure(layout=dict(title=dict(text='')))

    x_positions = np.arange(len(labels))
    successes_arr = np.array(successes, dtype=float).flatten()
    failures_arr = np.array(failures, dtype=float).flatten()
    colors_arr = np.array(colors).flatten()
    symbols_arr = np.array(symbols).flatten() if symbols is not None else None

    task_names = [l.split(' - ')[0] for l in labels]
    unique_tasks = []
    unique_indices = []

    for data_idx, (S_raw, F_raw, color) in enumerate(zip(successes_arr, failures_arr, colors_arr)):
        S = int(S_raw) if not np.isnan(S_raw) else None
        F = int(F_raw) if not np.isnan(F_raw) else None

        if S is None or F is None or (S + F) == 0:
            continue

        task_name = task_names[data_idx]
        if task_name not in unique_tasks:
            unique_tasks.append(task_name)
            unique_indices.append(data_idx)

        alpha_param = S + 1
        beta_param_val = F + 1

        y = np.linspace(0.001, 0.999, 300)
        density = beta.pdf(y, alpha_param, beta_param_val)

        if density.max() == 0:
            continue

        mask = density >= (density.max() * 0.0001)
        y_clipped = y[mask]
        density_clipped = density[mask]

        x = density_clipped / density.max() * 0.325

        # Build closed polygon: left edge going up, right edge going down
        x_violin = np.concatenate([
            x_positions[data_idx] - x,
            (x_positions[data_idx] + x)[::-1],
        ])
        y_violin = np.concatenate([y_clipped, y_clipped[::-1]])

        posterior_mean = alpha_param / (alpha_param + beta_param_val)
        label = labels[data_idx]

        fig.add_trace(go.Scatter(
            x=x_violin.tolist(),
            y=y_violin.tolist(),
            fill='toself',
            fillcolor=str(color),
            line=dict(color=str(color), width=0),
            mode='lines',
            showlegend=False,
            hoveron='fills',
            hovertemplate=f'{label}<br>Mean: {posterior_mean:.3f}<br>N: {S + F}<extra></extra>',
            name=label,
        ))

        # Posterior mean horizontal line
        fig.add_trace(go.Scatter(
            x=[x_positions[data_idx] - 0.105, x_positions[data_idx] + 0.105],
            y=[posterior_mean, posterior_mean],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hovertemplate=f'{label}<br>Mean: {posterior_mean:.3f}<br>N: {S + F}<extra></extra>',
            name=label,
        ))

        # Model marker above violin
        if symbols_arr is not None and data_idx < len(symbols_arr):
            marker = symbols_arr[data_idx]
            if marker:
                y_pos = float(y_clipped.max()) + 0.05 if len(y_clipped) > 0 else 1.05
                fig.add_trace(go.Scatter(
                    x=[float(x_positions[data_idx])],
                    y=[y_pos],
                    mode='markers',
                    marker=dict(symbol=mpl_marker_to_plotly(str(marker)), color='black', size=8),
                    showlegend=False,
                    hoverinfo='skip',
                ))

    shapes = []
    annotations = []

    for i, task_label in enumerate(unique_tasks):
        start_idx = unique_indices[i]
        end_idx = unique_indices[i + 1] if i + 1 < len(unique_indices) else len(labels)
        group_size = end_idx - start_idx

        if i > 0:
            shapes.append(dict(
                type='line',
                x0=float(x_positions[start_idx]) - 0.5,
                x1=float(x_positions[start_idx]) - 0.5,
                y0=0, y1=1.15,
                line=dict(color='gray', width=0.8, dash='dash'),
                xref='x', yref='y',
            ))

        center_x = float(start_idx + (group_size - 1) / 2)
        annotations.append(dict(
            x=center_x,
            y=-0.1,
            xref='x', yref='paper',
            text=task_label,
            showarrow=False,
            textangle=-45,
            font=dict(size=max(8, fontsize - 2)),
            xanchor='right',
            yanchor='top',
        ))

    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        yaxis=dict(
            range=[0, 1.15],
            title='Success Rate',
            title_font=dict(size=fontsize),
            gridcolor='lightgray',
        ),
        xaxis=dict(
            range=[-0.5, len(labels) - 0.5],
            showticklabels=False,
            showgrid=False,
        ),
        title=dict(
            text=title or 'Binary Success Rate (Bayesian Posterior)',
            font=dict(size=fontsize + 2),
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(b=150, t=60, l=60, r=20),
        height=420,
    )

    return fig
