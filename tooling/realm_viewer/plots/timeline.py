import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go

from ._markers import mpl_marker_to_plotly


def plot_task_progression_timesteps_per_task(df, model_to_marker=None, model_to_color=None, title=None, fontsize=12):
    col_name = None
    for col in ['task_progression_timesteps', 'task_progression_timestamps', 'task_progression_timstamps']:
        if col in df.columns:
            col_name = col
            break

    if col_name is None or 'task' not in df.columns:
        return go.Figure()

    if 'perturbation' in df.columns:
        df_filtered = df[~df['perturbation'].astype(str).str.contains('SB-VRB')].copy()
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        return go.Figure()

    df_plot = df_filtered.copy()
    unique_models = sorted(df_plot['model'].unique()) if 'model' in df_plot.columns else ['Unknown']
    unique_tasks = sorted(df_plot['task'].unique())

    try:
        df_plot['parsed_ts'] = df_plot[col_name].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        max_stages = df_plot['parsed_ts'].apply(lambda x: len(x) if isinstance(x, list) else 0).max()
        if max_stages == 0:
            return go.Figure()

        def get_durations(ts_list):
            if not isinstance(ts_list, list) or len(ts_list) == 0:
                return []
            durations = []
            prev_ts = 0
            for ts in ts_list:
                durations.append(ts - prev_ts)
                prev_ts = ts
            return durations

        # Build stage color list from viridis
        cmap = plt.get_cmap('viridis')
        stage_colors = [mcolors.to_hex(cmap(i / max_stages)) for i in range(max_stages)]

        n_models = len(unique_models)
        n_tasks = len(unique_tasks)
        x_offset_width = 0.8 / n_models

        # Compute mean durations per (task, model, stage)
        mean_durations = {}  # (model, stage_idx) -> list of mean durations per task
        for model in unique_models:
            model_df = df_plot[df_plot['model'] == model].copy()
            if model_df.empty:
                for s in range(max_stages):
                    mean_durations[(model, s)] = [0.0] * n_tasks
                continue

            model_df['durations'] = model_df['parsed_ts'].apply(get_durations)
            model_df['durations'] = model_df['durations'].apply(
                lambda x: x + [np.nan] * (max_stages - len(x))
            )
            durations_df = pd.DataFrame(model_df['durations'].tolist(), index=model_df.index)
            durations_df['task'] = model_df['task']

            task_means = durations_df.groupby('task').mean().reindex(unique_tasks).fillna(0)
            for s in range(max_stages):
                mean_durations[(model, s)] = task_means[s].values.tolist() if s in task_means.columns else [0.0] * n_tasks

        # Build (task, model) pairs for x positions
        task_model_pairs = [(task, model) for task in unique_tasks for model in unique_models]
        n_bars = len(task_model_pairs)
        x_positions = np.arange(n_bars)

        fig = go.Figure(layout=dict(title=dict(text='')))

        for stage_idx in range(max_stages):
            y_vals = []
            for task, model in task_model_pairs:
                t_idx = unique_tasks.index(task)
                y_vals.append(mean_durations[(model, stage_idx)][t_idx])

            fig.add_trace(go.Bar(
                x=x_positions.tolist(),
                y=y_vals,
                name=f'Stage {stage_idx + 1}',
                marker_color=stage_colors[stage_idx],
                marker_line_width=0,
                opacity=0.8,
                hovertemplate=[
                    f'{task} / {model}<br>Stage {stage_idx + 1}: {y_vals[j]:.1f} steps<extra></extra>'
                    for j, (task, model) in enumerate(task_model_pairs)
                ],
            ))

        # Markers above stacked bars
        if model_to_marker:
            bar_totals = []
            for j, (task, model) in enumerate(task_model_pairs):
                t_idx = unique_tasks.index(task)
                total = sum(mean_durations[(model, s)][t_idx] for s in range(max_stages))
                bar_totals.append(total)

            for j, (task, model) in enumerate(task_model_pairs):
                if model in model_to_marker and bar_totals[j] > 0:
                    fig.add_trace(go.Scatter(
                        x=[int(x_positions[j])],
                        y=[bar_totals[j] + max(bar_totals) * 0.02 + 1],
                        mode='markers',
                        marker=dict(symbol=mpl_marker_to_plotly(model_to_marker[model]), color='black', size=7),
                        showlegend=False,
                        hoverinfo='skip',
                    ))

        # Task group separators and labels
        shapes = []
        annotations = []
        for t_idx, task in enumerate(unique_tasks):
            bar_start = t_idx * n_models

            if t_idx > 0:
                shapes.append(dict(
                    type='line',
                    x0=float(x_positions[bar_start]) - 0.5,
                    x1=float(x_positions[bar_start]) - 0.5,
                    y0=0, y1=1,
                    line=dict(color='gray', width=0.8, dash='dash'),
                    xref='x', yref='paper',
                ))

            center_x = float(bar_start + (n_models - 1) / 2)
            annotations.append(dict(
                x=center_x,
                y=-0.12,
                xref='x', yref='paper',
                text=task,
                showarrow=False,
                textangle=-45,
                font=dict(size=max(8, fontsize - 2)),
                xanchor='right',
                yanchor='top',
            ))

        fig.update_layout(
            barmode='stack',
            shapes=shapes,
            annotations=annotations,
            xaxis=dict(
                range=[-0.5, n_bars - 0.5],
                showticklabels=False,
                showgrid=False,
            ),
            yaxis=dict(title='Average Timesteps', gridcolor='lightgray'),
            legend=dict(title='Stage Step', orientation='h', y=-0.55, x=0.5, xanchor='center'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(b=220, t=50, l=60, r=20),
            height=500,
        )
        if title:
            fig.update_layout(title=dict(text=title, font=dict(size=fontsize + 2)))

        return fig

    except Exception as e:
        import streamlit as st
        st.error(f"Error plotting average timesteps: {e}")
        return go.Figure()
