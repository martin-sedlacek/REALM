import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go

from ._markers import mpl_marker_to_plotly


def get_stage_progression_and_colors(df_plot):
    stage_progression = df_plot.groupby('simplified_stage')['task_progression'].mean().sort_values()
    ordered_labels = [l for l in stage_progression.index if l != 'Success']
    if 'Success' in stage_progression.index:
        ordered_labels.append('Success')

    unique_tps = sorted(stage_progression.unique())
    success_tp = stage_progression.get('Success', 1.0)
    failure_tps = [tp for tp in unique_tps if tp != success_tp]

    cmap = plt.get_cmap('Reds')
    tp_to_color = {}
    num_failure_tps = len(failure_tps)
    for i, tp in enumerate(failure_tps):
        rgba = cmap((i + 1) / (num_failure_tps + 1))
        tp_to_color[tp] = mcolors.to_hex(rgba)

    color_dict = {'Success': 'lightgreen'}
    for stage in ordered_labels:
        if stage == 'Success':
            continue
        tp = stage_progression[stage]
        color_dict[stage] = tp_to_color.get(tp, 'darkred')

    return stage_progression, ordered_labels, color_dict


def _group_annotations_and_shapes(unique_groups, unique_indices, n_labels, x_positions, fontsize, y_sep=1.1):

    shapes = []
    annotations = []
    for i, group_label in enumerate(unique_groups):
        start_idx = unique_indices[i]
        end_idx = unique_indices[i + 1] if i + 1 < len(unique_indices) else n_labels
        group_size = end_idx - start_idx

        if i > 0:
            shapes.append(dict(
                type='line',
                x0=float(x_positions[start_idx]) - 0.5,
                x1=float(x_positions[start_idx]) - 0.5,
                y0=0, y1=y_sep,
                line=dict(color='gray', width=0.8, dash='dash'),
                xref='x', yref='y',
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
    return shapes, annotations


def plot_stage_frequency(df, model_to_marker=None, model_to_color=None, title=None, fontsize=12):
    if df is None or df.empty or 'stage' not in df.columns or 'task_progression' not in df.columns:
        return go.Figure()

    df_plot = df.copy()
    df_plot['simplified_stage'] = df_plot['stage'].apply(
        lambda x: 'Success' if str(x).lower() == 'success' else str(x).split('|')[0].strip().split('_')[0].capitalize()
    )

    stage_progression, ordered_labels, color_dict = get_stage_progression_and_colors(df_plot)

    # Always show a "Reach" column, even when no reach failures are present in this data
    if 'Reach' not in ordered_labels:
        ordered_labels.insert(0, 'Reach')
        color_dict['Reach'] = mcolors.to_hex(plt.get_cmap('Reds')(0.25))

    unique_models = sorted(df_plot['model'].unique()) if 'model' in df_plot.columns else ['Unknown']

    n_models = len(unique_models)
    x_offset_width = 0.8 / n_models
    x_base = np.arange(len(ordered_labels))

    unique_tasks = sorted(df_plot['task'].unique()) if 'task' in df_plot.columns else []

    fig = go.Figure(layout=dict(title=dict(text='')))

    for i, model in enumerate(unique_models):
        model_df = df_plot[df_plot['model'] == model]

        # Macro-average: compute per-task stage proportions then average equally across tasks.
        per_task_props = []
        if unique_tasks:
            for task in unique_tasks:
                cell_df = model_df[model_df['task'] == task]
                if len(cell_df) == 0:
                    continue
                counts = cell_df['simplified_stage'].value_counts()
                total = len(cell_df)
                per_task_props.append([counts.get(l, 0) / total for l in ordered_labels])
            if per_task_props:
                proportions = list(np.mean(per_task_props, axis=0))
            else:
                proportions = [0.0] * len(ordered_labels)
        else:
            # Fallback: no task column available
            stage_counts = model_df['simplified_stage'].value_counts()
            total_rows = len(model_df)
            proportions = [stage_counts.get(l, 0) / total_rows if total_rows > 0 else 0 for l in ordered_labels]

        # Sanity-check printout: macro-averaged vs pooled proportions
        stage_counts_pooled = model_df['simplified_stage'].value_counts()
        total_rows = len(model_df)
        pooled = [stage_counts_pooled.get(l, 0) / total_rows if total_rows > 0 else 0 for l in ordered_labels]
        n_tasks_averaged = len(per_task_props) if unique_tasks else 'N/A'
        print(f"\n[stage_frequency] model={model}  tasks_averaged={n_tasks_averaged}")
        print(f"  {'Stage':<16}  {'Macro-avg':>10}  {'Pooled':>10}  {'Δ':>8}")
        for l, m_p, p_p in zip(ordered_labels, proportions, pooled):
            print(f"  {l:<16}  {m_p:>10.4f}  {p_p:>10.4f}  {m_p - p_p:>+8.4f}")

        stage_colors = [color_dict[label] for label in ordered_labels]
        x_pos = (x_base + (i - (n_models - 1) / 2) * x_offset_width).tolist()

        fig.add_trace(go.Bar(
            x=x_pos,
            y=proportions,
            name=model,
            marker_color=stage_colors,
            marker_line_width=0,
            width=x_offset_width,
            showlegend=False,
            opacity=0.8,
            hovertemplate=[
                f'{model} / {ordered_labels[j]}<br>Macro-avg proportion: {proportions[j]:.3f}<extra></extra>'
                for j in range(len(ordered_labels))
            ],
        ))

        if model_to_marker and model in model_to_marker:
            marker = model_to_marker[model]
            for j, p in enumerate(proportions):
                if p > 0:
                    fig.add_trace(go.Scatter(
                        x=[x_pos[j]],
                        y=[p + 0.025],
                        mode='markers',
                        marker=dict(symbol=mpl_marker_to_plotly(marker), color='black', size=7),
                        showlegend=False,
                        hoverinfo='skip',
                    ))

    note_text = (
        'Each task is weighted equally. Bar heights are the mean of per-task stage proportions, '
        'not a pooled proportion across all trials.'
    )

    fig.update_layout(
        barmode='group',
        xaxis=dict(
            tickvals=x_base.tolist(),
            ticktext=ordered_labels,
            tickangle=-45,
            showgrid=False,
        ),
        yaxis=dict(range=[0, 1.1], title='Frequency (Proportion)', gridcolor='lightgray'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(b=120, t=50, l=60, r=20),
        height=420,
        annotations=[dict(
            x=0.5, y=-0.28,
            xref='paper', yref='paper',
            text=note_text,
            showarrow=False,
            font=dict(size=10, color='gray'),
            xanchor='center',
            yanchor='top',
        )],
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=fontsize + 2)))

    return fig


def plot_stage_frequency_per_task(df, model_to_marker=None, model_to_color=None, title=None, fontsize=12):
    if df is None or df.empty or 'stage' not in df.columns or 'task' not in df.columns or 'task_progression' not in df.columns:
        return go.Figure()

    if 'perturbation' in df.columns:
        df_filtered = df[~df['perturbation'].astype(str).str.contains('SB-VRB')].copy()
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        return go.Figure()

    df_plot = df_filtered.copy()
    df_plot['simplified_stage'] = df_plot['stage'].apply(
        lambda x: 'Success' if str(x).lower() == 'success' else str(x).split('|')[0].strip().split('_')[0].capitalize()
    )

    stage_progression, ordered_stages, color_dict = get_stage_progression_and_colors(df_plot)
    unique_models = sorted(df_plot['model'].unique()) if 'model' in df_plot.columns else ['Unknown']
    unique_tasks = sorted(df_plot['task'].unique())

    n_models = len(unique_models)
    n_tasks = len(unique_tasks)

    # x positions: for each task, n_models side-by-side bars, each stacked by stage
    # x_labels: one entry per (task, model) pair
    task_model_pairs = [(task, model) for task in unique_tasks for model in unique_models]
    n_bars = len(task_model_pairs)
    x_positions = np.arange(n_bars)

    fig = go.Figure(layout=dict(title=dict(text='')))

    # Compute proportions per (task, model, stage)
    proportions_by_stage = {}
    for stage in ordered_stages:
        y_vals = []
        for task, model in task_model_pairs:
            model_task_df = df_plot[(df_plot['model'] == model) & (df_plot['task'] == task)]
            if model_task_df.empty:
                y_vals.append(0.0)
            else:
                stage_counts = model_task_df['simplified_stage'].value_counts()
                total = len(model_task_df)
                y_vals.append(stage_counts.get(stage, 0) / total)
        proportions_by_stage[stage] = y_vals

    for stage in ordered_stages:
        fig.add_trace(go.Bar(
            x=x_positions.tolist(),
            y=proportions_by_stage[stage],
            name=stage,
            marker_color=color_dict[stage],
            marker_line_width=0,
            opacity=0.8,
            hovertemplate=[
                f'{task} / {model}<br>{stage}: {proportions_by_stage[stage][j]:.3f}<extra></extra>'
                for j, (task, model) in enumerate(task_model_pairs)
            ],
        ))

    # Add markers above stacked bars
    if model_to_marker:
        bar_totals = [
            sum(proportions_by_stage[s][j] for s in ordered_stages)
            for j in range(n_bars)
        ]
        for j, (task, model) in enumerate(task_model_pairs):
            if model in model_to_marker and bar_totals[j] > 0:
                fig.add_trace(go.Scatter(
                    x=[int(x_positions[j])],
                    y=[bar_totals[j] + 0.025],
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
        bar_end = bar_start + n_models

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
        yaxis=dict(range=[0, 1.15], title='Frequency', gridcolor='lightgray'),
        legend=dict(title='Stage', orientation='h', y=-0.55, x=0.5, xanchor='center'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(b=220, t=50, l=60, r=20),
        height=500,
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=fontsize + 2)))

    return fig
