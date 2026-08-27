import ast
import io

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from plots.bayesian_violin import plot_bayesian_violin
from plots.bar_charts import plot_grouped_bars_with_symbols
from plots.stage_analysis import plot_stage_frequency, plot_stage_frequency_per_task
from plots.timeline import plot_task_progression_timesteps_per_task
from plots.legend import plot_model_legend


def prepare_report_data(df):

    df = df.copy()

    col_name = None
    for col in ['task_progression_timesteps', 'task_progression_timestamps', 'task_progression_timstamps']:
        if col in df.columns:
            col_name = col
            break

    if col_name:
        def get_comp_time(row):
            if row.get('task_progression') == 1:
                try:
                    ts = ast.literal_eval(row[col_name]) if isinstance(row[col_name], str) else row[col_name]
                    return ts[-1] if isinstance(ts, list) and len(ts) > 0 else np.nan
                except:
                    return np.nan
            return np.nan
        df['completion_time'] = df.apply(get_comp_time, axis=1)

    if 'perturbation' in df.columns:
        df['clean_pert'] = df['perturbation'].apply(lambda x: x.replace("['", "").replace("']", "") if isinstance(x, str) else str(x))

    return df


def generate_comprehensive_report(df_full, model_to_marker, model_to_color):

    df_report = prepare_report_data(df_full)

    metrics = [
        {'col': 'binary_SR', 'title': 'Success Rate', 'ylabel': 'Success Rate', 'type': 'bayesian'},
        {'col': 'task_progression', 'title': 'Task Progression', 'ylabel': 'Task Progression', 'type': 'bar'},
        {'col': 'completion_time', 'title': 'Time to Completion', 'ylabel': 'Average Completion Time', 'type': 'bar'},
        {'col': 'joint_vel_var', 'title': 'Joint Velocity Variance', 'ylabel': 'Mean Variance', 'type': 'bar'},
        {'col': 'joint_acc_var', 'title': 'Joint Acceleration Variance', 'ylabel': 'Mean Variance', 'type': 'bar'},
        {'col': 'joint_jerk', 'title': 'Joint Jerk', 'ylabel': 'Mean Jerk', 'type': 'bar'},
        {'col': 'cart_jerk', 'title': 'Cartesian Jerk', 'ylabel': 'Mean Jerk', 'type': 'bar'},
        {'col': 'joint_path_length', 'title': 'Joint Path Length', 'ylabel': 'Mean Path Length', 'type': 'bar'},
        {'col': 'cart_path_length', 'title': 'Cartesian Path Length', 'ylabel': 'Mean Path Length', 'type': 'bar'},
        {'col': 'collisions_env', 'title': 'Environment Collisions', 'ylabel': 'Count', 'type': 'bar'},
        {'col': 'object_drops', 'title': 'Object Drops', 'ylabel': 'Count', 'type': 'bar'},
    ]

    available_perts = sorted(df_report['clean_pert'].unique()) if 'clean_pert' in df_report.columns else []

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        # Title Page
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.5, 0.6, "REALM Toolkit: Comprehensive Experiment Report", ha='center', va='center', fontsize=24, fontweight='bold')
        plt.text(0.5, 0.4, f"Generated for {len(df_report)} trials across {len(available_perts)} perturbations", ha='center', va='center', fontsize=16)
        pdf.savefig(fig)
        plt.close(fig)

        # Legend Page
        fig, ax = plt.subplots(figsize=(11, 2))
        plot_model_legend(ax, model_to_marker, model_to_color)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        for m_config in metrics:
            m_col = m_config['col']
            if m_col not in df_report.columns and m_col != 'binary_SR':
                continue

            # All perturbations together
            fig, (ax, ax_leg) = plt.subplots(2, 1, figsize=(12, 8.5), gridspec_kw={'height_ratios': [7, 1]})
            if m_config['type'] == 'bayesian':
                task_stats = df_report.groupby(['task', 'model'])['binary_SR'].agg(['sum', 'count']).reset_index()
                task_stats = task_stats.sort_values(['task', 'model'])
                labels = [f"{row['task']} - {row['model']}" for _, row in task_stats.iterrows()]
                successes = task_stats['sum'].tolist()
                failures = (task_stats['count'] - task_stats['sum']).tolist()
                colors = [model_to_color[row['model']] for _, row in task_stats.iterrows()]
                symbols = [model_to_marker[row['model']] for _, row in task_stats.iterrows()]
                model_names = task_stats['model'].tolist()
                plot_bayesian_violin(ax, labels, successes, failures, colors, symbols=symbols, model_names=model_names, model_colors=model_to_color, title=f"{m_config['title']} (All Perturbations)", fontsize=14)
            else:
                metric_df = df_report[df_report[m_col].notna()].copy()
                if not metric_df.empty:
                    stats = metric_df.groupby(['task', 'model'])[m_col].mean().reset_index()
                    stats = stats.sort_values(['task', 'model'])
                    labels = [f"{row['task']} - {row['model']}" for _, row in stats.iterrows()]
                    values = stats[m_col].tolist()
                    colors = [model_to_color[row['model']] for _, row in stats.iterrows()]
                    symbols = [model_to_marker[row['model']] for _, row in stats.iterrows()]
                    model_names = stats['model'].tolist()
                    plot_grouped_bars_with_symbols(ax, labels, values, colors, symbols, model_names, model_colors=model_to_color, ylabel=m_config['ylabel'], title=f"{m_config['title']} (All Perturbations)", fontsize=10)

            plot_model_legend(ax_leg, model_to_marker, model_to_color)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Per perturbation
            for pert in available_perts:
                pert_df = df_report[df_report['clean_pert'] == pert]
                fig, (ax, ax_leg) = plt.subplots(2, 1, figsize=(12, 8.5), gridspec_kw={'height_ratios': [7, 1]})

                if m_config['type'] == 'bayesian':
                    task_stats = pert_df.groupby(['task', 'model'])['binary_SR'].agg(['sum', 'count']).reset_index()
                    if not task_stats.empty:
                        task_stats = task_stats.sort_values(['task', 'model'])
                        labels = [f"{row['task']} - {row['model']}" for _, row in task_stats.iterrows()]
                        successes = task_stats['sum'].tolist()
                        failures = (task_stats['count'] - task_stats['sum']).tolist()
                        colors = [model_to_color[row['model']] for _, row in task_stats.iterrows()]
                        symbols = [model_to_marker[row['model']] for _, row in task_stats.iterrows()]
                        model_names = task_stats['model'].tolist()
                        plot_bayesian_violin(ax, labels, successes, failures, colors, symbols=symbols, model_names=model_names, model_colors=model_to_color, title=f"{m_config['title']} - {pert}", fontsize=14)
                else:
                    metric_df = pert_df[pert_df[m_col].notna()].copy()
                    if not metric_df.empty:
                        stats = metric_df.groupby(['task', 'model'])[m_col].mean().reset_index()
                        stats = stats.sort_values(['task', 'model'])
                        labels = [f"{row['task']} - {row['model']}" for _, row in stats.iterrows()]
                        values = stats[m_col].tolist()
                        colors = [model_to_color[row['model']] for _, row in stats.iterrows()]
                        symbols = [model_to_marker[row['model']] for _, row in stats.iterrows()]
                        model_names = stats['model'].tolist()
                        plot_grouped_bars_with_symbols(ax, labels, values, colors, symbols, model_names, model_colors=model_to_color, ylabel=m_config['ylabel'], title=f"{m_config['title']} - {pert}", fontsize=10)

                if ax.has_data():
                    plot_model_legend(ax_leg, model_to_marker, model_to_color)
                    pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        # Stage-related metrics
        stage_metrics = [
            {'func': plot_stage_frequency, 'title': 'Failure Stage Frequency'},
            {'func': plot_stage_frequency_per_task, 'title': 'Failure Stage Frequency per Task'},
            {'func': plot_task_progression_timesteps_per_task, 'title': 'Stage Timesteps per Task'},
        ]

        for sm in stage_metrics:
            fig, (ax, ax_leg) = plt.subplots(2, 1, figsize=(12, 8.5), gridspec_kw={'height_ratios': [7, 1]})
            sm['func'](df_report, ax, model_to_marker=model_to_marker, model_to_color=model_to_color, title=f"{sm['title']} (All Perturbations)")
            plot_model_legend(ax_leg, model_to_marker, model_to_color)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            for pert in available_perts:
                pert_df = df_report[df_report['clean_pert'] == pert]
                if pert_df.empty:
                    continue
                fig, (ax, ax_leg) = plt.subplots(2, 1, figsize=(12, 8.5), gridspec_kw={'height_ratios': [7, 1]})
                sm['func'](pert_df, ax, model_to_marker=model_to_marker, model_to_color=model_to_color, title=f"{sm['title']} - {pert}")
                plot_model_legend(ax_leg, model_to_marker, model_to_color)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    buffer.seek(0)
    return buffer
