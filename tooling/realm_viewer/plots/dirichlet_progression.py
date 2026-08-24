import numpy as np
from scipy.stats import gaussian_kde
import plotly.graph_objects as go

from ._markers import mpl_marker_to_plotly

_RNG = np.random.default_rng(42)
_N_SAMPLES = 2000


def _infer_modes(values):
    """
    Return sorted unique task_progression values (rounded to 3 dp).

    Uses an adaptive count floor of max(1, N // 50) to filter out rare
    floating-point artefacts from mixed staging schemes without discarding
    legitimate modes in small-N datasets (e.g. N=15 → floor=1, N=400 → floor=8).
    """
    arr = np.round(np.asarray(values, dtype=float), 3)
    arr = arr[~np.isnan(arr)]
    unique, counts = np.unique(arr, return_counts=True)
    min_count = max(1, len(arr) // 50)
    return np.sort(unique[counts >= min_count])


def plot_dirichlet_progression_violin(
    df,
    group_col,
    model_to_color=None,
    model_to_marker=None,
    title=None,
    fontsize=12,
    show_average=False,
):
    """
    One violin per (group, model) showing the Bayesian posterior over
    *expected task progression* under a uniform Dirichlet prior.

    Model
    -----
    Task progression takes K discrete values (modes) inferred from the data,
    e.g. {0, 1/3, 2/3, 1} for a 3-stage task.  Each trial is a categorical
    outcome.

      Prior     :  Dir(1, …, 1)          (K components, uniform)
      Posterior :  Dir(1+n_0, …, 1+n_{K-1})

    The quantity plotted on the y-axis is the *expected task progression*

        E[TP | θ] = Σ_k  mode_k · θ_k,   θ ~ Dir(posterior)

    obtained by Monte Carlo: draw _N_SAMPLES vectors θ from the Dirichlet
    posterior, compute the dot product with the mode values, then show the
    resulting scalar distribution as a KDE violin.

    At K=2 (binary success/fail, modes={0,1}) this reduces to an estimate of
    the success rate, directly comparable to the Beta-conjugate SR violins.

    Layout: same as the binary SR violin — one violin per (group, model),
    grouped by task / perturbation with dashed separators.
    """
    if 'task_progression' not in df.columns or group_col not in df.columns:
        return go.Figure()

    model_to_color = model_to_color or {}
    model_to_marker = model_to_marker or {}

    unique_groups = sorted(df[group_col].unique())
    unique_models = sorted(df['model'].unique()) if 'model' in df.columns else ['Unknown']

    fig = go.Figure(layout=dict(title=dict(text='')))
    x_cursor = 0
    shapes = []
    annotations = []

    for g_idx, group in enumerate(unique_groups):
        group_df = df[df[group_col] == group]
        all_vals = group_df['task_progression'].dropna().values
        if len(all_vals) == 0:
            continue

        # Modes inferred once per group — shared across all models so that
        # K is consistent within a group.
        modes = _infer_modes(all_vals)
        K = len(modes)
        if K == 0:
            continue

        if g_idx > 0:
            shapes.append(dict(
                type='line',
                x0=x_cursor - 0.5, x1=x_cursor - 0.5,
                y0=0, y1=1.15,
                line=dict(color='gray', width=0.8, dash='dash'),
                xref='x', yref='y',
            ))

        group_start_x = x_cursor

        for model in unique_models:
            cell_df = group_df[group_df['model'] == model] if 'model' in df.columns else group_df
            cell_vals = np.round(cell_df['task_progression'].dropna().values, 3)
            N = len(cell_vals)
            if N == 0:
                x_cursor += 1
                continue

            # Count observations at each mode
            n_k = np.array([int(np.sum(cell_vals == m)) for m in modes], dtype=float)

            # Dirichlet posterior parameters
            alpha = 1.0 + n_k   # shape (K,)

            # Posterior mean of E[TP] = Σ mode_k * alpha_k / alpha_0
            posterior_mean = float(modes @ (alpha / alpha.sum()))

            # Monte Carlo: sample θ ~ Dir(alpha), compute E[TP] = modes · θ
            theta = _RNG.dirichlet(alpha, _N_SAMPLES)   # (N_SAMPLES, K)
            expected_tp = theta @ modes                  # (N_SAMPLES,)

            # KDE over the expected-TP samples.
            # When all samples collapse to a single value (K=1 or near-zero
            # variance), gaussian_kde raises LinAlgError. Fall back to a
            # narrow Gaussian spike centred on the posterior mean.
            y = np.linspace(0.0, 1.0, 300)
            try:
                kde = gaussian_kde(expected_tp, bw_method='scott')
                density = kde(y)
            except np.linalg.LinAlgError:
                spike_sigma = 0.015
                density = np.exp(-0.5 * ((y - posterior_mean) / spike_sigma) ** 2)

            if density.max() == 0:
                x_cursor += 1
                continue

            mask = density >= density.max() * 0.0001
            y_c = y[mask]
            d_c = density[mask]
            x_half = d_c / density.max() * 0.325

            x_poly = np.concatenate([x_cursor - x_half, (x_cursor + x_half)[::-1]])
            y_poly = np.concatenate([y_c, y_c[::-1]])
            model_color = model_to_color.get(model, '#1f77b4')

            fig.add_trace(go.Scatter(
                x=x_poly.tolist(),
                y=y_poly.tolist(),
                fill='toself',
                fillcolor=model_color,
                line=dict(color=model_color, width=0),
                mode='lines',
                showlegend=False,
                hoveron='fills',
                hovertemplate=(
                    f'<b>{group}</b><br>'
                    f'Model: {model}<br>'
                    f'N: {N} trials, {K} modes<br>'
                    f'Mean E[TP]: {posterior_mean:.3f}'
                    '<extra></extra>'
                ),
            ))

            # Posterior mean tick
            fig.add_trace(go.Scatter(
                x=[x_cursor - 0.105, x_cursor + 0.105],
                y=[posterior_mean, posterior_mean],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hovertemplate=(
                    f'<b>{group}</b><br>'
                    f'Model: {model}<br>'
                    f'N: {N} trials, {K} modes<br>'
                    f'Mean E[TP]: {posterior_mean:.3f}'
                    '<extra></extra>'
                ),
            ))

            # Model marker above violin
            marker = model_to_marker.get(model)
            if marker:
                y_top = float(y_c.max()) + 0.05 if len(y_c) > 0 else 1.05
                fig.add_trace(go.Scatter(
                    x=[float(x_cursor)],
                    y=[y_top],
                    mode='markers',
                    marker=dict(
                        symbol=mpl_marker_to_plotly(str(marker)),
                        color='black',
                        size=8,
                    ),
                    showlegend=False,
                    hoverinfo='skip',
                ))

            x_cursor += 1

        center_x = (group_start_x + x_cursor - 1) / 2
        annotations.append(dict(
            x=center_x,
            y=-0.1,
            xref='x', yref='paper',
            text=str(group),
            showarrow=False,
            textangle=-45,
            font=dict(size=max(8, fontsize - 2)),
            xanchor='right',
            yanchor='top',
        ))

    # --- Average column (all groups combined), appended on the right ---
    if show_average:
        all_vals_global = df['task_progression'].dropna().values
        if len(all_vals_global) > 0:
            modes_global = _infer_modes(all_vals_global)
            K_global = len(modes_global)
            if K_global > 0:
                # Separator before average block
                shapes.append(dict(
                    type='line',
                    x0=x_cursor - 0.5, x1=x_cursor - 0.5,
                    y0=0, y1=1.15,
                    line=dict(color='gray', width=0.8, dash='dash'),
                    xref='x', yref='y',
                ))

                avg_start_x = x_cursor
                for model in unique_models:
                    cell_df = df[df['model'] == model] if 'model' in df.columns else df
                    cell_vals = np.round(cell_df['task_progression'].dropna().values, 3)
                    N = len(cell_vals)
                    if N == 0:
                        x_cursor += 1
                        continue

                    n_k = np.array([int(np.sum(cell_vals == m)) for m in modes_global], dtype=float)
                    alpha = 1.0 + n_k
                    posterior_mean = float(modes_global @ (alpha / alpha.sum()))

                    theta = _RNG.dirichlet(alpha, _N_SAMPLES)
                    expected_tp = theta @ modes_global

                    y = np.linspace(0.0, 1.0, 300)
                    try:
                        kde = gaussian_kde(expected_tp, bw_method='scott')
                        density = kde(y)
                    except np.linalg.LinAlgError:
                        spike_sigma = 0.015
                        density = np.exp(-0.5 * ((y - posterior_mean) / spike_sigma) ** 2)

                    if density.max() == 0:
                        x_cursor += 1
                        continue

                    mask = density >= density.max() * 0.0001
                    y_c = y[mask]
                    d_c = density[mask]
                    x_half = d_c / density.max() * 0.325
                    model_color = model_to_color.get(model, '#1f77b4')

                    x_poly = np.concatenate([x_cursor - x_half, (x_cursor + x_half)[::-1]])
                    y_poly = np.concatenate([y_c, y_c[::-1]])

                    fig.add_trace(go.Scatter(
                        x=x_poly.tolist(),
                        y=y_poly.tolist(),
                        fill='toself',
                        fillcolor=model_color,
                        line=dict(color=model_color, width=0),
                        mode='lines',
                        showlegend=False,
                        hoveron='fills',
                        hovertemplate=(
                            f'<b>Average (all tasks)</b><br>'
                            f'Model: {model}<br>'
                            f'N: {N} trials, {K_global} modes<br>'
                            f'Mean E[TP]: {posterior_mean:.3f}'
                            '<extra></extra>'
                        ),
                    ))

                    fig.add_trace(go.Scatter(
                        x=[x_cursor - 0.105, x_cursor + 0.105],
                        y=[posterior_mean, posterior_mean],
                        mode='lines',
                        line=dict(color='black', width=2),
                        showlegend=False,
                        hovertemplate=(
                            f'<b>Average (all tasks)</b><br>'
                            f'Model: {model}<br>'
                            f'N: {N} trials, {K_global} modes<br>'
                            f'Mean E[TP]: {posterior_mean:.3f}'
                            '<extra></extra>'
                        ),
                    ))

                    marker = model_to_marker.get(model)
                    if marker:
                        y_top = float(y_c.max()) + 0.05 if len(y_c) > 0 else 1.05
                        fig.add_trace(go.Scatter(
                            x=[float(x_cursor)],
                            y=[y_top],
                            mode='markers',
                            marker=dict(
                                symbol=mpl_marker_to_plotly(str(marker)),
                                color='black',
                                size=8,
                            ),
                            showlegend=False,
                            hoverinfo='skip',
                        ))

                    x_cursor += 1

                avg_center_x = (avg_start_x + x_cursor - 1) / 2
                annotations.append(dict(
                    x=avg_center_x,
                    y=-0.1,
                    xref='x', yref='paper',
                    text='Average',
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
            title='Expected Task Progression — Dirichlet posterior',
            title_font=dict(size=fontsize),
            gridcolor='lightgray',
        ),
        xaxis=dict(
            range=[-0.5, x_cursor - 0.5],
            showticklabels=False,
            showgrid=False,
        ),
        title=dict(
            text=title or 'Task Progression (Uniform Dirichlet Prior)',
            font=dict(size=fontsize + 2),
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(b=150, t=60, l=60, r=20),
        height=420,
    )

    return fig
