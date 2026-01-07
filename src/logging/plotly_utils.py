import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Any


def plot_sella_trajectory_interactive(
    trajectory: Dict[str, List[Optional[float]]],
    sample_index: int,
    formula: str,
    start_from: str,
    initial_neg_num: int,
    final_neg_num: int,
    *,
    converged: bool = False,
    final_eig0: Optional[float] = None,
    final_eig1: Optional[float] = None,
    final_eig_product: Optional[float] = None,
) -> go.Figure:
    """Create interactive Plotly figure for Sella optimization trajectory.

    This function creates a 4x2 grid of plots (7 used) showing:
    - Row 1: Energy, Force Mean
    - Row 2: Eigenvalue Product, Eigenvalues (λ₀, λ₁)
    - Row 3: Displacement from Last, Displacement from Start
    - Row 4: Number of Negative Eigenvalues, Force Convergence (log)

    Args:
        trajectory: Dictionary with per-step metrics including:
            - energy, force_max, force_mean (always available)
            - eig0, eig1, eig_product, neg_vib (computed at intervals)
            - disp_from_last, disp_from_start (always available)
        sample_index: Sample index for title
        formula: Molecular formula for title
        start_from: Starting geometry type
        initial_neg_num: Initial number of negative eigenvalues
        final_neg_num: Final number of negative eigenvalues
        converged: Whether Sella converged
        final_eig0: Final lowest eigenvalue (optional, for title)
        final_eig1: Final second eigenvalue (optional, for title)
        final_eig_product: Final eigenvalue product (optional, for title)

    Returns:
        Plotly Figure object
    """
    # Helper to handle None/NaN for Plotly
    def get_data(key):
        return [x if x is not None else float('nan') for x in trajectory.get(key, [])]

    # Extract all data
    energy = get_data("energy")
    force_mean = get_data("force_mean")
    force_max = get_data("force_max")
    eig_product = get_data("eig_product")
    eig0 = get_data("eig0")
    eig1 = get_data("eig1")
    neg_vib = get_data("neg_vib")
    disp_from_last = get_data("disp_from_last")
    disp_from_start = get_data("disp_from_start")

    # Determine number of steps
    n = max(len(energy), len(force_mean), len(disp_from_last), 1)
    steps = np.arange(n)

    # Create 4x2 subplot grid (7 plots + 1 for force convergence)
    fig = make_subplots(
        rows=4, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
        subplot_titles=(
            "Energy (eV)", "Force Mean (eV/Å)",
            "Eigenvalue Product (λ₀·λ₁)", "Eigenvalues (λ₀, λ₁)",
            "Mean Atom Disp. from Last (Å)", "Mean Atom Disp. from Start (Å)",
            "Negative Eigenvalue Count", "Force Convergence (log)"
        )
    )

    # --- ROW 1: Energy & Force Mean ---
    if len(energy) > 0:
        fig.add_trace(go.Scatter(
            x=steps[:len(energy)], y=energy,
            mode='lines+markers', name='Energy',
            line=dict(color='blue', width=1.5),
            marker=dict(size=3)
        ), row=1, col=1)

    if len(force_mean) > 0:
        fig.add_trace(go.Scatter(
            x=steps[:len(force_mean)], y=force_mean,
            mode='lines+markers', name='Force Mean',
            line=dict(color='orange', width=1.5),
            marker=dict(size=3)
        ), row=1, col=2)

    # --- ROW 2: Eigenvalue Product & Eigenvalues ---
    if len(eig_product) > 0:
        fig.add_trace(go.Scatter(
            x=steps[:len(eig_product)], y=eig_product,
            mode='lines+markers', name='λ₀·λ₁',
            line=dict(color='purple', width=1.5),
            marker=dict(size=3),
            connectgaps=False  # Don't connect across None values
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Eigenvalues (two lines)
    if len(eig0) > 0:
        fig.add_trace(go.Scatter(
            x=steps[:len(eig0)], y=eig0,
            mode='lines+markers', name='λ₀',
            line=dict(color='red', width=1.5),
            marker=dict(size=3),
            connectgaps=False
        ), row=2, col=2)

    if len(eig1) > 0:
        fig.add_trace(go.Scatter(
            x=steps[:len(eig1)], y=eig1,
            mode='lines+markers', name='λ₁',
            line=dict(color='green', width=1.5),
            marker=dict(size=3),
            connectgaps=False
        ), row=2, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=2)

    # --- ROW 3: Displacements ---
    if len(disp_from_last) > 0:
        fig.add_trace(go.Scatter(
            x=steps[:len(disp_from_last)], y=disp_from_last,
            mode='lines+markers', name='Disp (Last)',
            line=dict(color='crimson', width=1.5),
            marker=dict(size=3)
        ), row=3, col=1)

    if len(disp_from_start) > 0:
        fig.add_trace(go.Scatter(
            x=steps[:len(disp_from_start)], y=disp_from_start,
            mode='lines+markers', name='Disp (Start)',
            line=dict(color='teal', width=1.5),
            marker=dict(size=3)
        ), row=3, col=2)

    # --- ROW 4: Neg Eigenvalue Count & Force Convergence ---
    if len(neg_vib) > 0:
        # Filter out NaN values for plotting
        valid_steps = [s for s, v in zip(steps[:len(neg_vib)], neg_vib) if not np.isnan(v)]
        valid_neg_vib = [v for v in neg_vib if not np.isnan(v)]
        if valid_neg_vib:
            fig.add_trace(go.Scatter(
                x=valid_steps, y=valid_neg_vib,
                mode='lines+markers', name='Neg. Eigenvalues',
                line=dict(color='darkviolet', width=1.5),
                marker=dict(size=4)
            ), row=4, col=1)
            # Add horizontal line at 1 (TS target)
            fig.add_hline(y=1, line_dash="dash", line_color="green",
                          opacity=0.7, row=4, col=1,
                          annotation_text="TS (n=1)", annotation_position="right")

    # Force convergence (log scale)
    if len(force_max) > 0:
        force_max_positive = [f if f and f > 0 else 1e-10 for f in force_max]
        fig.add_trace(go.Scatter(
            x=steps[:len(force_max_positive)], y=force_max_positive,
            mode='lines+markers', name='Max |F| (log)',
            line=dict(color='darkred', width=1.5),
            marker=dict(size=3)
        ), row=4, col=2)
        fig.update_yaxes(type="log", row=4, col=2)
        # Add convergence threshold
        fig.add_hline(y=0.03, line_dash="dash", line_color="green",
                      opacity=0.7, row=4, col=2,
                      annotation_text="fmax=0.03", annotation_position="right")

    # Build title
    status = "converged" if converged else "not converged"
    title_text = f"Sella Sample {sample_index}: {formula} [{initial_neg_num}→{final_neg_num} neg] ({status})"
    if "_noise" in start_from:
        title_text += f" ({start_from})"

    # Add eigenvalue product annotation if available
    if final_eig_product is not None:
        eig_sign = "< 0 (TS)" if final_eig_product < 0 else "> 0"
        title_text += f" | λ₀·λ₁={final_eig_product:.2e} {eig_sign}"

    # Global layout
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        height=1100,
        title_text=title_text,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=55, r=25, t=80, b=55),
        font=dict(size=11),
        hovermode="x unified",
        hoverlabel=dict(font_size=11),
    )

    # Configure axes for better readability
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor="rgba(0,0,0,0.35)",
        showline=True,
        mirror=True,
        ticks="outside",
        ticklen=4,
        tickwidth=1,
        gridcolor="rgba(0,0,0,0.08)",
    )
    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor="rgba(0,0,0,0.35)",
        showline=True,
        mirror=True,
        ticks="outside",
        ticklen=4,
        tickwidth=1,
        gridcolor="rgba(0,0,0,0.08)",
    )

    # Add step label to bottom row only
    fig.update_xaxes(title_text="Step", row=4, col=1)
    fig.update_xaxes(title_text="Step", row=4, col=2)

    return fig


def plot_gad_trajectory_interactive(
    trajectory: Dict[str, List[Optional[float]]],
    sample_index: int,
    formula: str,
    start_from: str,
    initial_neg_num: int,
    final_neg_num: int,
    *,
    steps_to_ts: Optional[int] = None,
) -> go.Figure:
    
    # 1. Prepare Data
    steps = np.arange(len(trajectory["energy"]))
    
    # Helper to handle None/NaN for Plotly (it handles None gracefully usually, but let's be safe)
    def get_data(key):
        return [x if x is not None else float('nan') for x in trajectory.get(key, [])]

    # 2. Create Subplots (3 rows, 2 columns)
    # shared_xaxes=True is the MAGIC setting. It links zooming across all plots.
    fig = make_subplots(
        rows=3, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "Energy (eV)", "Force Mean (eV/Å)", 
            "Eigenvalue Product", "Eigenvalues (λ₀, λ₁)", 
            "Mean Atom Disp. from Last (Å)", "Mean Atom Disp. from Start (Å)"
        )
    )

    # --- ROW 1: Energy & Force ---
    fig.add_trace(go.Scatter(x=steps, y=get_data("energy"), mode='lines+markers', name='Energy', 
                            line=dict(width=1.5), marker=dict(size=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=get_data("force_mean"), mode='lines+markers', name='Force Mean', 
                            line=dict(color='orange', width=1.5), marker=dict(size=3)), row=1, col=2)

    # --- ROW 2: Eig Product & Eigenvalues ---
    
    # Eig Product
    fig.add_trace(go.Scatter(x=steps, y=get_data("eig_product"), mode='lines+markers', name='λ₀·λ₁', 
                            line=dict(color='purple', width=1.5), marker=dict(size=3)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Eigenvalues (Two lines on one plot)
    fig.add_trace(go.Scatter(x=steps, y=get_data("eig0"), mode='lines+markers', name='λ₀', 
                            line=dict(color='red', width=1.5), marker=dict(size=3)), row=2, col=2)
    fig.add_trace(go.Scatter(x=steps, y=get_data("eig1"), mode='lines+markers', name='λ₁', 
                            line=dict(color='green', width=1.5), marker=dict(size=3)), row=2, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=2)

    # --- ROW 3: Displacements ---
    fig.add_trace(go.Scatter(x=steps, y=get_data("disp_from_last"), mode='lines+markers', name='Mean Atom Disp (Last)', 
                            line=dict(color='crimson', width=1.5), marker=dict(size=3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=steps, y=get_data("disp_from_start"), mode='lines+markers', name='Mean Atom Disp (Start)', 
                            line=dict(color='blue', width=1.5), marker=dict(size=3)), row=3, col=2)

    # 3. Add TS Marker (Vertical Line)
    if steps_to_ts is not None:
        # Add a vertical line to ALL rows
        for r in [1, 2, 3]:
            for c in [1, 2]:
                fig.add_vline(x=steps_to_ts, line_width=2, line_dash="dash", line_color="green", opacity=0.5, row=r, col=c)
        
        # Add annotation only to the first plot to avoid clutter
        fig.add_annotation(x=steps_to_ts, yref="y domain", y=1.05, text="TS Found", showarrow=False, row=1, col=1)

    # 4. Global Layout Updates
    title_text = f"Sample {sample_index}: {formula} [{initial_neg_num}→{final_neg_num} neg]"
    if "_noise" in start_from:
        title_text += f" ({start_from})"
        
    # Plot defaults tuned for readability in embedded viewers (e.g., W&B panels).
    # Keep `autosize=True` so the host container can control width.
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        height=950,
        title_text=title_text,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=55, r=25, t=80, b=55),
        font=dict(size=12),
        hovermode="x unified",  # show all values for a step
        hoverlabel=dict(font_size=12),
    )

    # Make zooming/reading easier: spikelines across subplots and clearer axes.
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor="rgba(0,0,0,0.35)",
        showline=True,
        mirror=True,
        ticks="outside",
        ticklen=4,
        tickwidth=1,
        gridcolor="rgba(0,0,0,0.08)",
    )
    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor="rgba(0,0,0,0.35)",
        showline=True,
        mirror=True,
        ticks="outside",
        ticklen=4,
        tickwidth=1,
        gridcolor="rgba(0,0,0,0.08)",
    )

    return fig
