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

    This function creates a 2x2 grid of plots showing:
    - Energy vs step
    - Force metrics (max and mean) vs step
    - Force max vs step (for convergence tracking)
    - Final eigenvalue information as annotation

    Args:
        trajectory: Dictionary with lists of per-step metrics (energy, force_max, force_mean)
        sample_index: Sample index for title
        formula: Molecular formula for title
        start_from: Starting geometry type
        initial_neg_num: Initial number of negative eigenvalues
        final_neg_num: Final number of negative eigenvalues
        converged: Whether Sella converged
        final_eig0: Final lowest eigenvalue (optional)
        final_eig1: Final second eigenvalue (optional)
        final_eig_product: Final eigenvalue product (optional)

    Returns:
        Plotly Figure object
    """
    # Helper to handle None/NaN for Plotly
    def get_data(key):
        return [x if x is not None else float('nan') for x in trajectory.get(key, [])]

    energy = get_data("energy")
    force_max = get_data("force_max")
    force_mean = get_data("force_mean")

    n = max(len(energy), len(force_max), len(force_mean))
    if n == 0:
        n = 1  # Avoid empty plots
    steps = np.arange(n)

    # Create 2x2 subplot grid
    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
        subplot_titles=(
            "Energy (eV)", "Force Metrics (eV/Å)",
            "Max Force (log scale)", "Eigenvalue Summary"
        )
    )

    # --- ROW 1, COL 1: Energy ---
    if len(energy) > 0:
        fig.add_trace(go.Scatter(
            x=steps[:len(energy)], y=energy,
            mode='lines+markers', name='Energy',
            line=dict(color='blue', width=1.5),
            marker=dict(size=3)
        ), row=1, col=1)

    # --- ROW 1, COL 2: Force metrics (max and mean) ---
    if len(force_max) > 0:
        fig.add_trace(go.Scatter(
            x=steps[:len(force_max)], y=force_max,
            mode='lines+markers', name='Max |F|',
            line=dict(color='red', width=1.5),
            marker=dict(size=3)
        ), row=1, col=2)

    if len(force_mean) > 0:
        fig.add_trace(go.Scatter(
            x=steps[:len(force_mean)], y=force_mean,
            mode='lines+markers', name='Mean |F|',
            line=dict(color='orange', width=1.5),
            marker=dict(size=3)
        ), row=1, col=2)

    # --- ROW 2, COL 1: Force max (log scale) for convergence ---
    if len(force_max) > 0:
        # Filter positive values for log scale
        force_max_positive = [f if f and f > 0 else 1e-10 for f in force_max]
        fig.add_trace(go.Scatter(
            x=steps[:len(force_max_positive)], y=force_max_positive,
            mode='lines+markers', name='Max |F| (log)',
            line=dict(color='crimson', width=1.5),
            marker=dict(size=3)
        ), row=2, col=1)
        fig.update_yaxes(type="log", row=2, col=1)

        # Add convergence threshold line (typical 0.03 eV/A)
        fig.add_hline(y=0.03, line_dash="dash", line_color="green",
                      opacity=0.7, row=2, col=1,
                      annotation_text="fmax=0.03", annotation_position="right")

    # --- ROW 2, COL 2: Eigenvalue summary (as annotation/bar) ---
    # Create a simple bar chart showing final eigenvalues
    if final_eig0 is not None and final_eig1 is not None:
        eig_labels = ['λ₀', 'λ₁']
        eig_values = [final_eig0, final_eig1]
        colors = ['red' if v < 0 else 'green' for v in eig_values]

        fig.add_trace(go.Bar(
            x=eig_labels, y=eig_values,
            name='Final Eigenvalues',
            marker_color=colors,
            text=[f'{v:.4f}' for v in eig_values],
            textposition='outside'
        ), row=2, col=2)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=2)
    else:
        # No eigenvalue data - add placeholder text
        fig.add_annotation(
            text="Eigenvalue data not available",
            xref="x4", yref="y4",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray"),
            row=2, col=2
        )

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
        height=700,
        title_text=title_text,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=55, r=25, t=80, b=55),
        font=dict(size=12),
        hovermode="x unified",
        hoverlabel=dict(font_size=12),
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

    # Add step label to bottom plots
    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_xaxes(title_text="", row=2, col=2)

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
