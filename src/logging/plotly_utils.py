import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Any

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
            "Disp. from Last (Å)", "Disp. from Start (Å)"
        )
    )

    # --- ROW 1: Energy & Force ---
    fig.add_trace(go.Scatter(x=steps, y=get_data("energy"), mode='lines+markers', name='Energy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=get_data("force_mean"), mode='lines+markers', name='Force Mean', line=dict(color='orange')), row=1, col=2)

    # --- ROW 2: Eig Product & Eigenvalues ---
    
    # Eig Product
    fig.add_trace(go.Scatter(x=steps, y=get_data("eig_product"), mode='lines+markers', name='λ₀·λ₁', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Eigenvalues (Two lines on one plot)
    fig.add_trace(go.Scatter(x=steps, y=get_data("eig0"), mode='lines+markers', name='λ₀', line=dict(color='red')), row=2, col=2)
    fig.add_trace(go.Scatter(x=steps, y=get_data("eig1"), mode='lines+markers', name='λ₁', line=dict(color='green')), row=2, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=2)

    # --- ROW 3: Displacements ---
    fig.add_trace(go.Scatter(x=steps, y=get_data("disp_from_last"), mode='lines+markers', name='Disp (Last)', line=dict(color='crimson')), row=3, col=1)
    fig.add_trace(go.Scatter(x=steps, y=get_data("disp_from_start"), mode='lines+markers', name='Disp (Start)', line=dict(color='blue')), row=3, col=2)

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
        
    fig.update_layout(
        height=900, 
        width=1000, 
        title_text=title_text,
        showlegend=True,
        hovermode="x unified"  # This shows all values for a specific step when hovering
    )

    return fig
