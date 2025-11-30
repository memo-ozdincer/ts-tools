"""
Saddle point detection and classification for transition state search.

This module provides utilities to classify geometries based on their Hessian
eigenvalue spectrum and recommend adaptive step sizes for optimization.
"""

import torch
from typing import Dict, Union


def classify_saddle_point(
    eigvals_vibrational: torch.Tensor,
    force_norm: float,
    neg_threshold: float = -0.001
) -> Dict[str, Union[int, float, bool, str]]:
    """
    Classify current geometry based on eigenvalue spectrum.

    Args:
        eigvals_vibrational: Vibrational eigenvalues (rigid modes already removed),
                            sorted in ascending order
        force_norm: Magnitude of force/gradient (eV/Å)
        neg_threshold: Threshold for counting negative eigenvalues (eV/Å²)
                      Default: -0.001 eV/Å²

    Returns:
        Dictionary containing:
            - saddle_order (int): Number of negative eigenvalues
            - num_negative (int): Same as saddle_order
            - num_positive (int): Number of positive eigenvalues
            - smallest_eigval (float): Most negative eigenvalue (eV/Å²)
            - second_eigval (float or None): Second smallest eigenvalue
            - is_converged (bool): True if forces < 1e-3 eV/Å
            - classification (str): 'minimum', 'ts', 'higher_order', or 'unconverged'

    Classification logic:
        - 'unconverged': Forces too large (not at stationary point)
        - 'minimum': 0 negative eigenvalues (local minimum)
        - 'ts': Exactly 1 negative eigenvalue (transition state)
        - 'higher_order': 2+ negative eigenvalues (higher-order saddle)
    """
    # Count negative and positive eigenvalues
    num_negative = (eigvals_vibrational < neg_threshold).sum().item()
    num_positive = (eigvals_vibrational >= neg_threshold).sum().item()

    saddle_order = num_negative
    is_converged = force_norm < 1e-3

    # Classify geometry
    if not is_converged:
        classification = 'unconverged'
    elif num_negative == 0:
        classification = 'minimum'
    elif num_negative == 1:
        classification = 'ts'
    else:  # num_negative >= 2
        classification = 'higher_order'

    # Extract key eigenvalues
    smallest_eigval = eigvals_vibrational[0].item()
    second_eigval = eigvals_vibrational[1].item() if len(eigvals_vibrational) > 1 else None

    return {
        'saddle_order': saddle_order,
        'num_negative': num_negative,
        'num_positive': num_positive,
        'smallest_eigval': smallest_eigval,
        'second_eigval': second_eigval,
        'is_converged': is_converged,
        'classification': classification,
    }


def compute_adaptive_step_scale(
    saddle_info: Dict[str, Union[int, float, bool, str]],
    base_scale: float = 1.0,
    higher_order_mult: float = 5.0,
    ts_mult: float = 0.5
) -> float:
    """
    Compute step size scaling factor based on saddle order.

    Strategy:
        - Higher-order saddles (order 2+): Use LARGE steps to escape
          the broad saddle region. Scale increases with saddle order.
        - Order-1 (TS candidate): Use SMALL steps for precise convergence
        - Minimum (order 0): Use NORMAL steps to continue searching

    Args:
        saddle_info: Output dictionary from classify_saddle_point()
        base_scale: Baseline scaling factor (default: 1.0)
        higher_order_mult: Multiplier for higher-order saddles (default: 5.0)
                          Order-2 → 5×, Order-3 → 10×, Order-4 → 15×, etc.
        ts_mult: Multiplier near TS (default: 0.5 for smaller, more precise steps)

    Returns:
        step_scale: Multiplicative factor for step size

    Examples:
        >>> saddle_info = {'saddle_order': 2, ...}
        >>> compute_adaptive_step_scale(saddle_info)  # Order-2
        5.0  # 5× larger steps

        >>> saddle_info = {'saddle_order': 3, ...}
        >>> compute_adaptive_step_scale(saddle_info)  # Order-3
        10.0  # 10× larger steps

        >>> saddle_info = {'saddle_order': 1, ...}
        >>> compute_adaptive_step_scale(saddle_info)  # TS
        0.5  # 0.5× smaller steps for refinement

        >>> saddle_info = {'saddle_order': 0, ...}
        >>> compute_adaptive_step_scale(saddle_info)  # Minimum
        1.0  # Normal steps
    """
    saddle_order = saddle_info['saddle_order']

    if saddle_order >= 2:
        # Higher-order saddle: scale increases linearly with order
        # Order-2 → higher_order_mult × 1
        # Order-3 → higher_order_mult × 2
        # Order-4 → higher_order_mult × 3, etc.
        scale = base_scale * higher_order_mult * (saddle_order - 1)
    elif saddle_order == 1:
        # Near TS (order-1): use smaller steps for precise convergence
        scale = base_scale * ts_mult
    else:
        # Minimum (order-0) or unconverged: use normal steps
        scale = base_scale

    return scale
