"""Kick direction experiments for understanding v₂ effectiveness.

This module will contain experiments to test different kick strategies:
1. v₂ kick (current) - kick along second vibrational eigenvector
2. Gradient descent kick - do N steps of gradient descent when stuck
3. Random kick - random direction as control experiment
4. Orthogonal to v₁ kick - any orthogonal direction
5. Higher mode kicks (v₃, v₄, etc.)
6. Adaptive k-reflection kick - reflect along full unstable subspace

These experiments aim to understand WHY v₂ kicking works and whether
there are scientifically justifiable alternatives.
"""
