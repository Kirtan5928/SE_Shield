"""
layer4/__init__.py
===================
Layer 4 — Sliding Window Context Engine.

Components
----------
RiskCounter    (layer4a) — per-conversation risk accumulator, between L2 and L3
SemanticWindow (layer4b) — per-conversation semantic pattern detector, after L3
HybridPipeline           — full L1→L2→L4a→L3→L4b pipeline
"""

from .layer4a_risk_counter    import RiskCounter
from .layer4b_semantic_window import SemanticWindow
from .hybrid_pipeline         import HybridPipeline

__all__ = ["RiskCounter", "SemanticWindow", "HybridPipeline"]