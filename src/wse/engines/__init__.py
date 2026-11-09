"""Inference engines for imported open-weight models (M2 plan Section 4.1).

``wse.engines.hf_torch`` runs ``transformers`` causal LMs with PyTorch. Engines
import ``transformers`` lazily, so importing this package needs no optional
extra.
"""
