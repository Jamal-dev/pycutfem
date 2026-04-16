"""Tensor-only entrypoint for the one-domain biofilm final-form builder.

This module intentionally delegates to the shared implementation in
``final_form.py`` and forces ``interface_formulation="tensor"``.

Keeping the tensor variant as a thin entrypoint avoids implementation drift
between the maintained builder and the Seboldt tensor-vs-decomposed
comparison path.
"""

from __future__ import annotations

from .final_form import build_biofilm_one_domain_final_form as _build_shared_final_form


def build_biofilm_one_domain_final_form(**kwargs):
    return _build_shared_final_form(interface_formulation="tensor", **kwargs)


__all__ = ["build_biofilm_one_domain_final_form"]
