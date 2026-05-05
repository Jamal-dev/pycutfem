"""Decomposed entrypoint for the bulk/interface-split builder."""

from __future__ import annotations

from .bulk_interface_split_form import (
    build_biofilm_one_domain_bulk_interface_split_form_decomposed
    as build_biofilm_one_domain_bulk_interface_split_form,
)

__all__ = ["build_biofilm_one_domain_bulk_interface_split_form"]
