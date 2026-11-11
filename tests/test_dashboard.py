"""
Smoke tests for the GeneWeaver terminal dashboard (Day 1).
"""

import pytest
from textual.widgets import Footer, Header

from src.dashboard import (
    GeneWeaverApp,
    ProcessingInfoPanel,
    ProjectStatusPanel,
    ResultsPanel,
)


def test_import_gene_weaver_app() -> None:
    """Verify that GeneWeaverApp imports successfully."""
    from src.dashboard import GeneWeaverApp as ImportedApp

    assert ImportedApp is not None
    assert issubclass(ImportedApp, object)


def test_instantiate_gene_weaver_app() -> None:
    """Verify that GeneWeaverApp can be instantiated with correct metadata."""
    app = GeneWeaverApp()
    assert app is not None
    assert isinstance(app, GeneWeaverApp)
    assert app.TITLE == "GeneWeaver"
    assert app.SUB_TITLE == "GPU-Accelerated CRISPR Alignment Engine"


def test_instantiate_dashboard_panels() -> None:
    """Verify that each dashboard panel component can be instantiated."""
    project_panel = ProjectStatusPanel()
    assert project_panel is not None
    assert project_panel.border_title == "Project Status"

    proc_panel = ProcessingInfoPanel()
    assert proc_panel is not None
    assert proc_panel.border_title == "Processing Information"

    results_panel = ResultsPanel()
    assert results_panel is not None
    assert results_panel.border_title == "Alignment Results"


def test_dashboard_composition() -> None:
    """Verify that compose yields the required top-level widgets."""
    app = GeneWeaverApp()
    composed_widgets = list(app.compose())

    # Should contain Header, Main Container, and Footer
    widget_types = [type(w) for w in composed_widgets]
    assert Header in widget_types
    assert Footer in widget_types

