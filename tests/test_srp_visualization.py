from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tandem import plotting as tandem_plotting


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(fig):
    plt.close(fig)


def test_srp_plotting_helpers_run_on_quick_csv_without_touching_mw_screening():
    srp_path = Path("results/heterogeneous_screening_with_srp.csv")
    mw_only_path = Path("results/heterogeneous_screening.csv")
    assert srp_path.exists()
    assert mw_only_path.exists()

    before = _sha256(mw_only_path)
    df = pd.read_csv(srp_path)

    for plotter in [
        tandem_plotting.plot_srp_iso_vs_tandem_gap_by_alignment,
        tandem_plotting.plot_srp_iso_vs_tandem_gap_by_regime,
        tandem_plotting.plot_srp_tandem_minus_iso_gap,
        tandem_plotting.plot_srp_pipeline_diagnostics,
    ]:
        fig, _ = plotter(df)
        assert fig is not None
        _close(fig)

    after = _sha256(mw_only_path)
    assert before == after


def test_srp_plotting_helpers_handle_missing_optional_columns():
    df = pd.read_csv("results/heterogeneous_screening_with_srp.csv")
    slim = df[
        [
            "policy",
            "alignment",
            "gap_vs_iso_lambda_pct",
            "total_VOQ_arrival_rate",
            "total_delivery_rate",
        ]
    ].copy()

    for plotter in [
        tandem_plotting.plot_srp_iso_vs_tandem_gap_by_alignment,
        tandem_plotting.plot_srp_iso_vs_tandem_gap_by_regime,
        tandem_plotting.plot_srp_tandem_minus_iso_gap,
        tandem_plotting.plot_srp_pipeline_diagnostics,
    ]:
        fig, _ = plotter(slim)
        assert fig is not None
        _close(fig)
