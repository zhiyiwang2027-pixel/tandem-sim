from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.heterogeneous_screening import (
    iter_screening_settings,
    run_heterogeneous_screening,
    screening_networks,
)


def test_screening_configuration_generation_is_deterministic():
    first = list(
        iter_screening_settings(
            N=4,
            weight_ratios=(10.0,),
            alignments=("aligned", "neutral", "conflict"),
            L_values=(3,),
            mu_values=(0.2, 0.4),
            p_levels=(0.95, 0.7, 0.4, 0.1),
        )
    )
    second = list(
        iter_screening_settings(
            N=4,
            weight_ratios=(10.0,),
            alignments=("aligned", "neutral", "conflict"),
            L_values=(3,),
            mu_values=(0.2, 0.4),
            p_levels=(0.95, 0.7, 0.4, 0.1),
        )
    )
    assert len(first) == 6
    for a, b in zip(first, second):
        assert a["alignment"] == b["alignment"]
        assert a["L"] == b["L"]
        assert a["mu"] == b["mu"]
        np.testing.assert_array_equal(a["w"], b["w"])
        np.testing.assert_array_equal(a["p"], b["p"])


def test_alignment_correlation_signs():
    network = screening_networks(
        N=8,
        weight_ratio=50.0,
        p_levels=(0.95, 0.85, 0.70, 0.55, 0.40, 0.28, 0.18, 0.10),
    )
    w = network["w"]
    aligned_p = network["channels"]["aligned"]
    conflict_p = network["channels"]["conflict"]
    c_aligned = 3 - 1.0 + 1.0 / aligned_p
    c_conflict = 3 - 1.0 + 1.0 / conflict_p
    assert float(np.corrcoef(w, c_aligned)[0, 1]) < 0.0
    assert float(np.corrcoef(w, c_conflict)[0, 1]) > 0.0


def test_screening_quick_smoke_run(tmp_path):
    output = tmp_path / "screening.csv"
    df = run_heterogeneous_screening(
        mode="quick",
        output_csv=output,
        N=4,
        weight_ratios=(10.0,),
        L_values=(3,),
        seeds=(0,),
        pilot_seeds=(100,),
        p_levels=(0.95, 0.7, 0.4, 0.1),
    )
    assert output.exists()
    assert len(df) == 20
    assert set(df["policy"]) == {
        "Joint FGMW",
        "iso1 + iso2",
        "iso1 + iso2-lambda",
        "Greedy",
        "Uniform",
    }
    assert np.all(np.isfinite(df["weighted_dest_aoi"]))
    assert np.all(np.isfinite(df["gap_vs_iso_lambda_pct"]))


def test_existing_quick_heterogeneous_csv_is_unchanged():
    path = Path("results/quick_heterogeneous_comparison.csv")
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == 15
    row = df[(df["config"] == "conflict") & (df["policy"] == "Greedy")].iloc[0]
    np.testing.assert_allclose(row["weighted_dest_aoi"], 35.97064954541356, atol=1e-12)
    np.testing.assert_allclose(row["gap_vs_iso_lambda_pct"], 16.693267287162872, atol=1e-12)
