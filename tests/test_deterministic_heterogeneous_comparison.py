from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tests.notebook_test_utils import NOTEBOOK
from experiments.quick_heterogeneous_comparison import quick_random_heterogeneous_comparison
from experiments.deterministic_heterogeneous_comparison import (
    NETWORK_ORDER,
    POLICY_ORDER,
    RAW_WEIGHTS,
    deterministic_networks,
    run_deterministic_heterogeneous_comparison,
)


EXPECTED_WEIGHTS = np.array([8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_deterministic_weights_are_exact_and_unnormalized():
    np.testing.assert_array_equal(RAW_WEIGHTS, EXPECTED_WEIGHTS)
    assert not np.isclose(float(RAW_WEIGHTS.mean()), 1.0)
    for block in deterministic_networks().values():
        np.testing.assert_array_equal(block["w"], EXPECTED_WEIGHTS)


def test_deterministic_alignment_correlation_signs():
    networks = deterministic_networks()
    for L in (3, 5):
        aligned = networks["det_aligned"]
        conflict = networks["det_conflict"]
        c_aligned = (L - 1.0) + 1.0 / aligned["p"]
        c_conflict = (L - 1.0) + 1.0 / conflict["p"]

        assert float(np.corrcoef(aligned["w"], c_aligned)[0, 1]) < 0.0
        assert float(np.corrcoef(conflict["w"], c_conflict)[0, 1]) > 0.0


def test_deterministic_quick_experiment_runs_without_duplicate_rows(tmp_path):
    output = tmp_path / "deterministic.csv"
    df = run_deterministic_heterogeneous_comparison(mode="quick", output_csv=output)

    assert output.exists()
    assert len(df) == len(NETWORK_ORDER) * 2 * 3 * len(POLICY_ORDER)
    assert not df.duplicated(["network", "L", "mu", "policy"]).any()
    assert set(df["network"]) == set(NETWORK_ORDER)
    assert set(df["policy"]) == set(POLICY_ORDER)
    assert np.all(np.isfinite(df["weighted_dest_aoi"]))
    assert np.all(np.isfinite(df["gap_vs_iso_lambda_pct"]))

    for value in df["w"].unique():
        np.testing.assert_array_equal(np.asarray(json.loads(value), float), EXPECTED_WEIGHTS)


def test_quick_heterogeneous_uses_deterministic_raw_networks_without_duplicate_rows(tmp_path):
    output = tmp_path / "quick_heterogeneous.csv"
    out = quick_random_heterogeneous_comparison(
        seeds=(0,),
        pilot_seeds=(100,),
        K=600,
        warmup=60,
        K_pilot=800,
        warmup_pilot=80,
        output_csv=output,
    )
    df = out["results"]

    assert output.exists()
    assert not df.duplicated(["network", "L", "mu", "policy"]).any()
    assert set(df["network"]) == set(NETWORK_ORDER)
    assert set(df["policy"]) == set(POLICY_ORDER)
    for value in df["w"]:
        np.testing.assert_array_equal(np.asarray(value, float), EXPECTED_WEIGHTS)
    assert np.isnan(df["weight_seed"]).all()
    assert np.isnan(df["channel_seed"]).all()


def test_deterministic_quick_does_not_modify_full_cached_csvs(tmp_path):
    full_paths = [
        Path("results/heterogeneous_screening.csv"),
        Path("results/heterogeneous_screening_with_srp.csv"),
    ]
    before = {path: _sha256(path) for path in full_paths}

    run_deterministic_heterogeneous_comparison(
        mode="quick",
        output_csv=tmp_path / "deterministic.csv",
    )

    after = {path: _sha256(path) for path in full_paths}
    assert before == after


def test_section8_embedded_sweep_uses_aggressive_raw_deterministic_metadata():
    nb = json.loads(NOTEBOOK.read_text())
    section8 = "".join(nb["cells"][21].get("source", []))
    namespace = {}
    exec("".join(nb["cells"][22].get("source", [])), namespace)
    sweep = json.loads(namespace["SWEEP_V3_JSON"])
    metadata = sweep["metadata"]

    assert "Aggressive deterministic heterogeneous sweeps" in section8
    assert "raw and unnormalized" in section8
    assert metadata["network"] == "det_conflict"
    assert metadata["weights_normalized"] is False
    assert metadata["random_sampling"] is False
    np.testing.assert_array_equal(np.asarray(metadata["raw_weights_base"], float), EXPECTED_WEIGHTS)

    all_text = section8.lower()
    assert "log-uniform" not in all_text
    assert "normalized" not in all_text.replace("unnormalized", "")
    assert "weight_ratio=10" not in all_text
