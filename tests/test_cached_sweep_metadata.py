from __future__ import annotations

import json
import warnings

from tests.notebook_test_utils import NOTEBOOK


def _cached_json_objects():
    namespace = {}
    nb = json.loads(NOTEBOOK.read_text())
    for variable in ("SWEEP_V3_JSON", "HETERO_V3_JSON"):
        source = next(
            "".join(cell.get("source", []))
            for cell in nb["cells"]
            if variable in "".join(cell.get("source", []))
        )
        exec(compile(source, f"{NOTEBOOK.name}:{variable}", "exec"), namespace)
    return {
        "SWEEP_V3_JSON": json.loads(namespace["SWEEP_V3_JSON"]),
        "HETERO_V3_JSON": json.loads(namespace["HETERO_V3_JSON"]),
    }


def test_cached_sweep_metadata_diagnostic_warns_without_requiring_recompute_match():
    cached = _cached_json_objects()

    sweep = cached["SWEEP_V3_JSON"]
    assert {"p", "L", "w", "mu", "N"}.issubset(sweep)
    missing = {"K", "warmup", "seeds", "code_version", "aggregation"} - set(sweep)
    if missing:
        warnings.warn(
            "SWEEP_V3_JSON lacks cache metadata "
            f"{sorted(missing)}; scalar policy values do not reveal whether "
            "they are single-seed or seed-averaged.",
            UserWarning,
        )

    hetero = cached["HETERO_V3_JSON"]
    assert {"A", "B"}.issubset(hetero)
    for name, block in hetero.items():
        first_mu = str(block["mu"][0])
        n_joint = len(block["joint"][first_mu])
        n_iso = len(block["iso"][first_mu])
        assert n_joint == n_iso
        missing = {"K", "warmup", "code_version"} - set(block)
        if missing:
            warnings.warn(
                f"HETERO_V3_JSON[{name!r}] stores {n_joint} per-seed values "
                f"but lacks cache metadata {sorted(missing)}.",
                UserWarning,
            )
