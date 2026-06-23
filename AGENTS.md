# AGENTS.md

This repository simulates a two-stage tandem information freshness system.

## Current main file
The current notebook is:
- tandem_combined.ipynb

## Research goal
Compare joint FGMW against two isolated MW policies and baseline policies.

## Important model conventions
- Slotted-time tandem system.
- Stage 1 is source-to-BS.
- Stage 2 is BS-to-edge/destination service.
- VOQ capacity is one per source.
- Destination age h_i increases by one each slot unless source i is delivered.
- Post-warmup averages and rates must use post-warmup counters only.
- Do not change mathematical formulas unless explicitly asked.

## Required checks
When modifying code, check:
1. Warmup accounting.
2. Stage-1 utilization definition.
3. Stage-2 utilization definition.
4. Optimizer feasibility.
5. Bridge invariant: h_i <= A_i^Q + Q_i + Y_i + 1.
6. Achieved rates satisfy physical capacity constraints up to simulation error.

## Commands
Run notebook execution check:
python -m nbconvert --to notebook --execute tandem_combined.ipynb --output executed_check.ipynb

If tests exist:
pytest -q

## Style
Prefer readable research code over clever code.
Preserve mathematical notation in comments when possible.
