# MODEL_SPEC.md

This repository simulates a slotted two-stage tandem information freshness system.

## 1. System model

There are N sources. Each source sends status updates through two stages:

1. Stage 1: source -> BS.
2. Stage 2: BS/edge -> destination.

Each source has a VOQ at the BS with capacity one. A newly completed Stage-1 packet overwrites the existing VOQ packet for the same source.

The destination age of source i is denoted h_i. It increases by one each slot unless source i is delivered to the destination.

The BS-side packet age / VOQ generation age is denoted A_i^Q.

The VOQ occupancy is V_i in {0,1}.

The useful downstream freshness gap is

Q_i = V_i (h_i - A_i^Q).

This quantity is zero when the VOQ is empty and positive when the VOQ has a packet that can reduce destination age.

Stage 1 has source-dependent channel success probability p_i. A successful Stage-1 gate attempt starts an L-slot transmission process. The effective Stage-1 capacity constraint used by the joint lower bound is

sum_i ((L - 1) + 1/p_i) q_i <= 1.

Stage 2 has geometric service success probability mu, so the edge-side capacity constraint is

sum_i q_i <= mu.

## 2. Important slot convention

The current notebook uses its own consistent slot convention. Do not change the slot order or the destination age reset convention unless explicitly asked.

In particular, deterministic N=1, p=1, mu=1 tests should validate the current trajectory convention, not impose a new convention.

## 3. Policies in the notebook

The main policies are:

### Joint FGMW

Joint FGMW is tuned using the full tandem lower-bound feasible region:

sum_i ((L - 1) + 1/p_i) q_i <= 1,
sum_i q_i <= mu.

It uses one joint set of Lyapunov coefficients and makes both Stage-1 and Stage-2 decisions.

The Stage-1 decision is BS-side and mostly depends on A_i^Q and joint lower-bound coefficients.

The Stage-2 decision is ES-side and depends on the downstream gap Q_i.

### Isolated Stage-1 MW

Isolated Stage-1 MW is designed for the source-to-BS subsystem only. It tunes q_i using the Stage-1 feasible region and schedules Stage 1 based mainly on A_i^Q.

It does not directly use downstream VOQ usefulness Q_i in the BS decision.

### Isolated Stage-2 MW

Isolated Stage-2 MW is designed for the BS/edge-to-destination subsystem. In the current notebook, its universal relaxed lower bound uses

sum_i q_i <= mu,

without assuming prior knowledge of the VOQ arrival rates lambda_i.

It schedules Stage 2 based on Q_i.

### Combined isolated policy

The main isolated baseline is:

isolated Stage-1 MW + isolated Stage-2 MW.

This represents two locally designed MW controllers without full joint tandem coordination.

## 4. What Codex may do

Codex may:
- add tests;
- refactor code for readability;
- add diagnostics;
- improve plotting;
- check optimizer feasibility;
- check post-warmup accounting;
- compare implemented indices against direct one-step drift enumeration.

## 5. What Codex must not do unless explicitly asked

Codex must not:
- change the mathematical policy formulas;
- change the slot order;
- change the age reset convention;
- change the lower-bound feasible regions;
- replace Joint FGMW with a heuristic;
- silently update cached sweep results;
- decide theoretical modeling choices.

If a formula appears suspicious, report it instead of changing it.

## 6. Required validation ideas

Useful tests include:
- post-warmup conservation checks;
- Stage-1 and Stage-2 resource capacity checks;
- optimizer feasibility and optimality checks;
- same-seed reproducibility;
- deterministic N=1, p=1, mu=1 trajectory tests;
- direct one-step Lyapunov drift oracle tests for policy decisions.
