# Milestone 5b progress

Running log for the reward-weight retune. See `_plans/milestone_5b_plan.md`
for scope and acceptance.

## 2026-05-12 — Kickoff

- Plan file written: `_plans/milestone_5b_plan.md`.
- Todo list refreshed for M5b (11 tasks).
- Open with the M5 finding: on seed=0, learned policy total reward
  -4440 vs constant u=1.0 total reward -3758 (constant-open wins).
  Diagnosis: queue contribution to total reward at u=1.0 is exactly
  0 (queue never grows from 0), so raising beta doesn't hurt that
  corner; it makes interior policies more attractive instead.
- Strategic decision: sweep beta ∈ {0.3, 1.0, 3.0}, 5 seeds each,
  pick smallest beta passing the 4/5-wins + 5%-margin acceptance bar.

## 5b.1 — CLI flags

_pending_

## 5b.2 — Baseline helper

_pending_

## 5b.3 — Sweep driver

_pending_

## 5b.4 — Full sweep

_pending_

## 5b.5 — Analysis and beta choice

_pending_

## 5b.6 — Defaults updated

_pending_

## 5b.7 — Verification (V1, V2)

_pending_

## Acceptance verdict

_pending_
