# M12 plan — Study: ablations, final evaluation, figures (E7, E8)

Source: `draft_pipeline.md` §10, §11. Status: `_progress/m12_study_progress.md`.

- `scripts/build_arms_manifest.py` collects every arm (policy, cumulative EE from the
  ledgers, wall-clock) into `runs/study/<study>/arms.json`.
- `scripts/run_final_evaluation.py` rolls every arm on T (30 × 3 seeds) and O (12 × 3)
  with `purpose eval_test / eval_ood` (not budgeted); per-episode JSONL + summaries with
  mean, p10, worst, breakdown / recovery rates, TTS, served vehicles, final queue and
  paired bootstrap 95 % CIs.
- `scripts/plot_sample_efficiency.py` draws Figs 1–7 (return vs EE, vs wall-clock,
  surrogate accuracy vs N_0 / data source, transfer gap per round, breakdown rates,
  policy structure, OOD).
- `scripts/run_study.sh` is the one-command driver (demo budgets by default; paper
  budgets through environment variables).

Exit: tables and figures in `_progress/figures/m12_<study>/`, README pipeline section,
proposal revision request.
