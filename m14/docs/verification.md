# Extraction verification

Checked on 2026-09-19. These checks validate the standalone workflow and the
preserved implementation; they do not reproduce historical M14 performance.

## Completed checks

- **73 tests passed**, including real SUMO integration tests. The three warnings
  were Stable-Baselines3 notices about the environment's absent render mode.
- A standalone one-hour SUMO simulation completed before training data existed.
- A reduced end-to-end smoke run completed: 18 data trajectories, two one-epoch
  GRU DeepONets, one surrogate-PPO aggregation round with fine-tuning, short
  direct-SUMO PPO, validation selection, controller tuning, and ID/OOD testing.
- The smoke run produced 16 final evaluation files covering eight controller
  arms on two held-out sets, plus five comparison figures in PNG and PDF.
- After evaluation/reuse and reporting hardening, real PPO and constant-policy
  evaluations were run and repeated. The repeat reused the two saved episodes
  without adding evaluation rows or ledger entries. Updated figures generated
  successfully; the selected-controller OOD figure was visually inspected.
- Original and extracted active GRU implementations produced identical state
  dictionaries and numerical outputs under the same configuration. All 288
  initial behavior-controller specifications, profile draws, and SUMO seeds
  matched the original implementation.
- Tests checked imports and commands from a detached copy, dry runs without
  output creation, seed propagation, and evaluation integrity/reuse guards.
- The parent project's source and existing work were left unchanged.

The full smoke preceded the final reuse/reporting hardening; the affected paths
were then checked separately as described above, followed by the full test suite.
Temporary smoke checkpoints, trajectories, figures, and caches were removed from
this delivered source folder. Run `python run.py smoke --workers 2` to create a
new isolated smoke run under `runs/smoke/`.

## Verified environment

| Component | Version |
|---|---|
| Python | 3.11.16 |
| SUMO, TraCI, sumolib | 1.27.1 |
| PyTorch | 2.13.0 |
| Stable-Baselines3 | 2.9.0 |
| Gymnasium | 1.3.0 |
| NumPy | 2.4.6 |
| Matplotlib | 3.11.1 |
| PyYAML | 6.0.3 |

The verification used the parent project's existing Python environment. The
package also supplies installation metadata for a dedicated environment; a
fresh dependency installation was not performed during this extraction.

```bash
python run.py check
python -m pytest tests -q --tb=short
python run.py smoke --workers 2
```

Ensure `sumo` and `netconvert` are on `PATH` when running pytest so simulator
tests run rather than skip. The public `run.py` also checks its Python
interpreter's binary directory.
