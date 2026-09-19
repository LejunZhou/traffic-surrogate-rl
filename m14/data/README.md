Generated SUMO data live here. `python run.py data` creates:

- `round0/`: trajectories, index, train/validation/test splits, normalization statistics.
- `networks/`: generated SUMO networks and detector files.

The frozen demand definitions are versioned in `configs/profiles/`.
Historical M14 data and trained weights are not bundled. Generate fresh data with
the commands in the main README; see `docs/provenance.md` for historical availability.
