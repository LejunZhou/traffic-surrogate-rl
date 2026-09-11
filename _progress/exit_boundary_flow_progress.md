# Exit boundary flow output progress

## 2026-09-10

- Added `exit_boundary_flow_vph` to the shared simulation result and both `.npz`
  save paths. The series is a float32 spatial mean over the last three detectors,
  with one value per control interval.
- Added a validation error for scenarios with fewer than three detectors.
- Documented the new field in the generator schema and README.
- Validation passed with installed SUMO 1.20.0 and its bundled TraCI bindings
  using the Python 3.11 Anaconda runtime. The project pin remains SUMO 1.27.1;
  this check validates output computation and persistence, not traffic-physics
  equivalence between versions.
- Ran one single-rollout CLI simulation and two dataset CLI simulations, each
  lasting 180 seconds (six control intervals), with all outputs in a temporary
  directory. All three saved arrays had dtype float32, shape `(6,)`, and exactly
  matched `(flow[-3] + flow[-2] + flow[-1]) / 3`. Checked that downstream flow
  was nonzero and that the spatial mean differed from the final detector alone.
- Confirmed the selected detector positions were 1700, 1800, and 1900 m, and
  that fewer than three detectors raised an error before starting SUMO.
- The first smoke attempt could not open a local TraCI socket in the sandbox;
  rerunning with approved local socket access passed. Logs and sample files:
  `/var/folders/4n/q95jyw_94kq_x0zf37fb2wch0000gn/T/sumo-exit-flow-bkcxb242/`.
- Existing training datasets were not modified.
