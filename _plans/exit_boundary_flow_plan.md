# Exit boundary flow output plan

Save the user's definition of average exit boundary flow: the arithmetic mean
of the final three detector flows at each control timestep.

- Return `exit_boundary_flow_vph = flow[-3:, :].mean(axis=0)` from the shared
  simulation runner; require at least three detectors.
- Save the series in both the dataset generator and single-rollout `.npz` writer.
- Document units, shape, detector locations, and compatibility with older files.
- Verify both save paths using temporary outputs and check the saved values
  against the detector arrays, including nonzero flow after traffic reaches the exit.

The change does not require regenerating existing data or changing RL rewards.
