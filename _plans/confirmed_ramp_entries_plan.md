# Confirmed ramp entries plan

Use SUMO-confirmed ramp insertions as passage through the modeled meter,
retaining the existing upstream virtual queue and network geometry.

- Track accepted requests until SUMO reports departure or discards the request.
- Keep pending demand in the queue and exclude it from new release requests.
- Compute ramp inflow and the metered surrogate input from confirmed departures.
- Save interval departure counts, end-of-interval pending counts, and measurement
  provenance in both rollout writers alongside the corrected queue and inflow.
- Test delayed entry across control intervals, failed and discarded requests,
  open-loop compatibility, and both saved `.npz` schemas; run a live SUMO check.

Preserve the earlier exit-boundary detector average and existing dataset files.
