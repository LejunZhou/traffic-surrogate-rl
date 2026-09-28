# Restore fixed demand dataset generator

Integrate the September 20 stash’s fixed one-hour demand template, scaled sampling, dataset generation, and discarded warmup into the current root pipeline. Preserve the existing ProfileFamily and explicit-block route API. Verify profile values, route timing, confirmed entries, append reproducibility, and a SUMO smoke run when available.

Follow-up: reset the upstream ramp queue, fractional arrival/release state, and pending ramp insertion requests at the recording boundary. Preserve already admitted traffic and pre-reset warmup diagnostics.

M14 adaptation: point the fixed dataset config at m14/configs/scenario.yaml; share density aggregation with the RL SUMO environment so occupancy, through-lane selection, and jam clipping match. Verify speed in the generated SUMO network and meter capacity in saved trajectories.
