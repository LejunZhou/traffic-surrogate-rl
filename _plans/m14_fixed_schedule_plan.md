# M14 fixed schedule across the entire study

Use the seven-segment demand from the saved local generator as the single
source in m14/configs/scenario.yaml. Keep all M14 road, vehicle, metering,
reward, and model settings. Wire training, E0/mixture data, baseline tuning,
aggregation, and evaluation through that source. Evaluate independent SUMO
seeds on the same schedule and label the legacy OOD slot as repeatability.
Verify exact interval values, source propagation, dataset serialization,
SUMO/SURROGATE environments, and CLI integration. Do not run full training.
