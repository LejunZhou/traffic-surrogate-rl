# What M14 implements

## Traffic and control

`configs/scenario.yaml` defines M14's v3b road. The mainline has one through
lane, is 2,000 m long, and has a speed limit of 33.33 m/s. A 200 m ramp joins
at 10° at x=1,300 m and continues along a 100 m auxiliary acceleration lane.
The meter controls insertion at the ramp entrance; released vehicles start
from rest. Vehicles waiting before insertion are represented by a virtual queue.

SUMO runs at one-second resolution. The controller acts every 30 seconds for
one hour: K=120 actions. Density is estimated from occupancy at 19 detector
stations, x=100,…,1,900 m. The merge detector measures only the through lane.
Every episode starts empty. There is no finite queue storage cap.

At each control interval, the physical action is u in [0,1]. PPO internally
uses a symmetric [-1,1] action mapped to u=(a+1)/2. A capacity of uD does not
guarantee this amount of discharge: released flow also depends on arrivals and
the existing queue. D=1,200 veh/h, and τ=30/3600 hours:

```text
arrivals = τ × ramp_demand
released = min(queue + arrivals, τ × u × D)
next_queue = queue + arrivals − released
ramp_inflow = released / τ
```

SUMO discretizes vehicles and confirms insertions before removing them from
the virtual queue. The surrogate uses the continuous analytic approximation.
The learned traffic model includes acceleration and merging after admission at
the ramp entrance; ramp inflow is not a measurement at the merge nose.

## DeepONet

The M14 operator maps mainline-demand and admitted ramp-flow histories into
detector density and downstream exit flow. It assumes the configured geometry,
initial-condition regime, temporal grid, and traffic-vehicle parameters.

| Component | Setting |
|---|---|
| Branch tensor | batch × 2 × 120 |
| Branch channels | offered mainline demand / 2500, admitted ramp flow / 1600 |
| Encoder | unidirectional GRU, 2 layers, 128 hidden units |
| Readout | 128 → 256 at every time index |
| Trunk input | (x/2000, t/3600) |
| Trunk | 3 hidden layers of 512, GELU; 512 output features |
| Output | separate density/flow dot products, divided by sqrt(256), plus biases |
| Density scale | training-split mean and standard deviation |
| Flow scale | 2500 veh/h |
| Ensemble | 5 members with independent initialization and bootstrap samples |

The two trunk heads each contain 256 features. They share the branch latent.
Density is supervised at detector coordinates; flow is supervised only at
x=L. The model does not learn queue length, or take the queue, raw action,
previous density, initial density field, or road geometry as separate inputs.
The analytic queue converts actions into the flow history the branch receives.

The ramp-flow scale of 1600 is normalization, not the physical discharge limit.
The physical limit is 1200 in both the scenario and shared PPO configuration.

### Sequence handling and causality

This is a causal GRU-branch DeepONet, following the sequential operator-learning
idea of S-DeepONet. It is not an exact reproduction of its encoder–decoder
architecture. There is no recurrent decoder in M14.

Prediction at index k reads the GRU representation at index k, so input values
after k cannot affect that prediction. A full trajectory can be used during
training, while online control has only a growing ramp-flow prefix. The actual
environment keeps a fixed-size history and recomputes the GRU when stepping.

GRU parameter shapes permit other sequence lengths, but the deployed ensemble
caches a fixed K=120 time grid. Training does not use variable-length minibatches
or irregular timestamps, and no accuracy claim is made for longer horizons,
different sampling intervals, or new road geometries.

The implementation labels interval-k outputs by the interval's **start** time,
k×30 s. Detector density and exit flow are collected over that interval and
returned afterward. Dataset creation and model inference use the same convention.

### Learning

Each epoch samples 512 density coordinates per rollout and supervises exit flow
at all 120 intervals. The loss is weighted MSE on normalized density plus MSE
on normalized exit flow, with relative coefficient 1. Congested samples where
density exceeds 40 veh/km and x≤1400 m receive weight 3; all others receive 1.

Initial training uses AdamW, 300 epochs, learning rate 0.001 with cosine decay,
five warmup epochs, batch size 16, weight decay 1e-6, and gradient clipping 1.
Member selection minimizes the sum of validation density and exit-flow relative
L2 errors. Aggregation fine-tunes for 20 epochs at learning rate 0.0003.

The neural loss has no PDE residual or hard conservation constraint. In the
surrogate environment, predicted density is clipped to [0,143] veh/km and exit
flow to [0,3000] veh/h. Queue conservation is explicit outside the network.

## PPO and the surrogate environment

The policy observes 23 values: 19 normalized densities, current mainline demand
/2500, current ramp demand /1000, normalized time k/120, and queue /100.
Density z-scores are clipped to [-3,25]. No demand lookahead is provided.
The density observation comes from the preceding interval; the demand features
describe the interval about to be controlled.

This is a partially observed control problem with a reactive policy. The GRU
belongs to the environment model, not the policy. Policy and value networks
each have three 512-unit tanh hidden layers. PPO uses learning rate 1e-4,
discount 0.99, GAE lambda 0.95, clipping 0.2, 480 steps/environment/update,
five optimization epochs, minibatches of 120, and target KL 0.02.

Sixteen surrogate environments run in a batch. Each episode samples one of the
five members and uses it for the entire episode; surrogate validation uses the
ensemble mean. PPO samples trajectories and does not differentiate through
the traffic model. During SUMO evaluation, the policy acts directly on SUMO
observations and does not call DeepONet.

After each policy-training round, the top three surrogate checkpoints are
evaluated on 18 SUMO validation profiles. These 54 trajectories update the
dataset, and all ensemble members are fine-tuned. Later policy rounds warm-start
from the selected checkpoint. The full study stops after two rounds with less
than 2 return units of improvement, or after the round limit (default 5).

## Objective and metrics

Training uses negative estimated TTS: density-derived road occupancy, virtual
ramp waiting, and conservation-estimated upstream backlog, integrated over
the control interval. Offered demand and downstream exits maintain the backlog
accounting. Reward scale is 1; the first 90 seconds have zero reward, discount
is 0.99, and there is no terminal queue penalty.

The extraction preserves the original M14 timing choices:

- SUMO reward charges interval-average ramp queue.
- Surrogate reward charges the updated/end queue.
- Both conservation backlog estimates use the end queue.
- `tts_veh_h` is a full-episode metric using density, end queue, and recorded
  pending-mainline counts. It is not simply negative training return.

`episode_queue_mean` and `episode_queue_max` use one-second queue samples in
SUMO. Across-episode reports should distinguish the mean of episode peaks from
the single largest queue observed anywhere. Served vehicles derive from network
exit counts. Final ID/OOD episodes do not count toward training simulation cost.

These details are intentional preservation of the existing implementation, not
a claim of exact physical integration or exact SUMO/surrogate reward parity.
