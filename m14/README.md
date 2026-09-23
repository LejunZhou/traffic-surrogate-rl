# M14 · DeepONet + PPO for ramp metering

A self-contained implementation of the **M14 v3b study**: one metered ramp,
one mainline through lane, a causal GRU DeepONet ensemble, and PPO trained
with iterative SUMO validation and data aggregation.

Start here rather than navigating the older milestone scripts. The folder can
be copied to another location without the parent repository. It contains code,
configuration, frozen demand profiles, and tests; generated data and models go
into `data/` and `runs/`.

## At a glance

```text
m14/
├── run.py                 One entry point for every stage
├── configs/               Seven configs + frozen validation/ID/OOD profiles
├── src/
│   ├── sumo_env/          Road, demand, detectors, queue, simulation data
│   ├── surrogate/         GRU DeepONet, dataset, training, model evaluation
│   ├── rl/                SUMO/surrogate environments, PPO, baselines
│   └── utils/             Configuration, logging, simulation-cost ledger
├── scripts/               Individual study stages behind run.py
├── tests/                 Model, queue, simulator, and integration contracts
├── docs/                  Model explanation, workflow, source provenance
├── data/                  Generated trajectories and SUMO networks
├── runs/                  Checkpoints, logs, and episode-level evaluations
└── reports/               Generated comparison figures
```

**Physical setting:** 2 km road at 120 km/h; merge nose at 1.3 km; a 200 m ramp at
60 km/h, joining at 10°, plus a 100 m acceleration lane; 1,200 veh/h meter; uncapped virtual ramp queue.
Episodes last one hour with 120 control intervals and 19 density detectors.

**Learned model:** demand/released-flow histories → two-layer GRU → 256 features,
combined with a space–time trunk to predict density and downstream flow.
Five bootstrap members provide the surrogate environment. PPO is a separate
feedforward policy; [the model guide](docs/model.md) explains the distinction.

## Start

From this folder, create a dedicated Python 3.11+ environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[sumo,dev]"
python run.py check
python run.py simulate --policy u=0.5
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
The `sumo` extra installs SUMO 1.27.1 and matching Python bindings.
If SUMO is managed separately, install `.[dev]` and put both `sumo` and
`netconvert` on `PATH`. W&B is optional and disabled by default.

The simulation command saves a complete trajectory and metrics in
`runs/simulation/`. It works before training data or checkpoints exist.

## Run the study

```bash
python run.py e0                     # 404 screening trajectories + capacity check vs the 120 km/h ramp
python run.py data                   # + 288 mixture trajectories
python run.py deeponet               # 5 models, 300 epochs; held-out model checks
python run.py surrogate-ppo          # up to 5 rounds of PPO + SUMO + fine-tuning
python run.py sumo-ppo               # direct PPO at 1000 training episodes (resumable)
python run.py baselines              # tune ALINEA/PI-ALINEA and constant rates
python run.py evaluate               # frozen ID/OOD profiles in SUMO, incl. Surrogate-MPC
python run.py tables                 # paper Tables I and II, headline TTS reductions
python run.py report                 # comparison figures
```

Or run `python run.py pipeline --parallel --workers 8`. This launches the full study and
can take hours; `--parallel` runs [deeponet → surrogate-ppo], sumo-ppo and baselines
concurrently after the data stage. Inspect its commands first with `python run.py pipeline --dry-run`.
Completed stage outputs are reused; aggregation resumes recorded rounds and the
direct PPO run continues from its latest checkpoint. On Google Colab use
`../colab/m14_ramp60.ipynb` (it adds `--recover-interrupted`, see the workflow notes).
Use a fresh copy for experiments with changed configurations so outputs do not
mix settings. [Workflow details](docs/workflow.md) cover custom commands and outputs.

## Verify the installation

```bash
python -m pytest -q
python run.py smoke --workers 2
```

The smoke command creates a separate project copy under `runs/smoke/` and runs
data generation, two short DeepONet trainings, one aggregation round, short
direct PPO training, baseline tuning, final evaluation, and plots. It preserves
the one-hour traffic horizon but uses tiny training budgets. Its scores are
integration diagnostics, not research results.

## Find or change something

| Concern | Start here |
|---|---|
| Road, simulator timing, discharge capacity | [configs/scenario.yaml](configs/scenario.yaml) |
| Demand distribution and frozen evaluation sets | [configs/demand.yaml](configs/demand.yaml), `configs/profiles/` |
| Initial trajectory mixture | [configs/dataset.yaml](configs/dataset.yaml) |
| DeepONet architecture and optimizer | [configs/deeponet.yaml](configs/deeponet.yaml) |
| PPO, observation, reward, shared meter settings | [configs/ppo.yaml](configs/ppo.yaml) |
| SUMO/surrogate-specific PPO settings | `configs/env_sumo.yaml`, `configs/env_surrogate.yaml` |
| Exact equations and implementation caveats | [docs/model.md](docs/model.md) |
| Original-file mapping and missing historical artifacts | [docs/provenance.md](docs/provenance.md) |

Raw evaluation files expose `tts_veh_h`. Training return and full-episode TTS
are different quantities; existing return plots label their metric explicitly.
The historical M14 Windows checkpoints/evaluations were unavailable when this
folder was created. This package provides a runnable workflow, not a bundled
reproduction of those saved results.
