"""A copied source/config tree must load without the parent experiment repo."""

import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_imports_and_configs_work_in_detached_copy(tmp_path):
    project = tmp_path / "copied_m14"
    shutil.copytree(ROOT / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "configs", project / "configs")
    program = r'''
import importlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "src"))
modules = ["surrogate.deeponet", "surrogate.datasets", "surrogate.train_plant",
           "rl.reward", "rl.baseline_controllers", "rl.surrogate_vec_env",
           "rl.sumo_env_wrapper", "sumo_env.rollout", "sumo_env.rollout_store"]
for name in modules:
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(root / "src"), name
from utils.config import load_config
from sumo_env.demand_profiles import ProfileFamily, load_profile_set
scenario = load_config(str(root / "configs/scenario.yaml"))
assert scenario["demand"]["ramp_discharge_vph"] == 1200
assert scenario["demand"]["ramp_queue_max_veh"] is None
assert ProfileFamily.load(root / "configs/demand.yaml").sample_by_key("train", 0).K == 120
assert len(load_profile_set(root / "configs/profiles/val.json")) == 18
print(json.dumps({"modules": len(modules), "root": str(root)}))
'''
    result = subprocess.run(
        [sys.executable, "-I", "-c", program, str(project)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1])["root"] == str(project)
