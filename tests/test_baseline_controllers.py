"""Unit tests for the ALINEA / PI-ALINEA baseline controllers (numpy-only)."""

from __future__ import annotations

import numpy as np
import pytest

from rl.baseline_controllers import (
    PIALINEAController,
    is_controller_spec,
    make_controller,
)

N_X = 19


def obs_vec(rho_phys, mean=0.0, std=1.0, queue=0.0, queue_scale=100.0,
            ramp_demand_norm=None):
    """Build an observation like SumoEnv: z-scored densities + scalars."""
    dens = (np.full(N_X, float(rho_phys)) - mean) / std
    scalars = [0.5]  # mainline demand norm
    if ramp_demand_norm is not None:
        scalars.append(ramp_demand_norm)
    scalars += [0.0, queue / queue_scale]  # time norm, queue norm
    return np.concatenate([dens, np.array(scalars)]).astype(np.float32)


def test_integral_sign():
    c = PIALINEAController(ki=40.0, rho_set=30.0, u_init=0.5, discharge_vph=1600.0)
    u_low = c(obs_vec(20.0))[0]  # below set-point -> open up
    assert u_low > 0.5
    c.reset()
    u_high = c(obs_vec(40.0))[0]  # above set-point -> close down
    assert u_high < 0.5
    # step sizes match ki * error / discharge
    assert u_low == pytest.approx(0.5 + 40.0 * 10.0 / 1600.0)
    assert u_high == pytest.approx(0.5 - 40.0 * 10.0 / 1600.0)


def test_anti_windup():
    c = PIALINEAController(ki=100.0, rho_set=30.0, u_init=0.5)
    for _ in range(50):  # far below set-point: saturates at u_max
        u = c(obs_vec(5.0))[0]
    assert u == pytest.approx(1.0)
    # one step above set-point must move off the bound immediately
    u = c(obs_vec(40.0))[0]
    assert u == pytest.approx(1.0 - 100.0 * 10.0 / 1600.0)


def test_pi_damps_rising_density():
    kwargs = dict(ki=40.0, rho_set=30.0, u_init=0.5)
    i_only = PIALINEAController(**kwargs)
    pi = PIALINEAController(kp=80.0, **kwargs)
    for c in (i_only, pi):
        c(obs_vec(25.0))
    u_i = i_only(obs_vec(29.0))[0]   # still below set-point, but rising
    u_pi = pi(obs_vec(29.0))[0]
    assert u_pi < u_i  # P-term brakes while density rises
    assert u_i == pytest.approx(0.5 + 40.0 * (5.0 + 1.0) / 1600.0)
    assert u_pi == pytest.approx(u_i - 80.0 * 4.0 / 1600.0)


def test_reset_restores_initial_state():
    c = PIALINEAController(ki=40.0, rho_set=30.0, u_init=0.4)
    first = c(obs_vec(20.0))[0]
    for _ in range(10):
        c(obs_vec(20.0))
    c.reset()
    assert c(obs_vec(20.0))[0] == pytest.approx(first)


def test_denormalization():
    # z-scored obs with mean 18.73 / std 5.971 must reproduce raw behaviour
    raw = PIALINEAController(ki=40.0, rho_set=30.0)
    z = PIALINEAController(ki=40.0, rho_set=30.0, density_mean=18.73, density_std=5.971)
    u_raw = raw(obs_vec(35.0))[0]
    u_z = z(obs_vec(35.0, mean=18.73, std=5.971))[0]
    assert u_z == pytest.approx(u_raw, abs=1e-6)


def test_queue_override_lower_bound():
    c = PIALINEAController(ki=40.0, rho_set=30.0, u_init=0.1, queue_max=100.0,
                           max_ramp_demand=800.0, dt_ctrl_s=30.0)
    # high density wants to close, but queue 150 > 100 forces release:
    # r_w = 800 + 50 * 120 = 6800 vph -> u = 1 (clamped)
    u = c(obs_vec(45.0, queue=150.0))[0]
    assert u == pytest.approx(1.0)
    c.reset()
    # queue below threshold: override inactive, integral closes as usual
    u = c(obs_vec(45.0, queue=10.0))[0]
    assert u < 0.1


def test_queue_override_reads_ramp_demand_from_obs():
    c = PIALINEAController(ki=0.0, rho_set=30.0, u_init=0.0, queue_max=1000.0,
                           observe_ramp_demand=True, min_ramp_demand=400.0,
                           max_ramp_demand=800.0)
    # queue == queue_max -> r_w = d_ramp exactly; ramp_demand_norm 0.5 -> 600 vph
    u = c(obs_vec(30.0, queue=1000.0, ramp_demand_norm=0.5))[0]
    assert u == pytest.approx(600.0 / 1600.0)


def test_spec_detection_and_parsing():
    assert is_controller_spec("alinea:ki=35,rho=30,det=14")
    assert is_controller_spec("pialinea:kp=4,ki=35")
    assert not is_controller_spec("u=0.5")
    assert not is_controller_spec("runs/rl/x/best_model.zip")

    env_cfg = {"density_mean": 18.73, "density_std": 5.971, "queue_norm_scale": 100.0,
               "observe_ramp_demand": True, "ramp_demand_levels": [400.0, 600.0, 800.0]}
    c = make_controller("alinea:ki=35,rho=32,det=13,u0=0.3", env_cfg)
    assert (c.ki, c.kp, c.rho_set, c.detector_index, c.u_init) == (35.0, 0.0, 32.0, 13, 0.3)
    assert c.density_mean == pytest.approx(18.73)
    assert c.min_ramp_demand == 400.0 and c.max_ramp_demand == 800.0
    assert "ALINEA" in c.label and "ki35" in c.label

    pi = make_controller("pialinea:kp=4,ki=35,rho=30", env_cfg)
    assert pi.kp == 4.0 and pi.label.startswith("PI-ALINEA")

    with pytest.raises(ValueError):
        make_controller("alinea:kp=4,ki=35", env_cfg)  # kp not valid for alinea
    with pytest.raises(ValueError):
        make_controller("alinea:rho=30", env_cfg)  # ki required
    with pytest.raises(ValueError):
        make_controller("alinea:bogus=1,ki=35", env_cfg)
