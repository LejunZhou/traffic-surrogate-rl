"""Open-loop random control samplers used by the M14 behavior mixture.

Rollout collection is handled by scripts/generate_round0_dataset.py.
"""

import numpy as np


def sample_ramp_control(
    control_type: str, T_ctrl: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample a ramp metering signal from the specified family.

    Args:
        control_type: One of "constant", "piecewise_constant", "smooth",
                      "ramp_step".
        T_ctrl: Number of control steps (120).
        rng: numpy random generator for reproducibility.

    Returns:
        shape (T_ctrl,) float32 array with values in [0, 1].
    """
    if control_type == "constant":
        return _sample_constant(T_ctrl, rng)
    elif control_type == "piecewise_constant":
        return _sample_piecewise_constant(T_ctrl, rng)
    elif control_type == "smooth":
        return _sample_smooth(T_ctrl, rng)
    elif control_type == "ramp_step":
        return _sample_ramp_step(T_ctrl, rng)
    else:
        raise ValueError(f"Unknown ramp control type: {control_type!r}")


def _sample_constant(T_ctrl: int, rng: np.random.Generator) -> np.ndarray:
    r = rng.uniform(0.0, 1.0)
    return np.full(T_ctrl, r, dtype=np.float32)


def _sample_piecewise_constant(
    T_ctrl: int, rng: np.random.Generator
) -> np.ndarray:
    n_segments = rng.integers(2, 7)  # 2 to 6 segments
    changepoints = np.sort(rng.integers(1, T_ctrl, size=n_segments - 1))
    levels = rng.uniform(0.0, 1.0, size=n_segments).astype(np.float32)

    signal = np.empty(T_ctrl, dtype=np.float32)
    prev = 0
    for i, cp in enumerate(changepoints):
        signal[prev:cp] = levels[i]
        prev = cp
    signal[prev:] = levels[-1]
    return signal


def _sample_smooth(T_ctrl: int, rng: np.random.Generator) -> np.ndarray:
    n_knots = rng.integers(4, 9)  # 4 to 8 knots
    knot_positions = np.linspace(0, T_ctrl - 1, n_knots)
    knot_values = rng.uniform(0.0, 1.0, size=n_knots)
    signal = np.interp(
        np.arange(T_ctrl), knot_positions, knot_values
    ).astype(np.float32)
    return np.clip(signal, 0.0, 1.0)


def _sample_ramp_step(T_ctrl: int, rng: np.random.Generator) -> np.ndarray:
    peak = rng.uniform(0.3, 1.0)
    pattern = rng.integers(0, 4)  # 4 patterns

    # Randomize transition timing
    t_start = rng.integers(T_ctrl // 6, T_ctrl // 3)
    t_end = rng.integers(2 * T_ctrl // 3, 5 * T_ctrl // 6)

    signal = np.zeros(T_ctrl, dtype=np.float32)

    if pattern == 0:  # monotone increase: 0 → peak
        signal[:t_start] = 0.0
        ramp = np.linspace(0.0, peak, t_end - t_start, dtype=np.float32)
        signal[t_start:t_end] = ramp
        signal[t_end:] = peak
    elif pattern == 1:  # monotone decrease: peak → 0
        signal[:t_start] = peak
        ramp = np.linspace(peak, 0.0, t_end - t_start, dtype=np.float32)
        signal[t_start:t_end] = ramp
        signal[t_end:] = 0.0
    elif pattern == 2:  # up-then-down: 0 → peak → 0
        mid = (t_start + t_end) // 2
        signal[:t_start] = 0.0
        signal[t_start:mid] = np.linspace(
            0.0, peak, mid - t_start, dtype=np.float32
        )
        signal[mid:t_end] = np.linspace(
            peak, 0.0, t_end - mid, dtype=np.float32
        )
        signal[t_end:] = 0.0
    else:  # down-then-up: peak → 0 → peak
        mid = (t_start + t_end) // 2
        signal[:t_start] = peak
        signal[t_start:mid] = np.linspace(
            peak, 0.0, mid - t_start, dtype=np.float32
        )
        signal[mid:t_end] = np.linspace(
            0.0, peak, t_end - mid, dtype=np.float32
        )
        signal[t_end:] = peak

    return np.clip(signal, 0.0, 1.0)
