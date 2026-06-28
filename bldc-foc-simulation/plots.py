"""Matplotlib visualization of FOC simulation results."""
from __future__ import annotations

import os

import matplotlib

import numpy as np

_HAS_DISPLAY = bool(os.environ.get("DISPLAY")) or (hasattr(os, "uname") and os.uname().sysname == "Darwin")
if not _HAS_DISPLAY:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_results(data: dict[str, np.ndarray], path: str = "foc_simulation.png") -> None:
    plt.style.use("seaborn-v0_8-whitegrid") if "seaborn-v0_8-whitegrid" in plt.style.available else plt.style.use("default")

    t = data["t"]
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=False)

    # Panel 1: speed
    ax = axes[0]
    ax.plot(t, data["omega_ref_rpm"], "k--", label="omega_ref", linewidth=1.2)
    ax.plot(t, data["omega_m_rpm"], "C0", label="omega_m", linewidth=1.2)
    ax.set_ylabel("Speed [rpm]")
    ax.set_title("Rotor speed (reference vs actual)")
    ax.legend(loc="best")
    ax.grid(True)

    # Panel 2: dq currents
    ax = axes[1]
    ax.plot(t, data["id"], "C1", label="id", linewidth=1.0)
    ax.plot(t, data["iq"], "C2", label="iq", linewidth=1.0)
    ax.set_ylabel("Current [A]")
    ax.set_title("dq-axis currents")
    ax.legend(loc="best")
    ax.grid(True)

    # Panel 3: three-phase currents (zoomed window around steady state before load step)
    ax = axes[2]
    t_end = float(t[-1])
    # Zoom window: 20 ms starting near middle of final speed segment
    if t_end >= 0.34:
        t0 = 0.30
    else:
        t0 = max(0.0, t_end - 0.04)
    t1 = min(t_end, t0 + 0.02)
    mask = (t >= t0) & (t <= t1)
    ax.plot(t[mask], data["ia"][mask], "C0", label="ia", linewidth=1.0)
    ax.plot(t[mask], data["ib"][mask], "C3", label="ib", linewidth=1.0)
    ax.plot(t[mask], data["ic"][mask], "C2", label="ic", linewidth=1.0)
    ax.set_ylabel("Current [A]")
    ax.set_title(f"Three-phase currents (zoom {t0*1e3:.0f}-{t1*1e3:.0f} ms)")
    ax.legend(loc="best", ncol=3)
    ax.grid(True)

    # Panel 4: electromagnetic torque
    ax = axes[3]
    ax.plot(t, data["Te"], "C4", label="Te", linewidth=1.0)
    if "TL" in data:
        ax.plot(t, data["TL"], "k--", label="TL", linewidth=1.0)
    ax.set_ylabel("Torque [N·m]")
    ax.set_title("Electromagnetic torque")
    ax.legend(loc="best")
    ax.grid(True)

    # Panel 5: dq voltages
    ax = axes[4]
    ax.plot(t, data["vd"], "C1", label="vd", linewidth=1.0)
    ax.plot(t, data["vq"], "C2", label="vq", linewidth=1.0)
    ax.set_ylabel("Voltage [V]")
    ax.set_xlabel("Time [s]")
    ax.set_title("dq-axis voltages")
    ax.legend(loc="best")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=120)

    if _HAS_DISPLAY:
        try:
            plt.show()
        except Exception:
            pass
