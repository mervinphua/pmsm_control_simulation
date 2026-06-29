"""Closed-loop FOC simulation driver for an SPMSM motor."""
from __future__ import annotations

import math

import numpy as np

from pmsm.controller import FOCController
from pmsm.inverter import ThreePhaseInverter
from pmsm.motor import PMSMMotor
from pmsm.transforms import clarke, inv_clarke, inv_park, park


RPM_TO_RADS = 2.0 * math.pi / 60.0


def run_simulation(
    t_end: float = 0.5,
    dt: float = 1e-5,
    omega_ref_rpm: float | np.ndarray = 2000.0,
    TL: float | np.ndarray = 0.0,
) -> dict[str, np.ndarray]:
    motor = PMSMMotor()
    controller = FOCController()
    inverter = ThreePhaseInverter()

    n = int(round(t_end / dt))
    t = np.arange(n) * dt

    if np.isscalar(omega_ref_rpm):
        omega_ref_arr = np.full(n, float(omega_ref_rpm))
    else:
        omega_ref_arr = np.asarray(omega_ref_rpm, dtype=float)
    if np.isscalar(TL):
        TL_arr = np.full(n, float(TL))
    else:
        TL_arr = np.asarray(TL, dtype=float)

    omega_m_log = np.zeros(n)
    id_log = np.zeros(n)
    iq_log = np.zeros(n)
    vd_log = np.zeros(n)
    vq_log = np.zeros(n)
    Te_log = np.zeros(n)
    theta_e_log = np.zeros(n)
    ia_log = np.zeros(n)
    ib_log = np.zeros(n)
    ic_log = np.zeros(n)

    for k in range(n):
        theta_e = motor.theta_e
        omega_m = motor.omega_m
        id_ = motor.id
        iq = motor.iq

        omega_ref_rads = omega_ref_arr[k] * RPM_TO_RADS
        vd_cmd, vq_cmd = controller.update(omega_ref_rads, omega_m, id_, iq, theta_e, dt)

        valpha, vbeta = inv_park(vd_cmd, vq_cmd, theta_e)
        da, db, dc = inverter.svpwm(valpha, vbeta)
        va, vb, vc = inverter.apply(da, db, dc)

        valpha_act, vbeta_act = clarke(va, vb, vc)
        vd_act, vq_act = park(valpha_act, vbeta_act, theta_e)

        motor.TL = float(TL_arr[k])
        state = motor.step(vd_act, vq_act, dt)

        ialpha, ibeta = inv_park(state["id"], state["iq"], state["theta_e"])
        ia, ib, ic = inv_clarke(ialpha, ibeta)

        omega_m_log[k] = state["omega_m"] / RPM_TO_RADS
        id_log[k] = state["id"]
        iq_log[k] = state["iq"]
        vd_log[k] = vd_cmd
        vq_log[k] = vq_cmd
        Te_log[k] = state["Te"]
        theta_e_log[k] = state["theta_e"]
        ia_log[k] = ia
        ib_log[k] = ib
        ic_log[k] = ic

    return {
        "t": t,
        "omega_ref_rpm": omega_ref_arr,
        "omega_m_rpm": omega_m_log,
        "id": id_log,
        "iq": iq_log,
        "vd": vd_log,
        "vq": vq_log,
        "Te": Te_log,
        "theta_e": theta_e_log,
        "ia": ia_log,
        "ib": ib_log,
        "ic": ic_log,
        "TL": TL_arr,
        "P": motor.P,
    }


def main() -> None:
    t_end = 0.5
    dt = 1e-5
    n = int(round(t_end / dt))
    t = np.arange(n) * dt

    omega_ref_rpm = np.where(t < 0.2, 1000.0, 2000.0)
    TL = np.where(t < 0.35, 0.0, 0.05)

    data = run_simulation(t_end=t_end, dt=dt, omega_ref_rpm=omega_ref_rpm, TL=TL)

    from plots import plot_results

    plot_results(data)


if __name__ == "__main__":
    main()
