# BLDC FOC Simulation — PI vs FCS-MPCC vs MPF-MPCC

Python numerical simulation of **field-oriented control (FOC)** for a three-phase
BLDC (SPM) motor, comparing three current control strategies:

1. **PI-FOC** — Cascaded PI control with SVPWM (baseline)
2. **Conventional FCS-MPCC** — Finite-control-set model predictive current control
3. **MPF-MPCC** — Motor-parameter-free model predictive current control (Zhang et al., IEEE TIE 2024)

Based on the reference paper: *X. Zhang, C. Zhang, Z. Wang, and J. Rodríguez,
"Motor-parameter-free model predictive current control for PMSM drives,"
IEEE Trans. Ind. Electron., vol. 71, no. 6, pp. 5443–5452, Jun. 2024.*

## Features

- Clarke/Park transforms (amplitude-invariant) and their inverses
- Space-vector PWM (SVPWM) with common-mode injection
- **PI-FOC**: cascaded speed + dq current PI loops with anti-windup
- **Conventional FCS-MPCC**: 8-state voltage-vector enumeration, delay compensation
- **MPF-MPCC**: parameter-free prediction using current/voltage differences, adaptive α balance factor
- dq-frame motor model with electromagnetic torque
- Parameter mismatch scenarios (2R, 2L, 2ψf) for robustness testing
- THD and RMS error metrics computation

## Files

### Core modules (`pmsm/`)

| File | Description |
|------|-------------|
| `motor.py` | PMSMMotor — dq-frame motor model (forward Euler) |
| `inverter.py` | ThreePhaseInverter + SVPWM modulation |
| `transforms.py` | Clarke, Park, inverse Park transforms |
| `controller.py` | **Three controllers**: `FOCController` (PI), `ConventionalMPCCController`, `MPFMPCCController` |

### Simulation scripts

| File | Description |
|------|-------------|
| `simulation.py` | Original PI-FOC only demo (speed step + load step) |
| `simulation_final.py` | **3-Experiment comparison**: nominal, mismatch (2R/2L/2ψf), load disturbance |
| `simulation_github.py` | Dynamic detail comparison (5-panel figure) |
| `simulation_metrics.py` | Quantitative metrics: id RMS, iq Std, THD |
| `plots.py` | Visualization utilities |

## Quick start

```bash
pip install -r requirements.txt
```

### Run individual simulations

```bash
# Original PI-FOC demo
python simulation.py

# Three-controller comparison (nominal + mismatch)
python simulation_final.py

# Dynamic response details
python simulation_github.py

# Performance metrics (id RMS, iq Std, THD)
python simulation_metrics.py
```

### Simulation results

| Figure | Description |
|--------|-------------|
| `fig_exp1.png` | **Experiment 1**: Nominal parameters — 3×3 panel (speed, id, iq) for PI / CMPC / MPF |
| `fig_exp2.png` | **Experiment 2**: Parameter mismatch (2R,2L,2ψf) — 3×3 panel |
| `fig_github.png` | **Experiment 3**: Dynamic response — 5-panel detail (speed, id, iq, torque, voltage) |
| `fig_metrics.png` | **Metrics**: id RMS / iq Std / THD bar chart (nominal vs mismatch) |
| `docs/foc_simulation.png` | Original PI-FOC demo output |

## Default motor parameters

| Symbol | Nominal | Mismatch (2×) | Description |
|--------|---------|---------------|-------------|
| Rs     | 0.5 Ω   | 1.0 Ω         | Stator resistance |
| Ld, Lq | 1.5 mH  | 3.0 mH        | d/q-axis inductances |
| Ke     | 0.01 V·s/rad | 0.02 V·s/rad | Back-EMF constant |
| J      | 1e-4 kg·m² | —            | Rotor inertia |
| B      | 1e-4 N·m·s | —            | Viscous friction |
| P      | 4        | 4             | Pole pairs |
| Vdc    | 48 V     | 48 V          | DC-bus voltage |

## Simulation settings

| Parameter | Value |
|-----------|-------|
| Time step | 80 μs (12.5 kHz) |
| Control frequency | 12.5 kHz |
| Load torque | 0.02 N·m (nominal) |
| Speed reference | 0 → 1000 rpm (t=0.5s) → 2000 rpm (t=2.0s) |

## Controllers overview

### PI-FOC
- Cascaded structure: speed PI → iq reference, id=0, dq current PI → vd/vq
- SVPWM continuous modulation
- Anti-windup via conditional integration
- **Requires tuning**: PI gains need manual adjustment

### Conventional FCS-MPCC
- Predicts future currents using motor model (Eq.1-7 in paper)
- Enumerates 8 switching states → selects optimal via cost function
- Includes one-step delay compensation
- **Requires motor parameters**: R, L, ψf must be accurate

### MPF-MPCC (parameter-free)
- Predicts current using only Δi(k) and Δu(k) — no motor parameters
- Balance factor α adapts via gradient descent (Eq.22, 31)
- Balance factor β = T_s·ω_e handles back-EMF coupling
- **No motor parameters needed**: robust to parameter mismatch

## Control signal flow (PI-FOC)

```
ω_ref ──► [speed PI] ──► iq_ref ──┐
                                  ▼
      id_ref = 0 ──► [id PI] ──► vd ──┐
                     [iq PI] ──► vq ──┤
                                      ▼
                                [inv Park] ──► vα, vβ
                                                 │
                                              [SVPWM] ──► da, db, dc
                                                 │
                                             [inverter] ──► va, vb, vc
                                                 │
                                          [Clarke → Park] ──► motor (dq)
                                                 │
                       id, iq, ω_m, θ_e  ◄───────┘ (feedback)
```

## License

MIT
