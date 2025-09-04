# Mass–Spring–Damper Simulation

This project simulates a **1-DOF mass–spring–damper system** using Python.

Equation of motion:
m * x'' + c * x' + k * x = F(t)

## Features
- Simulates displacement, velocity, and energy over time.
- Supports different input forces:
  - Step
  - Impulse
  - Sinusoidal
  - Chirp
- Uses RK4 integration (no SciPy needed).
- Plots results with Matplotlib.

## Requirements
- Python 3.9+
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install numpy matplotlib
