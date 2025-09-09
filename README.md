
# Mass–Spring–Damper System Simulation with PID Control

This project simulates a **1-DOF mass–spring–damper (MSD) system** in Python, with support for both open-loop and closed-loop (PID-controlled) operation.

Equation of motion:
```
m * x'' + c * x' + k * x = F(t)
```

## Features
- Simulates displacement, velocity, and energy over time
- Supports open-loop (preset force input) and closed-loop (PID control) simulations
- PID controller includes:
  - Output saturation and anti-windup (integrator clamping)
  - Derivative on measurement (better noise behavior)
  - Optional low-pass filter on derivative
- Multiple input/reference types:
  - Step
  - Impulse
  - Sinusoidal
  - Chirp
- Uses custom RK4 integration (no SciPy needed)
- Plots results (state, control, energy, reference tracking) with Matplotlib

## Requirements
- Python 3.9+
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install numpy matplotlib
```

## Run the Simulation

```bash
python msd_sim.py
```

Edit the main block in `msd_sim.py` to:
- Select open-loop or closed-loop simulation
- Choose force/reference input type
- Adjust PID parameters

## Example Output

The script will display plots showing system response, control input, and energy. Example:

![Example plot showing displacement, velocity, control, and energy](example_plot.png)