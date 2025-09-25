
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
- Comprehensive controller performance metrics:
  - Tracking performance: IAE, ITAE, RMSE
  - Dynamic response: Rise time, settling time, overshoot, steady-state error
  - Constraint handling: Control saturation, control rate
  - Automatic metrics calculation and display
- Uses custom RK4 integration (no SciPy needed)
- Plots results (state, control, energy, reference tracking) with Matplotlib
- cascade pid 

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
- Select run mode ("open-loop", "position", "velocity", or "comparison")

## Controller Performance Evaluation

The simulation includes a `ControlEvaluator` class that automatically calculates and displays:

### Tracking Performance
- **IAE (Integral Absolute Error)**: Measures cumulative error over time
- **ITAE (Integral Time Absolute Error)**: Weights errors by time (later errors penalized more)
- **RMSE (Root Mean Square Error)**: Measures the magnitude of error variations

### Dynamic Response
- **Rise Time**: Time to rise from 10% to 90% of the final value
- **Settling Time**: Time to settle within 2% of the final value
- **Overshoot**: Maximum excursion beyond the final value
- **Steady-state Error**: Average error at steady state

### Constraint Handling
- **Control Saturation**: Percentage of time the controller hits its limits
- **Control Rate**: Maximum and average rate of control signal change

### Visualization
- Reference tracking plots
- Error analysis plots

## Example Output

The script will display plots showing system response, control input, energy, and controller performance metrics. Example:

![Example plot showing displacement, velocity, control, and energy](example_plot.png)