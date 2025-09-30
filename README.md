
# Mass–Spring–Damper System Simulation with PID Control

This project simulates a **1-DOF mass–spring–damper (MSD) system** in Python, with support for open-loop, single-loop PID, and cascade PID control architectures.

Equation of motion:
```
m * x'' + c * x' + k * x = F(t)
```

## Features

### Control Architectures
- **Open-loop**: Predetermined force inputs
- **Single-loop PID**: Direct position or velocity control
- **Cascade PID**: Dual-loop controller with outer position loop and inner velocity loop
- **Direct velocity setpoint**: Bypass position loop for direct velocity control

### Advanced PID Implementation
- Output saturation and anti-windup (integrator clamping)
- Derivative on measurement (better noise behavior)
- Optional low-pass filter on derivative term
- Configurable velocity limits for cascade controller

### Input/Reference Functions
- Step inputs with configurable magnitude and delay
- Impulse (Gaussian approximation of Dirac delta)
- Sinusoidal signals with adjustable frequency and phase
- Linear frequency chirp signals
- Custom force/reference functions

### Performance Analysis
- Comprehensive controller performance metrics:
  - **Tracking performance**: IAE, ITAE, RMSE
  - **Dynamic response**: Rise time, settling time, overshoot, steady-state error
  - **Constraint handling**: Control saturation analysis, control rate limits
- Automatic metrics calculation and formatted display
- Error analysis plots with reference tracking visualization

### Technical Implementation
- Custom 4th-order Runge-Kutta (RK4) numerical integration
- No external dependencies beyond NumPy and Matplotlib
- Modular design with configurable system parameters
- Energy analysis (kinetic, potential, and total energy)

## Requirements
- Python 3.9+
- NumPy
- Matplotlib
- PyQt5 (recommended for interactive plotting)

Install dependencies:
```bash
pip install numpy matplotlib PyQt5
```

## Quick Start

```bash
python msd_sim.py
```

## Usage and Configuration

### Run Modes
Edit the `run_mode` variable in the main block of `msd_sim.py` to select:

- **`"open-loop"`**: Open-loop simulation with predetermined force input
- **`"position"`**: Single-loop PID position control
- **`"velocity"`**: Single-loop PID velocity control  
- **`"cascade"`**: Cascade PID with position reference (outer loop controls position → inner loop controls velocity)
- **`"velocity-setpoint"`**: Direct velocity setpoint control (bypass position loop)
- **`"comparison"`**: Compare multiple controllers side-by-side

### System Parameters
Configure the mass-spring-damper system in the `MSDParams` class:
```python
params = MSDParams(
    mass=20.0,        # Mass [kg]
    damping=20.0,     # Damping coefficient [N·s/m]  
    stiffness=0.0     # Spring stiffness [N/m]
)
```

### Controller Tuning
#### Single PID Controller
```python
pid = PID(
    Kp=200.0,         # Proportional gain
    Ki=60.0,          # Integral gain  
    Kd=10.0,          # Derivative gain
    u_min=-50.0,      # Minimum control force [N]
    u_max=50.0,       # Maximum control force [N]
    deriv_tau=0.002   # Derivative filter time constant
)
```

#### Cascade PID Controller
```python
cascade_pid = CascadePID(
    # Outer loop (position control)
    outer_Kp=50.0, outer_Ki=10.0, outer_Kd=5.0,
    # Inner loop (velocity control)  
    inner_Kp=100.0, inner_Ki=20.0, inner_Kd=2.0,
    # Constraints
    u_min=-50.0, u_max=50.0,
    velocity_limit=(-2.0, 2.0)  # Optional velocity saturation
)
```

### Reference Signals
Choose from predefined reference functions:
```python
# Step reference
ref = step_ref(magnitude=0.1, t_delay=0.5)

# Or create custom reference functions
ref = lambda t: 0.1 * np.sin(2*np.pi*0.5*t)  # Sinusoidal reference
```

## Cascade PID Control Architecture

The cascade controller implements a dual-loop structure:

```
Position Reference → [Outer PID] → Velocity Setpoint → [Inner PID] → Force Output
                                           ↑                 ↑
                                   Position Feedback    Velocity Feedback
```

### Benefits of Cascade Control
- **Improved disturbance rejection**: Inner loop quickly responds to velocity disturbances
- **Better transient response**: Separate tuning of position and velocity dynamics
- **Flexible operation**: Can operate in full cascade mode or direct velocity control mode

### Operating Modes
1. **Full Cascade Mode**: Position reference drives outer loop, which generates velocity setpoint for inner loop
2. **Direct Velocity Control**: Bypass outer loop and directly control velocity setpoint

## Controller Performance Evaluation

The simulation includes a comprehensive `ControlEvaluator` class that automatically calculates and displays performance metrics:

### Tracking Performance Metrics
- **IAE (Integral Absolute Error)**: `∫|e(t)|dt` - Measures cumulative tracking error
- **ITAE (Integral Time Absolute Error)**: `∫t|e(t)|dt` - Penalizes persistent errors more heavily  
- **RMSE (Root Mean Square Error)**: `√(1/N ∑e²)` - Quantifies error magnitude variations

### Dynamic Response Analysis
- **Rise Time (10-90%)**: Time to transition from 10% to 90% of final value
- **Settling Time (2%)**: Time to settle and remain within 2% of final value
- **Overshoot**: Maximum excursion beyond setpoint as percentage
- **Steady-state Error**: Average tracking error in final 10% of simulation

### Constraint Analysis  
- **Control Saturation**: Percentage of time at upper/lower control limits
- **Control Rate**: Maximum and average rate of change of control signal
- **Velocity Limiting**: Analysis of velocity saturation in cascade mode

### Visualization and Analysis
- **Multi-plot displays**: System states, control signals, energy, reference tracking
- **Error analysis plots**: Reference vs. output comparison with error visualization
- **Cascade-specific plots**: Shows both position and velocity loop performance
- **Formatted performance reports**: Automatic calculation and display of all metrics

## Example Output

### Performance Metrics Display
```
==================================================
CONTROLLER PERFORMANCE METRICS
==================================================

--- Tracking Performance ---
IAE:   0.01847
ITAE:  0.08129  
RMSE:  0.006841

--- Dynamic Response ---
Rise time (10-90%): 0.242 s
Settling time (2%): 4.63 s
Overshoot:          3.79 %
Steady-state error: 8.33e-06

--- Constraint Handling ---
Control saturation: 0.00 %
Max control rate:   2.156e+04
==================================================
```

### Plotting Features
- **System Response**: Displacement, velocity, control force, and energy plots
- **Reference Tracking**: Comparison of setpoint vs. actual response
- **Cascade Visualization**: Dual-loop controller with position and velocity references
- **Error Analysis**: Dedicated plots for tracking error evaluation

## Mathematical Background

### System Dynamics
The mass-spring-damper system is governed by the second-order differential equation:
```
m⋅ẍ + c⋅ẋ + k⋅x = F(t)
```

Where:
- `m`: Mass [kg]  
- `c`: Damping coefficient [N⋅s/m]
- `k`: Spring stiffness [N/m]
- `x`: Displacement [m]
- `F(t)`: Applied force [N]

### System Characterization
- **Natural frequency**: `ωₙ = √(k/m)` [rad/s]
- **Damping ratio**: `ζ = c/(2√(km))` [-]
- **Damped frequency**: `ωd = ωₙ√(1-ζ²)` [rad/s]

### PID Controller
The PID controller implements:
```
u(t) = Kₚe(t) + Kᵢ∫e(τ)dτ + Kd⋅de(t)/dt
```

With additional features:
- **Anti-windup**: Prevents integrator saturation during control limiting
- **Derivative filtering**: Low-pass filter with time constant `τd` to reduce noise
- **Output saturation**: Configurable force limits `[uₘᵢₙ, uₘₐₓ]`

## Controller Tuning Guidelines

### Single-Loop PID Tuning
1. **Start with P-only control**: Increase `Kₚ` until system responds with minor overshoot
2. **Add integral action**: Increase `Kᵢ` to eliminate steady-state error
3. **Add derivative action**: Increase `Kd` to reduce overshoot and improve settling time
4. **Enable derivative filtering**: Set `deriv_tau ≈ 1-5×dt` for noise reduction

### Cascade PID Tuning  
1. **Tune inner loop first**: Design velocity controller with good disturbance rejection
   - Typically faster response: higher bandwidth, smaller time constants
   - Focus on `Kₚ` and `Kᵢ` for velocity tracking, moderate `Kd`
2. **Tune outer loop**: Design position controller assuming well-tuned inner loop
   - Typically slower response: lower bandwidth, larger time constants  
   - Inner loop bandwidth should be 5-10× faster than outer loop
3. **Velocity limiting**: Set reasonable `velocity_limit` to prevent unrealistic setpoints

### Recommended Starting Values
```python
# For mass-spring-damper with m=20kg, c=20N⋅s/m, k=0N/m (pure integrator)
cascade_pid = CascadePID(
    outer_Kp=50.0,   # Position loop - moderate for stability
    outer_Ki=10.0,   # Position loop - eliminate steady-state error  
    outer_Kd=5.0,    # Position loop - improve transient response
    inner_Kp=100.0,  # Velocity loop - higher for fast response
    inner_Ki=20.0,   # Velocity loop - remove velocity tracking error
    inner_Kd=2.0,    # Velocity loop - dampen velocity oscillations
    velocity_limit=(-2.0, 2.0)  # Reasonable velocity constraints
)
```

## Applications and Use Cases

### Educational Applications
- Control systems education and demonstration
- PID tuning exercises and experimentation  
- Performance metrics analysis and interpretation
- Cascade control architecture understanding

### Research and Development
- Controller performance benchmarking
- Algorithm development and testing
- System identification and parameter estimation
- Disturbance rejection analysis

### Industrial Relevance
- Position control systems (actuators, robotics)
- Velocity control applications (motor drives)
- Cascade control in process industries
- Performance optimization and tuning methodologies

## File Structure

```
msd_sim.py              # Main simulation script
├── MSDParams           # System parameter dataclass
├── SimConfig           # Simulation configuration  
├── MassSpringDamper    # System dynamics and simulation methods
├── PID                 # Single-loop PID controller
├── CascadePID          # Dual-loop cascade PID controller  
├── ControlEvaluator    # Performance metrics and analysis
└── Helper functions    # Reference signals, plotting, utilities
requirements.txt        # Python dependencies
README.md              # This documentation
```

## Contributing


## License
