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

### Test Scenarios
The `Scenario` class provides comprehensive reference signals and disturbances for systematic controller testing:

#### 1. Steps and Ramps (Position and Velocity)
```python
# Position step
ref = Scenario.step_position(magnitude=0.1, t_start=0.5)

# Velocity step
ref = Scenario.step_velocity(magnitude=0.05, t_start=0.5)

# Position ramp
ref = Scenario.ramp_position(slope=0.02, t_start=0.5, t_end=5.0)

# Velocity ramp
ref = Scenario.ramp_velocity(slope=0.01, t_start=0.5)

# Multiple position steps
ref = Scenario.multi_step_position(
    steps=[0.05, 0.1, 0.15], 
    times=[1.0, 3.0, 5.0]
)
```

#### 2. Station-Keeping (with Disturbances)
Test controller's ability to maintain position under external disturbances:
```python
# Station-keeping reference (maintain position at 0.0)
ref = Scenario.station_keeping(position=0.0)

# Add step disturbance
disturbance = Scenario.disturbance_step(magnitude=5.0, t_start=2.0)

# Add impulse disturbance
disturbance = Scenario.disturbance_impulse(area=1.0, t_impulse=2.0, width=0.01)

# Add sinusoidal disturbance
disturbance = Scenario.disturbance_sinusoidal(amplitude=2.0, frequency=1.0)

# Add random walk disturbance
disturbance = Scenario.disturbance_random_walk(magnitude=0.1, seed=42)
```

#### 3. Straight-Line Cruise (Constant Velocity)
```python
# Constant velocity cruise
ref = Scenario.cruise(velocity=0.05)
```

#### 4. Path Following: Combined Maneuvers
```python
# Sinusoidal position tracking
ref = Scenario.sinusoidal_position(
    amplitude=0.05, 
    frequency=0.5, 
    phase=0.0, 
    offset=0.1
)

# Sinusoidal velocity tracking
ref = Scenario.sinusoidal_velocity(
    amplitude=0.05, 
    frequency=0.5
)

# Combined maneuver (position tracking → velocity cruise)
ref = Scenario.combined_maneuver(
    position_amplitude=0.1,
    velocity_offset=0.02,
    frequency=0.3,
    t_transition=2.0
)
```

#### Example: Station-Keeping Under Disturbance
```python
# Configure system for station-keeping test
params = MSDParams(mass=20.0, damping=20.0, stiffness=0.0)

# Create disturbance force
disturbance = Scenario.disturbance_step(magnitude=5.0, t_start=2.0)
system = MassSpringDamper(params, disturbance)

# Station-keeping reference
ref = Scenario.station_keeping(position=0.0)

# Simulate with cascade controller
cascade_pid = CascadePID(
    outer_Kp=50.0, outer_Ki=10.0, outer_Kd=5.0,
    inner_Kp=100.0, inner_Ki=20.0, inner_Kd=2.0,
    u_min=-50.0, u_max=50.0
)

t, x, v, u, E, KE, PE, x_ref, v_ref = system.simulate_cascade_loop(
    cfg=SimConfig(tf=10.0, dt=0.001),
    cascade_controller=cascade_pid,
    ref_fn=ref,
    mode="position",
    use_feedforward=False  # Don't use disturbance as feedforward
)
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

### Cost Aggregation for Optimization

The `ControlEvaluator` includes a `cost_from_metrics()` method that aggregates per-scenario metrics into a scalar cost J for controller optimization (e.g., Bayesian optimization):

**Pipeline**: `timeseries → metrics → normalized metrics → weighted sum → J`

#### Cost Function
For a given controller configuration φ, metrics from multiple scenarios are aggregated:

**Mean Aggregator** (default):
```
J(φ) = (1/|S|) Σ [w_r·t̂_r + w_s·t̂_s + w_o·M̂_p + w_e·ŜSE]
```

**Worst-Case Aggregator** (robust):
```
J_wc(φ) = max [w_r·t̂_r + w_s·t̂_s + w_o·M̂_p + w_e·ŜSE]
```

Where:
- `t̂_r`: Normalized rise time (10-90%)
- `t̂_s`: Normalized settling time (2% band)
- `M̂_p`: Normalized overshoot (%)
- `ŜSE`: Normalized steady-state error
- `w_r, w_s, w_o, w_e`: Weights reflecting mission priorities

#### Default Configuration (Defined in BOEvaluator)
```python
# Weights (should sum to 1.0)
weights = {
    'rise_time': 0.3,      # Fast response priority
    'settling_time': 0.3,  # Stability priority
    'overshoot': 0.2,      # Damping priority
    'sse': 0.2            # Accuracy priority
}

# Normalization references
normalization_refs = {
    'rise_time': 1.0,      # 1 second
    'settling_time': 5.0,  # 5 seconds
    'overshoot': 20.0,     # 20%
    'sse': 0.01           # 0.01 m or m/s
}
```

**Note**: `cost_from_metrics()` requires `weights` and `normalization_refs` to be explicitly passed (no defaults). The `BOEvaluator` class is the single source of truth for these configuration values.

#### Usage Example
```python
evaluator = ControlEvaluator()
metrics_list = []

# Run multiple scenarios
for scenario_ref, disturbance in test_scenarios:
    system = MassSpringDamper(params, disturbance)
    t, x, v, u, E, KE, PE, x_ref, v_ref = system.simulate_cascade_loop(
        cfg, cascade_pid, ref_fn=scenario_ref, mode="position"
    )
    
    # Compute metrics for this scenario
    metrics = evaluator.evaluate(t, x, x_ref, u, u_min=-50, u_max=50)
    metrics_list.append(metrics)

# Aggregate into scalar cost
# Note: Must explicitly provide weights and normalization_refs
J = evaluator.cost_from_metrics(
    metrics_list,
    weights={'rise_time': 0.3, 'settling_time': 0.3, 'overshoot': 0.2, 'sse': 0.2},
    normalization_refs={'rise_time': 1.0, 'settling_time': 5.0, 'overshoot': 20.0, 'sse': 0.01},
    use_worst_case=False  # Use mean aggregation
)

print(f"Aggregate cost J = {J:.4f}")  # Lower is better
```

#### Applications
- **Bayesian Optimization**: Use J as objective function for hyperparameter tuning
- **Controller Benchmarking**: Compare different controller configurations
- **Robust Design**: Use worst-case aggregator for conservative tuning
- **Multi-Scenario Testing**: Ensure performance across diverse operating conditions

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

## Bayesian Optimization for PID Tuning

The `BOEvaluator` class provides a complete framework for automated PID tuning using Bayesian Optimization. It manages multi-stage tuning, constraint checking, and provides a clean interface for optimization libraries.

### Architecture: Separation of Concerns

**BOEvaluator** (Constraint Checking & Safety):
- Performs **early-exit** constraint checking during simulation
- Prevents unnecessary computation for infeasible candidates
- Handles: energy divergence, actuator saturation, position envelope, NaN/Inf detection
- Returns large penalty (`1e6`) for constraint violations

**ControlEvaluator** (Pure Metric Aggregation):
- Purely focused on metric normalization and cost calculation
- NO constraint checking - assumes feasible inputs
- Converts multi-scenario metrics into scalar cost `J`

This design eliminates redundancy while maintaining safety through early exits in `BOEvaluator._run_scenarios()`.

### Constraint Checking (Safety Configuration)

Candidates are **immediately rejected** during simulation if any scenario exhibits:

1. **Numerical blow-up**: Energy divergence or NaN/Inf in metrics
2. **Excessive actuator saturation**: Beyond specified percentage (default: 95%)
3. **Position envelope violation**: Exceeds spatial bounds

```python
safety_cfg = {
    'max_saturation': 95.0,   # Max % saturation allowed
    'max_energy': 1e6,        # Energy divergence threshold
    'max_position': 100.0     # Position envelope limit [m]
}
```

These checks happen in `BOEvaluator._run_scenarios()` **before** calling `cost_from_metrics()`, preventing wasted computation on infeasible solutions.

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
├── Scenario            # Test scenarios and disturbances generator
├── ControlEvaluator    # Performance metrics and analysis
└── Helper functions    # Reference signals, plotting, utilities
requirements.txt        # Python dependencies
README.md              # This documentation
```

## Contributing


## License
