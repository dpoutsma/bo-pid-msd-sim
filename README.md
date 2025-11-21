
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
- **`"cascade"`**: Cascade PID testing with easy mode switching (see Cascade Mode Testing below)
- **`"velocity-setpoint"`**: Direct velocity setpoint control (bypass position loop, includes cost calculation for optimization)
- **`"comparison"`**: Compare multiple controllers side-by-side
- **`"bayesian_optimization"`**: Automated PID tuning using Bayesian Optimization

### Cascade Mode Testing

The cascade mode includes a convenient `TEST_MODE` variable for easy switching between testing scenarios:

```python
run_mode = "cascade"

# Change this single line to switch test scenarios:
TEST_MODE = "velocity"  # Options: "velocity" or "position"
```

**Velocity Mode** (Inner Loop Only):
- Tests inner velocity loop with outer position loop disabled
- Two built-in scenarios (uncomment to switch):
  - **Cruise + Disturbance** (default): Station-keeping with impulse disturbance
  - **Velocity Step**: Step tracking performance

**Position Mode** (Full Cascade):
- Tests complete cascade system with both loops active
- Two built-in scenarios (uncomment to switch):
  - **Position Step** (default): Step tracking with both loops
  - **Position + Disturbance**: Step with external disturbance

**Example**:
```python
# Test velocity disturbance rejection
TEST_MODE = "velocity"
reference = Scenario.cruise(velocity=0.0)
disturbance = Scenario.disturbance_impulse(area=1.0, t_impulse=2.0, width=0.01)

# Test position tracking  
TEST_MODE = "position"
reference = Scenario.step_position(magnitude=0.1, t_start=0.5)
disturbance = lambda t: 0.0
```

The code automatically:
- ✅ Sets correct control mode
- ✅ Selects appropriate controlled variable (velocity or position)
- ✅ Configures reference and disturbance
- ✅ Updates plots and performance output

### System Parameters
Configure the mass-spring-damper system in the `MSDParams` class:
```python
params = MSDParams(
    mass=20.0,        # Mass [kg]
    damping=20.0,     # Damping coefficient [N·s/m]  
    stiffness=0.0     # Spring stiffness [N/m]
)
```

### Simulation Configuration

```python
cfg = SimConfig(
    t0=0.0,      # Start time [s]
    tf=10.0,     # End time [s]
    dt=0.05,     # Time step [s] - affects stability limits!
    x0=0.0,      # Initial position [m]
    v0=0.0       # Initial velocity [m/s]
)
```

**Important**: The time step `dt` affects discrete-time stability:
- **Testing/validation**: Use `dt=0.001` (1ms) for high-fidelity simulation
- **Bayesian optimization**: Use `dt=0.05` (50ms) for faster evaluations
- **Stability impact**: Larger `dt` reduces stability margins (see stability analysis in cascade mode)

For the MSD system with `dt=0.05s`:
- Velocity loop stable up to: `Ki ≈ 79,000` (Nyquist limit)
- Recommended max gains reduced from `dt=0.001` limits
- Trade-off: 50× faster simulations vs. reduced stability margins

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

**Important**: To apply disturbances to the system dynamics, use `use_feedforward=True` in simulation:

```python
# Apply disturbance to system
msd.F = disturbance

# Enable feedforward to add disturbance force to dynamics
t, x, v, u, E, KE, PE, x_ref, v_ref = msd.simulate_cascade_loop(
    cfg, cascade_pid, ref_fn=ref, mode="position",
    use_feedforward=True  # Critical: enables disturbance application
)
```

**Note**: If `use_feedforward=False` (default for some modes), the disturbance is ignored even if set. Always use `True` when testing disturbance rejection.

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

The simulation includes a comprehensive `ControllerMetrics` class that automatically calculates and displays performance metrics:

### Tracking Performance Metrics
- **IAE (Integral Absolute Error)**: `∫|e(t)|dt` - Measures cumulative tracking error
- **ITAE (Integral Time Absolute Error)**: `∫t|e(t)|dt` - Penalizes persistent errors more heavily  
- **RMSE (Root Mean Square Error)**: `√(1/N ∑e²)` - Quantifies error magnitude variations

### Dynamic Response Analysis

The evaluator automatically detects the reference type and applies appropriate metric calculations:

#### Step Response Metrics (for step references)
- **Rise Time (10-90%)**: Time to transition from 10% to 90% of final value
- **Settling Time (2%)**: Time to settle and remain within 2% of final value
- **Overshoot**: Maximum excursion beyond setpoint as percentage
- **Steady-state Error**: Average tracking error in final 10% of simulation

#### Disturbance Response Metrics (for constant references)

For station-keeping and cruise scenarios with disturbances, specialized metrics measure disturbance rejection performance:

- **Rise Time (10-90%)**: Time to recover from peak error to 10% of peak (90% recovery)
- **Settling Time (2%)**: Time to reduce error to 2% of peak and maintain it
- **Peak Deviation**: Maximum error during disturbance as percentage of setpoint
- **Steady-state Error**: Residual error after disturbance rejection

**Key Features**:
- Automatic detection of constant vs. step references
- Disturbance detection threshold: 0.001 (detects even well-controlled disturbances)
- Normalized metrics relative to peak disturbance magnitude
- Returns NaN if no significant disturbance detected (max error < 0.001)

Example output for disturbance rejection:
```
--- Dynamic Response ---
Rise time (10-90%): 5.179 s    # Time to 90% recovery from peak
Settling time (2%): 8.587 s    # Time to 2% of peak and stable
Overshoot:          34.61 %    # Peak deviation from setpoint
Steady-state error: 2.17e-07   # Final residual error
```

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

The `ControllerMetrics` includes a `cost_from_metrics()` method that aggregates per-scenario metrics into a scalar cost J for controller optimization (e.g., Bayesian optimization):

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

#### Default Configuration (Defined in BayesianOptimizer)
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

**Note**: `cost_from_metrics()` requires `weights` and `normalization_refs` to be explicitly passed (no defaults). The `BayesianOptimizer` class is the single source of truth for these configuration values.

#### Usage Example
```python
evaluator = ControllerMetrics()
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

The `BayesianOptimizer` class provides a complete framework for automated PID tuning using Bayesian Optimization. It manages multi-stage tuning, constraint checking, and provides a clean interface for optimization libraries.

### Visualization Tools

The framework includes advanced visualization capabilities to understand BO behavior and Gaussian Process (GP) learning:

- **2D Progression Plots** (`plot_bo_progression_2d()`): Shows evolution of GP mean, uncertainty, and Expected Improvement (EI) acquisition function across iterations as 2D contour plots. Reveals how the optimizer explores the parameter space and converges to optimal regions.

- **1D Slice Analysis** (`plot_bo_progression_1d()`): Examines parameter sensitivity along single dimensions with automatic flat-region detection to identify parameter coupling and low-sensitivity directions.

- **Final State Visualization** (`plot_gp_and_acquisition_2d()`): Displays final GP state and acquisition landscape after optimization completes, showing the learned surrogate model.

- **EI Diagnostics**: Tracks Expected Improvement convergence statistics (min/max/mean) at each iteration. EI→0 indicates successful optimization and confidence in current optimum.

#### Understanding Acquisition Function Behavior

For deeper insights into optimization dynamics, see these explanation documents:
- `EI_CONVERGENCE_EXPLAINED.md` - Why Expected Improvement converges to near-zero values (this is desired behavior)
- `WHY_EI_STARTS_NEAR_ZERO.md` - How random initialization can find optimal quickly in tight bounds
- `WHY_1D_SLICES_APPEAR_FLAT.md` - Understanding parameter coupling effects in 1D slice visualizations

**Note**: The number of random initialization points affects convergence speed. Current configuration uses 2 random seeds before BO iterations begin. More random samples can find good solutions faster but may reduce the visibility of EI evolution in early iterations.

### Architecture: Separation of Concerns

**BayesianOptimizer** (Constraint Checking & Safety):
- Performs **early-exit** constraint checking during simulation
- Prevents unnecessary computation for infeasible candidates
- Handles: energy divergence, actuator saturation, position envelope, NaN/Inf detection
- Returns large penalty (`1e6`) for constraint violations

**ControllerMetrics** (Pure Metric Aggregation):
- Purely focused on metric normalization and cost calculation
- NO constraint checking - assumes feasible inputs
- Converts multi-scenario metrics into scalar cost `J`

This design eliminates redundancy while maintaining safety through early exits in `BayesianOptimizer._run_scenarios()`.

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

These checks happen in `BayesianOptimizer._run_scenarios()` **before** calling `cost_from_metrics()`, preventing wasted computation on infeasible solutions.

### Finding Appropriate Gain Bounds

Before running Bayesian Optimization, it's crucial to determine reasonable bounds for the gain search space. Using the **cascade mode test** (`run_mode = "cascade"`), you can empirically test different gain combinations to understand stability boundaries and performance characteristics.

#### Recommended Gain Bounds (Empirically Determined)

For the MSD system with `m=20kg`, `c=20 N·s/m`, `k=0 N/m`:

```python
bounds_log10 = {
    # Inner loop (velocity PI): tune first
    'inner': {
        'log10_Kp_v': [-1.0, 4.0],   # Kp_v ∈ [0.1, 10000]
        'log10_Ki_v': [-1.0, 4.0],   # Ki_v ∈ [0.1, 10000]
    },
    # Outer loop (position PI): tune second with fixed inner loop
    'outer': {
        'log10_Kp_x': [-1.0, 3.0],   # Kp_x ∈ [0.1, 1000]
        'log10_Ki_x': [-10.0, 2.0],  # Ki_x ∈ [1e-10, 100]
    },
}
```

#### How to Find These Bounds

**Step 1: Test Individual Loops in Cascade Mode**

Set `run_mode = "cascade"` and configure `TEST_MODE`:

```python
# Test velocity loop (inner loop)
TEST_MODE = "velocity"  # Isolates inner loop

# Test different gain combinations
cascade_pid = CascadePID(
    outer_Kp=50.0, outer_Ki=10.0, outer_Kd=0.0,    # Disabled in velocity mode
    inner_Kp=100.0, inner_Ki=50.0, inner_Kd=0.0,   # Active - test these!
    u_min=-50.0, u_max=50.0,
    velocity_limit=(-2.0, 2.0)
)
```

**Step 2: Vary Gains Systematically**

Test gains across multiple orders of magnitude:
- Very low: `Kp=0.1, Ki=0.1` (too slow)
- Low: `Kp=10, Ki=10` (slow response)
- Moderate: `Kp=100, Ki=50` (good balance)
- High: `Kp=1000, Ki=500` (aggressive)
- Very high: `Kp=10000, Ki=5000` (oscillatory/unstable)

**Step 3: Identify Stability Boundaries**

Run the test and observe:
- **Stable region**: Metrics compute successfully, J is finite
- **Marginally stable**: High oscillations, large overshoot
- **Unstable**: NaN metrics, saturation > 90%, position divergence

**Step 4: Set Conservative Bounds**

- **Lower bound**: Just above where response becomes too slow (high J)
- **Upper bound**: Well below instability threshold (safety margin)

**Example findings for inner loop (velocity)**:
- Stable up to: `Kp ≈ 1000, Ki ≈ 180` (oscillation boundary)
- Instability beyond: `Ki > 79,000` (Nyquist limit with dt=0.05s)
- **Recommended bounds**: `Kp ∈ [0.1, 10000], Ki ∈ [0.1, 10000]`

**Step 5: Repeat for Outer Loop**

```python
# Test position loop (outer loop)
TEST_MODE = "position"  # Full cascade

# With fixed, well-tuned inner loop
cascade_pid = CascadePID(
    outer_Kp=50.0, outer_Ki=10.0, outer_Kd=0.0,    # Test these!
    inner_Kp=100.0, inner_Ki=50.0, inner_Kd=0.0,   # Fixed from Step 1
    ...
)
```

The outer loop is **more sensitive** to instability due to cascade coupling. Use more conservative bounds.

#### Notes on Bound Selection

1. **Logarithmic space**: Search in `log10(gain)` for uniform exploration across orders of magnitude
2. **Safety margins**: Keep upper bounds 2-5× below observed instability
3. **Ki lower bound**: Can be very small (1e-10) to allow pure P control
4. **System-dependent**: These bounds are specific to the MSD parameters used

### Optimization Strategy: Sequential Two-Stage Approach

The current implementation uses a **sequential optimization** strategy rather than joint optimization:

**Stage 1: Inner Loop (Velocity PI)**
- Optimize `Kp_v` and `Ki_v` with outer loop disabled
- Test scenarios: velocity steps, cruise with disturbances
- Fix best inner gains before proceeding

**Stage 2: Outer Loop (Position PI)**  
- Optimize `Kp_x` and `Ki_x` with fixed inner loop
- Test scenarios: position steps, multi-steps
- Inner loop gains remain fixed at Stage 1 optimum

**Benefits of Sequential Approach**:
- ✅ Faster convergence (2D search vs 4D)
- ✅ Better exploration with fewer iterations
- ✅ Follows cascade tuning best practices (inner first, then outer)
- ✅ Easier to interpret results and debug issues

**Note**: A Stage 3 joint optimization (all 4 gains simultaneously) is available but commented out in the code, as sequential optimization typically provides sufficient results with much lower computational cost.

### Convergence Visualization

Bayesian Optimization convergence plots automatically filter extreme cost values for better visualization:

```python
# Y-axis limited to [0, 5] to focus on useful range
# Extreme outliers (J > 5) are clipped but still in data
axs[0].set_ylim([0, 5])  # Inner loop convergence
axs[1].set_ylim([0, 5])  # Outer loop convergence
```

This prevents infeasible candidates with very high costs (J > 100) from dominating the plot scale, making it easier to see the optimization trend in the feasible region.

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
├── ControllerMetrics   # Performance metrics and analysis
└── Helper functions    # Reference signals, plotting, utilities
requirements.txt        # Python dependencies
README.md              # This documentation
```

## Contributing


## License
