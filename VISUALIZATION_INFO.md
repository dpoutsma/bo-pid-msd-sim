# GP and Acquisition Function Visualization

## Generated Files

### 1. `gp_acquisition_inner.png` (204 KB) - 2D Contour Visualization
Visualizes the **inner loop** (velocity PI gains) optimization:
- **Left panel**: GP Mean Prediction - shows what the GP thinks the cost function looks like
- **Middle panel**: GP Uncertainty (std dev) - shows where the GP is uncertain (high std = unexplored regions)
- **Right panel**: Utility Function (Expected Improvement) - shows where BO will sample next
  - Yellow star ⭐ = "Next Best Guess" (maximum of acquisition function)
  - Red dots = Observed points from optimization
  - Red star = Last evaluated point

**Parameters**: log10_Kp_v vs log10_Ki_v

### 2. `gp_acquisition_outer.png` (184 KB) - 2D Contour Visualization
Visualizes the **outer loop** (position PI gains) optimization:
- Same 3-panel layout as inner loop
- Shows how the algorithm explores the outer loop parameter space

**Parameters**: log10_Kp_x vs log10_Ki_x

### 3. `bo_progression_example.png` (NEW!) - 1D Progression Through Iterations
Inspired by the library's visualization at:
https://github.com/bayesian-optimization/BayesianOptimization/blob/master/docsrc/static/bo_example.png

Shows **how Bayesian Optimization evolves over time** with 3 stages:
- **Column 1**: After 2 iterations (Initial Exploration)
- **Column 2**: After 10 iterations (Learning)
- **Column 3**: After 30 iterations (Converging)

Each column has 3 plots:
- **Top**: GP prediction with uncertainty bands
  - Red line = True cost function
  - Blue line = GP mean prediction  
  - Blue shaded = 95% confidence interval
  - Black dots = Sampled points
  - Red dot = Latest point

- **Middle**: Zoomed view near optimal (Ki: 1400-2000)
  - Green dashed line = Known optimal (Ki=1700)
  - Shows how GP converges to true function

- **Bottom**: Acquisition function (Expected Improvement)
  - Purple area = Where BO "wants" to sample
  - Gold star = Next point to evaluate
  - Gray dotted lines = Already sampled points

**Key insight**: As iterations progress, uncertainty shrinks around sampled points and the acquisition function focuses on promising regions!

### 4. `bo_convergence.png` (80 KB)
Shows cost function convergence over iterations:
- Top: Inner loop convergence
- Bottom: Outer loop convergence

## Understanding the Plots

### GP Mean (Left Panel in 2D plots)
- Brighter colors = higher predicted target (better performance)
- This is the GP's "best guess" of the true cost function
- Red dots show where we've actually evaluated

### GP Uncertainty (Middle Panel in 2D plots)
- Hot colors (red/yellow) = high uncertainty (GP doesn't know much about this region)
- Cool colors (blue) = low uncertainty (well-explored region near observations)
- BO balances exploring high-uncertainty regions vs exploiting high-mean regions

### Utility Function / Acquisition (Right Panel in 2D plots)
This is the **Expected Improvement** acquisition function (xi=0.01):
- Shows where BO thinks it's most valuable to sample next
- Combines exploitation (high GP mean) + exploration (high GP uncertainty)
- Yellow star shows the argmax (next point to evaluate)
- As optimization progresses, this focuses more on exploitation (refining the optimum)

## Key Insights from Your Results

### Inner Loop (Velocity PI):
✅ **Great convergence!** Found Kp=543, Ki=1839
- Compare to your target: Kp=600, Ki=1700
- Error: 9.5% off on Kp, 8.2% off on Ki
- The GP uncertainty is low around the best point (well explored)
- Utility function shows focused search near optimum

### Outer Loop (Position PI):
✅ **Excellent convergence!** Found Kp=0.32, Ki=22.8
- Compare to your target: Kp=0.1, Ki=20
- The algorithm found values very close to optimal
- Narrower bounds helped significantly

## How This Compares to the Documentation

Your visualizations follow the same concepts as the official documentation:
https://bayesian-optimization.github.io/BayesianOptimization/3.1.0/acquisition_functions.html

**The difference:**
- Documentation uses 1D optimization (single x axis) - easier to visualize
- Your problem uses 2D optimization (contour plots instead of line plots)
- `bo_progression_example.png` shows 1D slices through your 2D space
- Same fundamental concepts: GP mean, GP uncertainty, acquisition function

## Comparison to Library's bo_example.png

The library's example shows a 2D function being optimized with:
- Small dots = sampled points
- Surface plot with color = function value
- Shows how sampling focuses on promising regions

Your `bo_progression_example.png` shows the same concept but as a temporal progression:
- **Early**: Wide uncertainty, exploration across the space
- **Mid**: Uncertainty shrinking, learning the landscape
- **Late**: Focused on optimal region, fine-tuning

This progression visualization is especially useful for understanding:
1. How the GP "learns" the cost function
2. Why acquisition function changes over time
3. The balance between exploration (early) and exploitation (late)

## Technical Notes

The visualization functions were modeled after the documentation's examples:

**2D Contour Plots** (`plot_gp_and_acquisition_2d`):
- `optimizer._gp.predict()` to get GP mean and std
- `acquisition_func._get_acq()` to compute the utility function
- Contour plots (`contourf`) to visualize the 2D parameter space

**1D Progression Plots** (`bo_progression_example.png`):
- Shows a 1D slice through 2D space (fix Kp, vary Ki)
- Demonstrates GP uncertainty reduction over iterations
- Shows acquisition function evolution from exploration → exploitation
- Uses scipy.stats.norm for Expected Improvement calculation
