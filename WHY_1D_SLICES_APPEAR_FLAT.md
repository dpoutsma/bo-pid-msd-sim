# Why Do Some 1D Slices Show Flat Posteriors?

## The Question
You observed that **Kp_x** and **Ki_v** 1D progression plots show relatively flat GP posteriors and near-zero acquisition functions. Why does this happen?

## TL;DR
**1D slices are taken through the 2D optimization landscape by fixing one parameter at its final optimal value.** If the cost function is relatively insensitive to one parameter when the other is at its optimum, that slice will appear flat. This is actually **revealing important information** about the cost landscape!

---

## Understanding 1D Slices Through 2D Space

### What Is a 1D Slice?
A 1D slice is created by:
1. **Fixing one parameter** (e.g., Ki_x = 19, the final optimal value)
2. **Varying the other** (e.g., Kp_x from 0.05 to 0.32)
3. **Predicting** the GP mean and uncertainty along this line

**Visualization:**
```
2D Parameter Space (Kp_x, Ki_x):
     
Ki_x
 40 │                           
    │                           
 30 │        ╱────╲  ← Optimal region
    │      ╱        ╲
 20 │────▓▓▓▓▓▓▓▓▓▓──────  ← 1D slice at Ki_x=19 (optimal)
    │      ╲        ╱
 10 │        ╲────╱
    │
    └──────────────────────── Kp_x
      0.05      0.15     0.32

The slice passes through the optimal region horizontally.
If cost is similar along this slice → FLAT posterior!
```

### Why Kp_x Slice Appears Flat

**Your 1D slice setup:**
- Fix: **Ki_x = 19** (final optimal from BO)
- Vary: **Kp_x** from 0.05 to 0.32

**Possible reasons for flatness:**

#### 1. **Low Sensitivity When Other Parameter Is Optimal**
When Ki_x is at its optimal value (~19), the system might be relatively insensitive to Kp_x variations. This means:
```
Cost(Kp_x=0.1, Ki_x=19) ≈ Cost(Kp_x=0.2, Ki_x=19) ≈ Cost(Kp_x=0.3, Ki_x=19)
```

**Why?** For cascade PI control:
- Ki_x (integral gain) might dominate steady-state error elimination
- Once Ki_x is optimal, Kp_x variations have less impact
- The outer loop might be "robust" to Kp variations when Ki is well-tuned

#### 2. **Ridge or Valley in Cost Function**
The 2D cost landscape might have a **ridge** or **valley** structure:
```
Cost Function Cross-Section:

         Bad
          │  ╱╲    ╱╲
          │ ╱  ╲  ╱  ╲
          │╱    ╲╱    ╲
Good      │──────────────  ← Flat valley bottom (optimal region)
          │
          └─────────────── Kp_x
                ↑
          Your slice here (at optimal Ki_x)
```

If the slice cuts through the **bottom of a valley**, the function will be relatively flat!

#### 3. **Few Observations Near This Slice**
The GP was trained on observations scattered throughout 2D space. If few observations are actually **near** the line Ki_x=19:
- GP has high uncertainty
- GP "smooths out" to a relatively flat prediction
- True function variation is unknown

### Why Ki_v Slice Appears Flat

**Your 1D slice setup:**
- Fix: **Kp_v = 565** (final optimal from BO)  
- Vary: **Ki_v** from 1000 to 2512

**Possible reasons:**

#### 1. **Proportional Gain Dominates at This Operating Point**
When Kp_v is optimal (~565):
- The proportional term provides strong damping
- Variations in Ki_v (integral) have reduced impact
- Cost function becomes relatively insensitive to Ki_v changes

**Physical interpretation:**
```
For inner loop (velocity control):
- Kp_v = 565: Strong proportional feedback
- With good Kp, the system is well-damped
- Ki_v variations affect steady-state error but less so the overall cost
```

#### 2. **Optimal "Ridge" Running Parallel to Ki_v Axis**
```
2D Parameter Space (Kp_v, Ki_v):

Ki_v
2500│        
    │    ══════════════════  ← Optimal ridge (many good solutions)
1700│────▓▓▓▓▓▓▓▓▓▓▓▓▓▓────  ← 1D slice at Kp_v=565
    │    ══════════════════
1000│        
    └─────────────────────── Kp_v
       100    565      1000

The slice runs along/parallel to an optimal ridge!
```

If the optimal region is **elongated** along the Ki_v direction, a horizontal slice will show similar costs throughout.

---

## What Does This Tell Us?

### ✅ **Flat Slices Are INFORMATIVE, Not a Problem!**

A flat 1D slice reveals:

1. **Parameter Interaction:**
   - The two parameters are **coupled**
   - Optimal value of one parameter depends on the other
   - 1D slices don't capture this coupling well

2. **Cost Function Structure:**
   - **Flat valley:** Good! Multiple parameter combinations work well
   - **Robustness:** System is robust to one parameter when other is optimal
   - **Ridge structure:** Optimal region is elongated in parameter space

3. **Why BO Converged:**
   - If cost is similar across a parameter range → EI is low
   - BO correctly identified: "exploring this direction won't improve much"
   - Algorithm is working as intended!

---

## Why 2D Contour Plots Are Better

**2D plots show the FULL landscape:**
```
2D Contour Plot:                1D Slice (misleading):

Ki_x                            Cost
 40 │   ╱──╲                    High│
    │  ╱ ✱  ╲  ← Peak              │
 20 │═▓▓▓▓▓▓═                  Low │══════  ← Looks flat!
    │  ╲    ╱                       │
 10 │   ╲──╱                        └────── Kp_x
    └────────── Kp_x

✱ = optimal point
▓ = optimal region
```

The 1D slice **hides the valley shape** - it only shows one cut through the landscape!

---

## What Can We Do?

### Option 1: Accept That 1D Slices Have Limitations
- Use them to show **EI convergence over iterations** (still valuable!)
- Don't expect them to show full cost landscape
- Rely on 2D contour plots for landscape visualization

### Option 2: Show Multiple Slices at Different Fixed Values
Instead of one slice at optimal, show 3 slices:
```python
# For Kp_x slice, fix Ki_x at three values:
- Ki_x = 10 (lower bound)
- Ki_x = 19 (optimal)
- Ki_x = 40 (upper bound)
```

This would reveal how the landscape changes with the fixed parameter!

### Option 3: Slice Through Sub-Optimal Fixed Values
Fix the other parameter at a **non-optimal** value to show more variation:
```python
# Instead of:
other_param_value = np.log10(best_gains.get('Ki_x', 20))  # Optimal

# Use:
other_param_value = np.mean(bounds[other_param_idx])  # Middle of range
```

This would show more "interesting" slices with higher variation.

### Option 4: Add Diagnostic Text
Add text to plots explaining why slice is flat:
```
"Note: Flat posterior indicates low cost sensitivity to Kp_x 
when Ki_x is at optimal value (19). This suggests the outer 
loop is robust to Kp variations when Ki is well-tuned."
```

---

## Specific Analysis for Your Case

### Kp_x Slice (Outer Loop)
```
Fixed: Ki_x = 19 (optimal for steady-state error)
Vary:  Kp_x ∈ [0.05, 0.32]

Physical meaning:
- Ki_x=19 provides good integral action
- With this Ki, varying Kp from 0.05 to 0.32 might have minor impact
- System might be in "overac actuated" regime where both work

Result: Flat GP posterior along this slice
```

### Ki_v Slice (Inner Loop)
```
Fixed: Kp_v = 565 (optimal for damping)
Vary:  Ki_v ∈ [1000, 2512]

Physical meaning:
- Kp_v=565 provides strong proportional feedback
- With this Kp, the system is well-damped
- Ki_v mainly affects steady-state velocity error
- If steady-state error is small, Ki variations matter less

Result: Flat GP posterior along this slice
```

---

## Mathematical Explanation

### Why EI is Also Flat

If the GP posterior is flat (μ ≈ constant):
```
EI(x) = (μ(x) - y_max - ξ) × Φ(Z) + σ(x) × φ(Z)

If μ(x) ≈ μ_const for all x:
- μ(x) - y_max ≈ constant (negative if found optimum)
- First term ≈ 0 (improvement term is negative)
- Second term depends only on σ(x)
- But σ(x) is also low after many observations

Result: EI ≈ 0 everywhere along slice
```

**This is correct behavior!** The algorithm correctly identifies: "no point along this slice will improve the current best."

---

## Conclusion

**Flat 1D slices are not a bug - they're revealing true properties of your optimization problem:**

1. ✅ **Parameter coupling:** Optimal value of one parameter depends on the other
2. ✅ **Cost structure:** Flat valley/ridge in optimal region  
3. ✅ **Robustness:** System tolerates parameter variations when other is optimal
4. ✅ **BO effectiveness:** Algorithm correctly identified low-value exploration directions

**What you should do:**
- ✅ Keep the 1D plots for **EI convergence visualization** (still very useful!)
- ✅ Rely on **2D contour plots** for cost landscape understanding
- ✅ Add a note explaining that flat slices indicate **low sensitivity** or **parameter coupling**
- ✅ Consider showing the **full 2D evolution** instead (which you're already doing!)

**The 1D slices are still valuable for showing:**
- How GP uncertainty decreases over iterations
- When EI converges to zero
- The trajectory of observations
- How the algorithm focuses on promising regions

They just don't show the full 2D cost landscape - and that's okay! That's what the 2D progression plots are for. 🎯
