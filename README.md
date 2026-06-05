# RKHS-PI example code

This repository contains experiment scripts and helper modules for RKHS-based value-function surrogate construction in nonlinear infinite-horizon optimal control. The main RKHS-PI experiments do not learn a control law directly. Instead, they construct a surrogate for the optimal value function by solving a sequence of generalized Hamilton-Jacobi-Bellman (GHJB) equations with symmetric kernel collocation. The sequence is generated in a policy-iteration manner: for a fixed feedback policy, a linear GHJB equation is collocated in an RKHS; the gradient of the resulting value-function surrogate then induces the next feedback policy.

The repository also contains a constrained-versus-unconstrained experiment. This experiment compares RKHS-PI with a direct constrained RKHS recovery formulation for the value function. The constrained formulation follows the verification-condition viewpoint of the paper **"Recovery of the optimal control value function in reproducing kernel Hilbert spaces from verification conditions"**, arXiv:2512.07477. In that approach, the value function is recovered from HJB verification constraints and inequality constraints by solving a finite-dimensional nonlinear RKHS optimization problem.

## Organization of the repository

```text
.
├── AcademicToy.py
├── VanDerPol.py
├── CoupledDuffingOscillator4D.py
├── buildBasisForROM.py
├── HeatEquation.py
├── ConVsUncon.py
├── functions/
│   ├── auxFunctions.py
│   ├── kernel.py
│   ├── model.py
│   ├── observer.py
│   └── surrogate.py
└── data/
```

`functions/model.py` contains the optimal-control model classes. The examples cover an academic toy problem with an analytic value function, the Van der Pol oscillator, a four-dimensional coupled Duffing oscillator, a reduced-order nonlinear heat equation, and the two-dimensional constrained-versus-unconstrained benchmark `ConUnconExample`.

`functions/kernel.py` contains the standard radial kernels, the product kernels, the PDE Gramian builders for RKHS-PI, and the Hermite Gramian builder used by the constrained recovery experiment. A standard radial kernel has the form `k(x, y) = phi(||x-y||)`. In the implementation, `phi(r)` is the radial basis function, `phiR(r)` denotes `phi'(r)/r`, and `phiRR(r)` denotes `(phiR)'(r)/r`. A product kernel has the form `k(x, y) = phi(||x-y||)(y^T x)^case`; it is used here as an alternative RKHS ansatz for the value-function surrogate, not as a different target quantity. For `ConVsUncon.py`, the Hermite Gramian evaluates value and first-derivative functionals for the product kernel with `case = 2`.

`functions/surrogate.py` stores and fits the RKHS value-function surrogate. The greedy routines select collocation centers by the maximal GHJB residual and solve the symmetric kernel collocation system for the current GHJB equation.

`functions/auxFunctions.py` contains the main RKHS-PI workflow: gamma search, standard-kernel RKHS policy iteration, and product-kernel RKHS policy iteration. In each PI step, the current feedback defines the closed-loop vector field and the right-hand side of the GHJB equation; the surrogate is then recomputed by kernel collocation, and its gradient is used to update the feedback law.

`functions/observer.py` stores the residual, value-error, performance-error, stagnation, and quadratic-bound histories and creates the comparison plots.

The `data/` directory stores generated observer files, ROM basis files, snapshot data, and the optional constrained-experiment sweep file `data/ConUnconRun`.

## Mathematical workflow

For each fixed feedback policy, RKHS-PI constructs a value-function surrogate by approximately solving the associated GHJB equation in an RKHS. The GHJB equation is enforced through symmetric kernel collocation at greedily selected collocation points. After this collocation solve, the feedback law is updated from the gradient of the surrogate value function. Repeating these steps yields the RKHS-PI iteration.

The reference value-function data generated in the model scripts is used for diagnostics and plotting. The RKHS-PI surrogate itself is fitted through the GHJB collocation conditions, not by direct supervised regression on the reference value-function samples.

The two main RKHS-PI training variants are:

```text
standard kernel: solve the GHJB equation with a radial RKHS kernel
product kernel: solve the same GHJB equation with a product RKHS kernel
```

Both variants therefore approximate the value function. The induced feedback laws are obtained afterward from the gradients of the corresponding surrogate value functions.

The constrained experiment in `ConVsUncon.py` follows a different workflow. It directly searches for Hermite data of a value-function candidate by minimizing an RKHS norm subject to HJB verification equalities and a positivity constraint on the value samples. This is compared with unconstrained product-kernel RKHS-PI on the same two-dimensional benchmark. The benchmark has both a true value function and a spurious HJB solution, so the plot compares the relative error to both references.

## Workflows for the executable scripts

### `AcademicToy.py`

This script initializes the academic toy model, generates reference value-function data from the analytic value function, builds a Cartesian test grid for greedy collocation and diagnostics, runs gamma search for the standard and product kernels, trains the standard-kernel RKHS-PI value-function surrogate, trains the product-kernel RKHS-PI value-function surrogate, reloads the observer, and plots the comparison.

### `VanDerPol.py`

This script initializes the controlled Van der Pol oscillator, generates reference value-function data with the open-loop BVP solver, builds the two-dimensional test grid, runs gamma search for the standard and product kernels, trains the standard-kernel RKHS-PI value-function surrogate, trains the product-kernel RKHS-PI value-function surrogate, reloads the observer, and plots the comparison.

### `CoupledDuffingOscillator4D.py`

This script initializes the four-dimensional coupled Duffing oscillator, generates reference value-function data with the open-loop BVP solver, builds the four-dimensional test grid, runs gamma search for the standard and product kernels, trains the standard-kernel RKHS-PI value-function surrogate, trains the product-kernel RKHS-PI value-function surrogate, reloads the observer, and plots the comparison.

### `buildBasisForROM.py`

This script initializes the finite-difference heat-equation model, generates initial states from low Dirichlet-Laplacian modes, solves open-loop optimal-control problems to collect state snapshots, computes a POD basis by SVD, and saves the reduced basis to `data/ROMBasis.pkl`.

### `HeatEquation.py`

This script loads `data/ROMBasis.pkl`, initializes the reduced-order heat-equation model, generates reference value-function data with the iterative open-loop solver, builds the reduced test grid, runs gamma search for the standard and product kernels, trains the standard-kernel RKHS-PI value-function surrogate, trains the product-kernel RKHS-PI value-function surrogate, reloads the observer, and plots the ROM comparison.

### `ConVsUncon.py`

This script initializes the two-dimensional benchmark `ConUnconExample`, builds a Cartesian collocation grid, and compares two ways to approximate the value function. First, `ConRun(zeta)` builds Hermite value/gradient data, forms the Hermite Gram matrix in `functions/kernel.py`, and solves the direct constrained RKHS recovery problem with SLSQP. The constraints enforce the pointwise HJB verification equality and nonnegativity of the value samples. Second, `UnconRun(zeta)` runs product-kernel RKHS-PI on the same grid. The default setting is safe for a fresh repository: if no cached `data/ConUnconRun` file exists, the script runs a one-point demo with `zeta = 0.5` and a smaller grid. To reproduce the full sweep, set `RUN_FULL_SWEEP = True` in `ConVsUncon.py`; then the script sweeps the parameter `zeta`, stores `data/ConUnconRun`, and plots the relative error to the true value function and to the spurious HJB solution.

## Typical run order

```bash
python AcademicToy.py
python VanDerPol.py
python CoupledDuffingOscillator4D.py
python buildBasisForROM.py
python HeatEquation.py
python ConVsUncon.py
```

For the heat-equation ROM, run `buildBasisForROM.py` before `HeatEquation.py`, because the ROM script expects `data/ROMBasis.pkl` to exist. The constrained experiment is independent of the ROM workflow and can be run separately with `python ConVsUncon.py`. For the full constrained sweep, edit `ConVsUncon.py` and set `RUN_FULL_SWEEP = True`; otherwise the script runs the smaller one-point demo when no cache is available.

Data generation, gamma search, SLSQP solves, and RKHS-PI training can be computationally expensive. This is especially relevant for the four-dimensional, ROM heat-equation, and constrained-versus-unconstrained examples. The constrained script limits common BLAS thread counts inside the script to make the SLSQP run more stable on machines where many linear-algebra threads cause overhead.

## Main routines

- `findGamma()` in each model script tests candidate kernel parameters and selects a gamma based on the final greedy GHJB residual for the initial stabilizing policy.
- `auxFunctions.RKHSPI(...)` runs the RKHS policy iteration with the standard radial kernel.
- `auxFunctions.RKHSPIProductKernel(...)` runs the same RKHS policy iteration with the product kernel.
- `Surrogate.doFGreedy(...)` and `SurrogateProductKernel.doFGreedy(...)` select greedy collocation centers by the maximal GHJB residual.
- `ConRun(zeta)` in `ConVsUncon.py` runs the direct constrained Hermite RKHS recovery solve.
- `UnconRun(zeta)` in `ConVsUncon.py` runs the product-kernel RKHS-PI comparison solve.
- `Observer.plotObserver(...)`, `Observer.plotObserver2(...)`, and `plot_convergence_vs_parameter(...)` create the comparison plots.

## Requirements

The code uses NumPy, SciPy, Matplotlib, and a local `functions` package. The plotting routines use LaTeX rendering through Matplotlib, so a working LaTeX installation is needed for the final PDF plots. The optional `threadpoolctl` package is used by `ConVsUncon.py` when available to limit BLAS threads during the SLSQP solve.
