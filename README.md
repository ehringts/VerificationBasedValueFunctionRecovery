# RKHS-PI example code

This repository contains example implementations for RKHS policy iteration for nonlinear infinite-horizon optimal-control problems. The code builds value-function surrogates from generalized HJB residual data, compares a standard radial RKHS kernel with a product-kernel variant, and stores the resulting greedy residuals, value-function errors, performance errors, and quadratic-bound diagnostics.

## Organization of the repository

```text
.
├── AcademicToy.py
├── VanDerPol.py
├── CoupledDuffingOscillator4D.py
├── buildBasisForROM.py
├── HeatEquation.py
├── functions/
│   ├── auxFunctions.py
│   ├── kernel.py
│   ├── model.py
│   ├── observer.py
│   └── surrogate.py
└── data/
```

`functions/model.py` contains the mathematical model classes. The examples cover an academic toy problem with known value function, the Van der Pol oscillator, a four-dimensional coupled Duffing oscillator, and a reduced-order nonlinear heat equation.

`functions/kernel.py` contains the standard radial kernels and the product kernels. A standard radial kernel has the form `k(x, y) = phi(||x-y||)`. In the implementation, `phiR(r)` denotes `phi'(r)/r`, and `phiRR(r)` denotes `(phiR)'(r)/r`. The product-kernel class uses `k(x, y) = phi(||x-y||)(y^T x)^case`, where `case = 0` gives the radial kernel and `case > 0` adds a polynomial product factor.

`functions/surrogate.py` stores and fits the RKHS value-function surrogate. `functions/auxFunctions.py` runs gamma search, standard-kernel RKHS-PI, and product-kernel RKHS-PI. `functions/observer.py` saves the error histories and creates the comparison plots.

The `data/` directory stores generated observer files, ROM basis files, and snapshot data.

## Workflows for the executable scripts

### `AcademicToy.py`

The script initializes the academic toy model, generates random value-function test data from the analytic value function, builds a Cartesian greedy/test grid, runs gamma cross-validation, trains the standard RKHS-PI surrogate, trains the product-kernel RKHS-PI surrogate, reloads the observer, and plots the comparison.

### `VanDerPol.py`

The script initializes the controlled Van der Pol oscillator, generates reference value-function data with the open-loop BVP solver, builds the two-dimensional test grid, runs gamma cross-validation, trains the standard RKHS-PI surrogate, trains the product-kernel RKHS-PI surrogate, reloads the observer, and plots the comparison.

### `CoupledDuffingOscillator4D.py`

The script initializes the four-dimensional coupled Duffing oscillator, generates reference value-function data with the open-loop BVP solver, builds the four-dimensional test grid, runs gamma cross-validation, trains the standard RKHS-PI surrogate, trains the product-kernel RKHS-PI surrogate, reloads the observer, and plots the comparison.

### `buildBasisForROM.py`

The script initializes the finite-difference heat-equation model, generates initial data from low Laplacian modes, solves open-loop optimal-control problems to collect state snapshots, computes a POD basis by SVD, and saves the reduced basis to `data/ROMBasis.pkl`.

### `HeatEquation.py`

The script loads `data/ROMBasis.pkl`, initializes the reduced-order heat-equation model, generates reference value-function data with the iterative open-loop solver, builds the reduced test grid, runs gamma cross-validation, trains the standard RKHS-PI surrogate, trains the product-kernel RKHS-PI surrogate, reloads the observer, and plots the ROM comparison.

## Typical run order

```bash
python AcademicToy.py
python VanDerPol.py
python CoupledDuffingOscillator4D.py
python buildBasisForROM.py
python HeatEquation.py
```

For the heat-equation ROM, run `buildBasisForROM.py` before `HeatEquation.py`, because the ROM script expects `data/ROMBasis.pkl` to exist.

## Main routines

- `findGamma()` in each model script runs gamma cross-validation for the kernels used by that example.
- `auxFunctions.RKHSPI(...)` runs policy iteration with the standard radial kernel.
- `auxFunctions.RKHSPIProductKernel(...)` runs policy iteration with the product kernel.
- `Surrogate.doFGreedy(...)` and `SurrogateProductKernel.doFGreedy(...)` select greedy centers by the maximal GHJB residual.
- `Observer.plotObserver(...)` and `Observer.plotObserver2(...)` plot greedy residuals, value errors, performance errors, and quadratic bounds.

## Requirements

The code uses NumPy, SciPy, Matplotlib, and a local `functions` package. The plotting routines use LaTeX rendering through Matplotlib, so a working LaTeX installation is needed for the final PDF plots.
