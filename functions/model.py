"""Optimal-control model definitions used by the RKHS-PI example scripts."""
import abc
import itertools
import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as la
from scipy.integrate import solve_bvp, solve_ivp, simpson
from scipy.interpolate import CubicSpline, interp1d

class Model(metaclass=abc.ABCMeta):
    """Abstract base class for nonlinear infinite-horizon optimal-control models."""

    def __init__(self, stateWeight, controlWeight):
        """Initialize the object and precompute the quantities used later."""
        self.stateWeight   = stateWeight
        self.controlWeight = controlWeight

    @abc.abstractmethod
    def f(self, x):
        """Evaluate the uncontrolled drift dynamics."""
        pass

    @abc.abstractmethod
    def g(self, x):
        """Evaluate the control operator."""
        pass

    @abc.abstractmethod
    def jacobi_of_f_transposed_dot_p(self, x, p):
        """Evaluate the product (Df(x))^T p."""
        pass

    def solveOLBVP(self, startState, endT, numbOfEval, initSol=None):
        """Solve the open-loop optimal-control problem as a boundary value problem."""
        n = startState.shape[0]
        print(n)

        def fun(t, y):
            """Evaluate the BVP right-hand side for state, costate, and accumulated value."""
            x  = y[:n, :]
            p  = y[n:2 * n, :]
            u  = self.getMuFromSr(x, p)
            A1 = self.getF(x, u)
            A2 = -self.jacobi_of_f_transposed_dot_p(x, p) - self.gradh(x) - self.jacobi_of_gu_transposed_dot_p(x, u, p)
            A3 = np.atleast_2d(self.getRHS(x, u))
            return np.r_[A1, A2, A3]

        def bc(ya, yb):
            """Evaluate the BVP boundary conditions."""
            return np.r_[ya[0:n] - startState, yb[n:2 * n] - self.gradTerminalCost(yb[0:n]), yb[2 * n]]
        ti    = time.time()
        tSpan = np.linspace(0, endT, numbOfEval)
        if initSol is None:
            initSol = np.zeros((2 * n + 1, tSpan.size))
        if self.fun_jac is None:
            sol = solve_bvp(fun, bc, tSpan, initSol, max_nodes=10000000, tol=1e-08)
        else:
            sol = solve_bvp(fun, bc, tSpan, initSol, fun_jac=self.fun_jac, bc_jac=self.bc_jac, max_nodes=10000000, tol=1e-08)
        print('Solved open-loop control with ' + str(sol.x.shape[0]) + ' mesh points, maximal residual error of ' + str(np.amax(np.abs(sol.rms_residuals))) + '. It took ' + str(time.time() - ti) + ' seconds')
        return (sol.y[0:n, :], sol.y[n:2 * n, :], sol.y[-1, 0] + self.terminalCost(sol.y[0:n, -1]), sol.x[:], sol.sol)

    def solveMPCBVP(self, startState, endT, deltaT, numbOfEval):
        """Run receding-horizon BVP solves and collect the sampled trajectory."""
        state, costate, vf, time, sol = self.solveOLBVP(startState, endT, numbOfEval)
        tol                           = 10 ** (-4)
        stateSample                   = state[:, 0]
        costateSample                 = costate[:, 0]
        vfSample                      = vf
        while np.sum(state[0, :] ** 2) > tol:
            state, costate, vf, time, sol = self.solveOLBVP(sol(deltaT)[:state.shape[0]], endT, numbOfEval, initSol=sol(np.linspace(deltaT, endT + deltaT, numbOfEval)))
            stateSample                   = np.c_[stateSample, state[:, 0]]
            costateSample                 = np.c_[costateSample, costate[:, 0]]
            vfSample                      = np.c_[vfSample, vf]
        return (stateSample, costateSample, vfSample, time)

    def solveMPCIterativ(self, startState, endT, deltaT, numbOfEval):
        """Run receding-horizon iterative open-loop solves and collect the sampled trajectory."""
        state, costate, vf, time = self.solveOLIterativ(startState, endT, numbOfEval)
        tol                      = 10 ** (-4)
        stateSample              = state[:, 0]
        costateSample            = costate[:, 0]
        vfSample                 = vf
        while np.sum(state[0, :] ** 2) > tol:
            initSolCont              = interp1d(time, state, kind='cubic', fill_value='extrapolate')
            state, costate, vf, time = self.solveOLIterativ(initSolCont(deltaT), endT, numbOfEval)
            stateSample              = np.c_[stateSample, state[:, 0]]
            costateSample            = np.c_[costateSample, costate[:, 0]]
            vfSample                 = np.c_[vfSample, vf]
        return (stateSample, costateSample, vfSample, time)

    def solveOLIterativ(self, startState, endT, numbOfEval, ivpSolver='BDF'):
        """Solve the open-loop problem with forward-backward gradient iterations."""
        startState = np.asarray(startState, dtype=float)
        eps        = 1e-07
        pres       = 1e-06
        maxIter    = 100
        t_grid     = np.linspace(0.0, float(endT), int(numbOfEval))
        N          = t_grid.size
        m          = self.B.shape[1] if hasattr(self, 'B') else self.g(startState).shape[1]

        def make_spline(values):
            """Create a cubic spline on the fixed time grid."""
            return CubicSpline(t_grid, values, axis=1, extrapolate=True)
        u = np.zeros((m, N), dtype=float)
        if hasattr(self, 'matrixKGain') and hasattr(self, 'B'):
            P = np.asarray(self.matrixKGain, dtype=float)
            B = np.asarray(self.g(startState), dtype=float)
            R = self.controlWeight * np.eye(B.shape[1], dtype=float)
            K = np.linalg.solve(R, B.T @ P)

            def rhs_closed_loop(t, x):
                """Evaluate the Riccati-feedback closed-loop warm-start dynamics."""
                u_fb = -K @ x
                return self.f(x) + self.g(x) @ u_fb
            sol0    = solve_ivp(rhs_closed_loop, (0.0, endT), startState, t_eval=t_grid, method=ivpSolver, rtol=pres, atol=pres)
            x0_grid = sol0.y
            u       = -(K @ x0_grid)

        def forward(u_spline, rtol):
            """Integrate the state and running cost forward for one control spline."""
            y0 = np.concatenate([startState, [0.0]])

            def rhs(t, y):
                """Evaluate the local ODE right-hand side."""
                x  = y[:-1]
                ut = u_spline(t)
                dx = self.f(x) + self.g(x) @ ut
                dJ = float(self.h(x) + self.controlWeight * np.dot(ut, ut))
                return np.concatenate([dx, [dJ]])
            sol    = solve_ivp(rhs, (0.0, endT), y0, method=ivpSolver, t_eval=t_grid, rtol=rtol, atol=rtol)
            x_grid = sol.y[:-1, :]
            J_int  = sol.y[-1, -1]
            fe     = J_int + self.terminalCost(x_grid[:, -1])
            return (x_grid, fe)

        def backward(x_spline, u_grid, rtol):
            """Integrate the costate backward and compute the control gradient."""
            pend = np.asarray(self.gradTerminalCost(x_spline(endT)), dtype=float)

            def rhs(t, p):
                """Evaluate the local ODE right-hand side."""
                x = x_spline(t)
                return -self.jacobi_of_f_transposed_dot_p(x, p) - self.gradh(x)
            sol    = solve_ivp(rhs, (endT, 0.0), pend, method=ivpSolver, t_eval=t_grid[::-1], rtol=rtol, atol=rtol)
            p_grid = sol.y[:, ::-1]
            gt_p   = np.empty_like(u_grid)
            for k in range(N):
                xk         = x_spline(t_grid[k])
                pk         = p_grid[:, k]
                gt_p[:, k] = self.g(xk).T @ pk
            grad_grid = 2.0 * self.controlWeight * u_grid + gt_p
            dir_grid  = -grad_grid
            normGrad  = np.sqrt(simpson(np.sum(grad_grid * grad_grid, axis=0), t_grid))
            return (dir_grid, grad_grid, normGrad, p_grid)
        u_spline                          = make_spline(u)
        x_grid, fe                        = forward(u_spline, pres)
        x_spline                          = make_spline(x_grid)
        d_old, grad_old, normGrad, p_grid = backward(x_spline, u, pres)
        print(f'iter -2, fe = {fe}, normGrad = {normGrad}')
        u_old                     = u.copy()
        u                         = u_old - 1.0 / max(normGrad, 1e-16) * grad_old
        u_spline                  = make_spline(u)
        x_grid, fe                = forward(u_spline, pres)
        x_spline                  = make_spline(x_grid)
        d, grad, normGrad, p_grid = backward(x_spline, u, pres)
        print(f'iter -1, fe = {fe}, normGrad = {normGrad}')
        j = 0
        while normGrad > eps and j < maxIter:
            s     = u - u_old
            y     = d - d_old
            denom = simpson(np.sum(s * y, axis=0), t_grid)
            num   = simpson(np.sum(y * y, axis=0), t_grid)
            if abs(denom) < 1e-16:
                alpha = -max(normGrad, 1.0)
            else:
                alpha = num / denom
                alpha = min(alpha, -alpha)
            u_old = u
            d_old = d
            u     = u - 1.0 / alpha * d
            if normGrad < 10:
                pres = 1e-06
            if normGrad < 1:
                pres = 1e-06
            if normGrad < 0.1:
                pres = 1e-07
            if normGrad < 0.01:
                pres = 1e-08
            if normGrad < 0.001:
                pres = 1e-09
            u_spline                  = make_spline(u)
            x_grid, fe                = forward(u_spline, pres)
            x_spline                  = make_spline(x_grid)
            d, grad, normGrad, p_grid = backward(x_spline, u, pres)
            print(f'iter {j}, fe = {fe}, normGrad = {normGrad}, alpha = {alpha} , HJB = {self.HJB(x_grid[:, 0], p_grid[:, 0])} ')
            j += 1
        return (x_grid, p_grid, fe, t_grid)

    def solveSurrogate(self, startState, endT, surrogateEval):
        """Simulate the closed loop induced by a surrogate gradient and return the trajectory and cost."""
        x0    = np.asarray(startState)
        surr0 = surrogateEval(np.zeros_like(x0))

        def control(x):
            """Evaluate the feedback control induced by the surrogate gradient."""
            return self.getMuFromSr(x, surrogateEval(x) - surr0)

        def rhs(t, y_aug):
            """Evaluate the local ODE right-hand side."""
            x  = y_aug[:-1]
            J  = y_aug[-1]
            u  = control(x)
            dx = self.getF(x, u)
            dJ = self.getRHS(x, u) * -1
            return np.concatenate([dx, [dJ]])
        y0   = np.concatenate([x0, [0.0]])
        sol  = solve_ivp(rhs, (0.0, float(endT)), y0, method='DOP853', rtol=1e-08, atol=1e-08)
        xT   = sol.y[:-1, -1]
        cost = sol.y[-1, -1] + self.terminalCost(xT)
        return (sol.y[:-1], sol.t, cost)

class HEDirichlet1DFD(Model):
    """Finite-difference nonlinear heat equation with distributed controls and Dirichlet boundary conditions."""

    def __init__(self, NumbNodes, para):
        """Initialize the object and precompute the quantities used later."""
        self.NumbNodes     = NumbNodes
        self.alpha         = para[0]
        self.beta          = para[1]
        self.stateWeight   = para[2]
        self.controlWeight = para[3]
        self.controlDim    = 4
        self.makeFDSystem(NumbNodes)
        self.stateWeight = self.stateWeight * self.dx
        print(self.dx)
        if self.stateWeight < 0:
            raise ValueError('stateWeight must be nonnegative.')
        if self.controlWeight <= 0:
            raise ValueError('controlWeight must be positive.')
        Q                  = self.stateWeight * np.eye(self.A.shape[0])
        R                  = self.controlWeight * np.eye(self.B.shape[1])
        self.matrixKGain   = la.solve_continuous_are(self.alpha * self.A, self.B, Q, R, e=None, s=None, balanced=True)
        self.matrixKGain   = 0.5 * (self.matrixKGain + self.matrixKGain.T)
        eigs               = np.linalg.eigvalsh(self.matrixKGain)
        self.lambdaMin     = eigs[0]
        self.lambdaMax     = eigs[-1]
        self.fun_jac       = self.jacobi_of_f
        self.getMuFromSr   = lambda x, srX: -0.5 / self.controlWeight * (self.B.T @ srX)
        self.getF          = lambda x, muX: self.f(x) + self.B @ muX
        self.getRHS        = lambda x, muX: -self.stateWeight * np.sum(x ** 2, axis=0) - self.controlWeight * np.sum(muX ** 2, axis=0)
        self.HJB           = lambda x, p: self.stateWeight * np.sum(x ** 2, axis=0) - 0.25 * np.sum((self.B.T @ p) ** 2, axis=0) / self.controlWeight + np.sum(self.f(x) * p, axis=0)
        P0                 = np.eye(self.A.shape[0]) * (self.lambdaMin + self.lambdaMax) / 2
        self.V0            = lambda x: np.sum(x * (P0 @ x), axis=0)
        self.gradV_0       = lambda x: 2 * (P0 @ x)
        self.stableControl = lambda x: self.getMuFromSr(x, self.gradV_0(x))

    def h(self, x):
        """Evaluate the state running cost."""
        return self.stateWeight * np.sum(x ** 2, axis=0)

    def gradh(self, x):
        """Evaluate the gradient of the state running cost."""
        return 2 * self.stateWeight * x

    def terminalCost(self, x):
        """Evaluate the terminal LQR cost."""
        return np.sum(x * (self.matrixKGain @ x), axis=0)

    def gradTerminalCost(self, x):
        """Evaluate the gradient of the terminal LQR cost."""
        return 2 * self.matrixKGain @ x

    def f(self, x):
        """Evaluate the uncontrolled drift dynamics."""
        return self.alpha * (self.A @ x) + self.beta * (x ** 2 - x ** 3)

    def g(self, x):
        """Evaluate the control operator."""
        return self.B

    def jacobi_of_f(self, x):
        """Evaluate the Jacobian of the drift dynamics."""
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError('jacobi_of_f expects x to have shape (N,).')
        return self.alpha * self.A + self.beta * np.diag(2 * x - 3 * x ** 2)

    def jacobi_of_f_transposed_dot_p(self, x, p):
        """Evaluate the product (Df(x))^T p."""
        return self.alpha * self.A.T @ p + 2 * self.beta * (p * x) - 3 * self.beta * (p * x ** 2)

    def jacobi_of_gu_transposed_dot_p(self, x, u, p):
        """Evaluate the product (D_x(g(x)u))^T p."""
        return np.zeros_like(x)

    def makeFDSystem(self, NumbNodes):
        """Build the one-dimensional finite-difference Laplacian and actuator matrix."""
        dx        = 1.0 / (NumbNodes + 1)
        center    = np.linspace(dx, 1.0 - dx, NumbNodes)
        main_diag = -2.0 * np.ones(NumbNodes) / dx ** 2
        off_diag  = 1.0 * np.ones(NumbNodes - 1) / dx ** 2
        self.A    = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        self.B    = np.zeros((NumbNodes, 4))
        tol       = 1e-12
        self.B[(center >= 0.1 - tol) & (center <= 0.2 + tol), 0] = 1.0
        self.B[(center >= 0.3 - tol) & (center <= 0.4 + tol), 1] = 1.0
        self.B[(center >= 0.6 - tol) & (center <= 0.7 + tol), 2] = 1.0
        self.B[(center >= 0.8 - tol) & (center <= 0.9 + tol), 3] = 1.0
        self.dx     = dx
        self.center = center
        if np.any(np.linalg.norm(self.B, axis=0) == 0):
            raise ValueError('At least one control region contains no grid points. Increase NumbNodes or change the actuator intervals.')

    def plotSolution(self, solY):
        """Animate the finite-difference solution in physical space."""
        dx      = 1.0 / (self.NumbNodes + 1)
        center  = np.linspace(dx, 1.0 - dx, self.NumbNodes)
        x_plot  = np.concatenate(([0.0], center, [1.0]))
        y0      = np.concatenate(([0.0], solY[:, 0], [0.0]))
        fig, ax = plt.subplots()
        line, = ax.plot(x_plot, y0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-1.0, 1.0)

        def update_currentPlot(num):
            """Update the animated finite-difference solution at one frame."""
            y_plot = np.concatenate(([0.0], solY[:, num], [0.0]))
            line.set_ydata(y_plot)
            ax.set_title(f'Time step {num}')
            return (line,)
        line_ani = animation.FuncAnimation(fig, update_currentPlot, frames=solY.shape[1], interval=1, repeat=False)
        plt.show()
        return line_ani

    def makeInitialStatesFromLaplacianModes(self, numModes, coefficientGrids):
        """Generate initial states from low Dirichlet-Laplacian modes."""
        dx        = 1.0 / (self.NumbNodes + 1)
        x         = np.linspace(dx, 1.0 - dx, self.NumbNodes)
        k         = np.arange(1, numModes + 1)
        Phi        = np.sqrt(2.0) * np.sin(np.pi * np.outer(x, k))
        coeffArray = np.asarray(coefficientGrids)
        if coeffArray.ndim == 2 and coeffArray.shape[1] == numModes:
            A0 = coeffArray.T
        else:
            coeffList = list(itertools.product(*coefficientGrids))
            A0        = np.array(coeffList).T
        X0        = Phi @ A0
        mask      = np.ones(X0.shape[1], dtype=bool)
        X0        = X0[:, mask]
        A0        = A0[:, mask]
        return (X0, A0, Phi)

class VanDerPol(Model):
    """Controlled Van der Pol oscillator model."""

    def __init__(self, stateWeight, controlWeight):
        """Initialize the object and precompute the quantities used later."""
        self.stateWeight   = stateWeight
        self.controlWeight = controlWeight
        self.matrixKGain   = la.solve_continuous_are(np.c_[np.r_[0, 1], np.r_[1, 1]], np.atleast_2d(np.array([0, 1])).T, stateWeight * np.eye(2), controlWeight, e=None, s=None, balanced=True)
        self.lambdaMin     = np.linalg.eigvals(self.matrixKGain).min().real
        self.lambdaMax     = np.linalg.eigvals(self.matrixKGain).max().real
        self.stableControl = lambda x: np.atleast_2d(-(1 / self.controlWeight) * np.sum(self.g(x) * (self.matrixKGain @ x), 0))
        self.getMuFromSr   = lambda x, srX: -0.5 / self.controlWeight * (self.g(x).T @ srX)
        self.getF          = lambda x, muX: self.f(x) + self.g(x) @ muX
        self.getRHS        = lambda x, muX: -self.stateWeight * np.sum(x ** 2, 0) - self.controlWeight * np.sum(muX ** 2, 0)
        self.controlDim    = 2
        self.HJB           = lambda x, muX: self.stateWeight * np.sum(x ** 2, 0) - 1 / 4 * (self.g(x).T @ muX) ** 2 / self.controlWeight + self.f(x).T @ muX
        P0                 = np.eye(2) * self.lambdaMin
        self.V0            = lambda z: np.sum(z * (P0 @ z), axis=0)
        self.gradV_0       = lambda z: 2 * (P0 @ z)

    def h(self, x):
        """Evaluate the state running cost."""
        return self.stateWeight * np.sum(x ** 2, 0)

    def gradh(self, x):
        """Evaluate the gradient of the state running cost."""
        return 2 * self.stateWeight * x

    def terminalCost(self, x):
        """Evaluate the terminal LQR cost."""
        return x.T @ self.matrixKGain @ x

    def gradTerminalCost(self, x):
        """Evaluate the gradient of the terminal LQR cost."""
        return 2 * self.matrixKGain @ x

    def f(self, x):
        """Evaluate the uncontrolled drift dynamics."""
        return np.array([x[1], -x[0] + x[1] * (1 - x[0] ** 2)])

    def g(self, x):
        """Evaluate the control operator."""
        return np.array([[0, 1]]).T

    def jacobi_of_f_transposed_dot_p(self, x, p):
        """Evaluate the product (Df(x))^T p."""
        return np.array([-p[1] - 2 * p[1] * x[0] * x[1], p[0] + p[1] * (1 - x[0] ** 2)])

    def jacobi_of_gu_transposed_dot_p(self, x, u, p):
        """Evaluate the product (D_x(g(x)u))^T p."""
        return 0.0 * p

    def fun_jac(self, t, y):
        """Evaluate the Jacobian of the BVP right-hand side."""
        x1         = y[0, :]
        x2         = y[1, :]
        p2         = y[3, :]
        Q          = self.stateWeight
        R          = self.controlWeight
        m          = t.size
        J          = np.zeros((5, 5, m))
        J[0, 1, :] = 1.0
        J[1, 0, :] = -1.0 - 2.0 * x1 * x2
        J[1, 1, :] = 1.0 - x1 ** 2
        J[1, 3, :] = -1.0 / (2.0 * R)
        J[2, 0, :] = 2.0 * x2 * p2 - 2.0 * Q
        J[2, 1, :] = 2.0 * x1 * p2
        J[2, 3, :] = 1.0 + 2.0 * x1 * x2
        J[3, 0, :] = 2.0 * x1 * p2
        J[3, 1, :] = -2.0 * Q
        J[3, 2, :] = -1.0
        J[3, 3, :] = -1.0 + x1 ** 2
        J[4, 0, :] = -2.0 * Q * x1
        J[4, 1, :] = -2.0 * Q * x2
        J[4, 3, :] = -p2 / (2.0 * R)
        return J

    def bc_jac(self, ya, yb):
        """Evaluate the Jacobian of the BVP boundary conditions."""
        K               = self.matrixKGain
        dbc_dya         = np.zeros((5, 5))
        dbc_dyb         = np.zeros((5, 5))
        dbc_dya[0, 0]   = 1.0
        dbc_dya[1, 1]   = 1.0
        dbc_dyb[2, 0:2] = -2.0 * K[0, :]
        dbc_dyb[3, 0:2] = -2.0 * K[1, :]
        dbc_dyb[2, 2]   = 1.0
        dbc_dyb[3, 3]   = 1.0
        dbc_dyb[4, 4]   = 1.0
        return (dbc_dya, dbc_dyb)

class AcademicToyModel(Model):
    """Academic toy model with analytic value function used for verification."""

    def __init__(self, N: int, alpha: float, beta: float):
        """Initialize the object and precompute the quantities used later."""
        self.N             = int(N)
        self.alpha         = float(alpha)
        self.beta          = float(beta)
        self.controlWeight = float(beta)
        self.stateDim      = self.N
        self.controlDim    = self.N
        if self.beta <= 0:
            raise ValueError('beta must be > 0 (R = beta I).')
        self._s            = lambda x: np.exp(-np.sum(np.asarray(x) ** 2, axis=0))
        A                  = 0.25 * np.eye(self.N)
        B                  = np.eye(self.N)
        Q                  = self.alpha * np.eye(self.N)
        R                  = self.beta * np.eye(self.N)
        self.matrixKGain   = la.solve_continuous_are(A, B, Q, R, balanced=True)
        self.trueVF        = lambda x: np.exp(np.sum(x ** 2, 0)) - 1
        eigs               = np.linalg.eigvals(self.matrixKGain).real
        self.lambdaMin     = eigs.min()
        self.lambdaMax     = eigs.max()
        P0                 = np.eye(4) * 2
        self.V0            = lambda z: np.sum(z * (P0 @ z), axis=0)
        self.gradV_0       = lambda z: 2 * (P0 @ z)
        self.stableControl = lambda z: self.getMuFromSr(z, self.gradV_0(z))
        self.getMuFromSr   = lambda x, p: -(self._s(x) / (2.0 * self.beta)) * p
        self.getF          = lambda x, u: self.f(x) + self._s(x) * u
        self.getRHS        = lambda x, u: -self.alpha * np.sum(x ** 2, axis=0) - self.beta * np.sum(u ** 2, axis=0)
        self.HJB           = lambda x, p: self.alpha * np.sum(x ** 2, axis=0) - self._s(x) ** 2 * np.sum(p ** 2, axis=0) / (4.0 * self.beta) + np.sum(self.f(x) * p, axis=0)
        self.fun_jac       = None
        self.bc_jac        = None

    def h(self, x):
        """Evaluate the state running cost."""
        return self.alpha * np.sum(x ** 2, axis=0)

    def gradh(self, x):
        """Evaluate the gradient of the state running cost."""
        return 2.0 * self.alpha * x

    def terminalCost(self, x):
        """Evaluate the terminal LQR cost."""
        x = np.asarray(x)
        if x.ndim == 1:
            return float(x.T @ self.matrixKGain @ x)
        return np.sum(x * (self.matrixKGain @ x), axis=0)

    def gradTerminalCost(self, x):
        """Evaluate the gradient of the terminal LQR cost."""
        return 2.0 * (self.matrixKGain @ x)

    def f(self, x):
        """Evaluate the uncontrolled drift dynamics."""
        return 0.25 * self._s(x) * x

    def g(self, x):
        """Evaluate the control operator."""
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError('g(x) expects a single x with shape (N,). For batched x use self._s(x).')
        return self._s(x) * np.eye(self.N)

    def jacobi_of_f_transposed_dot_p(self, x, p):
        """Evaluate the product (Df(x))^T p."""
        x        = np.asarray(x)
        p        = np.asarray(p)
        s        = self._s(x)
        alpha_xp = np.sum(x * p, axis=0)
        return s / 4.0 * p - s / 2.0 * x * alpha_xp

    def jacobi_of_gu_transposed_dot_p(self, x, u, p):
        """Evaluate the product (D_x(g(x)u))^T p."""
        s   = self._s(x)
        utp = np.sum(u * p, axis=0)
        return -2.0 * s * x * utp

class HEDirichlet1DROM(Model):
    """POD reduced-order nonlinear heat equation model."""

    def __init__(self, NumbNodes, para, PhiPOD):
        """Initialize the object and precompute the quantities used later."""
        self.NumbNodes     = NumbNodes
        self.alpha         = para[0]
        self.beta          = para[1]
        self.stateWeight   = para[2]
        self.controlWeight = para[3]
        self.controlDim    = 4
        self.makeFDSystem(NumbNodes)
        self.stateWeight = self.stateWeight * self.dx
        self.Phi         = np.asarray(PhiPOD)
        self.r           = self.Phi.shape[1]
        if self.Phi.shape[0] != NumbNodes:
            raise ValueError('PhiPOD must have shape (NumbNodes, r).')
        if self.stateWeight < 0:
            raise ValueError('stateWeight must be nonnegative.')
        if self.controlWeight <= 0:
            raise ValueError('controlWeight must be positive.')
        self.Gram          = self.Phi.T @ self.Phi
        self.GramInv       = la.inv(self.Gram)
        self.A_full        = self.A
        self.B_full        = self.B
        self.A             = self.GramInv @ self.Phi.T @ self.A_full @ self.Phi
        self.B             = self.GramInv @ self.Phi.T @ self.B_full
        self.Q             = self.stateWeight * self.Gram
        self.R             = self.controlWeight * np.eye(self.controlDim)
        self.matrixKGain   = la.solve_continuous_are(self.alpha * self.A, self.B, self.Q, self.R, e=None, s=None, balanced=True)
        self.fun_jac       = self.jacobi_of_f
        self.getMuFromSr   = lambda z, srZ: -0.5 / self.controlWeight * (self.B.T @ srZ)
        self.getF          = lambda z, muZ: self.f(z) + self.B @ muZ
        self.getRHS        = lambda z, muZ: -self.h(z) - self.controlWeight * np.sum(muZ ** 2, axis=0)
        self.HJB           = lambda z, p: self.h(z) - 0.25 * np.sum((self.B.T @ p) ** 2, axis=0) / self.controlWeight + np.sum(self.f(z) * p, axis=0)
        self.matrixKGain   = 0.5 * (self.matrixKGain + self.matrixKGain.T)
        eigs               = np.linalg.eigvalsh(self.matrixKGain)
        self.lambdaMin     = eigs[0]
        self.lambdaMax     = eigs[-1]
        P0                 = np.eye(self.r) * self.lambdaMin
        self.V0            = lambda z: np.sum(z * (P0 @ z), axis=0)
        self.gradV_0       = lambda z: 2 * (P0 @ z)
        self.stableControl = lambda z: self.getMuFromSr(z, self.gradV_0(z))

    def lift(self, z):
        """Map reduced coordinates to the full finite-difference state."""
        return self.Phi @ z

    def project(self, x):
        """Project a full finite-difference state to reduced coordinates."""
        return self.GramInv @ self.Phi.T @ x

    def h(self, z):
        """Evaluate the state running cost."""
        return np.sum(z * (self.Q @ z), axis=0)

    def gradh(self, z):
        """Evaluate the gradient of the state running cost."""
        return 2 * (self.Q @ z)

    def terminalCost(self, z):
        """Evaluate the terminal LQR cost."""
        return np.sum(z * (self.matrixKGain @ z), axis=0)

    def gradTerminalCost(self, z):
        """Evaluate the gradient of the terminal LQR cost."""
        return 2 * self.matrixKGain @ z

    def f(self, z):
        """Evaluate the uncontrolled drift dynamics."""
        x        = self.lift(z)
        reaction = x ** 2 - x ** 3
        return self.alpha * (self.A @ z) + self.beta * self.project(reaction)

    def g(self, z):
        """Evaluate the control operator."""
        return self.B

    def jacobi_of_f(self, z):
        """Evaluate the Jacobian of the drift dynamics."""
        z = np.asarray(z)
        if z.ndim != 1:
            raise ValueError('jacobi_of_f expects z to have shape (r,).')
        x             = self.lift(z)
        diag_reaction = 2 * x - 3 * x ** 2
        J_reaction    = self.GramInv @ self.Phi.T @ (diag_reaction[:, None] * self.Phi)
        return self.alpha * self.A + self.beta * J_reaction

    def jacobi_of_f_transposed_dot_p(self, z, p):
        """Evaluate the product (Df(x))^T p."""
        return self.jacobi_of_f(z).T @ p

    def jacobi_of_gu_transposed_dot_p(self, z, u, p):
        """Evaluate the product (D_x(g(x)u))^T p."""
        return np.zeros_like(z)

    def makeFDSystem(self, NumbNodes):
        """Build the one-dimensional finite-difference Laplacian and actuator matrix."""
        dx        = 1.0 / (NumbNodes + 1)
        center    = np.linspace(dx, 1.0 - dx, NumbNodes)
        main_diag = -2.0 * np.ones(NumbNodes) / dx ** 2
        off_diag  = 1.0 * np.ones(NumbNodes - 1) / dx ** 2
        self.A    = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        self.B    = np.zeros((NumbNodes, 4))
        tol       = 1e-12
        self.B[(center >= 0.1 - tol) & (center <= 0.2 + tol), 0] = 1.0
        self.B[(center >= 0.3 - tol) & (center <= 0.4 + tol), 1] = 1.0
        self.B[(center >= 0.6 - tol) & (center <= 0.7 + tol), 2] = 1.0
        self.B[(center >= 0.8 - tol) & (center <= 0.9 + tol), 3] = 1.0
        self.dx     = dx
        self.center = center
        if np.any(np.linalg.norm(self.B, axis=0) == 0):
            raise ValueError('At least one control region contains no grid points. Increase NumbNodes or change the actuator intervals.')

class CoupledDuffingOscillator4D(Model):
    """Four-dimensional controlled coupled Duffing oscillator."""

    def __init__(self, k1=1.0, k2=1.4, kc=0.35, d1=0.15, d2=0.25, lam=0.2, lamc=0.05, controlWeight=0.05):
        """Initialize the object and precompute the quantities used later."""
        self.k1            = float(k1)
        self.k2            = float(k2)
        self.kc            = float(kc)
        self.d1            = float(d1)
        self.d2            = float(d2)
        self.lam           = float(lam)
        self.lamc          = float(lamc)
        self.stateDim      = 4
        self.controlDim    = 2
        self.stateWeight   = 1.0
        self.controlWeight = float(controlWeight)
        if self.controlWeight <= 0:
            raise ValueError('controlWeight must be positive.')
        self.D             = np.diag([self.d1, self.d2])
        self.K0            = np.array([[self.k1 + self.kc, -self.kc], [-self.kc, self.k2 + self.kc]])
        self.B             = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        self.A             = np.block([[np.zeros((2, 2)), np.eye(2)], [-self.K0, -self.D]])
        H                  = np.block([[self.K0, np.zeros((2, 2))], [np.zeros((2, 2)), np.eye(2)]])
        self.Q             = 0.5 * H
        R                  = self.controlWeight * np.eye(self.controlDim)
        self.matrixKGain   = la.solve_continuous_are(self.A, self.B, self.Q, R)
        self.fun_jac       = None
        self.bc_jac        = None
        eigs               = np.linalg.eigvalsh(self.matrixKGain)
        self.lambdaMin     = eigs[0]
        self.lambdaMax     = eigs[-1]
        P0                 = np.eye(4) * self.lambdaMin
        self.V0            = lambda z: np.sum(z * (P0 @ z), axis=0)
        self.gradV_0       = lambda z: 2 * (P0 @ z)
        self.stableControl = lambda z: self.getMuFromSr(z, self.gradV_0(z))

    def _as2d(self, x):
        """Convert a state input to a two-dimensional batch representation."""
        x            = np.asarray(x, dtype=float)
        vector_input = False
        if x.ndim == 1:
            x            = x.reshape(-1, 1)
            vector_input = True
        if x.shape[0] != 4:
            raise ValueError(f'Expected state dimension 4, got shape {x.shape}.')
        return (x, vector_input)

    def _potential(self, q):
        """Evaluate the coupled Duffing potential energy."""
        q1 = q[0]
        q2 = q[1]
        z  = q1 - q2
        return 0.5 * self.k1 * q1 ** 2 + 0.5 * self.k2 * q2 ** 2 + 0.5 * self.kc * z ** 2 + 0.25 * self.lam * (q1 ** 4 + q2 ** 4) + 0.25 * self.lamc * z ** 4

    def _grad_potential(self, q):
        """Evaluate the gradient of the coupled Duffing potential."""
        q1    = q[0]
        q2    = q[1]
        z     = q1 - q2
        dU_q1 = self.k1 * q1 + self.kc * z + self.lam * q1 ** 3 + self.lamc * z ** 3
        dU_q2 = self.k2 * q2 - self.kc * z + self.lam * q2 ** 3 - self.lamc * z ** 3
        return np.vstack([dU_q1, dU_q2])

    def _hessian_potential_entries(self, q):
        """Evaluate the Hessian entries of the coupled Duffing potential."""
        q1     = q[0]
        q2     = q[1]
        z      = q1 - q2
        common = 3.0 * self.lamc * z ** 2
        H11    = self.k1 + self.kc + 3.0 * self.lam * q1 ** 2 + common
        H22    = self.k2 + self.kc + 3.0 * self.lam * q2 ** 2 + common
        H12    = -self.kc - common
        return (H11, H12, H22)

    def f(self, x):
        """Evaluate the uncontrolled drift dynamics."""
        x, vector_input = self._as2d(x)
        q               = x[:2, :]
        v               = x[2:, :]
        gradU           = self._grad_potential(q)
        out             = np.vstack([v, -self.D @ v - gradU])
        if vector_input:
            return out[:, 0]
        return out

    def g(self, x):
        """Evaluate the control operator."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return self.B
        if x.ndim == 2:
            return np.broadcast_to(self.B[:, :, None], (self.stateDim, self.controlDim, x.shape[1]))
        raise ValueError(f'g(x) expects shape (4,) or (4,M), got {x.shape}.')

    def h(self, x):
        """Evaluate the state running cost."""
        x, vector_input = self._as2d(x)
        q               = x[:2, :]
        v               = x[2:, :]
        val             = self._potential(q) + 0.5 * np.sum(v * v, axis=0)
        if vector_input:
            return float(val[0])
        return val

    def gradh(self, x):
        """Evaluate the gradient of the state running cost."""
        x, vector_input = self._as2d(x)
        q               = x[:2, :]
        v               = x[2:, :]
        out             = np.vstack([self._grad_potential(q), v])
        if vector_input:
            return out[:, 0]
        return out

    def getMuFromSr(self, x, p):
        """Evaluate the feedback control induced by a surrogate gradient."""
        p = np.asarray(p, dtype=float)
        if p.ndim == 1:
            return -0.5 / self.controlWeight * p[2:4]
        if p.ndim == 2:
            return -0.5 / self.controlWeight * p[2:4, :]
        raise ValueError(f'p must have shape (4,) or (4,M), got {p.shape}.')

    def getF(self, x, muX):
        """Evaluate the closed-loop dynamics for a given control."""
        x, vector_input = self._as2d(x)
        out             = self.f(x)
        out             = np.asarray(out, dtype=float)
        if out.ndim == 1:
            out = out.reshape(4, 1)
        u = np.asarray(muX, dtype=float)
        if u.ndim == 1:
            u = u.reshape(2, 1)
        if u.ndim != 2 or u.shape[0] != 2:
            raise ValueError(f'Control must have shape (2,) or (2,M), got {u.shape}.')
        if u.shape[1] == 1 and x.shape[1] > 1:
            u = np.broadcast_to(u, (2, x.shape[1]))
        out[2:4, :] += u
        if vector_input:
            return out[:, 0]
        return out

    def getRHS(self, x, muX):
        """Evaluate the signed GHJB right-hand side."""
        u = np.asarray(muX, dtype=float)
        if u.ndim == 1:
            usq = np.sum(u * u)
        elif u.ndim == 2:
            usq = np.sum(u * u, axis=0)
        else:
            raise ValueError(f'Control has invalid shape {u.shape}.')
        return -self.h(x) - self.controlWeight * usq

    def terminalCost(self, x):
        """Evaluate the terminal LQR cost."""
        x, vector_input = self._as2d(x)
        val             = np.sum(x * (self.matrixKGain @ x), axis=0)
        if vector_input:
            return float(val[0])
        return val

    def gradTerminalCost(self, x):
        """Evaluate the gradient of the terminal LQR cost."""
        x, vector_input = self._as2d(x)
        out             = 2.0 * self.matrixKGain @ x
        if vector_input:
            return out[:, 0]
        return out

    def jacobi_of_f_transposed_dot_p(self, x, p):
        """Evaluate the product (Df(x))^T p."""
        x, vector_input = self._as2d(x)
        p               = np.asarray(p, dtype=float)
        if p.ndim == 1:
            p = p.reshape(4, 1)
        q             = x[:2, :]
        H11, H12, H22 = self._hessian_potential_entries(q)
        p_q1          = p[0, :]
        p_q2          = p[1, :]
        p_v1          = p[2, :]
        p_v2          = p[3, :]
        out1          = -H11 * p_v1 - H12 * p_v2
        out2          = -H12 * p_v1 - H22 * p_v2
        out3          = p_q1 - self.d1 * p_v1
        out4          = p_q2 - self.d2 * p_v2
        out           = np.vstack([out1, out2, out3, out4])
        if vector_input:
            return out[:, 0]
        return out

    def jacobi_of_gu_transposed_dot_p(self, x, u, p):
        """Evaluate the product (D_x(g(x)u))^T p."""
        x, vector_input = self._as2d(x)
        out             = np.zeros_like(x)
        if vector_input:
            return out[:, 0]
        return out

    def HJB(self, x, p):
        """Evaluate the Hamilton-Jacobi-Bellman residual."""
        x, vector_input = self._as2d(x)
        p               = np.asarray(p, dtype=float)
        if p.ndim == 1:
            p = p.reshape(4, 1)
        gt_p_sq = np.sum(p[2:4, :] * p[2:4, :], axis=0)
        f_dot_p = np.sum(self.f(x) * p, axis=0)
        val     = self.h(x) - 0.25 / self.controlWeight * gt_p_sq + f_dot_p
        if vector_input:
            return float(val[0])
        return val

    def trueVF(self, x):
        """Evaluate the analytic value function when available."""
        raise NotImplementedError('No analytic value function is available for CoupledDuffingOscillator4D.')

class ConUnconExample:
    """Two-dimensional verification-condition example with one true value function and one spurious HJB solution."""

    def __init__(self, zeta=1.0):
        """Initialize the constrained-versus-unconstrained comparison model."""
        self.zeta          = float(zeta)
        self.stateDim      = 2
        self.controlDim    = 1
        self.stateWeight   = 1.0
        self.controlWeight = 1.0
        A                  = np.array([[0.25, -1.0], [1.0, 0.0]])
        B                  = np.array([[1.0], [0.0]])
        Q                  = np.array([[0.5, 0.0], [0.0, 0.0]])
        R                  = np.array([[self.controlWeight]])
        self.matrixKGain   = la.solve_continuous_are(A, B, Q, R)
        self.lambdaMin     = np.linalg.eigvals(self.matrixKGain).min().real
        self.lambdaMax     = np.linalg.eigvals(self.matrixKGain).max().real
        P0                 = self.zeta * np.eye(self.matrixKGain.shape[0])
        self.V0            = lambda x: np.sum(x * (P0 @ x), axis=0)
        self.gradV_0       = lambda x: 2.0 * (P0 @ x)
        self.stableControl = lambda x: self.getMuFromSr(x, self.gradV_0(x))
        self.getMuFromSr   = lambda x, srX: -0.5 / self.controlWeight * np.sum(self.g(x) * srX, axis=0)
        self.getF          = lambda x, muX: self.f(x) + self.g(x) * muX
        self.getRHS        = lambda x, muX: -self.h(x) - self.controlWeight * muX ** 2

    def _as2d(self, x):
        """Return a state array with columns as states and remember whether the input was one vector."""
        x            = np.asarray(x, dtype=float)
        vector_input = False
        if x.ndim == 1:
            x            = x.reshape(-1, 1)
            vector_input = True
        if x.shape[0] != 2:
            raise ValueError(f'Expected state dimension 2, got shape {x.shape}.')
        return x, vector_input

    def h(self, x):
        """Evaluate the state running cost used in the verification-condition example."""
        x, vector_input = self._as2d(x)
        val             = 0.5 * np.exp(2.0 * np.sum(x ** 2, axis=0)) * x[0, :] ** 2
        if vector_input:
            return float(val[0])
        return val

    def trueVF(self, x):
        """Evaluate the true optimal value function."""
        x, vector_input = self._as2d(x)
        val             = np.exp(np.sum(x ** 2, axis=0)) - 1.0
        if vector_input:
            return float(val[0])
        return val

    def falseVF(self, x):
        """Evaluate the spurious HJB solution used for diagnostics."""
        x, vector_input = self._as2d(x)
        val             = 0.5 * (1.0 - np.exp(np.sum(x ** 2, axis=0)))
        if vector_input:
            return float(val[0])
        return val

    def f(self, x):
        """Evaluate the uncontrolled drift dynamics."""
        x, vector_input = self._as2d(x)
        r2              = np.sum(x ** 2, axis=0)
        out             = np.vstack([0.25 * np.exp(r2) * x[0, :] - x[1, :], x[0, :]])
        if vector_input:
            return out[:, 0]
        return out

    def g(self, x):
        """Evaluate the control vector field."""
        x, vector_input = self._as2d(x)
        out             = np.vstack([np.ones_like(x[0, :]), np.zeros_like(x[0, :])])
        if vector_input:
            return out[:, 0]
        return out

    def jacobi_of_f_transposed_dot_p(self, x, p):
        """Evaluate the product (Df(x))^T p."""
        x, vector_input = self._as2d(x)
        p               = np.asarray(p, dtype=float)
        if p.ndim == 1:
            p = p.reshape(2, 1)
        r2      = np.sum(x ** 2, axis=0)
        er2     = np.exp(r2)
        df1_dx1 = 0.25 * er2 * (1.0 + 2.0 * x[0, :] ** 2)
        df1_dx2 = 0.5 * er2 * x[0, :] * x[1, :] - 1.0
        out     = np.vstack([df1_dx1 * p[0, :] + p[1, :], df1_dx2 * p[0, :]])
        if vector_input:
            return out[:, 0]
        return out

    def HJB(self, x, p):
        """Evaluate the HJB residual for the comparison model."""
        x, vector_input = self._as2d(x)
        p               = np.asarray(p, dtype=float)
        if p.ndim == 1:
            p = p.reshape(2, 1)
        gt_p = np.sum(self.g(x) * p, axis=0)
        val  = self.h(x) - 0.25 / self.controlWeight * gt_p ** 2 + np.sum(self.f(x) * p, axis=0)
        if vector_input:
            return float(val[0])
        return val
