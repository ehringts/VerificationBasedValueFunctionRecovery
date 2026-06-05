"""Kernel definitions and Gramian builders used by the RKHS-PI surrogate models."""
import abc
import numpy as np

class Kernel(metaclass=abc.ABCMeta):
    """Base class for radial kernels k(x, y) = phi(||x-y||); phiR means phi'(r)/r and phiRR means (phiR)'(r)/r for the Gramian formulas."""

    def __init__(self, gamma):
        """Initialize the object and precompute the quantities used later."""
        self.gamma = gamma

    def setGamma(self, gamma):
        """Set the kernel shape parameter gamma."""
        self.gamma = gamma

    @abc.abstractmethod
    def phi(self, r):
        """Evaluate the radial basis function phi(r)."""
        pass

    @abc.abstractmethod
    def phiR(self, r):
        """Evaluate phi'(r)/r for the radial basis function."""
        pass

    @abc.abstractmethod
    def phiRR(self, r):
        """Evaluate the radial derivative (phi'(r)/r)'/r used in the PDE Gramian."""
        pass

    def getGramPDEStartColumn(self, f, y, fy=[]):
        """Build the PDE Gramian column associated with the origin."""
        x    = np.atleast_2d(y[:, 0] * 0).T
        diff = np.sqrt(np.abs(np.sum(x ** 2, axis=0, keepdims=True) + np.sum(y ** 2, axis=0, keepdims=True).T - 2 * y.T @ x))
        phiR = self.phiR(diff)
        if len(fy) == 0:
            fy = f(y)
        return phiR * np.atleast_2d(np.sum(fy * y, axis=0)).T

    def getGramPDE(self, f, y, x, fy=[]):
        """Build the PDE Gramian block between evaluation points and centers."""
        diff  = np.sqrt(np.abs(np.sum(x ** 2, axis=0, keepdims=True) + np.sum(y ** 2, axis=0, keepdims=True).T - 2 * y.T @ x))
        phiR  = self.phiR(diff)
        phiRR = self.phiRR(diff)
        fx    = f(x)
        if len(fy) == 0:
            fy = f(y)
        term1 = phiRR * (np.sum(fy * y, axis=0) - x.T @ fy).T * (np.sum(fx * x, axis=0) - y.T @ fx)
        term2 = -1 * phiR * (fy.T @ fx)
        return term1 + term2

    def evalGrad(self, fx, y, x, alpha):
        """Evaluate the gradient of the RKHS surrogate."""
        diff   = np.sqrt(np.abs(np.sum(x ** 2, axis=0, keepdims=True) + np.sum(y ** 2, axis=0, keepdims=True).T - 2 * y.T @ x))
        phiR   = self.phiR(diff)
        phiRR  = self.phiRR(diff)
        alpha0 = alpha[0]
        alpha  = alpha[1:]
        term1  = phiRR * (np.sum(fx * x, axis=0) - y.T @ fx)
        term3  = y * np.atleast_2d(term1 @ np.atleast_2d(alpha).T).T
        term4  = -1 * (term1 * np.atleast_2d(alpha) @ x.T).T
        term5  = -1 * (phiR * np.atleast_2d(alpha) @ fx.T).T
        x      = np.atleast_2d(x[:, 0] * 0).T
        diff   = np.sqrt(np.abs(np.sum(x ** 2, axis=0, keepdims=True) + np.sum(y ** 2, axis=0, keepdims=True).T - 2 * y.T @ x))
        term6  = self.phiR(diff).T * y * alpha0
        return term3 + term4 + term5 + term6

    def evalFunc(self, fx, y, x, alpha):
        """Evaluate the RKHS surrogate value function approximation."""
        diff  = np.sqrt(np.abs(np.sum(x ** 2, axis=0, keepdims=True) + np.sum(y ** 2, axis=0, keepdims=True).T - 2 * y.T @ x))
        phiR  = self.phiR(diff)
        term2 = phiR * (np.sum(fx * x, axis=0) - y.T @ fx)
        x     = np.atleast_2d(x[:, 0] * 0).T
        diff  = np.sqrt(np.abs(np.sum(x ** 2, axis=0, keepdims=True) + np.sum(y ** 2, axis=0, keepdims=True).T - 2 * y.T @ x))
        term1 = self.phi(diff)
        return np.c_[term1, term2] @ alpha

class QuadWendland(Kernel):
    """Compactly supported quadratic Wendland radial basis kernel."""

    def __init__(self, gamma, d):
        """Initialize the object and precompute the quantities used later."""
        self.l     = np.floor(d / 2) + 2 + 1
        self.gamma = gamma

    def phi(self, r):
        """Evaluate the radial basis function phi(r)."""
        return (self.gamma * r <= 1) * (1 - self.gamma * r) ** (self.l + 2) * ((self.l ** 2 + 4 * self.l + 3) * (self.gamma * r) ** 2 + (3 * self.l + 6) * self.gamma * r + 3)

    def phiR(self, r):
        """Evaluate phi'(r)/r for the radial basis function."""
        return (self.gamma * r <= 1) * (1 - self.gamma * r) ** (self.l + 1) * -self.gamma ** 2 * (12 + 7 * self.l + self.l ** 2) * (1 + (1 + self.l) * self.gamma * r)

    def phiRR(self, r):
        """Evaluate the radial derivative (phi'(r)/r)'/r used in the PDE Gramian."""
        return (self.gamma * r <= 1) * (1 - self.gamma * r) ** self.l * self.gamma ** 4 * (24 + 50 * self.l + 35 * self.l ** 2 + 10 * self.l ** 3 + self.l ** 4)

class QuadMatern(Kernel):
    """Quadratic Matérn radial basis kernel."""

    def phi(self, r):
        """Evaluate the radial basis function phi(r)."""
        return np.exp(-self.gamma * r) * (3 + 3 * self.gamma * r + self.gamma ** 2 * r ** 2)

    def phiR(self, r):
        """Evaluate phi'(r)/r for the radial basis function."""
        return -1 * np.exp(-self.gamma * r) * (1 + self.gamma * r) * self.gamma ** 2

    def phiRR(self, r):
        """Evaluate the radial derivative (phi'(r)/r)'/r used in the PDE Gramian."""
        return self.gamma ** 4 * np.exp(-self.gamma * r)

class Gauss(Kernel):
    """Gaussian radial basis kernel phi(r) = exp(-(gamma r)^2)."""

    def phi(self, r):
        """Evaluate the radial basis function phi(r)."""
        return np.exp(-(self.gamma * r) ** 2)

    def phiR(self, r):
        """Evaluate phi'(r)/r for the radial basis function."""
        return -2 * self.gamma ** 2 * np.exp(-(self.gamma * r) ** 2)

    def phiRR(self, r):
        """Evaluate the radial derivative (phi'(r)/r)'/r used in the PDE Gramian."""
        return 4 * self.gamma ** 4 * np.exp(-(self.gamma * r) ** 2)

class InvMulti(Kernel):
    """Inverse multiquadric radial basis kernel."""

    def phi(self, r):
        """Evaluate the radial basis function phi(r)."""
        return 1 / np.sqrt(1 + (self.gamma * r) ** 2)

    def phiR(self, r):
        """Evaluate phi'(r)/r for the radial basis function."""
        return -1 * self.gamma ** 2 * (1 / np.sqrt((1 + (self.gamma * r) ** 2) ** 3))

    def phiRR(self, r):
        """Evaluate the radial derivative (phi'(r)/r)'/r used in the PDE Gramian."""
        return 3 * self.gamma ** 4 * (1 / np.sqrt((1 + (self.gamma * r) ** 2) ** 5))

class LinMatern(Kernel):
    """Linear Matérn radial basis kernel."""

    def phi(self, r):
        """Evaluate the radial basis function phi(r)."""
        return np.exp(-self.gamma * r) * (1 + self.gamma * r)

    def phiR(self, r):
        """Evaluate phi'(r)/r for the radial basis function."""
        return -1 * np.exp(-self.gamma * r) * self.gamma ** 2

    def phiRR(self, r):
        """Evaluate the radial derivative (phi'(r)/r)'/r used in the PDE Gramian."""
        diffMask1 = r < 10 ** (-14)
        diffMask2 = r > 10 ** (-14)
        return self.gamma ** 3 * np.exp(-self.gamma * r) * (1 / (r + diffMask1)) * diffMask2

class KernelProduct(metaclass=abc.ABCMeta):
    """Base class for product kernels k(x, y) = phi(||x-y||)(y^T x)^case; case = 0 gives the radial kernel and case > 0 adds a polynomial product factor."""

    def __init__(self, gamma, case=0):
        """Initialize the object and precompute the quantities used later."""
        self.case  = case
        self.gamma = gamma

    def setGamma(self, gamma):
        """Set the kernel shape parameter gamma."""
        self.gamma = gamma

    @abc.abstractmethod
    def phi(self, r):
        """Evaluate the radial basis function phi(r)."""
        pass

    @abc.abstractmethod
    def phiR(self, r):
        """Evaluate phi'(r)/r for the radial basis function."""
        pass

    @abc.abstractmethod
    def phiRR(self, r):
        """Evaluate the radial derivative (phi'(r)/r)'/r used in the PDE Gramian."""
        pass

    def getGramPDE(self, f, y, x, fy=[]):
        """Build the PDE Gramian block between evaluation points and centers."""
        if self.case > 1:
            d     = self.case
            diff  = np.sqrt(np.abs(np.sum(x ** 2, axis=0, keepdims=True) + np.sum(y ** 2, axis=0, keepdims=True).T - 2 * y.T @ x))
            phi   = self.phi(diff)
            phiR  = self.phiR(diff)
            phiRR = self.phiRR(diff)
            lin   = y.T @ x
            fx    = f(x)
            if len(fy) == 0:
                fy = f(y)
            term1 = phiRR * lin ** d * (np.sum(fy * y, axis=0) - x.T @ fy).T * (np.sum(fx * x, axis=0) - y.T @ fx)
            term2 = d * phiR * lin ** (d - 1) * (np.sum(fy * y, axis=0) * 0 + x.T @ fy).T * (np.sum(fx * x, axis=0) - y.T @ fx)
            term3 = -1 * phiR * lin ** d * (fy.T @ fx)
            term4 = d * phiR * lin ** (d - 1) * (np.sum(fy * y, axis=0) - x.T @ fy).T * (np.sum(fx * x, axis=0) * 0 + y.T @ fx)
            term5 = d * (d - 1) * phi * lin ** (d - 2) * (np.sum(fy * y, axis=0) * 0 + x.T @ fy).T * (np.sum(fx * x, axis=0) * 0 + y.T @ fx)
            term6 = d * phi * lin ** (d - 1) * (fy.T @ fx)
            return term1 + term2 + term3 + term4 + term5 + term6
        elif self.case == 1:
            diff  = np.sqrt(np.abs(np.sum(x ** 2, axis=0, keepdims=True) + np.sum(y ** 2, axis=0, keepdims=True).T - 2 * y.T @ x))
            phi   = self.phi(diff)
            phiR  = self.phiR(diff)
            phiRR = self.phiRR(diff)
            lin   = y.T @ x
            fx    = f(x)
            if len(fy) == 0:
                fy = f(y)
            term1 = phiRR * lin * (np.sum(fy * y, axis=0) - x.T @ fy).T * (np.sum(fx * x, axis=0) - y.T @ fx)
            term2 = phiR * (np.sum(fy * y, axis=0) * 0 + x.T @ fy).T * (np.sum(fx * x, axis=0) - y.T @ fx)
            term3 = -1 * phiR * lin * (fy.T @ fx)
            term4 = phiR * (np.sum(fy * y, axis=0) - x.T @ fy).T * (np.sum(fx * x, axis=0) * 0 + y.T @ fx)
            term5 = phi * (fy.T @ fx)
            return term1 + term2 + term3 + term4 + term5
        elif self.case == 0:
            diff  = np.sqrt(np.abs(np.sum(x ** 2, axis=0, keepdims=True) + np.sum(y ** 2, axis=0, keepdims=True).T - 2 * y.T @ x))
            phiR  = self.phiR(diff)
            phiRR = self.phiRR(diff)
            fx    = f(x)
            if len(fy) == 0:
                fy = f(y)
            term1 = phiRR * (np.sum(fy * y, axis=0) - x.T @ fy).T * (np.sum(fx * x, axis=0) - y.T @ fx)
            term2 = -1 * phiR * (fy.T @ fx)
            return term1 + term2
        else:
            print('Case not implemented')

    def evalGrad(self, fx, y, x, alpha):
        """Evaluate the gradient of the RKHS surrogate."""
        if self.case > 1:
            d      = self.case
            diff   = np.sqrt(np.abs(np.sum(x ** 2, axis=0, keepdims=True) + np.sum(y ** 2, axis=0, keepdims=True).T - 2 * y.T @ x))
            phi    = self.phi(diff)
            phiR   = self.phiR(diff)
            phiRR  = self.phiRR(diff)
            lin    = y.T @ x
            term10 = phiRR * lin ** d * (np.sum(fx * x, axis=0) - y.T @ fx)
            term11 = y * np.atleast_2d(term10 @ np.atleast_2d(alpha).T).T
            term12 = -1 * (term10 * np.atleast_2d(alpha) @ x.T).T
            term20 = d * phiR * lin ** (d - 1) * (np.sum(fx * x, axis=0) - y.T @ fx)
            term21 = (term20 * np.atleast_2d(alpha) @ x.T).T
            term31 = (-1 * phiR * lin ** d * np.atleast_2d(alpha) @ fx.T).T
            term40 = d * phiR * lin ** (d - 1) * (np.sum(fx * x, axis=0) * 0 + y.T @ fx)
            term41 = y * np.atleast_2d(term40 @ np.atleast_2d(alpha).T).T
            term42 = -1 * (term40 * np.atleast_2d(alpha) @ x.T).T
            term50 = d * (d - 1) * phi * lin ** (d - 2) * (np.sum(fx * x, axis=0) * 0 + y.T @ fx)
            term51 = (term50 * np.atleast_2d(alpha) @ x.T).T
            term61 = (d * phi * lin ** (d - 1) * np.atleast_2d(alpha) @ fx.T).T
            return term11 + term12 + term21 + term31 + term41 + term42 + term51 + term61
        elif self.case == 1:
            diff      = np.sqrt(np.abs(np.sum(x ** 2, axis=0, keepdims=True) + np.sum(y ** 2, axis=0, keepdims=True).T - 2 * y.T @ x))
            phi       = self.phi(diff)
            phiR      = self.phiR(diff)
            phiRR     = self.phiRR(diff)
            lin       = y.T @ x
            termTemp1 = phiRR * lin * (np.sum(fx * x, axis=0) - y.T @ fx)
            term1     = y * np.atleast_2d(termTemp1 @ np.atleast_2d(alpha).T).T
            term2     = -1 * (termTemp1 * np.atleast_2d(alpha) @ x.T).T
            termTemp2 = phiR * (np.sum(fx * x, axis=0) - y.T @ fx)
            term3     = (termTemp2 * np.atleast_2d(alpha) @ x.T).T
            term4     = -1 * (phiR * lin * np.atleast_2d(alpha) @ fx.T).T
            termTemp3 = phiR * (np.sum(fx * x, axis=0) * 0 + y.T @ fx)
            term5     = y * np.atleast_2d(termTemp3 @ np.atleast_2d(alpha).T).T
            term6     = -1 * (termTemp3 * np.atleast_2d(alpha) @ x.T).T
            term7     = (phi * np.atleast_2d(alpha) @ fx.T).T
            return term1 + term2 + term3 + term4 + term5 + term6 + term7
        elif self.case == 0:
            diff  = np.sqrt(np.abs(np.sum(x ** 2, axis=0, keepdims=True) + np.sum(y ** 2, axis=0, keepdims=True).T - 2 * y.T @ x))
            phiR  = self.phiR(diff)
            phiRR = self.phiRR(diff)
            term1 = phiRR * (np.sum(fx * x, axis=0) - y.T @ fx)
            term3 = y * np.atleast_2d(term1 @ np.atleast_2d(alpha).T).T
            term4 = -1 * (term1 * np.atleast_2d(alpha) @ x.T).T
            term5 = -1 * (phiR * np.atleast_2d(alpha) @ fx.T).T
            return term3 + term4 + term5
        else:
            print('Case not implemented')

    def evalFunc(self, fx, y, x, alpha):
        """Evaluate the RKHS surrogate value function approximation."""
        if self.case > 1:
            d    = self.case
            diff = np.sqrt(np.abs(np.sum(x ** 2, axis=0, keepdims=True) + np.sum(y ** 2, axis=0, keepdims=True).T - 2 * y.T @ x))
            phi  = self.phi(diff)
            phiR = self.phiR(diff)
            lin  = y.T @ x
            return phiR * lin ** d * (np.sum(fx * x, axis=0) - y.T @ fx) @ alpha + d * (phi * lin ** (d - 1) * (np.sum(fx * x, axis=0) * 0 + y.T @ fx)) @ alpha
        elif self.case == 1:
            diff = np.sqrt(np.abs(np.sum(x ** 2, axis=0, keepdims=True) + np.sum(y ** 2, axis=0, keepdims=True).T - 2 * y.T @ x))
            phi  = self.phi(diff)
            phiR = self.phiR(diff)
            lin  = y.T @ x
            return phiR * lin * (np.sum(fx * x, axis=0) - y.T @ fx) @ alpha + phi * (np.sum(fx * x, axis=0) * 0 + y.T @ fx) @ alpha
        elif self.case == 0:
            diff = np.sqrt(np.abs(np.sum(x ** 2, axis=0, keepdims=True) + np.sum(y ** 2, axis=0, keepdims=True).T - 2 * y.T @ x))
            phiR = self.phiR(diff)
            return phiR * (np.sum(fx * x, axis=0) - y.T @ fx) @ alpha
        else:
            print('Case not implemented')

class QuadWendlandProduct(KernelProduct):
    """Product version of the compactly supported quadratic Wendland kernel."""

    def __init__(self, gamma, d, case=0):
        """Initialize the object and precompute the quantities used later."""
        self.l     = np.floor(d / 2) + 2 + 1
        self.case  = case
        self.gamma = gamma

    def phi(self, r):
        """Evaluate the radial basis function phi(r)."""
        return (self.gamma * r <= 1) * (1 - self.gamma * r) ** (self.l + 2) * ((self.l ** 2 + 4 * self.l + 3) * (self.gamma * r) ** 2 + (3 * self.l + 6) * self.gamma * r + 3)

    def phiR(self, r):
        """Evaluate phi'(r)/r for the radial basis function."""
        return (self.gamma * r <= 1) * (1 - self.gamma * r) ** (self.l + 1) * -self.gamma ** 2 * (12 + 7 * self.l + self.l ** 2) * (1 + (1 + self.l) * self.gamma * r)

    def phiRR(self, r):
        """Evaluate the radial derivative (phi'(r)/r)'/r used in the PDE Gramian."""
        return (self.gamma * r <= 1) * (1 - self.gamma * r) ** self.l * self.gamma ** 4 * (24 + 50 * self.l + 35 * self.l ** 2 + 10 * self.l ** 3 + self.l ** 4)

class QuadMaternProduct(KernelProduct):
    """Product version of the quadratic Matérn kernel."""

    def phi(self, r):
        """Evaluate the radial basis function phi(r)."""
        return np.exp(-self.gamma * r) * (3 + 3 * self.gamma * r + self.gamma ** 2 * r ** 2)

    def phiR(self, r):
        """Evaluate phi'(r)/r for the radial basis function."""
        return -1 * np.exp(-self.gamma * r) * (1 + self.gamma * r) * self.gamma ** 2

    def phiRR(self, r):
        """Evaluate the radial derivative (phi'(r)/r)'/r used in the PDE Gramian."""
        return self.gamma ** 4 * np.exp(-self.gamma * r)

class GaussProduct(KernelProduct):
    """Product version of the Gaussian radial basis kernel."""

    def phi(self, r):
        """Evaluate the radial basis function phi(r)."""
        return np.exp(-(self.gamma * r) ** 2)

    def phiR(self, r):
        """Evaluate phi'(r)/r for the radial basis function."""
        return -2 * self.gamma ** 2 * np.exp(-(self.gamma * r) ** 2)

    def phiRR(self, r):
        """Evaluate the radial derivative (phi'(r)/r)'/r used in the PDE Gramian."""
        return 4 * self.gamma ** 4 * np.exp(-(self.gamma * r) ** 2)

class InvMultiProduct(KernelProduct):
    """Product version of the inverse multiquadric kernel."""

    def phi(self, r):
        """Evaluate the radial basis function phi(r)."""
        return 1 / np.sqrt(1 + (self.gamma * r) ** 2)

    def phiR(self, r):
        """Evaluate phi'(r)/r for the radial basis function."""
        return -1 * self.gamma ** 2 * (1 / np.sqrt((1 + (self.gamma * r) ** 2) ** 3))

    def phiRR(self, r):
        """Evaluate the radial derivative (phi'(r)/r)'/r used in the PDE Gramian."""
        return 3 * self.gamma ** 4 * (1 / np.sqrt((1 + (self.gamma * r) ** 2) ** 5))

class LinMaternProduct(KernelProduct):
    """Product version of the linear Matérn kernel."""

    def phi(self, r):
        """Evaluate the radial basis function phi(r)."""
        return np.exp(-self.gamma * r) * (1 + self.gamma * r)

    def phiR(self, r):
        """Evaluate phi'(r)/r for the radial basis function."""
        return -1 * np.exp(-self.gamma * r) * self.gamma ** 2

    def phiRR(self, r):
        """Evaluate the radial derivative (phi'(r)/r)'/r used in the PDE Gramian."""
        diffMask1 = r < 10 ** (-14)
        diffMask2 = r > 10 ** (-14)
        return self.gamma ** 3 * np.exp(-self.gamma * r) * (1 / (r + diffMask1)) * diffMask2
