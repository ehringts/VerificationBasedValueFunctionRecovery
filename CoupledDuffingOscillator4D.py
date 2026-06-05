"""Workflow for the coupled Duffing oscillator: generate BVP reference data, run gamma cross-validation, train standard and product-kernel RKHS-PI surrogates, and plot the observer errors."""
import numpy as np
from functions import auxFunctions, kernel, model, observer, surrogate

nMaxPI   = 5
newModel = model.CoupledDuffingOscillator4D()
newOb    = observer.Observer('Duffing4D')

def make_grid(N, num=100, a=-0.5, b=0.5):
    """Create a Cartesian tensor grid and return the grid points as columns."""
    x     = np.linspace(a, b, num)
    grids = np.meshgrid(*[x] * N, indexing='ij')
    omega = np.stack([g.ravel() for g in grids], axis=0)
    return omega

VFinput          = (np.random.rand(4, 100) - 0.5) * 2
VFoutputTrue     = np.zeros(VFinput.shape[1])
VFinputPerf      = (np.random.rand(4, 10) - 0.5) * 2
VFoutputTruePerf = np.zeros(VFinputPerf.shape[1])
for i in range(VFinput.shape[1]):
    _, _, VF, _, _  = newModel.solveOLBVP(VFinput[:, i], 100, 1001)
    VFoutputTrue[i] = VF
for i in range(VFinputPerf.shape[1]):
    _, _, VF, _, _      = newModel.solveOLBVP(VFinputPerf[:, i], 100, 1001)
    VFoutputTruePerf[i] = VF
testPoint  = make_grid(4, 10, a=-1, b=1)
nMaxGreedy = 500

def findGamma():
    """Run gamma cross-validation and print the best standard and product-kernel parameters."""
    gammaList = [1.5, 1.4, 1.3, 1.2, 1.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    a, b      = auxFunctions.findBestGammaNew(newModel, surrogate.Surrogate(kernel.Gauss(1)), nMaxGreedy, newOb, testPoint, gammaList)
    print('Best gamma for Gauss is ', a, ' with value ', b)
    a, b = auxFunctions.findBestGammaNew(newModel, surrogate.SurrogateProductKernel(kernel.GaussProduct(1, 2)), nMaxGreedy, newOb, testPoint, gammaList)
    print('Best gamma for GaussProduct is ', a, ' with value ', b)
    
findGamma()
newSr = surrogate.Surrogate(kernel.Gauss(0.2))
auxFunctions.RKHSPI(newModel, newSr, nMaxPI, nMaxGreedy, newOb, testPoint, VFinput, VFoutputTrue, VFinputPerf, VFoutputTruePerf)
newSr = surrogate.SurrogateProductKernel(kernel.GaussProduct(0.2, 2))
auxFunctions.RKHSPIProductKernel(newModel, newSr, nMaxPI, nMaxGreedy, newOb, testPoint, VFinput, VFoutputTrue, VFinputPerf, VFoutputTruePerf)
newOb.loadObserver('Duffing4D')
newOb.plotObserver('DuffingPI', 'Coupled Duffing oscillator (\\texttt{CoupledDuffing})', 0.5 * newModel.lambdaMin, 2 * newModel.lambdaMax)
