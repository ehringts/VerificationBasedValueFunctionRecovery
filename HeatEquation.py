"""Workflow for the reduced heat-equation model: load the POD basis, generate ROM reference data, run gamma cross-validation, train standard and product-kernel RKHS-PI surrogates, and plot the observer errors."""
import pickle
import numpy as np
from functions import auxFunctions, kernel, model, observer, surrogate

N      = 99
r      = 8
nMaxPI = 5
with open('data/ROMBasis.pkl', 'rb') as f:
    PhiPOD = pickle.load(f)

def make_grid(N, num=100, a=-0.5, b=0.5):
    """Create a Cartesian tensor grid and return the grid points as columns."""
    x     = np.linspace(a, b, num)
    grids = np.meshgrid(*[x] * N, indexing='ij')
    omega = np.stack([g.ravel() for g in grids], axis=0)
    return omega

newModel         = model.HEDirichlet1DROM(N, [1 / 4, 1, 100, 1], PhiPOD)
newOb            = observer.Observer('NonHEROM')
testPoint        = make_grid(r, 6, a=-2, b=2)
nMaxGreedy       = 500
VFinput          = (np.random.rand(r, 100) - 0.5) * 4
VFoutputTrue     = np.zeros(VFinput.shape[1])
VFinputPerf      = (np.random.rand(r, 10) - 0.5) * 2
VFoutputTruePerf = np.zeros(VFinputPerf.shape[1])

for i in range(VFinput.shape[1]):
    _, _, VF, _     = newModel.solveOLIterativ(VFinput[:, i], 50, 1001)
    VFoutputTrue[i] = VF
for i in range(VFinputPerf.shape[1]):
    _, _, VF, _         = newModel.solveOLIterativ(VFinputPerf[:, i], 50, 1001)
    VFoutputTruePerf[i] = VF

def findGamma():
    """Run gamma cross-validation and print the best standard and product-kernel parameters."""
    gammaList = [0.1, 0.07, 0.05, 0.03, 0.01, 0.007]
    a, b      = auxFunctions.findBestGammaNew(newModel, surrogate.Surrogate(kernel.Gauss(1)), nMaxGreedy, newOb, testPoint, gammaList)
    print('Best gamma for Gauss is ', a, ' with value ', b)
    a, b = auxFunctions.findBestGammaNew(newModel, surrogate.SurrogateProductKernel(kernel.GaussProduct(1, 2)), nMaxGreedy, newOb, testPoint, gammaList)
    print('Best gamma for GaussProduct is ', a, ' with value ', b)
    a, b = auxFunctions.findBestGammaNew(newModel, surrogate.Surrogate(kernel.LinMatern(1)), nMaxGreedy, newOb, testPoint, gammaList)
    print('Best gamma for LinMatern is ', a, ' with value ', b)
    a, b = auxFunctions.findBestGammaNew(newModel, surrogate.SurrogateProductKernel(kernel.LinMaternProduct(1, 2)), nMaxGreedy, newOb, testPoint, gammaList)
    print('Best gamma for LinMaternProduct is ', a, ' with value ', b)
    a, b = auxFunctions.findBestGammaNew(newModel, surrogate.Surrogate(kernel.QuadMatern(1)), nMaxGreedy, newOb, testPoint, gammaList)
    print('Best gamma for QuadMatern is ', a, ' with value ', b)
    a, b = auxFunctions.findBestGammaNew(newModel, surrogate.SurrogateProductKernel(kernel.QuadMaternProduct(1, 2)), nMaxGreedy, newOb, testPoint, gammaList)
    print('Best gamma for QuadMaternProduct is ', a, ' with value ', b)
    
findGamma()
newSr = surrogate.Surrogate(kernel.Gauss(0.05))
auxFunctions.RKHSPI(newModel, newSr, nMaxPI, nMaxGreedy, newOb, testPoint, VFinput, VFoutputTrue, VFinputPerf, VFoutputTruePerf)
newSr = surrogate.SurrogateProductKernel(kernel.GaussProduct(0.01, 2))
auxFunctions.RKHSPIProductKernel(newModel, newSr, nMaxPI, nMaxGreedy, newOb, testPoint, VFinput, VFoutputTrue, VFinputPerf, VFoutputTruePerf)
newOb.loadObserver('NonHEROM')
newOb.plotObserver2('ROMHeatPI', 'Reduced-order nonlinear heat equation (\\texttt{ROMHeat})', 0.5 * newModel.lambdaMin, 2 * newModel.lambdaMax)
