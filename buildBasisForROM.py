"""Workflow for the heat-equation ROM basis: generate open-loop trajectories, collect snapshots, compute a POD basis, and save the reduced basis."""
import pickle
import numpy as np
from functions import model

N                  = 99
newModel           = model.HEDirichlet1DFD(N, [1 / 4, 1, 100, 1])
numModes           = 8
rng                = np.random.default_rng(seed=0)
coefficientSamples = rng.uniform(-1.0, 1.0, size=(1000, numModes))
X0, _, _           = newModel.makeInitialStatesFromLaplacianModes(numModes, coefficientSamples)
X0                 = X0 / np.max(np.abs(X0), axis=0, keepdims=True)
Xopt               = []
for i in range(X0.shape[1]):
    x, _, _, _ = newModel.solveOLIterativ(X0[:, i], 50, 1001)
    x          = np.asarray(x)
    Xopt.append(x)
with open('data/Xopt_N99.pkl', 'wb') as f:
    pickle.dump(Xopt, f)
Xlist = []
for x in Xopt:
    x = np.asarray(x)
    if x.shape[0] != 99:
        x = x.T
    Xlist.append(x)
X        = np.hstack(Xlist)
U, S, VT = np.linalg.svd(X, full_matrices=False)
PhiPOD   = U[:, :8]
with open('data/ROMBasis.pkl', 'wb') as f:
    pickle.dump(PhiPOD, f)
