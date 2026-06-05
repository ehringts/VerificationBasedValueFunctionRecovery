"""RKHS-PI training utilities: gamma search, standard-kernel policy iteration, and product-kernel policy iteration."""
import contextlib
import io
import numpy as np

def RKHSPI(model, surrogate, nMaxPI, nMaxGreedy, observer, testPoint, VFinput, VFoutputTrue, VFinputPerf, VFoutputTruePerf):
    """Run RKHS policy iteration with a standard radial kernel and store the observer errors."""
    costCum = 0
    for i in range(VFinputPerf.shape[1]):
        srGrad     = lambda x: surrogate.kernel.evalGrad(FCenterVal, np.atleast_2d(x).T, center, alpha)[:, 0]
        _, _, cost = model.solveSurrogate(VFinputPerf[:, i], 50, model.gradV_0)
        costCum    = np.max([costCum, np.abs(cost - VFoutputTruePerf[i])])
    print(costCum)
    observer.addPerfomErrorStandart(costCum)
    observer.addQuadraticStandart(np.min(model.V0(testPoint) / np.sum(testPoint ** 2, 0)), np.max(model.V0(testPoint) / np.sum(testPoint ** 2, 0)))
    observer.addTrueErrorStandart(np.max(np.abs(VFoutputTrue - model.V0(VFinput))))
    for j in range(nMaxPI):
        if j == 0:
            mu  = lambda x: model.stableControl(x)
            F   = lambda x: model.getF(x, mu(x))
            rhs = lambda x: model.getRHS(x, mu(x))
            surrogate.doFGreedy(F, rhs, testPoint, nMaxGreedy, observer, 10 ** (-8))
            maxFGreedyError = surrogate.trainError[-1]
            oldSRVal        = 0
        else:
            mu  = lambda x: model.getMuFromSr(x, surrogate.kernel.evalGrad(FCenterVal, x, center, alpha))
            F   = lambda x: model.getF(x, mu(x))
            rhs = lambda x: model.getRHS(x, mu(x))
            surrogate.fit(F, rhs, surrogate.center, iterativ=False)
            maxFGreedyError = np.max(np.abs(np.c_[surrogate.kernel.getGramPDEStartColumn(F, testPoint), surrogate.kernel.getGramPDE(F, testPoint, surrogate.center)] @ surrogate.alpha - rhs(testPoint)))
        FCenterVal = F(surrogate.center).copy()
        center     = surrogate.center.copy()
        alpha      = surrogate.alpha.copy()
        costCum    = 0
        for i in range(VFinputPerf.shape[1]):
            srGrad     = lambda x: surrogate.kernel.evalGrad(FCenterVal, np.atleast_2d(x).T, center, alpha)[:, 0]
            _, _, cost = model.solveSurrogate(VFinputPerf[:, i], 50, srGrad)
            costCum    = np.max([costCum, np.abs(cost - VFoutputTruePerf[i])])
        print(costCum)
        observer.addPerfomErrorStandart(costCum)
        trueError     = np.max(np.abs(VFoutputTrue - surrogate.kernel.evalFunc(FCenterVal, VFinput, center, alpha)))
        evalSR        = surrogate.kernel.evalFunc(FCenterVal, testPoint, center, alpha)
        isPos         = np.min(evalSR) >= 0
        hasLowerBound = np.min(evalSR - 0.5 * model.lambdaMin * np.sum(testPoint ** 2, 0)) >= 0
        hasUpperBound = np.max(evalSR - 2 * model.lambdaMax * np.sum(testPoint ** 2, 0)) <= 0
        observer.addQuadraticStandart(np.min(evalSR / np.sum(testPoint ** 2, 0)), np.max(evalSR / np.sum(testPoint ** 2, 0)))
        print(str(j) + ' Iteration: True-Error = ' + str(trueError) + ', Residual-Error = ' + str(maxFGreedyError) + ', stagnation-Error = ' + str(np.max(np.abs(oldSRVal - surrogate.kernel.evalFunc(FCenterVal, testPoint, center, alpha)))) + ', is positive = ' + str(isPos) + ', Lower Bound = ' + str(hasLowerBound) + ', Upper Bound = ' + str(hasUpperBound))
        observer.addTrueErrorStandart(trueError)
        observer.addObjectStandart(0, maxFGreedyError, np.max(np.abs(oldSRVal - surrogate.kernel.evalFunc(FCenterVal, testPoint, center, alpha))))
        oldSRVal = surrogate.kernel.evalFunc(FCenterVal, testPoint, center, alpha)
    surrogate.FCenterVal = FCenterVal

def findBestGammaNew(model, surrogate, nMaxGreedy, observer, testPoint, gammaList):
    """Test a list of gamma values with suppressed greedy output and return the best residual."""
    gammaListVal = np.full(len(gammaList), np.inf)
    mu           = lambda x: model.stableControl(x)
    F            = lambda x: model.getF(x, mu(x))
    rhs          = lambda x: model.getRHS(x, mu(x))
    for i, gamma in enumerate(gammaList):
        surrogate.kernel.setGamma(gamma)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                surrogate.doFGreedy(F, rhs, testPoint, nMaxGreedy, observer, 10 ** (-8))
            gammaListVal[i] = surrogate.trainError[-1]
            print(f'Gamma: {gamma}, Error: {gammaListVal[i]}')
        except Exception:
            gammaListVal[i] = np.inf
        finally:
            surrogate.C = np.array([])
        valid_idx = np.where(np.isfinite(gammaListVal))[0]
        if len(valid_idx) > 0:
            bestGammaIndex = valid_idx[np.argmin(gammaListVal[valid_idx])]
    valid_idx = np.where(np.isfinite(gammaListVal))[0]
    if len(valid_idx) == 0:
        raise RuntimeError('All gamma values failed in surrogate.doFGreedy().')
    bestGammaIndex = valid_idx[np.argmin(gammaListVal[valid_idx])]
    return (gammaList[bestGammaIndex], gammaListVal[bestGammaIndex])

def findBestGamma(model, surrogate, nMaxGreedy, observer, testPoint, gammaList):
    """Test a list of gamma values and return the best standard-kernel residual."""
    gammaListVal = 1000 + np.zeros(len(gammaList))
    for i in range(len(gammaList)):
        surrogate.kernel.setGamma(gammaList[i])
        mu  = lambda x: model.stableControl(x)
        F   = lambda x: model.getF(x, mu(x))
        rhs = lambda x: model.getRHS(x, mu(x))
        surrogate.doFGreedy(F, rhs, testPoint, nMaxGreedy, observer, 10 ** (-8))
        surrogate.C     = np.array([])
        gammaListVal[i] = surrogate.trainError[-1]
        bestGammaIndex  = np.argmin(gammaListVal)
        print('Gamma: ' + str(gammaList[i]) + ', Error: ' + str(gammaListVal[i]))
        print('The current best Gamma is: ' + str(gammaList[bestGammaIndex]) + ' with a value of ' + str(gammaListVal[bestGammaIndex]))
    bestGammaIndex = np.argmin(gammaListVal)
    print('The best Gamma is: ' + str(gammaList[bestGammaIndex]) + ' with a value of ' + str(gammaListVal[bestGammaIndex]))
    return (gammaList[bestGammaIndex], gammaListVal[bestGammaIndex])

def RKHSPIProductKernel(model, surrogate, nMaxPI, nMaxGreedy, observer, testPoint, VFinput, VFoutputTrue, VFinputPerf, VFoutputTruePerf):
    """Run RKHS policy iteration with a product kernel and store the observer errors."""
    costCum = 0
    for i in range(VFinputPerf.shape[1]):
        srGrad     = lambda x: surrogate.kernel.evalGrad(FCenterVal, np.atleast_2d(x).T, center, alpha)[:, 0]
        _, _, cost = model.solveSurrogate(VFinputPerf[:, i], 50, model.gradV_0)
        costCum    = np.max([costCum, np.abs(cost - VFoutputTruePerf[i])])
    observer.addPerfomErrorQuadKernel(costCum)
    print(costCum)
    observer.addQuadraticQuadKernel(np.min(model.V0(testPoint) / np.sum(testPoint ** 2, 0)), np.max(model.V0(testPoint) / np.sum(testPoint ** 2, 0)))
    observer.addTrueErrorQuadKernel(np.max(np.abs(VFoutputTrue - model.V0(VFinput))))
    print('Initial True-Error = ' + str(np.max(np.abs(VFoutputTrue - model.V0(VFinput)))))
    for j in range(nMaxPI):
        if j == 0:
            mu  = lambda x: model.stableControl(x)
            F   = lambda x: model.getF(x, mu(x))
            rhs = lambda x: model.getRHS(x, mu(x))
            surrogate.doFGreedy(F, rhs, testPoint, nMaxGreedy, observer, 10 ** (-8))
            maxFGreedyError = surrogate.trainError[-1]
            oldSRVal        = 0
        else:
            mu  = lambda x: model.getMuFromSr(x, surrogate.kernel.evalGrad(FCenterVal, x, center, alpha))
            F   = lambda x: model.getF(x, mu(x))
            rhs = lambda x: model.getRHS(x, mu(x))
            surrogate.fit(F, rhs, surrogate.center, iterativ=False)
            maxFGreedyError = np.max(np.abs(surrogate.kernel.getGramPDE(F, testPoint, surrogate.center) @ surrogate.alpha - rhs(testPoint)))
        FCenterVal = F(surrogate.center).copy()
        center     = surrogate.center.copy()
        alpha      = surrogate.alpha.copy()
        costCum    = 0
        for i in range(VFinputPerf.shape[1]):
            srGrad     = lambda x: surrogate.kernel.evalGrad(FCenterVal, np.atleast_2d(x).T, center, alpha)[:, 0]
            _, _, cost = model.solveSurrogate(VFinputPerf[:, i], 50, srGrad)
            costCum    = np.max([costCum, np.abs(cost - VFoutputTruePerf[i])])
        observer.addPerfomErrorQuadKernel(costCum)
        print(costCum)
        trueError = np.max(np.abs(VFoutputTrue - surrogate.kernel.evalFunc(FCenterVal, VFinput, center, alpha)))
        evalSR    = surrogate.kernel.evalFunc(FCenterVal, testPoint, center, alpha)
        isPos     = np.min(evalSR) >= 0
        observer.addQuadraticQuadKernel(np.min(evalSR / np.sum(testPoint ** 2, 0)), np.max(evalSR / np.sum(testPoint ** 2, 0)))
        hasLowerBound = np.min(evalSR - 0.5 * model.lambdaMin * np.sum(testPoint ** 2, 0)) >= 0
        hasUpperBound = np.max(evalSR - 2 * model.lambdaMax * np.sum(testPoint ** 2, 0)) <= 0
        print(str(j) + ' Iteration: True-Error = ' + str(trueError) + ', Residual-Error = ' + str(maxFGreedyError) + ', stagnation-Error = ' + str(np.max(np.abs(oldSRVal - surrogate.kernel.evalFunc(FCenterVal, testPoint, center, alpha)))) + ', is positive = ' + str(isPos) + ', Lower Bound = ' + str(hasLowerBound) + ', Upper Bound = ' + str(hasUpperBound))
        observer.addTrueErrorQuadKernel(trueError)
        observer.addObjectQuadKernel(0, maxFGreedyError, np.max(np.abs(oldSRVal - surrogate.kernel.evalFunc(FCenterVal, testPoint, center, alpha))))
        oldSRVal = surrogate.kernel.evalFunc(FCenterVal, testPoint, center, alpha)
    surrogate.FCenterVal = FCenterVal

def findBestGammaProductKernel(model, surrogate, nMaxGreedy, observer, testPoint, gammaList):
    """Test a list of gamma values and return the best product-kernel residual."""
    gammaListVal = 1000 + np.zeros(len(gammaList))
    for i in range(len(gammaList)):
        surrogate.kernel.setGamma(gammaList[i])
        mu  = lambda x: model.stableControl(x)
        F   = lambda x: model.getF(x, mu(x))
        rhs = lambda x: model.getRHS(x, mu(x))
        surrogate.doFGreedy(F, rhs, testPoint, nMaxGreedy, observer, 10 ** (-8))
        surrogate.C     = np.array([])
        gammaListVal[i] = surrogate.trainError[-1]
        bestGammaIndex  = np.argmin(gammaListVal)
        print('Gamma: ' + str(gammaList[i]) + ', Error: ' + str(gammaListVal[i]))
        print('The current best Gamma is: ' + str(gammaList[bestGammaIndex]) + ' with a value of ' + str(gammaListVal[bestGammaIndex]))
    bestGammaIndex = np.argmin(gammaListVal)
    print('The best Gamma is: ' + str(gammaList[bestGammaIndex]) + ' with a value of ' + str(gammaListVal[bestGammaIndex]))
    return (gammaList[bestGammaIndex], gammaListVal[bestGammaIndex])
