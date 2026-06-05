"""Surrogate model classes that fit RKHS value-function approximations by greedy center selection."""
import pickle
import numpy as np

class Surrogate:
    """Standard radial-kernel RKHS surrogate for the value function."""

    def __init__(self, kernel):
        """Initialize the object and precompute the quantities used later."""
        self.trainError = []
        self.alpha      = []
        self.center     = []
        self.kernel     = kernel
        self.C          = np.array([])

    def saveSr(self, name):
        """Save the surrogate object to disk."""
        with open('data/' + name, 'wb') as outp:
            pickle.dump(self, outp, protocol=4)

    def loadSr(self, name):
        """Load surrogate data from disk."""
        with open('data/' + name, 'rb') as inp:
            oldSelf         = pickle.load(inp)
            self.trainError = oldSelf.trainError
            self.alpha      = oldSelf.alpha
            self.center     = oldSelf.center
            self.kernel     = oldSelf.kernel
            self.C          = oldSelf.C
            self.FCenterVal = oldSelf.FCenterVal

    def doFGreedy(self, F, rhs, testPoints, nMaxGreedy, observer, eps=10 ** (-16), outputFlag=True):
        """Select greedy centers and fit the RKHS surrogate on the GHJB residual."""
        rhsTestPoints       = rhs(testPoints)
        fy                  = F(testPoints)
        KYX                 = np.zeros((testPoints.shape[1], nMaxGreedy + 1))
        center              = np.zeros((testPoints.shape[0], nMaxGreedy + 1))
        idx                 = np.argmax(np.abs(rhs(testPoints)))
        self.trainError     = np.atleast_1d(np.abs(rhs(testPoints))[idx])
        self.center         = np.atleast_2d(testPoints[:, idx]).T
        center[:, 0]        = testPoints[:, idx]
        i                   = 0
        self.trainError[-1] = eps + 1
        while i < nMaxGreedy and self.trainError[-1] > eps:
            self.fit(F, rhs, center[:, :i + 1])
            if i == 0:
                KYX[:, 0] = self.kernel.getGramPDEStartColumn(F, testPoints)[:, 0]
                KYX[:, 1] = self.kernel.getGramPDE(F, testPoints, center[:, :i + 1])[:, 0]
            else:
                KYX[:, i + 1] = self.kernel.getGramPDE(F, testPoints, np.atleast_2d(center[:, i]).T, fy)[:, 0]
            preOut           = KYX[:, :i + 2] @ self.alpha
            res              = np.abs(preOut - rhsTestPoints)
            idx              = np.argmax(res)
            self.trainError  = np.r_[self.trainError, res[idx]]
            center[:, i + 1] = testPoints[:, idx]
            observer.addObjectGreedyErrorStandart(self.trainError[-1])
            if outputFlag:
                print(str(i) + ' Epoch: Error = ' + str(self.trainError[-1]))
            i = i + 1
        self.fit(F, rhs, center)

    def fit(self, F, rhs, center, iterativ=True):
        """Fit the RKHS surrogate coefficients for the current centers."""
        if iterativ:
            if self.C.shape[0] == 0:
                K0          = np.atleast_2d(self.kernel.phi(0))
                K1          = self.kernel.getGramPDEStartColumn(F, center)
                K           = np.c_[np.r_[K0, K1.T], np.r_[K1, self.kernel.getGramPDE(F, center, center)]]
                rhsVal      = np.r_[0, rhs(center)]
                self.alpha  = np.linalg.solve(K, rhsVal)
                self.center = center
                L           = np.linalg.cholesky(K)
                self.C      = np.linalg.solve(L, np.eye(L.shape[0]))
            else:
                rhsVal      = np.r_[0, rhs(center)]
                KX          = np.r_[self.kernel.getGramPDEStartColumn(F, np.atleast_2d(center[:, -1]).T), self.kernel.getGramPDE(F, center[:, :-1], np.atleast_2d(center[:, -1]).T)]
                KXX         = self.kernel.getGramPDE(F, np.atleast_2d(center[:, -1]).T, np.atleast_2d(center[:, -1]).T)
                dm          = self.C @ KX
                dmm         = np.sqrt(KXX - dm.T @ dm)
                cmm         = 1 / dmm
                cm          = -self.C.T @ (dm @ cmm.T)
                self.C      = np.c_[np.r_[self.C, cm.T], np.r_[cm * 0, cmm]]
                self.alpha  = self.C.T @ (self.C @ rhsVal)
                self.center = center
        else:
            K0          = np.atleast_2d(self.kernel.phi(0))
            K1          = self.kernel.getGramPDEStartColumn(F, center)
            K           = np.r_[np.c_[K0, K1.T], np.c_[K1, self.kernel.getGramPDE(F, center, center)]]
            rhsVal      = np.r_[0, rhs(center)]
            self.alpha  = np.linalg.solve(K, rhsVal)
            self.center = center

class SurrogateProductKernel:
    """Product-kernel RKHS surrogate for the value function."""

    def __init__(self, kernel):
        """Initialize the object and precompute the quantities used later."""
        self.trainError = []
        self.alpha      = []
        self.center     = []
        self.kernel     = kernel
        self.C          = np.array([])

    def saveSr(self, name):
        """Save the surrogate object to disk."""
        with open('data/' + name, 'wb') as outp:
            pickle.dump(self, outp, protocol=4)

    def loadSr(self, name):
        """Load surrogate data from disk."""
        with open('data/' + name, 'rb') as inp:
            oldSelf         = pickle.load(inp)
            self.trainError = oldSelf.trainError
            self.alpha      = oldSelf.alpha
            self.center     = oldSelf.center
            self.kernel     = oldSelf.kernel
            self.C          = oldSelf.C
            self.FCenterVal = oldSelf.FCenterVal

    def doFGreedy(self, F, rhs, testPoints, nMaxGreedy, observer, eps=10 ** (-16), outputFlag=True):
        """Select greedy centers and fit the RKHS surrogate on the GHJB residual."""
        rhsTestPoints       = rhs(testPoints)
        fy                  = F(testPoints)
        KYX                 = np.zeros((testPoints.shape[1], nMaxGreedy))
        center              = np.zeros((testPoints.shape[0], nMaxGreedy + 1))
        idx                 = np.argmax(np.abs(rhs(testPoints)))
        self.trainError     = np.atleast_1d(np.abs(rhs(testPoints))[idx])
        self.center         = np.atleast_2d(testPoints[:, idx]).T
        center[:, 0]        = testPoints[:, idx]
        i                   = 0
        self.trainError[-1] = eps + 1
        while i < nMaxGreedy and self.trainError[-1] > eps:
            self.fit(F, rhs, center[:, :i + 1])
            if i == 0:
                KYX[:, 0] = self.kernel.getGramPDE(F, testPoints, center[:, :i + 1])[:, 0]
            else:
                KYX[:, i] = self.kernel.getGramPDE(F, testPoints, np.atleast_2d(center[:, i]).T, fy)[:, 0]
            preOut           = KYX[:, :i + 1] @ self.alpha
            res              = np.abs(preOut - rhsTestPoints)
            idx              = np.argmax(res)
            self.trainError  = np.r_[self.trainError, res[idx]]
            center[:, i + 1] = testPoints[:, idx]
            observer.addObjectGreedyErrorQuadKernel(self.trainError[-1])
            if outputFlag:
                print(str(i) + ' Epoch: Error = ' + str(self.trainError[-1]))
            i = i + 1
        self.fit(F, rhs, center)

    def fit(self, F, rhs, center, iterativ=True):
        """Fit the RKHS surrogate coefficients for the current centers."""
        if iterativ:
            if self.C.shape[0] == 0:
                K           = self.kernel.getGramPDE(F, center, center)
                rhsVal      = rhs(center)
                self.alpha  = np.linalg.solve(K, rhsVal)
                self.center = center
                L           = np.linalg.cholesky(K)
                self.C      = np.linalg.solve(L, np.eye(L.shape[0]))
            else:
                rhsVal      = rhs(center)
                KX          = self.kernel.getGramPDE(F, center[:, :-1], np.atleast_2d(center[:, -1]).T)
                KXX         = self.kernel.getGramPDE(F, np.atleast_2d(center[:, -1]).T, np.atleast_2d(center[:, -1]).T)
                dm          = self.C @ KX
                dmm         = np.sqrt(KXX - dm.T @ dm)
                cmm         = 1 / dmm
                cm          = -self.C.T @ (dm @ cmm.T)
                self.C      = np.c_[np.r_[self.C, cm.T], np.r_[cm * 0, cmm]]
                self.alpha  = self.C.T @ (self.C @ rhsVal)
                self.center = center
        else:
            K           = self.kernel.getGramPDE(F, center, center)
            rhsVal      = rhs(center)
            self.alpha  = np.linalg.solve(K, rhsVal)
            self.center = center
