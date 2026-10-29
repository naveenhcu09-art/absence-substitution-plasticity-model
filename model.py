"""Use-dependent plasticity under absence and substitution: a two-pathway model with synaptic consolidation.

A population of learners is simulated in parallel. Each learner's internal pathway maps stimuli x (N-dim, standard
normal) to a binary response through synaptic weights with two coupled timescales: an expressed weight u that is
changed by practice, and a consolidated weight s that tracks u slowly and decays very slowly (a two-variable
reduction of cascade/chain models of synaptic consolidation). A decision neuron combines the internal pathway with an
external advice input; the two input gains share a fixed synaptic resource and are updated by an outcome-modulated
(three-factor) rule. Time unit: days.

Modes of a practice trial
  'unaided'    the learner generates a response; the correct answer is then revealed (feedback)
  'tutor'      as unaided: hints may be given but the answer does not reach the decision neuron
  'substitute' an advice input supplying an answer (correct with probability q) reaches the decision neuron;
               the internal pathway is engaged only at level kappa (0 <= kappa <= 1)
Absence: no trials; only consolidation and decay run.
"""
import numpy as np
from scipy.linalg import expm
from scipy.stats import norm

DEFAULTS = dict(N=200, eta=0.05, eta_g=0.01, tau_u=5.0, tau_s=30.0, tau_d=3650.0, sigma=0.1, g_floor=1e-3)

class Params:
    def __init__(self, **kw):
        for k, v in {**DEFAULTS, **kw}.items():
            setattr(self, k, v)
    def asdict(self):
        return {k: getattr(self, k) for k in DEFAULTS}

def consolidation_matrix(p, days, tau_d=None):
    td = p.tau_d if tau_d is None else tau_d
    A = np.array([[-1 / p.tau_u, 1 / p.tau_u], [1 / p.tau_s, -1 / p.tau_s - 1 / td]])
    return expm(A * days)

def make_task(p, rng, n_test=2000):
    B = rng.standard_normal(p.N); B *= np.sqrt(p.N) / np.linalg.norm(B)
    X = rng.standard_normal((n_test, p.N)); T = np.where(X @ B >= 0, 1.0, -1.0)
    return B, X, T

class Population:
    def __init__(self, p, rng, B, n, eta_scale=None, tau_d=None):
        self.p, self.rng, self.B, self.n = p, rng, B, n
        self.u = np.zeros((n, p.N)); self.s = np.zeros((n, p.N))
        self.gI = np.ones(n); self.gA = np.full(n, p.g_floor)
        self._norm_gains()
        self.eta = p.eta * (np.ones(n) if eta_scale is None else np.asarray(eta_scale))
        self.tau_d = np.full(n, p.tau_d) if tau_d is None else np.asarray(tau_d, dtype=float)
        self._M = {}

    def _norm_gains(self):
        self.gI = np.maximum(self.gI, self.p.g_floor); self.gA = np.maximum(self.gA, self.p.g_floor)
        S = self.gI + self.gA; self.gI /= S; self.gA /= S

    def trials(self, mode, active, q=0.9, kappa=0.2):
        """One practice trial for every learner flagged in `active` (boolean array)."""
        p = self.p; N = p.N
        if not active.any():
            return
        idx = np.where(active)[0]
        x = self.rng.standard_normal((len(idx), N))
        t = np.where(x @ self.B >= 0, 1.0, -1.0)
        h = np.einsum('ij,ij->i', self.u[idx], x) / np.sqrt(N)
        e = kappa if mode == 'substitute' else 1.0
        self.u[idx] += ((self.eta[idx] / np.sqrt(N)) * e * (t - np.tanh(h)))[:, None] * x
        vI = np.where(h >= 0, 1.0, -1.0)
        self.gI[idx] += p.eta_g * e * np.where(vI == t, 1.0, -1.0)
        if mode == 'substitute':
            a = np.where(self.rng.random(len(idx)) < q, t, -t)
            self.gA[idx] += p.eta_g * np.where(a == t, 1.0, -1.0)
        self._norm_gains()

    def consolidate(self, days, active=None):
        """Advance consolidation/decay by `days` for learners flagged in `active` (all by default)."""
        act = np.ones(self.n, bool) if active is None else active
        for td in np.unique(self.tau_d[act]):
            key = (round(days, 6), td)
            if key not in self._M:
                self._M[key] = consolidation_matrix(self.p, days, td)
            M = self._M[key]; sel = act & (self.tau_d == td)
            u, s = self.u[sel], self.s[sel]
            self.u[sel], self.s[sel] = M[0, 0] * u + M[0, 1] * s, M[1, 0] * u + M[1, 1] * s

    def practice_days(self, days, mode, nu, active=None, **kw):
        """Practise for `days` days at nu trials per day (nu: scalar or per-learner array)."""
        nu = np.broadcast_to(np.asarray(nu), (self.n,))
        act = np.ones(self.n, bool) if active is None else active
        for _ in range(int(days)):
            for k in range(int(nu[act].max()) if act.any() else 0):
                self.trials(mode, act & (nu > k), **kw)
            self.consolidate(1.0, act)

    def unaided(self, X, T):
        """Expected unaided accuracy of each learner on the test set (advice input absent)."""
        H = (self.u @ X.T) / np.sqrt(self.p.N)
        return norm.cdf(T[None, :] * self.gI[:, None] * H / self.p.sigma).mean(axis=1)

    def state(self):
        return dict(u=self.u.copy(), s=self.s.copy(), gI=self.gI.copy(), gA=self.gA.copy())

    def load(self, st):
        self.u, self.s, self.gI, self.gA = st['u'].copy(), st['s'].copy(), st['gI'].copy(), st['gA'].copy()

def z(pacc):
    return norm.ppf(np.clip(pacc, 1e-9, 1 - 1e-9))
