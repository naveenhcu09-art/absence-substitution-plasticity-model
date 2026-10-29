"""Fit the absence branch of the model to within-player change in chess performance after absences.

Reduction used for fitting (derived in the paper, checked against full simulation in check_reduction.py).
For a long-practised expert near steady state, the expressed weight u and consolidated weight s along the learned
direction satisfy s/u = phi = tau_d / (tau_d + tau_s). During absence (no practice), (u, s) evolve by the linear
consolidation dynamics, so the expressed skill relative to its pre-absence level is
    r(t) = M11(t) + M12(t) * phi,     M(t) = expm(A t),  A = [[-1/tau_u, 1/tau_u], [1/tau_s, -1/tau_s - 1/tau_d]].
Unaided performance is approximately linear in expressed skill over the observed range, so the predicted change is
    Delta(t) = -lambda * (1 - r(t)).
When tau_d is much longer than the observation window, lambda and tau_d enter only through the late loss rate b
(pp per year) and the early curvature, set by tau_s. We therefore fit b and tau_s (tau_u fixed
at 5 days), with lambda fixed at a large value; results are invariant to lambda (checked below).
Usage: python -I fit_absence.py <fide_results.json> <fide_bin_mean_gap.csv> <out_dir>
"""
import sys, json
import numpy as np, pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

RES, GAPS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
fr = json.load(open(RES)); gaps = pd.read_csv(GAPS)
DPM = 30.44; TAU_U = 5.0
BINS = ['1-2', '3-5', '6-11', '12-23', '24-47', '48+']

def data_for(group):
    rows = {r['gapb']: r for r in fr['primary_by_agegrp'] if r['agegrp'] == group}
    g = gaps[gaps.agegrp == group].set_index('gapb')['gap']
    return pd.DataFrame({'bin': BINS, 'months': [float(g[b]) for b in BINS], 'did': [rows[b]['did'] for b in BINS],
                         'se': [rows[b]['did_se'] for b in BINS], 'players': [rows[b]['players'] for b in BINS]})

def delta(months, b_pp_per_year, tau_s, lam=200.0, tau_u=TAU_U):
    # Parameterised by the late (slow-mode) loss rate b: the slow eigenvalue of the consolidation dynamics is
    # approximately (1/tau_d) * tau_s / (tau_u + tau_s) when tau_d >> tau_s, so tau_d is set to give slope b.
    tau_d = lam / b_pp_per_year * 365.0 * tau_s / (tau_u + tau_s)   # days
    A = np.array([[-1 / tau_u, 1 / tau_u], [1 / tau_s, -1 / tau_s - 1 / tau_d]])
    phi = tau_d / (tau_d + tau_s)
    out = []
    for m in np.atleast_1d(months):
        M = expm(A * m * DPM)
        out.append(-lam * (1 - (M[0, 0] + M[0, 1] * phi)))
    return np.array(out)

B_BOUNDS, LTS_BOUNDS = (0.05, 10.0), (np.log(TAU_U), np.log(1500.0))   # pp/year; days (consolidated variable slower than expressed)

def chi2(theta, D, lam=200.0):
    b, log_ts = theta
    if not (B_BOUNDS[0] <= b <= B_BOUNDS[1]) or not (LTS_BOUNDS[0] - 1e-9 <= log_ts <= LTS_BOUNDS[1] + 1e-9):
        return 1e9
    pred = delta(D.months.values, b, np.exp(log_ts), lam)
    return float(np.sum(((D.did.values - pred) / D.se.values) ** 2))

def fit(D, tau_s_fixed=None):
    if tau_s_fixed is None:
        best = None
        for ts0 in [15, 45, 90, 180, 365]:
            r = minimize(chi2, x0=[1.0, np.log(ts0)], args=(D,), method='L-BFGS-B', bounds=[B_BOUNDS, LTS_BOUNDS])
            best = r if best is None or r.fun < best.fun else best
        return best.x[0], float(np.exp(best.x[1])), best.fun
    r = minimize(lambda th: chi2([th[0], np.log(tau_s_fixed)], D), x0=[1.0], method='L-BFGS-B', bounds=[B_BOUNDS])
    return r.x[0], tau_s_fixed, r.fun

def profile_ts(D, grid):
    return [(ts, fit(D, ts)[2]) for ts in grid]

if __name__ == '__main__':
    res = {'tau_u_days': TAU_U}
    TS = np.exp(np.linspace(np.log(TAU_U), np.log(1500), 60))
    for group in ['20-59', '60 and over']:
        D = data_for(group)
        b, ts, c = fit(D)
        prof = profile_ts(D, TS)
        cmin = min(cc for _, cc in prof + [(ts, c)])
        inside = [t for t, cc in prof if cc - cmin <= 3.84]
        # linear-only comparison model (no fast phase): Delta = -b t
        lin = minimize(lambda th: float(np.sum(((D.did.values + th[0] * D.months.values / 12) / D.se.values) ** 2)), x0=[1.0], method='Nelder-Mead')
        lam_check = {lam: float(chi2([b, np.log(ts)], D, lam)) for lam in [50.0, 200.0, 1000.0]}
        res[group] = {'b_pp_per_year': float(b), 'tau_s_days': float(ts), 'chi2': float(c), 'df': len(D) - 2,
                      'tau_s_ci95_profile': [float(min(inside)), float(max(inside))],
                      'linear_only_b': float(lin.x[0]), 'linear_only_chi2': float(lin.fun),
                      'lambda_invariance_chi2': lam_check, 'data': D.to_dict('records'),
                      'pred': delta(D.months.values, b, ts).tolist(), 'profile': [(float(a), float(bb)) for a, bb in prof]}
        # bootstrap-free CI for b: profile with tau_s at its estimate
        bs = np.linspace(0.2, 4, 381); cb = [chi2([bb, np.log(ts)], D) for bb in bs]
        res[group]['b_ci95_profile'] = [float(bs[i]) for i in (min(i for i, v in enumerate(cb) if v - min(cb) <= 3.84), max(i for i, v in enumerate(cb) if v - min(cb) <= 3.84))]
        print(group, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[group].items() if k in ('b_pp_per_year', 'tau_s_days', 'chi2', 'df', 'tau_s_ci95_profile', 'b_ci95_profile', 'linear_only_b', 'linear_only_chi2', 'lambda_invariance_chi2')})
    # joint model: shared tau_s, group-specific b  vs  fully separate
    DA, DO = data_for('20-59'), data_for('60 and over')
    def joint(th):
        bA, bO, lts = th
        return chi2([bA, lts], DA) + chi2([bO, lts], DO)
    best = None
    for ts0 in [15, 45, 90, 180, 365]:
        r = minimize(joint, x0=[1.0, 2.0, np.log(ts0)], method='L-BFGS-B', bounds=[B_BOUNDS, B_BOUNDS, LTS_BOUNDS])
        best = r if best is None or r.fun < best.fun else best
    bA, bO, lts = best.x
    # constrained: shared b (age difference must come from tau_s only)
    best2 = None
    for ts0 in [15, 45, 90, 180, 365]:
        for ts1 in [15, 90, 365]:
            r = minimize(lambda th: chi2([th[0], th[1]], DA) + chi2([th[0], th[2]], DO), x0=[1.5, np.log(ts0), np.log(ts1)], method='L-BFGS-B', bounds=[B_BOUNDS, LTS_BOUNDS, LTS_BOUNDS])
            best2 = r if best2 is None or r.fun < best2.fun else best2
    res['joint_shared_tau_s'] = {'b_adult': float(bA), 'b_older': float(bO), 'ratio_older_adult': float(bO / bA), 'tau_s_days': float(np.exp(lts)), 'chi2': float(best.fun), 'df': 12 - 3}
    res['joint_shared_b'] = {'b': float(best2.x[0]), 'tau_s_adult': float(np.exp(best2.x[1])), 'tau_s_older': float(np.exp(best2.x[2])), 'chi2': float(best2.fun), 'df': 12 - 3}
    res['empirical_regression_slopes_pp_per_year'] = {'20-59': fr['slope_regression']['gy'], '60 and over': fr['slope_60plus']}
    json.dump(res, open(OUT + '/fit_absence.json', 'w'), indent=1)
    print('joint shared tau_s', {k: round(v, 3) for k, v in res['joint_shared_tau_s'].items()})
    print('joint shared b', {k: round(v, 3) for k, v in res['joint_shared_b'].items()})
