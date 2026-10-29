"""Simulations of the full model: experts (absence vs substitution, reserve, recovery) and learners (single session,
months of practice). Usage: python -I sim_substitution.py <out_dir> [fast]
All results are written to <out_dir>/sim_results.json; expert states are cached in <out_dir>/experts_*.npz.
"""
import sys, os, json, time
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Params, Population, make_task, z

OUT = sys.argv[1]; FAST = len(sys.argv) > 2
SEED = 20261027
N_EXP = 60 if FAST else 150
N_LEARN = 200 if FAST else 400
EMP_SD_PER_YEAR = {'adult': 0.727 * 6.9487 / 304.33, 'older': 1.832 * 6.9487 / 304.33}   # from fit_absence (pp/yr -> SD/yr)

def train_experts(p, tau_d_days, seed=SEED, n=N_EXP):
    path = f"{OUT}/experts_{int(tau_d_days)}_{n}.npz"
    rng = np.random.default_rng(seed); B, X, T = make_task(p, rng)
    years = rng.uniform(5, 20, n); nu = np.clip(np.round(rng.lognormal(np.log(8), 0.5, n)), 2, 20).astype(int)
    eta_scale = rng.lognormal(0.0, 0.3, n)
    pop = Population(p, rng, B, n, eta_scale=eta_scale, tau_d=np.full(n, tau_d_days))
    if os.path.exists(path):
        d = np.load(path); pop.load({k: d[k] for k in ('u', 's', 'gI', 'gA')}); return pop, X, T, nu
    total = int(np.ceil(years.max() * 365)); start = total - np.round(years * 365).astype(int)
    for day in range(total):
        act = day >= start
        for k in range(int(nu[act].max()) if act.any() else 0):
            pop.trials('unaided', act & (nu > k))
        pop.consolidate(1.0)
    np.savez(path, **pop.state()); return pop, X, T, nu

def sd_units(pop, X, T, z0, sd0):
    return float(np.mean(z(pop.unaided(X, T)) - z0) / sd0)

def absence_loss(p, tau_d_days, days=365):
    pop, X, T, nu = train_experts(p, tau_d_days)
    st = pop.state(); z0 = z(pop.unaided(X, T)); sd0 = z0.std(ddof=1)
    pop.consolidate(days); return -sd_units(pop, X, T, z0, sd0)

def calibrate_tau_d(p, target_sd_per_year):
    # loss over 365 days is monotone decreasing in tau_d; search on a log grid then refine by interpolation
    grid = [8, 16, 32, 64, 128, 256, 512]
    vals = [absence_loss(p, y * 365) for y in grid]
    lg = np.log(grid); f = np.array(vals) - target_sd_per_year
    i = next(k for k in range(len(grid) - 1) if f[k] * f[k + 1] <= 0)
    y = float(np.exp(lg[i] + (lg[i + 1] - lg[i]) * f[i] / (f[i] - f[i + 1])))
    return y, dict(zip(grid, vals))

def run_regime(pop, X, T, nu, z0, sd0, days, mode, f=0.0, q=0.9, kappa=0.2, record_every=30):
    """Run `days` of practice. Each trial is unaided with probability f (practice reserve), else in `mode`."""
    traj = []; rng = pop.rng
    for d in range(days):
        if mode == 'absence':
            pop.consolidate(1.0)
        else:
            for k in range(int(nu.max())):
                act = nu > k
                if not act.any():
                    continue
                res = rng.random(pop.n) < f
                pop.trials('unaided', act & res)
                pop.trials(mode, act & ~res, q=q, kappa=kappa)
            pop.consolidate(1.0)
        if (d + 1) % record_every == 0:
            traj.append((d + 1, sd_units(pop, X, T, z0, sd0), float(pop.gI.mean())))
    return traj

def recovery(pop, X, T, nu, z0, sd0, max_days=365):
    """Unaided practice after a regime; days to recover 50% and 90% of the loss."""
    start = sd_units(pop, X, T, z0, sd0); out = {'start_sd': start, 'curve': []}
    for d in range(1, max_days + 1):
        for k in range(int(nu.max())):
            pop.trials('unaided', nu > k)
        pop.consolidate(1.0)
        v = sd_units(pop, X, T, z0, sd0); out['curve'].append((d, v))
        for frac in (0.5, 0.9):
            key = f'days_to_{int(frac * 100)}'
            if key not in out and start < 0 and v >= start * (1 - frac):
                out[key] = d
        if 'days_to_90' in out:
            break
    return out

def learners_session(p, n, mode, kappa, q=0.9, n_trials=30, seed=SEED + 7):
    """One practice session of n_trials in `mode`, then an unaided test. The same seed gives the same learners, prior
    practice and stimulus sequence in every condition (common random numbers)."""
    rng = np.random.default_rng(seed); B, X, T = make_task(p, rng)
    eta_scale = rng.lognormal(0.0, 0.4, n)
    pop = Population(p, rng, B, n, eta_scale=eta_scale)
    prior = rng.integers(120, 361, n)                   # days of earlier practice at 5 trials/day: heterogeneous prior skill
    for d in range(int(prior.max())):
        act = d >= prior.max() - prior
        for k in range(5):
            pop.trials('unaided', act)
        pop.consolidate(1.0)
    sess_rng = np.random.default_rng(seed + 1000); pop.rng = sess_rng
    for _ in range(n_trials):
        pop.trials(mode, np.ones(n, bool), q=q, kappa=kappa)
    acc = pop.unaided(X, T)
    gI = pop.gI.copy(); pop.gI[:] = 1.0 - p.g_floor; acc_rep = pop.unaided(X, T); pop.gI = gI
    return acc, acc_rep, float(np.mean(gI))

def cohen_d(a, b):
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s)

if __name__ == '__main__':
    t0 = time.time(); p = Params(); R = {'params': p.asdict(), 'seed': SEED, 'n_experts': N_EXP, 'n_learners': N_LEARN}
    # 1. learners: calibrate the gain learning rate eta_g so that one session with an answer-supplying assistant
    #    (kappa = 0.2) gives the pooled learner estimate for answer-supplying assistants, d = -0.27
    KAPPA = 0.2; target = -0.27
    def d_session(eta_g, kappa=KAPPA, mode='substitute'):
        pp = Params(eta_g=eta_g)
        a, ar, g = learners_session(pp, N_LEARN, mode, kappa); b, br, _ = learners_session(pp, N_LEARN, 'unaided', 1.0)
        return cohen_d(a, b), cohen_d(ar, br), g, float(b.mean()), float(a.mean())
    eg_grid = [0.0, 0.002, 0.005, 0.01, 0.02, 0.04]
    dg = {e: d_session(e)[0] for e in eg_grid}
    eg = None
    for a_, b_ in zip(eg_grid[:-1], eg_grid[1:]):
        if (dg[a_] - target) * (dg[b_] - target) <= 0:
            eg = brentq(lambda e: d_session(e)[0] - target, a_, b_, xtol=1e-4); break
    p = Params(eta_g=eg)
    d_sub, d_rep, g_end, acc_b, acc_s = d_session(eg)
    d_tut = d_session(eg, 1.0, 'tutor')[0]
    sens = {k: d_session(eg, k)[:2] for k in (0.0, 0.2, 0.5, 1.0)}
    def eg_for(tgt):
        try:
            return brentq(lambda e: d_session(e)[0] - tgt, 0.0, 0.04, xtol=1e-4)
        except ValueError:
            return None
    R['learners'] = {'kappa': KAPPA, 'target_d': target, 'd_by_eta_g': dg, 'eta_g_calibrated': eg,
                     'eta_g_range_for_ci': [eg_for(-0.09), eg_for(-0.45)],
                     'd_substitute': d_sub, 'd_substitute_representation_only': d_rep, 'gI_after_session': g_end,
                     'd_tutor': d_tut, 'acc_unaided_mean': acc_b, 'acc_sub_mean': acc_s,
                     'kappa_sensitivity_total_and_representation': sens}
    R['params'] = p.asdict(); kap = KAPPA
    print('learners', R['learners'], round(time.time() - t0))
    # 2. experts: calibrate tau_d to the empirical adult absence loss (SD units per year)
    ty, grid_vals = calibrate_tau_d(p, EMP_SD_PER_YEAR['adult'])
    R['experts'] = {'tau_d_years_adult': ty, 'calibration_grid': grid_vals, 'target_sd_per_year': EMP_SD_PER_YEAR}
    pop, X, T, nu = train_experts(p, ty * 365); st = pop.state()
    z0 = z(pop.unaided(X, T)); sd0 = z0.std(ddof=1)
    R['experts']['baseline_acc_mean'] = float(pop.unaided(X, T).mean())
    def rep_only(pop):
        g = pop.gI.copy(); pop.gI[:] = 1.0 - p.g_floor; v = sd_units(pop, X, T, z0, sd0); pop.gI = g; return v
    EG = {'none': 0.0, 'central': R['learners']['eta_g_calibrated'], 'upper': R['learners']['eta_g_range_for_ci'][1]}
    R['experts']['eta_g_values'] = EG
    regimes = {'absence': dict(mode='absence'), 'tutor': dict(mode='tutor'),
               'substitution_f0': dict(mode='substitute', f=0.0), 'substitution_f20': dict(mode='substitute', f=0.20)}
    R['experts']['traj'] = {}
    for egname, egv in EG.items():
        pop.p = Params(eta_g=egv)
        for name, kw in regimes.items():
            if egname != 'central' and name in ('absence', 'tutor'):
                continue
            pop.load(st); pop.rng = np.random.default_rng(SEED + 11)
            tr = run_regime(pop, X, T, nu, z0, sd0, 365, kappa=kap, **kw)
            R['experts']['traj'][f'{name}|{egname}'] = {'traj': tr, 'representation_only_end': rep_only(pop)}
            print(egname, name, tr[-1], round(time.time() - t0))
    # recovery after one year of full substitution (central eta_g): total and representation-only
    pop.p = Params(eta_g=EG['central']); pop.load(st); pop.rng = np.random.default_rng(SEED + 11)
    run_regime(pop, X, T, nu, z0, sd0, 365, mode='substitute', f=0.0, kappa=kap)
    rec = {'start_total': sd_units(pop, X, T, z0, sd0), 'start_rep': rep_only(pop), 'curve': []}
    for d in range(1, 731):
        for k in range(int(nu.max())):
            pop.trials('unaided', nu > k)
        pop.consolidate(1.0)
        if d % 10 == 0:
            rec['curve'].append((d, sd_units(pop, X, T, z0, sd0), rep_only(pop), float(pop.gI.mean())))
    tot = [c[1] for c in rec['curve']]
    for frac in (0.5, 0.9):
        hit = [c[0] for c in rec['curve'] if c[1] >= rec['start_total'] * (1 - frac)]
        rec[f'days_to_{int(frac * 100)}'] = hit[0] if hit else None
    R['experts']['recovery_after_substitution'] = rec
    pop.p = Params(eta_g=EG['central'])
    # 3. reserve: steady state (3 years) gain share and unaided change vs f and q, against the analytic fixed point
    R['reserve'] = []
    for q in (0.75, 0.9, 0.99):
        for f in (0.0, 0.1, 0.2, 0.3, 0.5):
            pop.load(st); pop.rng = np.random.default_rng(SEED + 13)
            tr = run_regime(pop, X, T, nu, z0, sd0, 1095, mode='substitute', f=f, q=q, kappa=kap, record_every=365)
            H = (pop.u @ X.T) / np.sqrt(p.N); cI = float(np.mean(np.mean(np.sign(H) == T[None, :], axis=1) * 2 - 1))
            E = f + (1 - f) * kap; cA = 2 * q - 1
            R['reserve'].append({'q': q, 'f': f, 'sd_change_by_year': [x[1] for x in tr], 'gI_by_year': [x[2] for x in tr],
                                 'gI_analytic': E * cI / (E * cI + (1 - f) * cA), 'cI': cI})
        print('reserve q', q, round(time.time() - t0))
    # 4. learners over months: never-skilling
    R['learners_months'] = {}
    for name, kw in {'unaided': dict(mode='unaided'), 'tutor': dict(mode='tutor'), 'substitute': dict(mode='substitute'),
                     'substitute_f20': dict(mode='substitute', f=0.2)}.items():
        rng = np.random.default_rng(SEED + 21); B2, X2, T2 = make_task(p, rng)
        lp = Population(p, rng, B2, N_LEARN // 2, eta_scale=rng.lognormal(0, 0.4, N_LEARN // 2)); nu2 = np.full(lp.n, 5)
        traj = []
        for d in range(180):
            for k in range(5):
                res = rng.random(lp.n) < kw.get('f', 0.0)
                if kw['mode'] == 'substitute':
                    lp.trials('unaided', res); lp.trials('substitute', ~res, kappa=kap)
                else:
                    lp.trials(kw['mode'], np.ones(lp.n, bool))
            lp.consolidate(1.0)
            if (d + 1) % 15 == 0:
                acc = lp.unaided(X2, T2); traj.append((d + 1, float(acc.mean()), float(acc.std(ddof=1)), float(lp.gI.mean())))
        R['learners_months'][name] = traj
    R['seconds'] = time.time() - t0
    json.dump(R, open(f'{OUT}/sim_results.json', 'w'), indent=1)
    print('done', round(R['seconds']))
