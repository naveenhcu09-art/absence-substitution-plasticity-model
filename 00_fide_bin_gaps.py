"""Study 1: chess performance after absences from rated play (FIDE monthly standard lists, Jul 2012 - Dec 2019).
Usage: python -I 02_fide_layoff.py <panel.parquet> <out_dir>

Outcome: per-game score residual s = dR / (K x G) for the month (K = development coefficient on the preceding list),
relative to continuously active players in the same month, 100-point rating band, age band and games band (pp of score).
Estimators of the change associated with a layoff of g months (bounds, from most to least favourable to "no loss"):
  (i)  one-month baseline: return residual minus the residual in the last active month before the layoff;
  (ii) PRIMARY: return residual minus the mean residual in the 2nd-6th previous active months (>= 3 months; excludes the
       last pre-layoff month, which shows a pre-exit dip), differenced against continuously active players' identical
       quantity (difference-in-differences);
  (iii) cross-sectional residual on return (includes all selection into absence).
"""
import sys, json
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
PANEL, OUT = sys.argv[1], sys.argv[2]  # same construction as the Absence and Substitution analysis
P = pd.read_parquet(PANEL, columns=['id', 't', 'rating', 'games', 'k', 'byear', 'sex'])
P = P.astype({'id': 'int64', 't': 'int16', 'rating': 'int16', 'games': 'int16', 'k': 'float32', 'byear': 'float32'})
P = P.sort_values(['id', 't'], kind='stable').reset_index(drop=True)
same = P['id'].values[1:] == P['id'].values[:-1]
def lagv(col):
    v = P[col].values.astype('float64'); out = np.full(len(v), np.nan); out[1:] = np.where(same, v[:-1], np.nan); return out
P['r_prev'] = lagv('rating'); P['t_prev'] = lagv('t'); P['k_prev'] = lagv('k')
A = P.loc[P.games > 0, ['id', 't']]
sa = A['id'].values[1:] == A['id'].values[:-1]
la = np.full(len(A), np.nan); la[1:] = np.where(sa, A['t'].values[:-1], np.nan)
P['last_act'] = np.nan; P.loc[A.index, 'last_act'] = la
P['n_act_before'] = (P['games'] > 0).astype('int32').groupby(P['id']).cumsum() - (P['games'] > 0).astype('int32')
E = P[(P.games > 0) & (P.t_prev == P.t - 1) & P.k_prev.notna() & (P.k_prev > 0) & P.last_act.notna()].copy()
endt = int(P['t'].max())
act_sd = P[(P.games > 0)].assign(age=lambda d: 2012 + d.t // 12 - d.byear)
rating_sd_adults = float(act_sd[(act_sd.age >= 20) & (act_sd.age <= 59)].groupby('t')['rating'].std().mean())
del P, A, act_sd
E['gap'] = (E['t'] - E['last_act'] - 1).astype(int)
E['s'] = (E['rating'] - E['r_prev']) / (E['k_prev'] * E['games'])
E = E[np.abs(E['s']) <= 1]
E['age'] = (2012 + E['t'] // 12) - E['byear']; E = E[(E.age >= 6) & (E.age <= 90)]
age_edges = [5, 12, 15, 19, 24, 39, 59, 90]; age_lab = ['6-12', '13-15', '16-19', '20-24', '25-39', '40-59', '60+']
E['ageb'] = pd.cut(E.age, age_edges, labels=age_lab)
E['agegrp'] = np.select([E.age < 20, E.age >= 60], ['under 20', '60 and over'], '20-59')
E['rb'] = (E['r_prev'] // 100).astype(int); E['gb'] = pd.cut(E.games, [0, 2, 5, 9, 1000], labels=False)
ctrl = E[E.gap == 0].groupby(['t', 'ageb', 'rb', 'gb'], observed=True)['s'].mean().rename('s_exp')
E = E.join(ctrl, on=['t', 'ageb', 'rb', 'gb']).dropna(subset=['s_exp'])
E['res'] = (E['s'] - E['s_exp']) * 100
E = E.sort_values(['id', 't']).reset_index(drop=True).rename(columns={'id': 'pid'})
h = E.groupby('pid')['res']
E['last1'] = h.shift(1)
lag = pd.concat([h.shift(k) for k in range(2, 7)], axis=1)
E['base26'] = lag.mean(axis=1).where(lag.notna().sum(axis=1) >= 3)
E['r_base_start'] = E.groupby('pid')['r_prev'].shift(6)          # rating entering the earliest baseline month (for elite subset)
gap_edges = [-1, 0, 2, 5, 11, 23, 47, 200]; gap_lab = ['0', '1-2', '3-5', '6-11', '12-23', '24-47', '48+']
E['gapb'] = pd.cut(E.gap, gap_edges, labels=gap_lab)
ELO = 400 / (np.log(10) * 0.25) / 100

def summ(d, keys, col):
    pl = d.groupby(keys + ['pid'], observed=True)[col].mean().reset_index()
    o = pl.groupby(keys, observed=True)[col].agg(['mean', 'std', 'count']).reset_index()
    o['se'] = o['std'] / np.sqrt(o['count']); return o.rename(columns={'count': 'players'})
def did(d, keys):
    """Primary estimator: change vs months 2-6, minus the same quantity for continuously active players (gap 0) in the same cell."""
    t = summ(d.dropna(subset=['chg']), keys + ['gapb'], 'chg')
    if keys:
        base = t[t.gapb == '0'].set_index(keys)[['mean', 'se']]
        t = t.join(base.rename(columns={'mean': 'm0', 'se': 'se0'}), on=keys)
    else:
        b = t[t.gapb == '0']; t['m0'] = float(b['mean'].iloc[0]); t['se0'] = float(b['se'].iloc[0])
    t['did'] = t['mean'] - t['m0']; t['did_se'] = np.sqrt(t['se'] ** 2 + t['se0'] ** 2)
    t['did_elo'] = t['did'] * ELO; t['lo'] = t['did'] - 1.96 * t['did_se']; t['hi'] = t['did'] + 1.96 * t['did_se']
    return t[t.gapb != '0']
E['chg'] = E['res'] - E['base26']

D = E.dropna(subset=['chg'])
D = D[D.gap > 0]
pl = D.groupby(['agegrp', 'gapb', 'pid'], observed=True)['gap'].mean().reset_index()
m = pl.groupby(['agegrp', 'gapb'], observed=True)['gap'].mean().reset_index()
pl2 = D.groupby(['gapb', 'pid'], observed=True)['gap'].mean().reset_index()
m2 = pl2.groupby('gapb', observed=True)['gap'].mean().reset_index(); m2['agegrp'] = 'all'
out = pd.concat([m, m2]); out.to_csv(OUT + '/fide_bin_mean_gap.csv', index=False); print(out.to_string())
