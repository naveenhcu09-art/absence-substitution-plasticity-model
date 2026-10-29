"""Fill every number in the manuscript template from the result files, so text and outputs cannot drift.
Usage: python -I fill_numbers.py <template.md> <fit_absence.json> <sim_results.json> <fide_results.json> <out.md>"""
import sys, json, re
import numpy as np

TPL, FIT, SIM, FIDE, OUT = sys.argv[1:6]
F = json.load(open(FIT)); S = json.load(open(SIM)); D = json.load(open(FIDE))
A, O = F['20-59'], F['60 and over']; L = S['learners']; E = S['experts']; T = E['traj']

def n(x, k=2):
    s = f"{x:.{k}f}"
    return s.replace('-', '−')
def sg(x, k=2):
    return ('+' if x > 0 else '') + n(x, k)
def ci(v, k=2):
    return f"{n(v[0], k)} to {n(v[1], k)}"
def pct(x):
    return f"{100 * x:.0f}%"

cI = float(np.mean([r['cI'] for r in S['reserve']])); kap = L['kappa']; sc = 0.5
def fstar(q):
    cA = 2 * q - 1
    return (sc * cA - kap * cI * (1 - sc)) / ((1 - kap) * cI * (1 - sc) + sc * cA)
res = {(r['q'], r['f']): r for r in S['reserve']}
LM = S['learners_months']
V = {
    'n_returns': f"{D['primary_n_returns']:,}", 'n_players': f"{D['primary_n_players']:,}",
    'b_adult': n(A['b_pp_per_year']), 'b_older': n(O['b_pp_per_year']),
    'b_adult_ci': ci(A['b_ci95_profile']), 'b_older_ci': ci(O['b_ci95_profile']),
    'b_ratio': n(O['b_pp_per_year'] / A['b_pp_per_year'], 1),
    'chi_adult': n(A['chi2'], 1), 'df_adult': str(A['df']), 'chi_older': n(O['chi2'], 1), 'df_older': str(O['df']),
    'ts_adult_hi': f"{A['tau_s_ci95_profile'][1]:.0f}", 'ts_older_hi': f"{O['tau_s_ci95_profile'][1]:.0f}",
    'chi_lin_adult': n(A['linear_only_chi2'], 1), 'chi_lin_older': n(O['linear_only_chi2'], 1),
    'chi_shared_ts': n(F['joint_shared_tau_s']['chi2'], 1), 'chi_shared_b': n(F['joint_shared_b']['chi2'], 1),
    'n_learners': f"{S['n_learners']:,}", 'n_experts': str(S['n_experts']), 'kappa': n(kap, 1),
    'target_sd': n(E['target_sd_per_year']['adult'], 3), 'tau_d_years': f"{E['tau_d_years_adult']:.0f}",
    'eta_g': f"{L['eta_g_calibrated']:.5f}", 'eta_g_hi': f"{L['eta_g_range_for_ci'][1]:.4f}",
    'd_sub': n(L['d_substitute']), 'd_rep': n(L['d_substitute_representation_only']),
    'd_k0': n(L['kappa_sensitivity_total_and_representation']['0.0'][0]),
    'd_k1': n(L['kappa_sensitivity_total_and_representation']['1.0'][0]),
    'd_tutor': n(L['d_tutor']),
    'abs_12': n(T['absence|central']['traj'][-1][1], 3), 'tut_12': sg(T['tutor|central']['traj'][-1][1]),
    'sub_none_12': sg(T['substitution_f0|none']['traj'][-1][1]),
    'gI_sub_central': n(T['substitution_f0|central']['traj'][-1][2]),
    'sub_central_12': n(-T['substitution_f0|central']['traj'][-1][1], 1),
    'sub_upper_12': n(-T['substitution_f0|upper']['traj'][-1][1], 1),
    'sub20_central_12': n(-T['substitution_f20|central']['traj'][-1][1], 1),
    'rep_end': sg(T['substitution_f0|central']['representation_only_end']),
    'rec50': str(E['recovery_after_substitution']['days_to_50']), 'rec90': str(E['recovery_after_substitution']['days_to_90']),
    'rec_end': n(E['recovery_after_substitution']['curve'][-1][1]),
    'res_maxdiff': n(max(abs(r['gI_by_year'][-1] - r['gI_analytic']) for r in S['reserve'])),
    'res_q75_f0': n(res[(0.75, 0.0)]['gI_by_year'][-1]), 'res_q99_f0': n(res[(0.99, 0.0)]['gI_by_year'][-1]),
    'fstar_75': pct(fstar(0.75)), 'fstar_90': pct(fstar(0.9)), 'fstar_99': pct(fstar(0.99)),
    'lm_un_end': n(LM['unaided'][-1][1]), 'lm_sub_end': n(LM['substitute'][-1][1]),
    'lm_sub_g': n(LM['substitute'][-1][3]), 'lm_r20_end': n(LM['substitute_f20'][-1][1]),
}
s = open(TPL).read()
missing = sorted(set(re.findall(r'\{\{(\w+)\}\}', s)) - set(V))
assert not missing, missing
s = re.sub(r'\{\{(\w+)\}\}', lambda m: V[m.group(1)], s)
open(OUT, 'w').write(s)
json.dump(V, open(OUT.rsplit('.', 1)[0] + '_numbers.json', 'w'), indent=1, ensure_ascii=False)
for k, v in V.items():
    print(k, v)
