"""Figures for the manuscript. Usage: python -I figures.py <out_dir>"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = sys.argv[1]
FA = json.load(open(f'{OUT}/fit_absence.json')); SR = json.load(open(f'{OUT}/sim_results.json'))
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 0.6, 'lines.linewidth': 1.6, 'legend.frameon': False, 'figure.dpi': 150})
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight'); fig.savefig(f'{OUT}/{name}.png', dpi=300, bbox_inches='tight'); plt.close(fig)
def label(ax, s):
    ax.text(-0.2, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=10, color=INK)

# ---------- Fig 1: schematic ----------
fig, ax = plt.subplots(figsize=(6.85, 2.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis('off')
def box(x, y, w, h, txt, ec=INK2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12', fc='white', ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=7.5, color=INK)
def arrow(x0, y0, x1, y1, txt=None, col=INK2, dy=0.12):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=9, lw=1, color=col))
    if txt: ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, txt, ha='center', va='bottom', fontsize=6.8, color=col)
box(0.1, 1.6, 1.3, 0.8, 'Stimulus\n$\\mathbf{x}$')
box(2.2, 2.35, 2.4, 1.3, 'Internal pathway\nexpressed weights $\\mathbf{u}$\n⇅ consolidated $\\mathbf{s}$\n(decay $\\tau_d$)', ec=C1)
box(2.2, 0.35, 2.4, 1.0, 'Advice input $a$\n(correct with prob. $q$)', ec=C2)
box(6.0, 1.45, 1.9, 1.1, 'Decision neuron\n$g_I h + g_A a + \\xi$\n$g_I + g_A = G$')
box(8.5, 1.6, 1.35, 0.8, 'Response\n& outcome')
arrow(1.4, 2.0, 2.2, 3.0); arrow(1.4, 2.0, 2.2, 0.85)
arrow(4.6, 3.0, 6.0, 2.25, '$g_I$', C1); arrow(4.6, 0.85, 6.0, 1.75, '$g_A$', C2)
arrow(7.9, 2.0, 8.5, 2.0)
ax.text(3.4, 3.83, 'engagement-gated delta rule (gate $e$: 1 unaided, $\\kappa$ when advice supplies the answer)', ha='center', fontsize=6.6, color=C1)
ax.text(7.0, 0.95, 'outcome-modulated credit to each active input,\nthen normalisation (competition for resource)', ha='center', va='top', fontsize=6.6, color=INK2)
save(fig, 'Fig1')

# ---------- Fig 2: absence fits ----------
fig, axs = plt.subplots(1, 3, figsize=(6.85, 2.3), gridspec_kw={'width_ratios': [1, 1, 0.9]})
from scipy.linalg import expm
def delta(months, b, ts, lam=200.0, tu=5.0):
    td = lam / b * 365.0 * ts / (tu + ts); A = np.array([[-1 / tu, 1 / tu], [1 / ts, -1 / ts - 1 / td]]); phi = td / (td + ts)
    return np.array([-lam * (1 - (lambda M: M[0, 0] + M[0, 1] * phi)(expm(A * m * 30.44))) for m in months])
for ax, grp, col, ttl, lab in [(axs[0], '20-59', C1, 'Adults 20–59', 'a'), (axs[1], '60 and over', C2, 'Players 60 and over', 'b')]:
    r = FA[grp]; D = r['data']; m = np.array([d['months'] for d in D]); y = np.array([d['did'] for d in D]); se = np.array([d['se'] for d in D])
    ax.axhline(0, color=GRID, lw=0.8)
    ax.errorbar(m, y, yerr=1.96 * se, fmt='o', ms=4, color=INK2, ecolor=INK2, elinewidth=0.8, capsize=0, zorder=3)
    mm = np.linspace(0.3, 60, 200); ax.plot(mm, delta(mm, r['b_pp_per_year'], r['tau_s_days']), color=col)
    ax.set_xscale('log'); ax.set_xticks([1, 3, 6, 12, 24, 48]); ax.set_xticklabels(['1', '3', '6', '12', '24', '48'])
    ax.set_xlabel('Absence (months, log scale)'); ax.set_ylim(-16, 3)
    ax.set_title(ttl, fontsize=8, color=INK, loc='left'); label(ax, lab)
    ax.text(0.03, 0.04, f"slow loss {r['b_pp_per_year']:.2f} pp/yr\n95% CI [{r['b_ci95_profile'][0]:.2f}, {r['b_ci95_profile'][1]:.2f}]", transform=ax.transAxes, fontsize=6.6, color=INK2)
axs[0].set_ylabel('Change in score per game (pp)')
ax = axs[2]
for grp, col, nm in [('20-59', C1, '20–59'), ('60 and over', C2, '60+')]:
    pr = np.array(FA[grp]['profile']); cmin = pr[:, 1].min()
    ax.plot(pr[:, 0], pr[:, 1] - cmin, color=col, label=nm)
ax.axhline(3.84, color=INK2, lw=0.8, ls='--'); ax.text(6, 4.3, '95% threshold', fontsize=6.5, color=INK2)
ax.set_xscale('log'); ax.set_ylim(0, 12); ax.set_xlabel(r'Consolidation time $\tau_s$ (days)'); ax.set_ylabel(r'$\Delta\chi^2$ (profile)')
ax.legend(loc='upper left', fontsize=6.8); label(ax, 'c')
fig.tight_layout(); save(fig, 'Fig2')

# ---------- Fig 3: learners ----------
L = SR['learners']; fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.4))
ax = axs[0]; ks = sorted(float(k) for k in L['kappa_sensitivity_total_and_representation'])
KS = {float(k): v for k, v in L['kappa_sensitivity_total_and_representation'].items()}
tot = [KS[k][0] for k in ks]; rep = [KS[k][1] for k in ks]
ax.axhspan(-0.45, -0.09, color=GRID, alpha=0.6, lw=0); ax.axhline(-0.27, color=INK2, lw=0.8, ls='--')
ax.text(0.98, -0.44, 'grey band: pooled learner estimate, answer-supplying\nassistants, d = −0.27 [−0.45, −0.09] (dashed line)', ha='right', va='bottom', fontsize=6.0, color=INK2)
ax.plot(ks, tot, 'o-', color=C1, ms=4, label='total'); ax.plot(ks, rep, 's--', color=C3, ms=4, label='storage only (gain held fixed)')
ax.axhline(0, color=GRID, lw=0.8)
ax.set_xlabel(r'Engagement when the answer is supplied, $\kappa$'); ax.set_ylabel("Effect on unaided test (d)")
ax.legend(loc='upper left', fontsize=6.5); label(ax, 'a'); ax.set_ylim(-0.6, 0.15)
ax = axs[1]
for nm, col, ls, lab in [('unaided', C1, '-', 'unaided practice'), ('tutor', C3, ':', 'tutor (answer withheld)'),
                         ('substitute_f20', C4, '--', 'answers supplied, 20% reserve'), ('substitute', C2, '-', 'answers supplied')]:
    tr = np.array(SR['learners_months'][nm]); ax.plot(tr[:, 0], tr[:, 1], color=col, ls=ls, label=lab)
ax.set_xlabel('Days of practice (5 trials/day)'); ax.set_ylabel('Unaided accuracy'); ax.legend(fontsize=6.5, loc='upper left'); label(ax, 'b')
fig.tight_layout(); save(fig, 'Fig3')

# ---------- Fig 4: experts ----------
E = SR['experts']; fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.5))
ax = axs[0]; ax.axhline(0, color=GRID, lw=0.8)
def tr(k): a = np.array(E['traj'][k]['traj']); return a[:, 0] / 30.44, a[:, 1]
x, lo = tr('substitution_f0|upper'); _, hi = tr('substitution_f0|none'); _, mid = tr('substitution_f0|central')
ax.fill_between(x, lo, hi, color=C2, alpha=0.15, lw=0); ax.plot(x, mid, color=C2, label='answers supplied, no reserve')
x, lo = tr('substitution_f20|upper'); _, hi = tr('substitution_f20|none'); _, mid = tr('substitution_f20|central')
ax.fill_between(x, lo, hi, color=C4, alpha=0.18, lw=0); ax.plot(x, mid, color=C4, ls='--', label='answers supplied, 20% reserve')
x, y = tr('absence|central'); ax.plot(x, y, color=C1, label='absence')
x, y = tr('tutor|central'); ax.plot(x, y, color=C3, ls=':', label='tutor / unaided practice')
ax.set_xlabel('Months'); ax.set_ylabel('Unaided skill change (expert SD units)'); ax.legend(fontsize=6.3, loc='lower left', title='bands: arbitration learning rate from 0 to upper bound', title_fontsize=5.8); label(ax, 'a')
ax = axs[1]; rc = E['recovery_after_substitution']; c = np.array(rc['curve'])
ax.axhline(0, color=GRID, lw=0.8)
ax.plot(c[:, 0], c[:, 1], color=C2, label='expressed (unaided performance)'); ax.plot(c[:, 0], c[:, 2], color=C3, ls='--', label='stored (gain held fixed)')
ax.set_xlabel('Days of unaided practice after 12 months of substitution'); ax.set_ylabel('Skill change (expert SD units)')
ax.legend(fontsize=6.3, loc='lower right'); label(ax, 'b')
fig.tight_layout(); save(fig, 'Fig4')

# ---------- Fig 5: reserve ----------
fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.5)); kap = SR['learners']['kappa']
ax = axs[0]
for q, col in [(0.75, C1), (0.9, C2), (0.99, C3)]:
    rows = [r for r in SR['reserve'] if r['q'] == q]; f = [r['f'] for r in rows]
    ax.plot(f, [r['gI_analytic'] for r in rows], color=col, label=f'q = {q}')
    ax.plot(f, [r['gI_by_year'][-1] for r in rows], 'o', color=col, ms=4, mfc='white')
ax.set_xlabel('Reserve: share of trials unaided, f'); ax.set_ylabel(r'Internal gain share $g_I/G$')
ax.legend(fontsize=6.5, loc='upper left', title='line: analytic; circle: 3-year simulation', title_fontsize=6.2); label(ax, 'a')
ax = axs[1]; cI = np.mean([r['cI'] for r in SR['reserve']]); qq = np.linspace(0.55, 1.0, 200); cA = 2 * qq - 1
for sc, col in [(0.3, C1), (0.5, C2), (0.7, C3)]:
    fs = (sc * cA - kap * cI * (1 - sc)) / ((1 - kap) * cI * (1 - sc) + sc * cA)
    ax.plot(qq, np.clip(fs, 0, 1), color=col, label=fr'$\sigma_c$ = {sc}')
ax.set_xlabel('Advisor accuracy q'); ax.set_ylabel('Minimum reserve f*'); ax.set_ylim(0, 1)
ax.legend(fontsize=6.5, loc='upper left', title='internal share to keep', title_fontsize=6.2); label(ax, 'b')
fig.tight_layout(); save(fig, 'Fig5')
print('figures done')
