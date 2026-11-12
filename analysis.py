"""Scenario, sensitivity, Monte Carlo and spatial analyses for the article."""
import json, numpy as np, pandas as pd
from copy import deepcopy
from scipy.stats import spearmanr, rankdata
from pv_eol_model import *
import os; os.makedirs("outputs", exist_ok=True)

ex, pipe = load_existing(), load_pipeline()
OUT = {}

def S_(P, e=ex, p=pipe):
    return scenario(e, p, P)[3]

def row(name, s, group=''):
    return dict(group=group, scenario=name, cum_existing_t=s['cum_existing'], cum_pipeline_t=s['cum_pipeline'],
                cum_total_t=s['cum_total'], peak_year_2050=s['peak_year'], peak_t_2050=s['peak_t'],
                peak_year_ext=s['peak_year_ext'], peak_t_ext=s['peak_t_ext'], retired_mw_2050=s['retired_total_mw'],
                operating_mw_2050=s['op_2050'], post2050_t=s['post2050_t'])

# ------------------------------------------------ 1. bridge from the conventional formulation to the central case
b = []
P = deepcopy(CONVENTIONAL); b.append(row('Conventional formulation (76 t/MW, single 2027 cohort)', S_(P), 'bridge'))
P['mass'] = 'vintage'; b.append(row('+ Vintage mass intensity', S_(P), 'bridge'))
P['lead'] = (2, 3, 4); b.append(row('+ Staggered lead time (2-4 yr) = central case', S_(P), 'bridge'))
bridge = pd.DataFrame(b)

# ------------------------------------------------ 2. one-at-a-time sensitivity around central
base = S_(CENTRAL)
oat_defs = [
    ('Mean service life', '25 yr', [('30 yr (IRENA)', dict(mu=30.0))]),
    ('Early-loss Weibull shape', '3.5', [('2.4928 (IRENA)', dict(beta_early=2.4928))]),
    ('Early-loss share alpha', '0.20', [('0.10', dict(alpha=0.10)), ('0.30', dict(alpha=0.30))]),
    ('Repowering share rho', '0.10', [('0', dict(rho=0.0)), ('0.20', dict(rho=0.20))]),
    ('Repowering window', 'ages 15-20', [('ages 12-17', dict(rep_start=12)), ('ages 18-23', dict(rep_start=18))]),
    ('Island stress factors', '0.95/0.93/0.90', [('none (1.0)', dict(sf={'Luzon':1,'Visayas':1,'Mindanao':1})),
                                                  ('uniform 0.85', dict(sf={'Luzon':.85,'Visayas':.85,'Mindanao':.85}))]),
    ('Module mass intensity', 'vintage fit', [('constant 76 t/MW', dict(mass=76.0)), ('fit -10%', dict(mass_mult=0.9)),
                                               ('fit +10%', dict(mass_mult=1.1))]),
    ('Pipeline lead time', '2-4 yr staggered', [('+2 yr only', dict(lead=(2,))), ('+4 yr only', dict(lead=(4,)))]),
    ('Pipeline realization', '100%', [('75%', dict(realization=0.75)), ('50%', dict(realization=0.50))]),
    ('COA capacity basis', 'MW = MWdc', [('MWac x 1.27 DC/AC', dict(dcac=1.27))]),
]
oat = [row('Central case', base, 'central')]
for name, central_val, alts in oat_defs:
    for lab, ch in alts:
        P = deepcopy(CENTRAL); P.update(ch); r = row(lab, S_(P), name); r['central_value'] = central_val; oat.append(r)
oat = pd.DataFrame(oat)
oat['d_cum_total_pct'] = 100 * (oat.cum_total_t / base['cum_total'] - 1)
oat['d_peak_pct'] = 100 * (oat.peak_t_ext / base['peak_t_ext'] - 1)

# ------------------------------------------------ 3. literature bounding cases (IRENA-style single curves)
lit = []
for lab, ch in [('IRENA regular-loss (beta 5.3759, mean 30 yr, no early/repower)', dict(mu=30, alpha=0, rho=0, sf={'Luzon':1,'Visayas':1,'Mindanao':1})),
                ('IRENA early-loss (beta 2.4928, mean 30 yr)', dict(mu=30, alpha=1.0, rho=0, beta_early=2.4928, sf={'Luzon':1,'Visayas':1,'Mindanao':1})),
                ('Kim & Park early-loss 25 yr (beta 3.5)', dict(mu=25, alpha=1.0, rho=0, sf={'Luzon':1,'Visayas':1,'Mindanao':1})),
                ('Mixture of this study, no stress, 30 yr', dict(mu=30, sf={'Luzon':1,'Visayas':1,'Mindanao':1}))]:
    P = deepcopy(CENTRAL); P.update(ch); lit.append(row(lab, S_(P), 'literature'))
lit = pd.DataFrame(lit)

# ------------------------------------------------ 4. Monte Carlo
rng = np.random.default_rng(20261017)
N = 3000
draws = pd.DataFrame(dict(
    mu=rng.uniform(25, 30, N), beta_early=rng.uniform(2.4928, 3.5, N), alpha=rng.uniform(0.10, 0.30, N),
    rho=rng.uniform(0.0, 0.20, N), rep_start=rng.integers(12, 19, N), stress_k=rng.uniform(0.0, 1.5, N),
    mass_mult=rng.uniform(0.9, 1.1, N), lead_mean=rng.uniform(2.0, 4.0, N),
    realization=rng.uniform(0.5, 1.0, N), dcac=rng.uniform(1.0, 1.3, N)))
SF0 = CENTRAL['sf']
years = YEARS
ann_param, ann_full, cum_param, cum_full, pk_year_full, ret_full = [], [], [], [], [], []
u_ex_cache = None
for i, d in draws.iterrows():
    P = deepcopy(CENTRAL)
    P.update(mu=d.mu, beta_early=d.beta_early, alpha=d.alpha, rho=d.rho, rep_start=int(d.rep_start),
             mass_mult=d.mass_mult, sf={g: 1 - d.stress_k * (1 - v) for g, v in SF0.items()})
    # lead time: mix of +2,+3,+4 with mean lead_mean (two adjacent integers)
    lo = int(np.floor(d.lead_mean)); fr = d.lead_mean - lo
    leads = (lo,) if fr < 1e-9 or lo == 4 else (lo, lo + 1)
    # parametric-only run (full pipeline, central lead)
    u, R, S = (lambda uu: (uu, *run(uu, P)))(build_units(ex, pipe, P))
    cp = S[:, years <= 2050].sum(); ann_param.append(S.sum(0)); cum_param.append(cp)
    # full run: + lead, realization, dc/ac
    P2 = deepcopy(P); P2.update(realization=d.realization, dcac=d.dcac, lead=leads)
    u2 = build_units(ex, pipe, P2)
    if len(leads) == 2:   # weight the two lead years to hit lead_mean
        w = np.where(u2.inst_year.values - pipe.award_year.iloc[0] == lo, (1 - fr) * 2, fr * 2)
        w = np.where(u2.stream.values == 'pipeline', w, 1.0)
        u2['cap_mw'] *= w; u2['mass_t'] = u2.cap_mw * u2.mass_t_per_mw
    R2, S2 = run(u2, P2)
    a2 = S2.sum(0); ann_full.append(a2); cum_full.append(a2[years <= 2050].sum())
    pk_year_full.append(int(years[np.argmax(a2)])); ret_full.append(R2[:, years <= 2050].sum())
ann_param, ann_full = np.array(ann_param), np.array(ann_full)
draws['cum_param_t'] = cum_param; draws['cum_full_t'] = cum_full; draws['peak_year_full'] = pk_year_full
draws['retired_mw_full'] = ret_full; draws['peak_t_full'] = ann_full.max(1)
q = lambda A: np.percentile(A, [5, 50, 95], axis=0)
mc_band = pd.DataFrame({'year': years})
for lab, A in [('param', ann_param), ('full', ann_full)]:
    p5, p50, p95 = q(A); mc_band[f'{lab}_p5'] = p5; mc_band[f'{lab}_p50'] = p50; mc_band[f'{lab}_p95'] = p95
mc_band['central'] = base['nat'].waste_total_t.values
# variance attribution: squared standardized rank-regression coefficients on cumulative (full)
X = np.column_stack([rankdata(draws[c]) for c in ['mu','beta_early','alpha','rho','rep_start','stress_k','mass_mult','lead_mean','realization','dcac']])
Xs = (X - X.mean(0)) / X.std(0)
for ycol in ['cum_full_t', 'peak_t_full']:
    y = rankdata(draws[ycol]); ys = (y - y.mean()) / y.std()
    beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    r2 = 1 - ((ys - Xs @ beta) ** 2).sum() / (ys ** 2).sum()
    OUT[f'srrc_{ycol}'] = dict(zip(['mu','beta_early','alpha','rho','rep_start','stress_k','mass_mult','lead_mean','realization','dcac'], beta.round(4).tolist()))
    OUT[f'srrc_r2_{ycol}'] = round(float(r2), 4)
mc_summary = {}
for c in ['cum_param_t', 'cum_full_t', 'peak_t_full', 'retired_mw_full']:
    mc_summary[c] = dict(zip(['p5','p50','p95'], np.percentile(draws[c], [5, 50, 95]).round(1).tolist()))
mc_summary['peak_year_full'] = dict(zip(['p5','p50','p95'], np.percentile(draws.peak_year_full, [5, 50, 95]).tolist()))
mc_summary['share_peak_after_2050'] = float((draws.peak_year_full > 2050).mean())
OUT['mc_summary'] = mc_summary

# ------------------------------------------------ 5. spatial-allocation test (plant-resolved vs capacity-share allocation)
u, R, S, s = scenario(ex, pipe, CENTRAL)
exu = u.stream.values == 'existing'
prov_ex = expand_alloc(u[exu], S[exu])               # plant-resolved provincial existing waste
cap_share = ex.groupby('province').mwdc.sum() / ex.mwdc.sum()
sp = []
for y in [2030, 2035, 2040, 2045, 2050]:
    pr = prov_ex[y]; nat_y = pr.sum(); td = cap_share.reindex(pr.index).fillna(0) * nat_y
    mis = 0.5 * (pr - td).abs().sum() / nat_y
    top_pr = set(pr.sort_values(ascending=False).index[:5]); top_td = set(td.sort_values(ascending=False).index[:5])
    sp.append(dict(year=y, national_existing_t=nat_y, misallocated_share=mis, top5_overlap=len(top_pr & top_td),
                   rank_corr=spearmanr(pr, td).statistic))
spatial = pd.DataFrame(sp)
cum_pr = prov_ex.loc[:, 2025:2040].sum(1); cum_td = cap_share.reindex(cum_pr.index).fillna(0) * cum_pr.sum()
OUT['spatial_cum2040_misalloc'] = float(0.5 * (cum_pr - cum_td).abs().sum() / cum_pr.sum())

# ------------------------------------------------ save
for n, df in [('bridge', bridge), ('oat', oat), ('literature', lit), ('mc_band', mc_band), ('mc_draws', draws), ('spatial', spatial)]:
    df.to_csv(f'outputs/{n}.csv', index=False)
json.dump(OUT, open('outputs/summary.json', 'w'), indent=1)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(bridge.round(1).to_string()); print(oat[['group','scenario','cum_total_t','d_cum_total_pct','peak_year_ext','peak_t_ext','d_peak_pct','retired_mw_2050']].round(1).to_string())
print(lit[['scenario','cum_total_t','peak_year_ext','peak_t_ext','retired_mw_2050']].round(0).to_string())
print(json.dumps(OUT, indent=1)); print(spatial.round(3).to_string())
