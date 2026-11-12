"""
pv_eol_model.py
End-of-life PV module waste and capacity attrition model for Philippine utility-scale solar PV.
Code for the article "End-of-Life Photovoltaic Module Waste and Capacity Attrition in the Philippines: A Plant-Resolved Simulation of
Existing and Pipeline Utility-Scale Solar Plants" by Christopher John T. Raagas and Ambrosio B. Cultura II.

Model chain (one unit = one existing plant or one pipeline sub-cohort):
  installed capacity (MWdc) --> annual retirement fraction p_total,g(a) --> retired capacity R (MWdc/yr)
  retired capacity x vintage mass intensity m(y) --> module waste S (t/yr)

The retirement fraction is the mixture model of the article (Eqs. 1-5):
  p_total(a) = (1 - alpha - rho) * p_reg(a) + alpha * p_early(a) + rho * p_rep(a)
with Weibull components whose scale is derived from a mean life mu and contracted by island stress factor SF_g,
and a uniform repowering window.

Run analysis.py, then tables.py, then figures.py (see README.txt).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from math import gamma
from copy import deepcopy

YEARS = np.arange(2025, 2076)          # computation horizon (reporting horizon is 2025-2050)
REPORT_END = 2050

# ---------------------------------------------------------------- parameters
CENTRAL = dict(
    mu=25.0,                 # mean module service life (years)
    beta_reg=5.3759,         # Weibull shape, planned wear-out (IRENA/Kuitsche)
    beta_early=3.5,          # Weibull shape, early loss (Kim & Park 2018)
    alpha=0.20,              # early-loss share
    rho=0.10,                # repowering share
    rep_start=15, rep_len=6, # repowering window: ages 15-20, uniform
    sf={'Luzon': 0.95, 'Visayas': 0.93, 'Mindanao': 0.90},   # island stress factors (time-scale contraction)
    mass='vintage',          # 'vintage' (Nieto-Morone et al. 2023 fit) or a constant t/MW (e.g. 76.0)
    mass_mult=1.0,
    lead=(2, 3, 4),          # award-to-installation lead times (years), equal weights
    realization=1.0,         # share of awarded COA capacity that is built
    dcac=1.0,                # multiplier converting COA "potential capacity" to MWdc
)
CONVENTIONAL = deepcopy(CENTRAL); CONVENTIONAL.update(mass=76.0, lead=(2,))   # constant mass, single-cohort reference case

def mass_intensity(year, P):
    """Module-only mass intensity (t/MWdc) for modules installed in `year`."""
    if P['mass'] == 'vintage':
        # exponential decay fit of mass-to-power ratio, Nieto-Morone et al. (2023), A = 2.72e25 t/MW, B = 37.11 yr
        m = 2.72e25 * np.exp(-np.asarray(year, float) / 37.11)
    else:
        m = np.full_like(np.asarray(year, float), float(P['mass']))
    return m * P['mass_mult']

def _eta(mu, beta):
    return mu / gamma(1 + 1 / beta)

def p_total(ages, island, P):
    a = np.asarray(ages, float)
    sf = P['sf'][island]
    er, ee = _eta(P['mu'], P['beta_reg']) * sf, _eta(P['mu'], P['beta_early']) * sf
    F = lambda x, b, e: np.where(x > 0, 1 - np.exp(-(np.clip(x, 0, None) / e) ** b), 0.0)
    p_reg = F(a, P['beta_reg'], er) - F(a - 1, P['beta_reg'], er)
    p_early = F(a, P['beta_early'], ee) - F(a - 1, P['beta_early'], ee)
    lo, hi = P['rep_start'], P['rep_start'] + P['rep_len'] - 1
    p_rep = np.where((a >= lo) & (a <= hi), 1.0 / P['rep_len'], 0.0)
    out = (1 - P['alpha'] - P['rho']) * p_reg + P['alpha'] * p_early + P['rho'] * p_rep
    return np.where(a >= 1, out, 0.0)

# ---------------------------------------------------------------- data
REGION_OF_PROV = {
 'Batangas':'IV-A','Laguna':'IV-A','Rizal':'IV-A','Cavite':'IV-A','Quezon':'IV-A',
 'Pampanga':'III','Tarlac':'III','Bataan':'III','Zambales':'III','Bulacan':'III','Nueva Ecija':'III',
 'Ilocos Norte':'I','Ilocos Sur':'I','Pangasinan':'I','La Union':'I',
 'Cagayan':'II','Isabela':'II','Nueva Vizcaya':'II','Ifugao':'CAR','Benguet':'CAR','Metro Manila':'NCR',
 'Camarines Sur':'V',
 'Iloilo':'VI','Capiz':'VI','Cebu':'VII','Bohol':'VII','Leyte':'VIII',
 'Negros Occidental':'NIR','Negros Oriental':'NIR','Siquijor':'NIR',
 'Zamboanga del Norte':'IX','Zamboanga del Sur':'IX','Zamboanga Sibugay':'IX',
 'Bukidnon':'X','Misamis Oriental':'X','Lanao del Norte':'X',
 'Davao del Sur':'XI','Davao de Oro':'XI','Davao del Norte':'XI','Davao Occidental':'XI',
 'South Cotabato':'XII','Cotabato':'XII','Sarangani':'XII',
}
ISLAND_OF_REGION = {**{r:'Luzon' for r in ['I','II','III','IV-A','V','CAR','NCR']},
                    **{r:'Visayas' for r in ['VI','VII','VIII','NIR']},
                    **{r:'Mindanao' for r in ['IX','X','XI','XII']}}

def load_existing(path='data/existing_plants.csv'):
    return pd.read_csv(path)

def load_pipeline(path='data/doe_coa_pipeline.csv'):
    return pd.read_csv(path)

def build_units(ex: pd.DataFrame, pipe: pd.DataFrame, P) -> pd.DataFrame:
    """One row per existing plant and per pipeline sub-cohort (project x lead time)."""
    rows = []
    for r in ex.itertuples():
        rows.append(dict(unit_id=r.plant_id, name=r.facility, stream='existing', cap_mw=r.mwdc,
                         inst_year=int(r.cod_year), island=r.island, alloc=r.province, project=r.plant_id))
    w = 1.0 / len(P['lead'])
    for r in pipe.itertuples():
        for L in P['lead']:
            rows.append(dict(unit_id=f'{r.coa_no}|+{L}', name=r.project, stream='pipeline',
                             cap_mw=r.capacity_mw * P['dcac'] * P['realization'] * w,
                             inst_year=int(r.award_year) + L, island=r.island, alloc=r.province_alloc,
                             project=r.coa_no))
    u = pd.DataFrame(rows)
    u['mass_t_per_mw'] = mass_intensity(u.inst_year.values, P)
    u['mass_t'] = u.cap_mw * u.mass_t_per_mw
    return u

def run(units: pd.DataFrame, P, years=YEARS):
    """Returns retired-capacity matrix R (MW/yr) and waste matrix S (t/yr), shape units x years."""
    ages = np.arange(-60, 101)                       # lookup table of p_total by island and age
    table = {g: p_total(ages, g, P) for g in set(units.island.values)}
    R = np.zeros((len(units), len(years)))
    for i, (cap, y0, isl) in enumerate(zip(units.cap_mw.values, units.inst_year.values, units.island.values)):
        R[i] = cap * table[isl][years - y0 + 60]
    S = R * units.mass_t_per_mw.values[:, None]
    return R, S

def expand_alloc(units: pd.DataFrame, M: np.ndarray, key='province'):
    """Split each unit's row across provinces listed in `alloc` ('A;B' = equal split); aggregate by province/region."""
    out = {}
    for i, a in enumerate(units.alloc.values):
        provs = a.split(';')
        for p in provs:
            out.setdefault(p, np.zeros(M.shape[1]))
            out[p] += M[i] / len(provs)
    df = pd.DataFrame(out, index=YEARS).T
    if key == 'region':
        df = df.groupby(df.index.map(REGION_OF_PROV)).sum()
    return df

def summarize(units, R, S, years=YEARS, end=REPORT_END):
    m = years <= end
    ex = (units.stream == 'existing').values
    nat = pd.DataFrame({'year': years,
                        'waste_existing_t': S[ex].sum(0), 'waste_pipeline_t': S[~ex].sum(0),
                        'retired_existing_mw': R[ex].sum(0), 'retired_pipeline_mw': R[~ex].sum(0)})
    nat['waste_total_t'] = nat.waste_existing_t + nat.waste_pipeline_t
    nat['retired_total_mw'] = nat.retired_existing_mw + nat.retired_pipeline_mw
    # operating capacity (end of year): installed-to-date minus cumulative retired (original-cohort accounting)
    inst = np.array([[c if y0 <= y else 0 for y in years] for c, y0 in zip(units.cap_mw, units.inst_year)])
    nat['operating_existing_mw'] = inst[ex].sum(0) - np.cumsum(R[ex].sum(0))
    nat['operating_pipeline_mw'] = inst[~ex].sum(0) - np.cumsum(R[~ex].sum(0))
    nat['operating_total_mw'] = nat.operating_existing_mw + nat.operating_pipeline_mw
    rep = nat[m]
    pk = rep.waste_total_t.idxmax()
    return dict(nat=nat,
                cum_existing=rep.waste_existing_t.sum(), cum_pipeline=rep.waste_pipeline_t.sum(),
                cum_total=rep.waste_total_t.sum(), retired_total_mw=rep.retired_total_mw.sum(),
                retired_existing_mw=rep.retired_existing_mw.sum(), retired_pipeline_mw=rep.retired_pipeline_mw.sum(),
                peak_year=int(rep.year[pk]), peak_t=float(rep.waste_total_t[pk]),
                peak_year_ext=int(nat.year[nat.waste_total_t.idxmax()]), peak_t_ext=float(nat.waste_total_t.max()),
                op_2050=float(rep.operating_total_mw.iloc[-1]),
                post2050_t=float(nat[~m].waste_total_t.sum()))

def scenario(ex, pipe, P):
    u = build_units(ex, pipe, P); R, S = run(u, P); return u, R, S, summarize(u, R, S)
