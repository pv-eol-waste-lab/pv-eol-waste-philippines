import numpy as np, pandas as pd, json
from pv_eol_model import *
ex, pipe = load_existing(), load_pipeline()
u, R, S, s = scenario(ex, pipe, CENTRAL)
Y = YEARS; m50 = Y <= 2050
exm = (u.stream == 'existing').values
# per plant
pl = ex[['plant_id','facility','province','region','island','mwdc','cod_year']].copy()
pl['t_per_mw'] = u.mass_t_per_mw.values[exm]; pl['module_mass_t'] = u.mass_t.values[exm]
pl['cum_waste_2050_t'] = S[exm][:, m50].sum(1); pl['retired_mw_2050'] = R[exm][:, m50].sum(1)
pl['surviving_mw_2050'] = pl.mwdc - pl.retired_mw_2050; pl['waste_post2050_t'] = S[exm][:, ~m50].sum(1)
pl.to_csv('outputs/per_plant.csv', index=False)
# per project (sum sub-cohorts)
pu = u[~exm].copy(); pu['cum'] = S[~exm][:, m50].sum(1); pu['ret'] = R[~exm][:, m50].sum(1); pu['post'] = S[~exm][:, ~m50].sum(1)
pj = pu.groupby('project').agg(cum_waste_2050_t=('cum','sum'), retired_mw_2050=('ret','sum'), waste_post2050_t=('post','sum'), module_mass_t=('mass_t','sum')).reset_index()
pj = pipe.merge(pj, left_on='coa_no', right_on='project', suffixes=('','_x')).drop(columns='project_x')
pj['surviving_mw_2050'] = pj.capacity_mw - pj.retired_mw_2050
pj.to_csv('outputs/per_project.csv', index=False)
# national annual
nat = s['nat']; nat['cum_total_t'] = nat.waste_total_t.cumsum(); nat['cum_existing_t'] = nat.waste_existing_t.cumsum(); nat['cum_pipeline_t'] = nat.waste_pipeline_t.cumsum()
nat.to_csv('outputs/national_annual.csv', index=False)
# island annual by stream
isl = {}
for stream, mask in [('existing', exm), ('pipeline', ~exm)]:
    for g in ['Luzon','Visayas','Mindanao']:
        mm = mask & (u.island.values == g)
        isl[(stream,g,'waste_t')] = S[mm].sum(0); isl[(stream,g,'retired_mw')] = R[mm].sum(0)
isl = pd.DataFrame(isl, index=Y); isl.columns = ['_'.join(c) for c in isl.columns]
isl.to_csv('outputs/island_annual.csv')
# province and region annual (existing, pipeline, total)
for key in ['province','region']:
    e = expand_alloc(u[exm], S[exm], key); p = expand_alloc(u[~exm], S[~exm], key)
    t = e.add(p, fill_value=0)
    for n, d in [('existing', e), ('pipeline', p), ('total', t)]:
        d.to_csv(f'outputs/{key}_annual_{n}.csv')
# island capacity balance
cap = []
for g in ['Luzon','Visayas','Mindanao']:
    me, mp = exm & (u.island.values==g), (~exm) & (u.island.values==g)
    cap.append(dict(island=g, initial_existing_mw=u.cap_mw[me].sum(), initial_pipeline_mw=u.cap_mw[mp].sum(),
        retired_existing_mw=R[me][:, m50].sum(), retired_pipeline_mw=R[mp][:, m50].sum(),
        waste_existing_t=S[me][:, m50].sum(), waste_pipeline_t=S[mp][:, m50].sum()))
cap = pd.DataFrame(cap); cap['surviving_2050_mw'] = cap.initial_existing_mw+cap.initial_pipeline_mw-cap.retired_existing_mw-cap.retired_pipeline_mw
cap['waste_total_t'] = cap.waste_existing_t+cap.waste_pipeline_t
cap.to_csv('outputs/island_capacity_balance.csv', index=False)
pd.set_option('display.width',250)
print(cap.round(1).to_string())
print(pl.sort_values('cum_waste_2050_t', ascending=False).head(10)[['facility','province','mwdc','cod_year','cum_waste_2050_t']].round(0).to_string())
print(pj.sort_values('cum_waste_2050_t', ascending=False).head(10)[['project','province','capacity_mw','cum_waste_2050_t']].round(0).to_string())
t = pd.read_csv('outputs/province_annual_total.csv', index_col=0); tc = t.loc[:, [str(y) for y in range(2025,2051)]].sum(1).sort_values(ascending=False)
print(tc.round(0).head(15).to_string())
r = pd.read_csv('outputs/region_annual_total.csv', index_col=0); print(r.loc[:, [str(y) for y in range(2025,2051)]].sum(1).sort_values(ascending=False).round(0).to_string())
re_ = pd.read_csv('outputs/region_annual_existing.csv', index_col=0); print(re_.loc[:, [str(y) for y in range(2025,2051)]].sum(1).sort_values(ascending=False).round(0).to_string())
print('existing check', pl.cum_waste_2050_t.sum(), 'pipe', pj.cum_waste_2050_t.sum())
