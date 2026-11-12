"""Figures 1-8 of the article: 174 mm wide, lettering >= 8 pt, patterns and line styles in addition to colour,
no transparency; saved as EPS (vector, embedded fonts), 600 dpi TIFF (LZW) and 300 dpi PNG preview."""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pv_eol_model import *

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8, 'hatch.linewidth': 0.7, 'ps.fonttype': 42, 'pdf.fonttype': 42, 'axes.edgecolor': '#9a9994', 'axes.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.labelcolor': '#2b2a27', 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'legend.frameon': False,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.06, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold', 'lines.linewidth': 1.2})
BLUE, ORANGE, AQUA, GREY, LGREY, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#9a9994', '#d6d5cf', '#0b0b0b', '#52514e'
RED = '#e34948'
BLUE_L, ORANGE_L, RED_L, BAND = '#b4d0f1', '#f6bba4', '#f09b9a', '#d9e7f8'
W = 6.85   # 174 mm
fmt = lambda v: f'{v:,.0f}'
thou = matplotlib.ticker.FuncFormatter(lambda v, p: f'{v:,.0f}')
F = 'figs/'
for sub in ('eps', 'tiff', 'png'): os.makedirs(F + sub, exist_ok=True)
def save(fig, name):
    stem = name.replace('.png', '')
    fig.savefig(F + 'png/' + stem + '.png', dpi=300)
    fig.savefig(F + 'eps/' + stem + '.eps', format='eps')
    fig.savefig(F + 'tiff/' + stem + '.tif', format='tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
ex, pipe = load_existing(), load_pipeline()
u, R, S, s = scenario(ex, pipe, CENTRAL)
nat = pd.read_csv('outputs/national_annual.csv')
def panel(ax, k): ax.text(-0.09, 1.04, k, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='left', color=INK)

# Fig 1 workflow
steps = [('1  Inventory development and correction', 'Existing plants: commissioning year, MWdc, province, region, island group'),
         ('2  Pipeline cohort construction', 'DOE-COA solar projects ≥ 1 MW, non-rooftop; installation year = award year + 2, 3, 4'),
         ('3  Mixed retirement model', r'$p_{\mathrm{total},g}(a) = 0.70\,p_{\mathrm{reg},g}(a) + 0.20\,p_{\mathrm{early},g}(a) + 0.10\,p_{\mathrm{rep}}(a)$'),
         ('4  Capacity layer', 'Retired capacity (MWdc/yr) and surviving operating capacity (MWdc)'),
         ('5  Mass layer', 'Module waste (t/yr) = retired MWdc × m(installation year)'),
         ('6  Aggregation and verification', 'Plant → province → region → island group → nation; closure checks'),
         ('7  Sensitivity and uncertainty', 'One-at-a-time, literature cases, 3,000-run Monte Carlo, spatial-allocation test')]
fig = plt.figure(figsize=(W - 0.15, 4.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02); ax.axis('off')
h = 0.11; gap = (1 - len(steps) * h) / (len(steps) - 1)
for k, (t, d) in enumerate(steps):
    y = 1 - (k + 1) * h - k * gap
    ax.add_patch(FancyBboxPatch((0.02, y), 0.96, h, boxstyle='round,pad=0.004,rounding_size=0.01', fc='#eef4fc' if k < 6 else '#fdf0ea', ec='#b9b8b2', lw=0.7))
    ax.text(0.045, y + h * 0.67, t, fontsize=9.5, fontweight='bold', va='center', color=INK)
    ax.text(0.045, y + h * 0.28, d, fontsize=8.5, va='center', color=INK2)
    if k < len(steps) - 1:
        ax.annotate('', xy=(0.5, y - gap + 0.004), xytext=(0.5, y - 0.002), arrowprops=dict(arrowstyle='-|>', color=GREY, lw=1.1))
save(fig, 'Fig1')

# Fig 2 (a) retirement components, (b) vintage mass intensity
ages = np.arange(1, 46)
def comp(alpha, rho):
    Q = dict(CENTRAL); Q.update(alpha=alpha, rho=rho); return p_total(ages, 'Luzon', Q)
p_reg, p_early = comp(0, 0), comp(1, 0)
p_rep = np.where((ages >= 15) & (ages <= 20), 1 / 6, 0)
tot = p_total(ages, 'Luzon', CENTRAL)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 5.4), gridspec_kw=dict(hspace=0.42))
a1.plot(ages, 0.70 * p_reg * 100, color=GREY, lw=1.4, ls='--', label='Planned wear-out × 0.70')
a1.plot(ages, 0.20 * p_early * 100, color=AQUA, lw=1.4, ls='-.', label='Early loss × 0.20')
a1.step(ages, 0.10 * p_rep * 100, where='mid', color=ORANGE, lw=1.4, ls=':', label='Repowering × 0.10 (ages 15–20)')
a1.plot(ages, tot * 100, color=BLUE, lw=2.0, label='Total retirement fraction')
a1.set_xlabel('Module age (years)'); a1.set_ylabel('Share of cohort retiring (%/yr)'); a1.yaxis.grid(True); a1.set_axisbelow(True)
a1.legend(fontsize=8, loc='upper left'); a1.set_xlim(0, 45); a1.set_ylim(0, 8); panel(a1, 'a')
yy = np.arange(2003, 2031); m = mass_intensity(yy, CENTRAL)
a2.plot(yy, m, color=BLUE, lw=1.8, label='Vintage mass intensity m(y)')
a2.axhline(76, color=GREY, lw=1); a2.text(2003.3, 77.6, 'Constant 76 t/MW', color=INK2, fontsize=8)
exy = ex.groupby('cod_year').mwdc.sum()
a2.scatter(exy.index, mass_intensity(exy.index.values, CENTRAL), s=np.sqrt(exy.values) * 5, color=BLUE_L, ec=BLUE, lw=0.6, zorder=3, label='Existing plants (marker area scales with MWdc)')
for y in (2027, 2028, 2029):
    a2.scatter([y], mass_intensity(np.array([y]), CENTRAL), s=np.sqrt(3081) * 5, marker='s', color=ORANGE_L, ec=ORANGE, lw=0.6, zorder=3)
a2.scatter([], [], s=50, marker='s', color=ORANGE_L, ec=ORANGE, lw=0.6, label='Pipeline cohorts 2027–2029')
a2.annotate('69.4 t/MW (2016)', (2016, 69.4), (2017.3, 80), fontsize=8, color=INK2, arrowprops=dict(arrowstyle='-', color=GREY, lw=0.6))
a2.annotate('51.6 t/MW (2027)', (2027, 51.6), (2022.8, 42.0), fontsize=8, color=INK2, arrowprops=dict(arrowstyle='-', color=GREY, lw=0.6))
a2.set_ylim(35, 100); a2.set_xlabel('Installation year'); a2.set_ylabel('Module mass (t/MWdc)'); a2.yaxis.grid(True); a2.set_axisbelow(True)
a2.legend(fontsize=8, loc='lower left'); panel(a2, 'b')
save(fig, 'Fig2')

# Fig 3 national by stream to 2070
n70 = nat[nat.year <= 2070]
fig, ax = plt.subplots(figsize=(W, 3.0))
ax.bar(n70.year, n70.waste_existing_t, color=BLUE, width=0.72, label='Existing fleet')
ax.bar(n70.year, n70.waste_pipeline_t, bottom=n70.waste_existing_t + 120, color=ORANGE, width=0.72, hatch='////', edgecolor='white', lw=0, label='Pipeline (DOE-COA)')
ax.axvline(2050.5, color=INK2, lw=0.8)
ax.text(2051.0, 43200, 'Beyond the 2025–2050 reporting horizon →', fontsize=8, color=INK2, va='top')
pk = nat.loc[nat.waste_total_t.idxmax()]
ax.annotate(f'Peak {fmt(pk.waste_total_t)} t ({int(pk.year)})', (pk.year, pk.waste_total_t), (2057.5, 38500), fontsize=8, color=INK2, arrowprops=dict(arrowstyle='-', color=GREY, lw=0.6))
ax.set_ylim(0, 44000); ax.set_xlim(2024, 2071); ax.set_ylabel('Module waste (t/yr)'); ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.legend(loc='upper left', bbox_to_anchor=(0, 0.93), fontsize=8); ax.yaxis.set_major_formatter(thou)
save(fig, 'Fig3')

# Fig 4 combined provinces (stacked)
yrs = [str(y) for y in range(2025, 2051)]
pe = pd.read_csv('outputs/province_annual_existing.csv', index_col=0).loc[:, yrs].sum(axis=1)
pp = pd.read_csv('outputs/province_annual_pipeline.csv', index_col=0).loc[:, yrs].sum(axis=1)
tt = pe.add(pp, fill_value=0).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(W, 0.27 * 15 + 0.8))
y = np.arange(15)[::-1]
e_, p_ = pe.reindex(tt.index).fillna(0).values, pp.reindex(tt.index).fillna(0).values
ax.barh(y, e_, color=BLUE, height=0.62, label='Existing fleet')
ax.barh(y, p_, left=e_ + tt.max() * 0.004, color=ORANGE, height=0.62, hatch='////', edgecolor='white', lw=0, label='Pipeline (DOE-COA)')
for yi, v in zip(y, tt.values): ax.text(v + tt.max() * 0.012, yi, fmt(v), va='center', fontsize=8, color=INK2)
ax.set_yticks(y, tt.index); ax.tick_params(axis='y', length=0); ax.set_xlim(0, tt.max() * 1.15)
ax.xaxis.grid(True); ax.set_axisbelow(True); ax.legend(loc='lower right', fontsize=8)
ax.set_xlabel('Cumulative module waste 2025–2050 (t)'); ax.xaxis.set_major_formatter(thou)
save(fig, 'Fig4')

# Fig 5 capacity and waste
fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 4.3), sharex=True, gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.12))
polys = a1.stackplot(n70.year, n70.operating_existing_mw, n70.operating_pipeline_mw, colors=[BLUE, ORANGE], labels=['Existing fleet', 'Pipeline (DOE-COA)'], edgecolor='white', linewidth=0.6)
polys[1].set_hatch('////')
a1.set_ylabel('Operating capacity\n(MWdc, end of year)'); a1.yaxis.grid(True); a1.set_axisbelow(True); a1.legend(loc='upper right', fontsize=8)
a1.annotate('12,021 MWdc (2029)', (2029, 12021), (2033, 13300), fontsize=8, color=INK2, arrowprops=dict(arrowstyle='-', color=GREY, lw=0.6))
a1.annotate('5,634 MWdc (2050)', (2050, 5634), (2052.5, 8800), fontsize=8, color=INK2, arrowprops=dict(arrowstyle='-', color=GREY, lw=0.6))
a1.set_ylim(0, 14500); panel(a1, 'a')
a2.bar(n70.year, n70.waste_total_t, color=GREY, width=0.72)
a2.set_ylabel('Module waste (t/yr)'); a2.yaxis.grid(True); a2.set_axisbelow(True); panel(a2, 'b')
for a in (a1, a2): a.axvline(2050.5, color=INK2, lw=0.8); a.yaxis.set_major_formatter(thou)
a2.set_xlim(2024, 2071)
save(fig, 'Fig5')

# Fig 6 bridge
br = pd.read_csv('outputs/bridge.csv'); vals = br.cum_total_t.values
labels = ['Conventional\nformulation', 'Vintage mass\nintensity', 'Staggered\nlead time', 'Central\ncase']
steps_ = [vals[0], vals[1] - vals[0], vals[2] - vals[1], vals[2]]
fig, ax = plt.subplots(figsize=(W * 0.8, 3.0)); run_ = 0
for i, v in enumerate(steps_):
    if i in (0, 3):
        ax.bar(i, v, color=GREY, width=0.6); ax.text(i, v + 8000, fmt(v), ha='center', fontsize=8, color=INK); run_ = v
    else:
        bottom = run_ if v > 0 else run_ + v
        ax.bar(i, abs(v), bottom=bottom, color=RED_L, edgecolor=RED, lw=0.8, hatch='////', width=0.6)
        ax.text(i, run_ + max(v, 0) + 8000, ('+' if v > 0 else '−') + fmt(abs(v)), ha='center', fontsize=8, color=INK); run_ += v
ax.set_xticks(range(4), labels); ax.tick_params(axis='x', length=0); ax.set_ylim(0, 600000)
ax.set_ylabel('Cumulative waste 2025–2050 (t)'); ax.yaxis.grid(True); ax.set_axisbelow(True); ax.yaxis.set_major_formatter(thou)
save(fig, 'Fig6')


# Fig 7 Monte Carlo band
mc = pd.read_csv('outputs/mc_band.csv'); mc = mc[mc.year <= 2070]
fig, ax = plt.subplots(figsize=(W, 2.9))
ax.fill_between(mc.year, mc.full_p5, mc.full_p95, color=BAND, lw=0, label='Monte Carlo 5th–95th percentile')
ax.plot(mc.year, mc.full_p50, color=BLUE, lw=1.8, label='Monte Carlo median')
ax.plot(mc.year, mc.central, color=INK, lw=1.2, ls='--', label='Central case')
ax.axvline(2050.5, color=INK2, lw=0.8)
ax.set_ylabel('Module waste (t/yr)'); ax.yaxis.grid(True); ax.set_axisbelow(True); ax.legend(fontsize=8, loc='upper left')
ax.yaxis.set_major_formatter(thou); ax.set_xlim(2024, 2071)
save(fig, 'Fig7')

# Fig 8 spatial dumbbell 2035
prov_ex = expand_alloc(u[u.stream == 'existing'], S[(u.stream == 'existing').values])
cap_share = ex.groupby('province').mwdc.sum() / ex.mwdc.sum()
yv = 2035; pr = prov_ex[yv]; td = cap_share.reindex(pr.index).fillna(0) * pr.sum()
d = pd.DataFrame({'plant': pr, 'share': td}); d = d.loc[(d.plant + d.share).sort_values(ascending=False).index[:12]]
fig, ax = plt.subplots(figsize=(W * 0.85, 0.28 * len(d) + 0.9))
yy = np.arange(len(d))[::-1]
for k, (p, r) in zip(yy, d.iterrows()): ax.plot([r.share, r.plant], [k, k], color=LGREY, lw=2, zorder=1)
ax.scatter(d.share, yy, color=GREY, s=30, zorder=2, label='Allocated by capacity share')
ax.scatter(d.plant, yy, color=BLUE, marker='s', s=30, zorder=3, label='Plant-resolved (this study)')
ax.set_yticks(yy, d.index); ax.tick_params(axis='y', length=0); ax.xaxis.grid(True); ax.set_axisbelow(True)
ax.set_xlabel(f'Existing-fleet module waste in {yv} (t)'); ax.legend(fontsize=8, loc='lower right'); ax.xaxis.set_major_formatter(thou)
save(fig, 'Fig8')
print('Figures done')
