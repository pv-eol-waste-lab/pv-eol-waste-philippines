# End-of-Life PV Module Waste and Capacity Attrition in the Philippines

Data and code for the article **"End-of-Life Photovoltaic Module Waste and Capacity Attrition in the Philippines: A Plant-Resolved Simulation of Existing and Pipeline Utility-Scale Solar Plants"** by Christopher John T. Raagas and Ambrosio B. Cultura II.

- Christopher John T. Raagas (corresponding author): Department of Electrical Engineering, College of Engineering and Architecture, University of Science and Technology of Southern Philippines, Cagayan de Oro City, Philippines; Department of Electrical Engineering, College of Engineering, Liceo de Cagayan University, Cagayan de Oro City, Philippines
- Ambrosio B. Cultura II: Department of Electrical Engineering, College of Engineering and Architecture, University of Science and Technology of Southern Philippines, Cagayan de Oro City, Philippines

The model converts 67 existing utility-scale solar plants (2,811 MWdc) and 98 solar projects awarded Certificates of Authority by the Department of Energy in 2025 (9,244 MW) into annual retired capacity and end-of-life module waste over 2025–2050, computed to 2075.

## Contents

| Path | Description |
|---|---|
| `data/existing_plants.csv` | Corrected inventory of the 67 existing utility-scale plants (DOE list of 28 February 2025), 2,811.07 MWdc |
| `data/doe_coa_pipeline.csv` | The 98 DOE Certificate of Authority solar projects awarded in 2025, 9,244.01 MW |
| `outputs/Model_Inputs_and_Outputs.xlsx` | Inputs and all central-case outputs: by plant, project, province, region, island group and year (2025–2075); conventional-to-central bridge; one-at-a-time sensitivity; literature bounding cases; Monte Carlo percentiles and draws; spatial-allocation test |
| `pv_eol_model.py` | Model: mixed Weibull retirement, island stress factors, vintage mass intensity, staggered pipeline commissioning, capacity–waste balance |
| `analysis.py` | Bridge, sensitivity, literature cases, Monte Carlo (3,000 runs, seed 20261017) and spatial test |
| `tables.py` | Plant, project, province, region, island and national tables |
| `figures.py` | Figures 1–8 of the article (EPS, 600 dpi TIFF and PNG) |

## Requirements

Python 3.9 or later with numpy, pandas, scipy and matplotlib:

```
pip install numpy pandas scipy matplotlib
```

## How to run

From the repository folder, in this order:

```
python analysis.py   # bridge, sensitivity, literature cases, Monte Carlo (about 1 min), spatial test -> outputs/
python tables.py     # plant, project, province, region, island and national tables -> outputs/
python figures.py    # Figures 1-8 -> figs/
```

Central parameters are set in `CENTRAL` in `pv_eol_model.py` (Table 1 of the article). `CONVENTIONAL` is the reference case of Section 3.7 (constant 76 t/MW, all pipeline projects installed at award + 2 years).

## Check values (central case)

- Module waste 2025–2050: 350,669 t (141,780 t existing plants; 208,889 t pipeline)
- Retired capacity 2025–2050: 6,421 MWdc
- Monte Carlo median 247,259 t (90% interval 174,256–351,042 t)
- Peak annual waste: 35,352 t in 2052
