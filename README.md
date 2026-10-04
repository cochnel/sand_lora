# sand_lora

Simulations couplées vent / sable (formation de dunes) en PyTorch, basées sur le code `windblown_sand-main`.

## Structure

```
.
├── simulation_stable_fluids.py   # vent par Stable Fluids (advection + projection de pression)
├── couplage_energy.py            # vent par minimisation d'énergie de Stokes (Adam)
├── simulation_energy_opti.py     # Stokes + étapes d'advection (sillage asymétrique)
├── simulation_rans_opti.py       # énergie RANS variationnelle (visc + pression + advection)
├── load_height_map_download/
│   └── load.py                   # téléchargement de DEM réels (dunes) via OpenTopography / SRTM
├── windblown_sand-main/          # bibliothèque de base : fluides, érosion/dépôt, avalanches
└── results/                      # sorties générées (ignoré par git)
```

## Installation

```bash
pip install torch numpy matplotlib
# pour load.py uniquement :
pip install rasterio requests
```

Un GPU CUDA est utilisé s'il est disponible, sinon le CPU.

## Utilisation

Chaque script de simulation prend un numéro d'expérience `--nb_xp` :

```bash
python simulation_stable_fluids.py --nb_xp 1 --res_x 128 --steps 500
python simulation_rans_opti.py     --nb_xp 2 --steps 500 --nu_wind 0.5
```

`python <script>.py --help` liste tous les paramètres.

Les résultats (images `.png` et hauteurs de sable `.npy`) sont écrits dans `results/sand_height_<nb_xp>/`.

Pour télécharger une heightmap réelle, choisir `LOCATION` en haut de `load_height_map_download/load.py` puis :

```bash
python load_height_map_download/load.py
```

Le DEM (`.tif`) et l'aperçu (`.png`) sont écrits dans `results/heightmaps/`.
