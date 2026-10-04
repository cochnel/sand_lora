"""
Heightmap - Star Dunes (multi-localisation)

Source : ALOS World 3D (AW3D30) via OpenTopography — 30m, aucun compte requis
Dépendances :
    pip install rasterio numpy matplotlib requests
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import zipfile
import gzip
import shutil
import requests
import rasterio
from rasterio.merge import merge

# ── Config ────────────────────────────────────────────────────────────────────
LOCATION = "peru"   # "peru" | "namibia" | "china" | "algeria" | "morocco"

LOCATIONS = {
    # Côte péruvienne (Atacama) — plus grosses barchanes du monde
    # Jusqu'à 100 m de haut, envergure cornes ~1–2 km, terrain hyperaride
    "peru": {
        "bounds":      (-75.80, -14.15, -75.55, -13.90),  # ~22×28 km — région d'Ica
        "output_tif":  "peru_barchans_dem.tif",
        "output_png":  "peru_barchans_heightmap.png",
        "title":       "Barchanes · Côte Atacama, Pérou",
        "subtitle":    "Plus grosses barchanes du monde  |  14°S 75.7°W  |  ~22×28 km",
    },
    # Namib Desert, Namibie — barchanes côtières (petites, 5–30 m)
    # Zone correcte : près de Walvis Bay, pas de Lüderitz
    "namibia": {
        "bounds":      (14.30, -23.15, 14.60, -22.85),  # ~26×33 km — Walvis Bay
        "output_tif":  "namibia_barchans_dem.tif",
        "output_png":  "namibia_barchans_heightmap.png",
        "title":       "Barchanes · Namib Desert, Namibie",
        "subtitle":    "Corridor barchanes Walvis Bay  |  23°S 14.5°E  |  ~26×33 km",
    },
    # Plus grand champ de star dunes du monde — Badain Jaran, Mongolie-Intérieure
    # Dunes atteignant 500 m de haut, record mondial
    "china": {
        "bounds":      (102.30, 39.65, 102.39, 39.8),  # ~17×18 km — 1:1 avec panel matplotlib
        "output_tif":  "china_badain_jaran_dem.tif",
        "output_png":  "china_badain_jaran_heightmap.png",
        "title":       "Star Dunes · Désert Badain Jaran, Chine",
        "subtitle":    "Plus grand champ de star dunes du monde  |  39.4°N 102.2°E  |  ~45×50 km",
    },
    # Grand Erg Oriental — Algérie/Tunisie
    "algeria": {
        "bounds":      (7.78, 29.73, 7.96, 29.83),
        "output_tif":  "algeria_star_dunes_dem.tif",
        "output_png":  "algeria_star_dunes_heightmap.png",
        "title":       "Star Dunes · Grand Erg Oriental, Algérie",
        "subtitle":    "Erg Oriental  |  29.8°N 7.9°E  |  ~18×10 km",
    },
    # Lala Lallia — Erg Chebbi, Maroc
    "morocco": {
        "bounds":      (-4.10, 35.00, -3.95, 35.12),
        "output_tif":  "morocco_lala_lallia_dem.tif",
        "output_png":  "morocco_lala_lallia_heightmap.png",
        "title":       "Star Dune Lala Lallia · Erg Chebbi, Maroc",
        "subtitle":    "Lala Lallia  |  35.06°N 4.01°W  |  ~13×13 km",
    },
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "heightmaps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_cfg      = LOCATIONS[LOCATION]
BOUNDS    = _cfg["bounds"]
OUTPUT_TIF = os.path.join(OUTPUT_DIR, _cfg["output_tif"])
OUTPUT_PNG = os.path.join(OUTPUT_DIR, _cfg["output_png"])
TITLE     = _cfg["title"]
SUBTITLE  = _cfg["subtitle"]

# ── Téléchargement ALOS AW3D30 via OpenTopography (pas de compte requis) ─────
def download_alos_opentopo(bounds, output_tif):
    """
    Télécharge le DEM ALOS AW3D30 (30m) via l'API publique OpenTopography.
    Aucun compte, aucune clé API requise pour cette source.
    """
    if os.path.exists(output_tif):
        os.remove(output_tif)
        print(f"🗑  DEM précédent supprimé : {output_tif}")

    lon_min, lat_min, lon_max, lat_max = bounds

    url = (
        "https://portal.opentopography.org/API/globaldem"
        f"?demtype=AW3D30"
        f"&south={lat_min}&north={lat_max}"
        f"&west={lon_min}&east={lon_max}"
        f"&outputFormat=GTiff"
        f"&API_Key=demoapikeyot2022"  # clé demo publique OpenTopography
    )

    print("⬇  Téléchargement ALOS AW3D30 - Grand Erg Oriental Algérie...")
    print(f"   Zone : {bounds}")

    resp = requests.get(url, stream=True, timeout=120)
    if resp.status_code != 200:
        print(f"❌ Erreur OpenTopography (status {resp.status_code})")
        print("   Tentative fallback SRTM 1-arcsec (AWS)...")
        return download_srtm_fallback(bounds, output_tif)

    with open(output_tif, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    # Vérifier que c'est bien un GeoTIFF valide
    try:
        with rasterio.open(output_tif) as src:
            _ = src.read(1)
        print(f"✓ DEM téléchargé → {output_tif}")
        return True
    except Exception as e:
        print(f"⚠  Fichier invalide ({e}), fallback SRTM...")
        os.remove(output_tif)
        return download_srtm_fallback(bounds, output_tif)


# ── Fallback : SRTM 1-arcsec depuis AWS (si OpenTopography échoue) ────────────
def get_srtm_tile_name(lat, lon):
    lp = "N" if lat >= 0 else "S"
    lop = "E" if lon >= 0 else "W"
    return f"{lp}{abs(lat):02d}{lop}{abs(lon):03d}"

def download_srtm_fallback(bounds, output_tif):
    lon_min, lat_min, lon_max, lat_max = bounds
    tiles = []
    # np.floor() au lieu de int() : int(-26.95)=-26 mais la tuile est S27
    for lat in range(int(np.floor(lat_min)), int(np.floor(lat_max)) + 1):
        for lon in range(int(np.floor(lon_min)), int(np.floor(lon_max)) + 1):
            tiles.append(get_srtm_tile_name(lat, lon))

    print(f"   Tuiles SRTM : {tiles}")
    tif_paths = []
    for tile in tiles:
        path = _fetch_srtm_tile(tile)
        if path:
            tif_paths.append(path)

    if not tif_paths:
        print("❌ Aucune tuile téléchargée.")
        return False

    print("🔧 Assemblage...")
    datasets = [rasterio.open(p) for p in tif_paths]
    mosaic, transform = merge(datasets, bounds=bounds)
    profile = datasets[0].profile.copy()
    profile.update(driver="GTiff", height=mosaic.shape[1],
                   width=mosaic.shape[2], transform=transform)
    for ds in datasets:
        ds.close()
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(mosaic)
    for p in tif_paths:
        os.remove(p)
    print(f"✓ DEM assemblé → {output_tif}")
    return True

def _fetch_srtm_tile(tile_name, out_dir=OUTPUT_DIR):
    tif_path = os.path.join(out_dir, f"{tile_name}.tif")
    if os.path.exists(tif_path):
        return tif_path

    url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{tile_name[:3]}/{tile_name}.hgt.gz"
    print(f"   ⬇  {tile_name}")
    resp = requests.get(url, stream=True, timeout=60)
    if resp.status_code != 200:
        print(f"   ❌ {tile_name} introuvable (status {resp.status_code})")
        return None

    gz_path = tif_path.replace(".tif", ".hgt.gz")
    hgt_path = tif_path.replace(".tif", ".hgt")
    with open(gz_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    with gzip.open(gz_path, "rb") as gz, open(hgt_path, "wb") as out:
        shutil.copyfileobj(gz, out)
    os.remove(gz_path)

    with rasterio.open(hgt_path) as src:
        data = src.read()
        profile = src.profile.copy()
    profile.update(driver="GTiff")
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(data)
    os.remove(hgt_path)
    return tif_path


# ── Load ──────────────────────────────────────────────────────────────────────
def load_heightmap(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        crs = src.crs
        transform = src.transform

    if nodata is not None:
        data[data == nodata] = np.nan

    print(f"\n── Infos heightmap ─────────────────────────────")
    print(f"   Dimensions  : {data.shape[1]} x {data.shape[0]} px")
    print(f"   Élév. min   : {np.nanmin(data):.1f} m")
    print(f"   Élév. max   : {np.nanmax(data):.1f} m")
    print(f"   Δ élévation : {np.nanmax(data) - np.nanmin(data):.1f} m")
    print(f"   CRS         : {crs}")
    print(f"────────────────────────────────────────────────\n")
    return data, transform


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_heightmap(data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(f"{TITLE}\n{SUBTITLE}",
                 color="white", fontsize=13, fontweight="bold", y=1.02)

    # Vue 1 — hillshade greyscale pur (bonne lecture morphologique)
    ax = axes[0]
    ax.set_facecolor("#0d0d0d")
    data_exag0 = np.where(np.isnan(data), np.nanmean(data), data) * 4.0
    dx0, dy0 = np.gradient(data_exag0)
    slope0  = np.arctan(np.sqrt(dx0**2 + dy0**2))
    aspect0 = np.arctan2(-dy0, dx0)
    hs0 = (np.sin(np.radians(18)) * np.cos(slope0)
           + np.cos(np.radians(18)) * np.sin(slope0) * np.cos(np.radians(315) - aspect0))
    hs0 = np.clip(hs0, 0, 1)
    im = ax.imshow(hs0, cmap="gray", interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="Intensité ombrage", shrink=0.8)
    ax.set_title("Hillshade greyscale · exag ×4", color="white", fontsize=10)
    ax.axis("off")

    # Vue 2 — colormap terrain
    ax = axes[1]
    ax.set_facecolor("#0d0d0d")
    im2 = ax.imshow(data, cmap="terrain", interpolation="bilinear")
    plt.colorbar(im2, ax=ax, label="Élévation (m)", shrink=0.8)
    ax.set_title("Colormap terrain", color="white", fontsize=10)
    ax.axis("off")

    # Vue 3 — hillshade multi-directionnel avec exagération verticale
    ax = axes[2]
    ax.set_facecolor("#0d0d0d")
    data_filled = np.where(np.isnan(data), np.nanmean(data), data)

    # ── Exagération verticale ×4 : amplifie les petits reliefs sableux ──
    VERT_EXAG = 4.0
    data_exag = data_filled * VERT_EXAG

    def _hillshade_single(dem, azimuth_deg, altitude_deg):
        dx, dy = np.gradient(dem)
        azimuth  = np.radians(azimuth_deg)
        altitude = np.radians(altitude_deg)
        slope    = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect   = np.arctan2(-dy, dx)
        hs = (
            np.sin(altitude) * np.cos(slope)
            + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect)
        )
        return np.clip(hs, 0, 1)

    # Multi-directionnel : 4 azimuths, soleil rasant 18° — révèle tous les bras
    azimuths = [225, 270, 315, 360]
    hillshade = np.mean(
        [_hillshade_single(data_exag, az, 18) for az in azimuths], axis=0
    )

    norm = mcolors.Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
    rgba = plt.cm.copper(norm(data_filled))
    rgba[..., :3] *= hillshade[..., np.newaxis]
    ax.imshow(rgba, interpolation="bilinear")
    ax.set_title("Hillshade multi-dir · exag ×4 · soleil 18°", color="white", fontsize=10)
    ax.axis("off")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"✓ Image sauvegardée → {OUTPUT_PNG}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if download_alos_opentopo(BOUNDS, OUTPUT_TIF):
        heightmap, transform = load_heightmap(OUTPUT_TIF)
        plot_heightmap(heightmap)
    else:
        print("\nInstalle les dépendances : pip install rasterio numpy matplotlib requests")