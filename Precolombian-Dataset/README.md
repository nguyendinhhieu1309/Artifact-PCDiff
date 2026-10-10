# Precolombian Dataset

This directory contains source meshes and scripts for making point-cloud completion examples. The pipeline samples complete meshes, combines source groups, removes a portion from the bottom of each object, and saves the observed and removed point sets.

Run the scripts in Linux or WSL. The CSV generation code uses POSIX path separators; running the full pipeline natively in Windows can produce manifests the project loader cannot parse correctly.

## Source folders

Place source meshes in the corresponding folders under `precol/`:

- `SN_bowl/`
- `native/`
- `scanned/`
- `MD40_bowl/` (optional; this checkout currently has no meshes here)

The preprocessing script accepts OBJ, STL, or OFF meshes. This checkout contains OBJ sources.

## Requirements

Use Python with NumPy, Open3D, tqdm, and PyMCubes:

```bash
python3 -m venv ~/venvs/precol
source ~/venvs/precol/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy open3d tqdm PyMCubes
```

## Generate the dataset

From WSL, go to this directory. For this repository on drive D:, the path is:

```bash
cd /mnt/d/Paper_Submit/Cultural_Heritage_Journal/Cultural_Heritage/Precolombian-Dataset
```

### 1. Sample each complete mesh

```bash
python scripts/preproc_dataset.py --dataset SN_bowl --threads 8
python scripts/preproc_dataset.py --dataset native --threads 8
python scripts/preproc_dataset.py --dataset scanned --threads 8
```

The script writes a sampled `.npy` beside each mesh. By default it uses Poisson-disk sampling with 200,000 points and normalizes coordinates into a 512-unit cube, centering the selected axes and flooring the others. Run each source group once; use `--forced True` only when you intend to regenerate existing samples.

### 2. Combine meshes and create the split lists

```bash
python scripts/collect_complete.py
```

This collects the configured source groups into `precol/collection/` and writes intermediate manifests. The scanned group is assigned to test; other groups use a random 70/30 train/test split unless their folder structure already marks test data. Preserve the generated CSVs if you need to keep the same split on later runs.

### 3. Generate broken and repair pairs

```bash
python scripts/degrade_cloud_bottom.py --dataset precol --breakage 0.20 --variance 0.10 --n_breaks 1 --threads 8
```

The default breakage removes roughly 20% of the object's vertical extent from the bottom, with 10% variation. The script writes point clouds and visualization meshes under `precol/complete/`, `precol/broken/`, and `precol/repair/`. Keep `--n_breaks 1`: the current training and evaluation loader expects the `_0.npy` target.

### 4. Assemble the runtime dataset

The model reads files from the repository's top-level `data/precol/`. From the repository root in WSL, copy the generated pairs and original meshes:

```bash
cd /mnt/d/Paper_Submit/Cultural_Heritage_Journal/Cultural_Heritage
mkdir -p data/precol/{complete,broken,repair}
cp Precolombian-Dataset/precol/broken/*.npy data/precol/broken/
cp Precolombian-Dataset/precol/repair/*.npy data/precol/repair/
cp Precolombian-Dataset/precol/collection/*.obj data/precol/complete/
```

Generated split manifests contain names like `<name>_0.npy`. For this project's loader, each runtime CSV row must point to the unsuffixed complete mesh, such as `../data/precol/complete/<name>.obj`; the loader adds `_0.npy` for the matching broken and repair files. Generate the runtime CSVs from the saved split as follows:

```bash
python - <<'PY'
from pathlib import Path

source = Path("Precolombian-Dataset/precol")
target = Path("data/precol")

for split in ("train", "test"):
    rows = []
    for line in (source / f"{split}.csv").read_text(encoding="utf-8").splitlines():
        stem = Path(line.replace("\\", "/")).stem
        if stem.endswith("_0"):
            stem = stem[:-2]
        rows.append(fr"..\data\precol\complete\{stem}.obj")
    (target / f"{split}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
PY
```

After assembly, verify that every CSV row has matching files in `data/precol/broken/`; training also requires matching files in `data/precol/repair/`. The training preflight reports missing paths before an epoch starts.

## Parameters to adjust

- `--points`: samples per source mesh (default 200,000).
- `--center`: axes to center; default `xz`.
- `--offset`: lift above the floor after scaling, from 0 to 1.
- `--breakage` and `--variance`: bottom break height and variation.
- `--maxx`, `--maxy`, `--maxz`: maximum random rotation in degrees.
- `--seed`: breakage random seed.

The preprocessing implementation currently fixes its normalization cube at 512 internally, so changing its `--size` argument does not change that value.
