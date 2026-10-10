<p align="center">
  <img src="assets/artifact-pcdiff-logo.svg" alt="Artifact-PCDiff logo" width="720">
</p>

<p align="center">Missing-region completion of fragmented 3D artifacts</p>

<p align="center">

  <img src="https://img.shields.io/badge/Task-3D%20Completion-C7793A" alt="3D completion">
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
</p>

Artifact-PCDiff restores missing geometry from partial 3D scans of cultural heritage objects. It combines conditional point-cloud diffusion with fracture-aware refinement so the generated fragment follows the observed boundary.

**Authors:** Nguyen Dinh Hieu, Phan Duy Hung · **Affiliation:** FPT University

[Paper](#paper) · [Method](#method) · [Results](#qualitative-results) · [Setup](#setup) · [Dataset](#dataset) · [Train and test](#train-and-test) · [Metrics](#metrics)

## Paper

**Artifact-PCDiff: Conditional Point-Cloud Diffusion for Missing-Region Completion of Fragmented 3D Artifacts**
- [Official Paper]()


## Method

The model keeps the observed fragment as context and learns to generate the missing repair points with a conditional diffusion process. A PVCNN2-style point-cloud backbone extracts geometric features. The fracture-aware refinement uses boundary information and reliability gating to focus updates near the repair interface; an interface loss encourages a coherent join.

The repository implements the training, inference, and artifact-metric paths in `Artifact-PCDiff/Artifact-PCDiff/`.

## Qualitative results

**Method overview**

![Artifact-PCDiff method overview](Figures/figure1.png)

**Qualitative comparison**

![Qualitative comparison with DRDAP and PCDiff](Figures/figure4.png)

## Setup

Use a CUDA-enabled PyTorch environment. The CUDA extension needs a compatible CUDA toolkit and C++ compiler; on Windows, install the Visual Studio C++ build tools. Match the toolkit/compiler to the installed PyTorch build before running the model.

```powershell
cd Artifact-PCDiff
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install a CUDA-enabled PyTorch build compatible with your toolkit/compiler first.
python -m pip install -r requirements.txt
```

The custom CUDA operations are built when the model imports them.

## Dataset

Keep the generated runtime dataset under `data/precol/`:

```text
data/precol/
├── train.csv
├── test.csv
├── complete/   # Original complete object meshes, such as <name>.obj
├── broken/     # Observed point clouds, <name>_0.npy
└── repair/     # Removed ground-truth fragments, <name>_0.npy
```

Each CSV row points to a complete object. The loader derives the matching `broken/<name>_0.npy` path; training also loads `repair/<name>_0.npy`. Test inference needs the broken point cloud, while metric evaluation needs the repair ground truth.

See [Precolombian-Dataset/README.md](Precolombian-Dataset/README.md) for generation steps and CSV naming. The top-level `.gitignore` excludes `data/`, so dataset files stay local.

## Train and test

From PowerShell:

```powershell
cd Artifact-PCDiff
.\train.ps1 -BatchSize 8 -Epochs 4000 -Workers 0
.\test.ps1 "path\to\checkpoint.pth" -EvalPath "results\precol" -ComputeMetrics
```

From Linux or WSL, the Bash launchers are also available:

```bash
cd Artifact-PCDiff
bash train.sh
bash test.sh /path/to/checkpoint.pth
```

The test launcher detects the method from checkpoint metadata by default. Use an Artifact-PCDiff checkpoint to evaluate the method described in the paper. `-ComputeMetrics` runs the CDF, HDF, and BCD evaluator after inference; complete repair ground truth must be present for a full test-set report.

## Metrics

CDF, HDF, and BCD are calculated by `Artifact-PCDiff/Artifact-PCDiff/evaluate_artifact_metrics.py` against saved predictions and repair ground truth.

## Repository layout

```text
.
├── Artifact-PCDiff/
│   ├── Artifact-PCDiff/       # Model, dataset loader, train/test and metric code
│   ├── train.ps1 / test.ps1   # Windows launchers
│   └── train.sh / test.sh     # Linux/WSL launchers
├── Precolombian-Dataset/
│   ├── precol/                # Source meshes and generated intermediate files
│   └── scripts/               # Preprocessing and breakage-generation scripts
├── Figures/                   # Paper figures embedded above
├── assets/                    # Project logo
└── data/                      # Local runtime data; ignored by Git
```

## Citation

Please cite the paper if this project supports your research:

Citation details will be added when the paper is publicly available.
