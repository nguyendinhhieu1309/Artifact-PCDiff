# Conditional diffusion restores intricate artifact shapes

**Author**: Nguyen Dinh Hieu, Phan Duy Hung  
**Affiliation**: FPT University  
**Contact**: hieundhe180318@fpt.edu.vn, hungpd2@fe.edu.vn

---

## ✨ Abstract

Digital restoration of ancient artifacts—especially those with intricate and irregular geometries—poses significant challenges. This project proposes a method leveraging **conditional diffusion models** to perform high-fidelity 3D shape completion from partial point clouds. By training on both general and culturally specific datasets (e.g., Precolumbian Pottery), our model learns detailed geometric priors and enables robust reconstruction. We report strong performance using metrics like Chamfer Distance and Hausdorff Distance, highlighting its relevance in AI-assisted cultural heritage preservation.

---

## 📄 Paper

- [Official Paper](https://doi.org/10.1007/978-981-96-4606-7_21)

---
## 🔧 Dependencies

| **Framework / Library** | **Version** |
|:------------------------|:------------|
| [![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-red?logo=pytorch&logoColor=white)](https://pytorch.org) | 2.2.1+cu118 |
| [![TorchVision](https://img.shields.io/badge/TorchVision-0.17.1-yellow?logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/index.html) | 0.17.1+cu118 |
| [![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org) | 3.12 |
| [![matplotlib](https://img.shields.io/badge/matplotlib-latest-orange?logo=plotly&logoColor=white)](https://matplotlib.org) | latest |
| [![tqdm](https://img.shields.io/badge/tqdm-latest-brightgreen)](https://tqdm.github.io) | latest |
| [![trimesh](https://img.shields.io/badge/trimesh-latest-lightgrey)](https://trimsh.org) | latest |
| [![ninja](https://img.shields.io/badge/ninja-build-blue)](https://ninja-build.org) | latest |
| [![pynrrd](https://img.shields.io/badge/pynrrd-latest-blueviolet)](https://github.com/mhe/pynrrd) | latest |
| [![open3d](https://img.shields.io/badge/open3d-0.17-green?logo=open3d)](http://www.open3d.org) | 0.17 |
| [![PyMCubes](https://img.shields.io/badge/PyMCubes-latest-lightblue)](https://github.com/pmneila/PyMCubes) | latest |

---

## 📦 Pretrained Checkpoint & Processed Dataset

- 🔗 **Trained Model Checkpoint**:  
  [📥 Download epoch_14999.pth](https://drive.google.com/file/d/1SuvpvF4McbvXkK8ek6faN1Is7AaPD2U3/view?usp=sharing)  
  *(stored under `experiments/train_completion/2025-05-10-20-14-58/checkpoints/`)*

- 📂 **Processed Dataset (Precol)**:  
  [📥 Download precol](https://drive.google.com/drive/folders/1NDaXU_wFQ61KaLWu44Y5EhlaDGa7WvCm?usp=sharing)  
  *(includes train/test CSVs and ready-to-use `.ply` files)*

## 📂 Table of Contents
- [🛠️ Environment Setup](#environment-setup)
- [📁 Dataset Preparation](#dataset-preparation)
- [🚀 Training](#training)
- [🧪 Evaluation](#evaluation)
- [👀 Visualization](#visualization)
- [📐 Model Architecture](#model-architecture)
- [📊 Metrics](#metrics)
- [📎 Project Structure](#project-structure)

---

## 🛠️ Environment Setup

### Requirements
- Python >= 3.10
- PyTorch (CUDA-enabled)
- NVIDIA GPU with CUDA toolkit (recommended: RTX 4090+ 48GB)
- g++, nvcc for custom extensions

### Installation
```bash
cd /workspace/pcdiff-method
python3 -m venv env
source env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If custom CUDA ops are used (e.g., in `pvcnn_completion.py`):
```bash
pip install -e .
```

---

## 📁 Dataset Preparation

Supports CSV-formatted datasets.

- Format (`precol/train.csv`):
```csv
/path/to/fragmented_object1.ply
/path/to/fragmented_object2.ply
```

Organize files under `data/precol/raw` and update CSVs accordingly.

---

## 🚀 Training

Train on `precol` dataset:
```bash
python -m pcdiff.train_completion --path data/precol/train.csv --dataset precol --bs 8 --output_dir experiments/train_completion/$(date +%Y-%m-%d-%H-%M-%S)
```

- Batch size: 8
- Optimizer: Adam (LR=1e-4)
- 1000 diffusion steps
- Epochs: 500 (Geometric), 4000 (Precolumbian)

---

## 🧪 Evaluation

Run inference on test set:
```bash
python -m pcdiff.test_completion --path data/precol/test.csv --dataset precol --model experiments/train_completion/<RUN_ID>/checkpoints/epoch_14999.pth --eval_path results/precol_test/<RUN_ID> --bs 8
```

Outputs:
- `input.npy`, `sample.npy`, `.ply` files
- `metrics.json`

---

## 👀 Visualization

To visualize `.npy` or export `.ply`:
```bash
python scripts/visualize_npy_pointcloud.py results/precol_test/<RUN_ID>/syn/<sample>/sample.npy
```

Use MeshLab or CloudCompare for `.ply`.

---

## 📐 Model Architecture

- **Encoder**: PointNet++
- **Decoder**: Feature propagation + MLP
- **Diffusion**: Conditional on partial input (`c_0`), denoises `~x_0` over 1000 steps

---

## 📊 Metrics

- **Chamfer Distance Factor (CDF)**
- **Hausdorff Distance Factor (HDF)**

Evaluate closeness to ground truth, normalized by object scale.

---

## 📎 Project Structure
```
/workspace/pcdiff-method/
├── pcdiff/                     # This seems to be your main Python package for the core logic.
│   ├── __init__.py             # Makes 'pcdiff' a Python package.
│   ├── model/                  # Where your neural network model definitions live.
│   │   ├── __init__.py
│   │   └── pvcnn_completion.py # For example, your PVCNN2Base model.
│   │   └── ...                 # Other model-related files.
│   ├── datasets/               # Code for loading and handling your datasets.
│   │   ├── __init__.py
│   │   └── precol_dataset.py   # A custom dataset class for 'precol'.
│   │   └── ...                 # Other dataset handlers.
│   ├── utils/                  # Helper functions and utility scripts.
│   │   ├── __init__.py
│   │   ├── file_utils.py       # For file operations.
│   │   ├── visualize.py        # For visualization tasks.
│   │   └── ...                 # Other general utilities.
│   ├── train_completion.py     # The script you use to train your completion models.
│   ├── test_completion.py      # The script for evaluating your trained models.
│   └── diffusion.py            # Core diffusion model logic (like GaussianDiffusion).
│
├── data/                       # A general place to store raw and processed datasets.
│   └── precol/                 # Specific to your 'precol' dataset.
│       ├── train.csv           # CSV file listing paths/info for training samples.
│       ├── test.csv            # CSV file listing paths/info for test samples.
│       └── raw/                # (Optional) Original raw data files (e.g., .ply, .obj).
│       └── processed/          # (Optional) Data after any preprocessing steps.
│
├── experiments/                # To store outputs from training runs.
│   └── train_completion/       # Specific to completion training.
│       └── 2025-05-10-20-14-58/ # A unique ID for each training run (timestamp is good!).
│           ├── checkpoints/    # Where model checkpoints like .pth files are saved.
│           │   ├── epoch_00000.pth
│           │   └── epoch_14999.pth
│           ├── logs/           # Training logs, tensorboard events, etc.
│           │   └── training_log.txt
│           └── config.yaml     # (Optional but good) The configuration used for this run.
│
├── results/                    # For outputs from your testing/evaluation runs.
│   └── precol_test/            # Results for the 'precol' test set.
│       └── 2025-05-10-20-14-58_epoch_14999/ # ID for which model was tested.
│           └── syn/            # Synthesized/completed outputs.
│               └── NA-native_10_0/ # Output for a specific sample.
│                   ├── input.npy
│                   ├── sample.npy
│                   ├── input.ply   # (If you save .ply files too)
│                   └── sample.ply
│           └── metrics.json    # (Optional) Quantitative evaluation metrics.
│
├── scripts/                    # Any standalone helper scripts.
│   ├── download_dataset.sh     # Script to download data, if applicable.
│   ├── preprocess_data.py      # Script for any data preprocessing steps.
│   └── visualize_npy_pointcloud.py # Your script for viewing .npy outputs.
│
├── env/                        # Your Python virtual environment (e.g., if using venv).
│
├── .gitignore                  # Tells Git which files/directories to ignore (like env/, __pycache__/, *.pyc, large data files if not tracked).
├── requirements.txt
```
---
## 💡 Citation

If you use this model or dataset, please cite:

```bibtex
@inproceedings{nguyendinhhieu1309/Cultural_Heritage,
  title={Conditional diffusion restores intricate artifact shapes},
  author={Nguyen Dinh Hieu, Phan Duy Hung},
  booktitle={Proceedings of the ...},
  year={2025},
  organization={FPT University, Hanoi}
}
```

## 📚 References

1. Pablo Jaramillo et al. *PCDiff* (arXiv 2024)  
2. Ivan Sipiran et al. *DRDAP* (IJCV 2022)  

## 🪪 License
Released under the **MIT License**.  
You may use, modify, and distribute the model and dataset with attribution.

## 🧩 Acknowledgment
Developed by **FPT University AI Research Group**, Hanoi, Vietnam  
