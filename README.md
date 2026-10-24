# IE4476 — Dimensionality Reduction on COIL-20

Reduce COIL-20 images to various dimensionalities with PCA and PCA→LDA, classify with 1-NN, minimum Euclidean and minimum Mahalanobis distance, and plot accuracy against dimensionality. All fitting uses training data only.

See [IE4476_Code_Implementation_Plan.md](IE4476_Code_Implementation_Plan.md) for the full plan.

## Dataset

Download `coil-20-proc.zip` (processed version) from
https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php
and unzip so the images are at `data/coil-20-proc/obj1__0.png` … `obj20__71.png`. The dataset is not committed.

## Setup

Python 3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install
```

Select `.venv` as the notebook kernel in VS Code.

## Run order

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_pca_experiments.ipynb`
3. `notebooks/03_lda_experiments.ipynb`
4. `notebooks/04_extra_experiments.ipynb`
5. `notebooks/05_figures_and_tables.ipynb`

Results are written to `results/arrays/` and `results/figures/`.
