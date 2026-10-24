# IE4476 Assignment — Code Implementation Plan (COIL-20, VS Code, GitHub)

Goal: reduce COIL-20 images to various dimensionalities with PCA and PCA→LDA, classify, and plot accuracy against dimensionality. Everything fitted on training data only.

Tooling: VS Code + Jupyter notebooks, local Python virtual environment, Git/GitHub for version control. CPU only, no GPU needed.

---

## 1. Dataset

| Item | Value |
|---|---|
| Name | COIL-20, **processed** version |
| Source | https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php |
| Content | 20 objects × 72 poses (one image every 5°) = 1440 images |
| Raw size | 128×128 grayscale PNG |
| File naming | `obj<k>__<p>.png` — object k (1–20), pose p (0–71). Confirm after unzipping |

Download `coil-20-proc.zip` and unzip into `data/coil-20-proc/`. The dataset is not committed to Git; the README says where to get it.

The loader must assert exactly 1440 files, 20 objects and 72 poses per object, and fail loudly otherwise (wrong folder, nested unzip directory, stray files).

---

## 2. Environment setup

1. VS Code extensions: **Python** and **Jupyter** (Microsoft).
2. Clone the repo and create the virtual environment:

```bash
git clone https://github.com/arshad705/ie4476.git
cd ie4476
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install numpy scipy matplotlib pillow pandas jupyter ipykernel nbstripout scikit-learn
pip freeze > requirements.txt
nbstripout --install
```

3. In VS Code, open a notebook and select `.venv` as the kernel (top-right kernel picker).
4. scikit-learn is installed for sanity-check cells only — never for the actual PCA/LDA/classifiers.
5. Record the Python version in the README (e.g. `python --version` → 3.11.x).

`nbstripout --install` strips notebook outputs on commit, so diffs show only code changes. It is configured per clone (in `.git/`), so rerun it after any fresh clone.

**Do not keep the repo (and especially `.venv/`) inside a OneDrive-synced folder.** OneDrive will try to sync thousands of venv files and can lock or corrupt `.git/` objects. Use a local path such as `C:\dev\ie4476`.

---

## 3. Repository structure

```
ie4476/
├── data/                       # COIL-20 images (gitignored)
├── src/
│   ├── __init__.py
│   ├── config.py               # IMG_SIZE, STEP, OFFSETS, paths
│   ├── data.py                 # load_coil20, make_split
│   ├── reduction.py            # fit_pca, pca_project, fit_lda
│   ├── classifiers.py          # nn_predict, min_euclid_predict, min_mahal_predict
│   └── evaluate.py             # accuracy, experiment runners, validation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pca_experiments.ipynb
│   ├── 03_lda_experiments.ipynb
│   ├── 04_extra_experiments.ipynb   # training size, optional E5/E6
│   └── 05_figures_and_tables.ipynb
├── results/
│   ├── arrays/                 # .npz accuracy arrays + d_list
│   └── figures/                # .png at 300 dpi
├── requirements.txt
├── .gitignore
└── README.md
```

**Rule:** reusable functions live in `src/`; notebooks only import, run and plot. A bug fixed in `src/` is fixed everywhere.

`.gitignore`:

```
.venv/
data/
__pycache__/
.ipynb_checkpoints/
.vscode/
*.zip
```

Commit `results/figures/` and `results/arrays/` (small files, useful for report history and reproducibility).

Importing `src` from inside `notebooks/` — add this to the first cell of each notebook:

```python
import sys, pathlib
sys.path.append(str(pathlib.Path.cwd().parent))
%load_ext autoreload
%autoreload 2
```

`autoreload` picks up edits to `src/` without restarting the kernel. (VS Code sets the notebook's working directory to the notebook folder by default, so `cwd().parent` is the repo root.)

### 3.1 Module responsibilities

Flow: `config` sets parameters → `data` produces `X`, `y`, `pose` and splits → `reduction` maps images to low-dimensional features → `classifiers` predict labels → `evaluate` sweeps dimensions and splits and returns accuracy arrays.

| File | Responsibility | Public functions |
|---|---|---|
| `__init__.py` | Marks `src/` as a package | — |
| `config.py` | All constants and paths | `ROOT`, `DATA_DIR`, `RESULTS_DIR`, `IMG_SIZE`, `STEP`, `OFFSETS`, `SEED`, `NORMALISE` |
| `data.py` | Load, pre-process (per image only), split | `load_coil20(normalise=NORMALISE)`, `make_split(pose, offset, step)` |
| `reduction.py` | PCA and LDA; fit on training data only | `fit_pca(Xtr)`, `pca_project(X, m, V, d)`, `fit_lda(Z, y)` |
| `classifiers.py` | Fit-and-predict in one call; fit uses training data only | `nn_predict`, `min_euclid_predict`, `min_mahal_predict(..., eps=0.0)` |
| `evaluate.py` | Metrics, experiment runners, training-only validation | `accuracy`, `error_rate`, `run_pca_experiment`, `run_lda_experiment`, `run_trainsize_experiment`, `select_by_validation` |

Design rules:

- `fit_*` functions take training data only; `*_project` / `*_predict` apply learned parameters to any data.
- Functions are pure: no global state, no file I/O except in notebooks.
- Every function has type hints and a docstring stating input/output shapes.
- No hard-coded 240 / 220 / 239 / 1200 anywhere in `src/`: derive N, c, N − c and N_test from the arrays passed in. These change in E4 and inside validation folds.

---

## 4. `src/config.py`

```python
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "coil-20-proc"
RESULTS_DIR = ROOT / "results"
IMG_SIZE = 32        # 32x32 -> n = 1024
STEP = 6             # every 6th pose for training -> 12 per object
OFFSETS = range(6)   # 6 disjoint splits
SEED = 0             # only used for optional random-split experiment E5
NORMALISE = True     # per-image brightness normalisation (main experiments)
```

---

## 5. `src/data.py` — load, pre-process, split

| Step | What | Output |
|---|---|---|
| Load | Read every PNG; parse object k and pose p from filename with a regex; sort by (k, p) | images, labels, poses |
| Resize | 128×128 → 32×32 with `Image.resize((32, 32), Image.Resampling.BOX)` (area averaging, fixed for reproducibility) | n = 1024 |
| Scale | Divide by 255 → [0, 1] | float64 |
| Brightness normalise (flag, default on) | Per image: subtract own mean, divide by own L2 norm | removes global illumination differences |
| Vectorise | Flatten each image | `X` (1440 × 1024), `y` (1440,) with labels 0–19, `pose` (1440,) |

Use float64 throughout: the LDA generalised eigenproblem and Mahalanobis near the rank cap are numerically sensitive.

Leakage rule: per-image operations are fine here. Anything using statistics across images belongs inside the fit functions on training data only.

Pose-spaced split (adjacent poses are 5° apart and nearly identical, so a random split makes 1-NN score ~100% at every d):

```python
def make_split(pose, offset, step=STEP):
    train = (pose % step) == offset
    return np.where(train)[0], np.where(~train)[0]
```

With step = 6, every test image is at most 15° from a training pose of the same object.

| Quantity | Value (step = 6) |
|---|---|
| Training poses per object | 12 (30° apart) |
| N_train / N_test | 240 / 1200 |
| Splits | 6 offsets (0–5), disjoint training sets |
| Max PCA dims | N − 1 = 239 |
| Rank of $S_w$ | ≤ N − c = 220 < 1024 → singular in pixel space |
| Max LDA dims | c − 1 = 19 |

---

## 6. `src/reduction.py` — PCA and LDA

### PCA (Wk9 slides 61–72)

```python
def fit_pca(Xtr):
    m = Xtr.mean(axis=0)
    Xc = Xtr - m
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals = s**2 / Xtr.shape[0]
    eigvecs = Vt.T            # columns, sorted largest first
    r = Xtr.shape[0] - 1      # centering removes one degree of freedom
    return m, eigvecs[:, :r], eigvals[:r]

def pca_project(X, m, eigvecs, d):
    return (X - m) @ eigvecs[:, :d]
```

SVD avoids forming the 1024×1024 covariance. The N-th singular value is ~0 because centering removes one degree of freedom, so `fit_pca` keeps only the first N − 1 components.

Efficiency: in sweeps, project once to the maximum d and slice columns (`Z[:, :d]`) instead of re-projecting for every d.

### LDA via PCA→LDA (Fisherfaces, Wk9 slides 73–77)

1. PCA to m = N − c dims (220 with default settings; **computed from the fitting set**, so it is 200 inside a validation fold and different in E4). This makes $S_w$ invertible.
2. In that space:

$S_w = \sum_i \sum_{\mathbf{x} \in \omega_i} (\mathbf{x} - \boldsymbol{\mu}_i)(\mathbf{x} - \boldsymbol{\mu}_i)^T$

$S_b = \sum_i N_i (\boldsymbol{\mu}_i - \mathbf{m})(\boldsymbol{\mu}_i - \mathbf{m})^T$

3. Solve $S_b \mathbf{w} = \lambda S_w \mathbf{w}$.

```python
from scipy.linalg import eigh

def fit_lda(Z, y):
    classes = np.unique(y)
    m = Z.mean(axis=0)
    Sw = np.zeros((Z.shape[1],) * 2)
    Sb = np.zeros_like(Sw)
    for c in classes:
        Zc = Z[y == c]
        mu = Zc.mean(axis=0)
        Sw += (Zc - mu).T @ (Zc - mu)
        Sb += len(Zc) * np.outer(mu - m, mu - m)
    vals, vecs = eigh(Sb, Sw)          # ascending; vecs satisfy W^T Sw W = I
    order = np.argsort(vals)[::-1]
    return vecs[:, order[:len(classes) - 1]], vals[order[:len(classes) - 1]]
```

At m = N − c exactly, $S_w$ is full rank in theory but can be badly conditioned. Log `np.linalg.cond(Sw)`; if `eigh` raises `LinAlgError` (not positive definite), add a tiny ridge `Sw + 1e-10 * np.trace(Sw)/m * I` and note it in the report.

Full projection: `pca_project(X, m, V, N - c) @ W[:, :d]` for d = 1 … c − 1 (19).

Optional: vary the intermediate PCA dim m ∈ {20, 50, 100, 150, N − c}. m must be ≥ c − 1 = 19.

---

## 7. `src/classifiers.py` (Wk9 slides 17–56)

| Function | Rule | Fitted from |
|---|---|---|
| `nn_predict` | Label of closest training sample (Euclidean) | training projections |
| `min_euclid_predict` | Closest class mean | class means |
| `min_mahal_predict` | Closest class mean under pooled within-class covariance | class means + $\Sigma_w = S_w/(N - c)$ |

Mahalanobis distance:

$d_\Sigma(\mathbf{z}, \boldsymbol{\mu}_i) = (\mathbf{z} - \boldsymbol{\mu}_i)^T \Sigma_w^{-1} (\mathbf{z} - \boldsymbol{\mu}_i)$

Notes:

- **1-NN distances:** do not broadcast `Zte[:, None, :] - Ztr[None]` — at d = 239 that is 1200 × 240 × 239 floats ≈ 550 MB. Use `‖a‖² + ‖b‖² − 2 a·b` (one matrix product), or `scipy.spatial.distance.cdist`.
- Class means (20 of them) can be broadcast safely: `((Zte[:, None, :] - M[None]) ** 2).sum(-1)`.
- **Mahalanobis via whitening:** `L = cholesky(Sigma_w, lower=True)`; whitened data = `solve_triangular(L, (Z - mu).T, lower=True)`, then use Euclidean. This avoids forming an explicit inverse.
- In PCA space $\Sigma_w$ has rank min(d, N − c), so cap Mahalanobis at d ≤ N − c (220). Compute the cap from data. Expect accuracy to drop as d approaches the cap: the smallest within-class eigenvalues are tiny and get amplified. This motivates E6.
- **In LDA space, min-Mahalanobis = min-Euclidean.** `eigh(Sb, Sw)` returns W with $W^T S_w W = I$, so $\Sigma_w = I/(N-c)$ and both rank classes identically. Identical curves are expected, not a bug — explain it in the report.
- Per-class covariances are impossible (12 samples each → singular once d ≥ 12), so pooled only.
- Optional regularised version (slide 93): $\Sigma_{reg} = \Sigma_w + \epsilon I$, ε chosen from training data only (scale ε relative to `trace(Sigma_w)/d`).

---

## 8. Notebooks and experiments

| Notebook | Experiments | Saves to `results/arrays/` |
|---|---|---|
| `01_data_exploration` | Load data, shape checks, sample image grid, verify split sizes and $S_w$ rank | `../figures/samples.png` |
| `02_pca_experiments` | E0 raw-pixel 1-NN and min-Euclidean baseline; E1 PCA + 3 classifiers (d = 1 … N − 1, Mahalanobis ≤ N − c); E3 eigenspectrum | `pca_acc.npz`, `eigvals.npy`, `baseline.npy` |
| `03_lda_experiments` | E2 PCA(N − c)→LDA + 3 classifiers (d = 1 … 19); optional m sweep | `lda_acc.npz` |
| `04_extra_experiments` | E4 training size (step = 4, 6, 9, 12 → 18, 12, 8, 6 per class); optional E5 random vs spaced split; optional E6 regularised Mahalanobis | `trainsize_acc.npz`, etc. |
| `05_figures_and_tables` | Loads arrays only; makes all report figures and the results table | `../figures/*.png`, `../table.csv` |

Accuracy array shape: `acc[offset, classifier, d_index]`, classifier order `["1-NN", "min-Euclid", "min-Mahal"]`. Entries that are undefined (Mahalanobis at d > N − c) are `np.nan`; aggregate with `np.nanmean` / `np.nanstd`. Save with `np.savez(path, acc=acc, d_list=d_list, classifiers=names)` so the d values always travel with the array.

Report mean ± std over the offsets. Accuracy = correct / N_test (1200 with step = 6).

E4: the number of offsets equals the step (4, 6, 9 or 12 splits), and N, N − c, N_test change per step. Use `range(step)` and derive all caps from data.

Loop skeleton (notebook 02):

```python
for k, off in enumerate(OFFSETS):
    tr, te = make_split(pose, off)
    m, V, lam = fit_pca(X[tr])                 # training only
    Ztr_all = pca_project(X[tr], m, V, V.shape[1])
    Zte_all = pca_project(X[te], m, V, V.shape[1])
    cap = len(tr) - len(np.unique(y[tr]))      # N - c
    for j, d in enumerate(d_list):
        Ztr, Zte = Ztr_all[:, :d], Zte_all[:, :d]
        acc[k, 0, j] = np.mean(nn_predict(Ztr, y[tr], Zte) == y[te])
        acc[k, 1, j] = np.mean(min_euclid_predict(Ztr, y[tr], Zte) == y[te])
        acc[k, 2, j] = (np.mean(min_mahal_predict(Ztr, y[tr], Zte) == y[te])
                        if d <= cap else np.nan)
np.savez(RESULTS_DIR / "arrays" / "pca_acc.npz", acc=acc, d_list=d_list)
```

d grid: every d from 1 to 50, then steps of 5, **plus** N − c (220) and N − 1 (239) explicitly so the cap and the endpoint are always included: `d_list = sorted(set(range(1, 51)) | set(range(55, 240, 5)) | {220, 239})`.

Why save arrays: notebook 05 can restyle figures without rerunning experiments.

---

## 8.1 Training-only validation (required for any "chosen" value)

Plotting test accuracy vs d is what the brief asks for. But picking a single "best d" (or the intermediate PCA dim m, or ε) by looking at test accuracy uses test data to set a parameter — not allowed.

Rule: any single selected value is chosen by validation **inside the training set**.

```python
def select_by_validation(X, y, pose, train_idx, candidates, score_fn):
    """Leave-one-pose-out within the training set.
    For each training pose p: validate on images at pose p, fit on the
    other training poses. Return the candidate with the best mean score."""
```

- Training set has 12 poses per object → 12 validation folds (each fold: 220 fit / 20 validate).
- **Inside a fold N = 220, so N − 1 = 219, N − c = 200.** PCA candidates are limited to d ≤ 219, Mahalanobis to d ≤ 200, and the LDA intermediate PCA dim becomes 200. Express m as "N − c of the fitting set" (not the literal 220), and drop or clip candidates that are invalid in a fold.
- The held-out pose is 30° from its nearest remaining training pose, while real test images are at most 15° away. Validation accuracy will therefore be pessimistic. That is fine for *ranking* candidates; do not report it as an estimate of test accuracy.
- Use it for: best d (PCA and LDA), intermediate PCA dim m, ε for regularised Mahalanobis.
- Report: full test curves as results, plus test accuracy at the validation-selected value.

## 8.2 Accuracy vs error rate

The brief allows either. Prof. Jiang's slides 103–104 plot **error rate**, so plot error rate = 1 − accuracy in the report figures to match. Store accuracy in arrays; convert at plotting time.

## 9. Figures (generated in notebook 05, 300 dpi)

1. Sample grid: 5 objects × 6 poses
2. Eigenspectrum, log-scale y, plus cumulative variance curve
3. E1: error rate vs d, 3 classifiers, dashed baseline, shaded ± std, marker at validation-selected d; vertical line at the Mahalanobis cap
4. E2: error rate vs d (1–19), 3 classifiers (min-Euclid and min-Mahal overlap — see section 7)
5. E4: error rate vs training samples per class
6. Table: per method × classifier — validation-selected d and its test accuracy (mean ± std); best test accuracy on the curve for reference, labelled "oracle" (pandas → CSV)

Use log-scale x for figure 3 or a broken axis, since most of the change happens at small d.

---

## 10. Sanity tests (notebook 01 or a `tests/` cell)

- `X.shape == (1440, 1024)`, 20 labels, 72 poses each
- Each split: 240 train, 1200 test, no overlap; union covers all 1440
- Each training set has exactly 12 images per class
- Eigenvalues sorted descending; the (dropped) N-th singular value ≈ 0
- `np.linalg.matrix_rank(Sw_pixel) <= 220`
- PCA eigenvalues × N/(N − 1) match `sklearn.decomposition.PCA().explained_variance_`
- PCA eigenvectors orthonormal: `V.T @ V ≈ I`
- LDA: `W.T @ Sw @ W ≈ I`; only c − 1 non-zero generalised eigenvalues
- LDA subspace matches `sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver="eigen")` on the same PCA(220) features (compare classification accuracy, not raw vectors — signs and scaling differ)
- Classifiers on a toy 2-class, 2-D dataset give the expected labels

---

## 11. Git workflow

| When | Commit message example |
|---|---|
| Plan | `docs: implementation plan` |
| Repo + env set up | `chore: project structure, requirements, gitignore` |
| Loading + split working | `feat: COIL-20 loader and pose-spaced split` |
| PCA + classifiers | `feat: PCA and three classifiers` |
| E1 results | `exp: PCA accuracy vs dimension (E1)` |
| LDA | `feat: PCA->LDA` |
| Figures | `fig: report figures and results table` |

- Commit small and often. Keep the GitHub repo **private** until after grading (Settings → General → Danger Zone → Change visibility).
- Commit author must be the student's own GitHub identity (`git config user.name` / `user.email`). No co-author trailers.
- README: project aim, how to download the dataset, how to set up the env, Python version, notebook run order.
- Tag the final version used for the report: `git tag v1.0-submission && git push origin v1.0-submission`.

---

## 12. Using GeeksforGeeks and sklearn

- GfG PCA/LDA articles use sklearn's `PCA` and `LinearDiscriminantAnalysis`. Fine for understanding and cross-checking.
- Don't use sklearn LDA as the implementation: it handles the singular $S_w$ internally and hides the small-sample-size problem you need to analyse.
- Useful GfG parts: reading images with Pillow, NumPy SVD/eig, Matplotlib subplots.

---

## 13. Common bugs

| Bug | Symptom | Fix |
|---|---|---|
| Fitting PCA mean on all 1440 images | Slightly inflated accuracy | Fit on `X[tr]` only |
| Eigenvectors as rows vs columns mixed up | Accuracy near chance (5%) | `Vt.T` gives columns |
| `eigh` returns ascending order | LDA d = 1 uses the worst direction | Reverse the order |
| Mahalanobis at d > N − c | `LinAlgError` or nonsense | Cap d at N − c, store `nan` |
| Hard-coded 220 inside validation folds / E4 | `LinAlgError` in fold, wrong caps | Derive N − c from the fitting set |
| Broadcast 1-NN distances at large d | Out-of-memory / very slow | `‖a‖² + ‖b‖² − 2ab` or `cdist` |
| Random split | ~100% everywhere, flat curves | Pose-spaced split |
| Choosing best d / m / ε from the test curve | Test data used to set parameters (violates brief) | `select_by_validation` on training data |
| Identical min-Euclid and min-Mahal curves in LDA space | Looks like a bug | Expected: $W^T S_w W = I$ |
| `np.mean` over arrays with `nan` | `nan` in plots/table | `np.nanmean` / `np.nanstd` |
| `ModuleNotFoundError: src` in notebooks | Import fails | Add repo root to `sys.path` (section 3) |
| Edits to `src/` not taking effect | Old behaviour persists | `%autoreload 2` or restart kernel |
| Repo inside OneDrive | Sync conflicts, locked `.git` files | Keep repo in a local, non-synced folder |

---

## 14. Code appendix for the report

Include the `src/` files plus the key loop cells from notebooks 02–04. This reads more cleanly than full notebook dumps.

---

## 15. Implementation constraints

- Implement PCA, LDA and all three classifiers with NumPy/SciPy only. sklearn is allowed only in a sanity-check cell, never in `src/`.
- Never pass test indices to any `fit_*` function or to `select_by_validation`.
- Keep the pose-spaced split exactly as in section 5; do not introduce random splits except optional E5.
- Mahalanobis must be capped at d ≤ N − c (220 with default settings); compute the cap from data, don't hard-code it.
- Accuracy arrays: shape `[offset, classifier, d_index]`, saved in one `.npz` with the matching `d_list`.
- Build in this order: `config` → `data` + notebook 01 → `reduction` (PCA) → `classifiers` → notebook 02 → LDA → notebook 03 → `evaluate.select_by_validation` → notebooks 04–05.
- After each module, add a quick test cell (shapes, sorted eigenvalues, no train/test overlap).

## 16. Coding timeline

| Dates | Task |
|---|---|
| 24–26 Oct | Repo, env, `config.py`, `data.py`, notebook 01 with sanity checks |
| 27–30 Oct | `reduction.py` (PCA), `classifiers.py`, notebook 02 (E0, E1, E3) |
| 31 Oct – 2 Nov | LDA, notebook 03 (E2), notebook 04 (E4) |
| 3–5 Nov | Optional E5/E6, notebook 05 figures and table, tag `v1.0-submission` |
