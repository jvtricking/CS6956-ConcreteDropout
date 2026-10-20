# Standard Deep Ensembles vs. Concrete Dropout

This project compares independently trained **standard deep ensembles** against **Concrete Dropout** on a synthetic 2-feature regression task and the 10-feature scikit-learn diabetes dataset.

## 1. Install Conda (one time only)

Install **Miniconda** or **Anaconda** from the official documentation: https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html

**On Windows:** open **Anaconda Prompt** or **Miniconda Prompt** after installing. If PowerShell says `conda is not recognized`, start from Anaconda Prompt instead; you don't need to add Conda to your PATH.

## 2. Put these three files in the same project folder

- `concrete_dropout_ensembles_report.ipynb`
- `environment.yml`
- `README.md`

In Anaconda Prompt, go to that folder (replace the example path with yours):

```bat
cd /d C:\SchoolDev\concrete-dropout
```

## 3. Create and activate the environment

Run once:

```bat
conda env create -f environment.yml
```

Each time you work on the project, activate it:

```bat
conda activate concrete-dropout
```

Optional: verify imports and check PyTorch's device:

```bat
python -c "import torch, numpy, pandas, matplotlib, sklearn; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The environment installs CPU-capable PyTorch by default. The notebook works without an NVIDIA GPU. GPU-enabled CUDA builds can be installed separately using the instructions on https://pytorch.org/get-started/locally/ .

## 4. Register the Jupyter kernel (once)

```bat
python -m ipykernel install --user --name concrete-dropout --display-name "Python (Concrete Dropout)"
```

## 5. Open the notebook

**Option A: JupyterLab**

```bat
jupyter lab
```

Open `concrete_dropout_ensembles_report.ipynb` and select the **Python (Concrete Dropout)** kernel.

**Option B: VS Code**

Open the project folder in VS Code, open the `.ipynb` file, click **Select Kernel** (top right), and choose **Python (Concrete Dropout)**. Install VS Code's Python and Jupyter extensions if prompted.

## 6. First test, then full experiment

In the notebook's setup code cell, set:

```python
QUICK_RUN = True
```

Choose **Run All**. This is a fast *smoke test* of imports, training, tables, and figures—not results you should report.

For the final experiment, change to:

```python
QUICK_RUN = False
```

Then **Restart Kernel and Run All**. The full configuration trains 5 random seeds for each model family on both datasets and tests ensemble sizes 1, 2, 3, and 5. Timing depends on your computer.

## 7. What to analyze for the report

- Synthetic 2D task: visual fit, known noise and predictive uncertainty, including OOD behavior.
- Standard ensembles vs. Concrete Dropout: RMSE, negative log-likelihood (NLL), 90% interval coverage and width, aleatoric and epistemic uncertainty.
- Ensemble size: 1, 2, 3, 5 independently trained members.
- Monte Carlo passes: number of sampled Concrete Dropout predictions.
- Training vs. test metrics and curves, plus learned dropout probabilities.
- Higher-dimensional real dataset: diabetes regression, 10 input features.

**Method note:** Both model families use Gaussian NLL and L2 weight regularization (a Gaussian-weight-prior interpretation). Concrete Dropout additionally learns dropout rates. The real dataset is provided by scikit-learn, so there is no separate dataset download.

## NumPy/PyTorch compatibility

This environment pins `numpy=1.26.4` because some PyTorch builds compiled against NumPy 1.x cannot initialize NumPy 2.x. If you already created the environment with NumPy 2.x, repair it in Anaconda Prompt:

```bat
conda activate concrete-dropout
conda install numpy=1.26.4 -y
python -c "import numpy, torch; print(numpy.__version__, torch.__version__); print(torch.tensor([1.0]).numpy())"
```

Restart the notebook kernel afterward.

## Troubleshooting

**`conda` not recognized**: Launch Anaconda Prompt / Miniconda Prompt, not plain PowerShell. Confirm with `conda --version`.

**Kernel not listed**: Activate the environment, rerun the `ipykernel install` command, then restart VS Code/Jupyter.

**`ModuleNotFoundError`**: Verify you selected the **Python (Concrete Dropout)** kernel instead of the system Python interpreter.

**Environment already exists**: Use `conda activate concrete-dropout` instead of creating it again. To update packages after changing `environment.yml`:

```bat
conda env update -f environment.yml --prune
```

**Notebook takes too long**: Set `QUICK_RUN = True` to check that it works, then use `False` for report-quality experiments.
