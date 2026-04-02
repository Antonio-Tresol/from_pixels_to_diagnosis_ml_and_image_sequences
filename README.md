# From Pixels to Diagnosis: ML and Image Sequences

In this project, we focused on intracranial hemorrhage diagnosis using medical image sequences through state-of-the-art vision models (ViViT and ConvNeXT). We explore how effective video models can be at classifying patients based on a sequence of images.

## Quick Start

### For Machine Learning Experiments
Follow the [ML Setup](#ml-setup-and-usage) section below.

### For LaTeX Paper Compilation (Recommended: Dev Container)
The easiest way to compile the research paper is using VS Code Dev Containers:

**Prerequisites:**
- [Docker](https://docs.docker.com/get-docker/)
- [Visual Studio Code](https://code.visualstudio.com/)
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

**Setup:**
1. Open this repository in VS Code
2. Click "Reopen in Container" when prompted (or press `F1` -> "Dev Containers: Reopen in Container")
3. Wait for the container to build (5-10 minutes first time)
4. Compile: `cd paper && latexmk -pdf main.tex`

The compiled PDF will be at `paper/main.pdf`.

## Project Overview

This repository contains implementations of two deep learning approaches for hemorrhage diagnosis:

- ViViT (Video Vision Transformer): Processes sequences of medical images to make predictions.
- ConvNeXT: Analyses individual images and aggregates results for patient-level diagnosis.

## Project Structure

```
├── src/                       # Python source code
│   ├── config.py              # Configuration parameters and constants
│   ├── clean.py               # Dataset preprocessing and organisation
│   ├── convert_nifti_to_jpg.py # NIfTI to JPG conversion
│   ├── image_processing.py    # Image loading and sampling utilities
│   ├── vivit.py               # ViViT model initialisation
│   ├── vivit_data_handling.py # ViViT dataset preprocessing
│   ├── vivit_experiment.py    # ViViT training and evaluation
│   ├── convnext.py            # ConvNeXT model initialisation
│   ├── convnext_data_handling.py # ConvNeXT dataset preprocessing
│   ├── convnext_experiment.py # ConvNeXT training and evaluation
│   └── logging_and_model_evaluation.py # Metrics and logging
├── notebooks/                 # Jupyter notebooks
│   ├── analysis.ipynb         # Results analysis
│   ├── analysis_latex.ipynb   # LaTeX-formatted analysis
│   └── eda.ipynb              # Exploratory data analysis
├── data/                      # Experiment results
│   ├── convnext_metrics_new/  # ConvNeXT metrics per run
│   ├── vivit_metrics_new/     # ViViT metrics per run
│   └── wandb_experiment_data.csv # Exported wandb data
├── paper/                     # LaTeX source for research paper
│   ├── main.tex               # Main document
│   ├── configuration.tex      # Package imports and settings
│   ├── sections/              # Paper sections
│   ├── references/            # Bibliography files (.bib)
│   └── imgs/                  # Figures and images
├── .devcontainer/             # Dev container configuration
├── pyproject.toml             # Python dependencies
├── uv.lock                    # Dependency lock file
├── reviewer_response.md       # Reviewer response document
├── README.md
└── LICENSE
```

## ML Setup and Usage

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Prepare your dataset:**
   ```bash
   uv run src/clean.py
   ```

4. **Run experiments:**

   Make sure you have a `src/key.py` file with your wandb API key and that you changed the project name in `src/config.py`. Then:

   - For ViViT model:
     ```bash
     uv run src/vivit_experiment.py
     ```
   - For ConvNeXT model:
     ```bash
     uv run src/convnext_experiment.py
     ```

## Model Configuration

The project uses the following pre-trained models:
- ViViT: `google/vivit-b-16x2-kinetics400`
- ConvNeXT: `facebook/convnext-tiny-224`

## Paper Compilation

### Manual Setup

If you prefer not to use Dev Containers:

```bash
cd paper
latexmk -pdf main.tex
```

Or with Docker:
```bash
docker build -t latex-env .devcontainer/
docker run --rm -v $(pwd)/paper:/workspace latex-env latexmk -pdf main.tex
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors
- Antonio Badilla-Olivas
- Enrique Vilchez-Lizano
- Kenneth Villalobos-Solis
- Brandon Mora-Umaña
