# From Pixels to Diagnosis: ML and Image Sequences

In this project, we focused on intracranial hemorrhage diagnosis using medical image sequences through state-of-the-art vision models (ViViT and ConvNeXT). We explore how effective video models can be at classifying patients based on a sequence of images.

## 🚀 Quick Start

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
2. Click "Reopen in Container" when prompted (or press `F1` → "Dev Containers: Reopen in Container")
3. Wait for the container to build (5-10 minutes first time)
4. Compile: `cd latex && latexmk -pdf main.tex`

The compiled PDF will be at `latex/main.pdf`.

## Project Overview

This repository contains implementations of two deep learning approaches for hemorrhage diagnosis:

- ViViT (Video Vision Transformer): Processes sequences of medical images to make predictions.
- ConvNeXT: Analyzes individual images and aggregates results for patient-level diagnosis.

## Features

- Modern Python project structure with `pyproject.toml` and `uv` package manager
- Automated dataset preprocessing and organization
- Implementation of both ViViT and ConvNeXT models
- Patient-level and image-level prediction capabilities
- Comprehensive evaluation metrics (accuracy, precision, recall)
- Integration with Weights & Biases (wandb) for experiment tracking
- Confusion matrix visualization for model performance analysis

## Requirements

- Python
- PyTorch
- Transformers library
- pandas
- numpy
- wandb (for experiment tracking)
- evaluate (for metrics computation)

## Project Structure

```
├── .devcontainer/          # Development container configuration
│   ├── Dockerfile          # Custom LaTeX environment
│   └── devcontainer.json   # VS Code container settings
├── latex/                  # LaTeX source files for research paper
│   ├── main.tex           # Main document
│   ├── configuration.tex  # Package imports and settings
│   ├── sections/          # Paper sections
│   ├── references/        # Bibliography files (.bib)
│   └── imgs/              # Figures and images
├── pyproject.toml          # Python project configuration and dependencies
├── config.py               # Configuration parameters and constants
├── preview_from_pixels_to_diagnosis_using_machine_learning_to_classify_medical_image_sequences.pdf  # Research paper preview
├── clean.py                # Dataset preprocessing and organization
├── vivit_data_handling.py  # Dataset preprocessing for the vivit model
├── convnext_data_handling.py  # Dataset preprocessing for convnext model
├── vivit_experiment.py     # ViViT model implementation and training
├── convnext_experiment.py  # ConvNeXT model implementation and training
├── logging_and_model_evaluation.py # Evaluation metrics and logging utilities
├── convnext_metrics_new/  # ConvNeXT model results
├── vivit_metrics_new/     # ViViT model results
└── dataset/                # Directory containing the medical image data
```

## ML Setup and Usage

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Setup

1. **Install uv** (if not already installed):
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Or with pip
   pip install uv
   ```

2. **Install dependencies** using uv and pyproject.toml:
   ```bash
   uv sync
   ```

3. **Prepare your dataset:**
   ```bash
   uv run clean.py
   ```

4. **Run experiments:**
   
   Make sure you have a `key.py` file with your wandb API key and that you changed the project name in the `config.py` file. Then:
   
   - For ViViT model:
     ```bash
     uv run vivit_experiment.py
     ```
   - For ConvNeXT model:
     ```bash
     uv run convnext_experiment.py
     ```

### Adding New Dependencies

To add new packages to the project:
```bash
uv add <package-name>
```

To add development dependencies:
```bash
uv add --dev <package-name>
```

## Model Configuration

The project uses the following pre-trained models:
- ViViT: `google/vivit-b-16x2-kinetics400`
- ConvNeXT: `facebook/convnext-tiny-224`

## 📝 LaTeX Paper Compilation

### Manual Setup (Advanced Users)

If you prefer not to use Dev Containers, you can set up LaTeX manually:

#### Installing TeX Live

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install texlive-full
```

**macOS:**
```bash
# Using MacTeX (recommended)
brew install --cask mactex
```

**Windows:**
Download and install [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)

#### Required LaTeX Packages

```bash
tlmgr install cite fancyvrb ieeetran paralist csvsimple titlesec soul \
              markdown adjustbox booktabs multirow pgfplots float natbib \
              times psnfss xcolor hyperref graphicx amsmath amsfonts amssymb
```

#### Compilation Commands

```bash
cd latex
latexmk -pdf main.tex
```

### Troubleshooting LaTeX

**Missing packages:**
```bash
tlmgr install <package-name>
```

**Font errors:**
```bash
updmap-sys && texhash
```

**Clean build files:**
```bash
latexmk -C
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors
- Antonio Badilla-Olivas
- Enrique Vilchez-Lizano
- Kenneth Villalobos-Solis
- Brandon Mora-Umaña

