# From Pixels to Diagnosis: ML and Image Sequences

**TL;DR:** We compared a video model (ViViT) against an image model (ConvNeXT) for detecting intracranial hemorrhage from CT scan sequences. ViViT won (72% accuracy, 62% recall vs 60% accuracy, 13% recall). The image model struggles because decomposing sequences into individual slices creates a heavily imbalanced training set (~90% negative). Video models avoid this by consuming the whole sequence at once, which also lets them pick up on temporal and spatial patterns across slices.

We explore how effective video models can be at classifying patients based on a sequence of medical images, using the [CT-ICH dataset](https://physionet.org/content/ct-ich/1.3.1/) (82 patients, 36 with hemorrhage).

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
├── .devcontainer/             # Dev container configuration
├── pyproject.toml             # Python dependencies
├── uv.lock                    # Dependency lock file
├── README.md
└── LICENSE
```

## Setup and Usage

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

3. **Download the dataset:**

   The dataset is [Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation](https://physionet.org/content/ct-ich/1.3.1/) from PhysioNet. You need to sign the Restricted Health Data Use Agreement, then download:

   ```bash
   wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/ct-ich/1.3.1/
   ```

4. **Convert and prepare the dataset:**

   The download contains NIfTI files. Convert them to JPGs and generate the required CSVs:

   ```bash
   uv run src/convert_nifti_to_jpg.py
   uv run src/clean.py
   ```

   This creates the `dataset/` folder with `Positives/` and `Negatives/` patient directories.

5. **Run experiments:**

   Create a `src/key.py` file with your [Weights & Biases](https://wandb.ai) API key:

   ```python
   WANDB_KEY = "your_key_here"
   ```

   ```bash
   # ViViT (video model)
   uv run src/vivit_experiment.py

   # ConvNeXT (image model)
   uv run src/convnext_experiment.py
   ```

   Results are saved to `data/vivit_metrics_new/` and `data/convnext_metrics_new/`.

## Model Configuration

The project uses the following pre-trained models:
- ViViT: `google/vivit-b-16x2-kinetics400`
- ConvNeXT: `facebook/convnext-tiny-224`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors
- Antonio Badilla-Olivas
- Enrique Vilchez-Lizano
- Kenneth Villalobos-Solis
- Brandon Mora-Umaña
