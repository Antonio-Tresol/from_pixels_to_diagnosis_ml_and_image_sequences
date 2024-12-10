# From Pixels to Diagnosis: ML and Image Sequences

In this project, we focused on intracranial hemorrhage diagnosis using medical image sequences through state-of-the-art vision models (ViViT and ConvNeXT). We explore how effective video models can be at classifying patients based on a sequence of images.

## Project Overview

This repository contains implementations of two deep learning approaches for hemorrhage diagnosis:

- ViViT (Video Vision Transformer): Processes sequences of medical images to make predictions.
- ConvNeXT: Analyzes individual images and aggregates results for patient-level diagnosis.

## Features

- Automated dataset preprocessing and organization.
- Implementation of both ViViT and ConvNeXT models.
- Patient-level and image-level prediction capabilities.
- Comprehensive evaluation metrics (accuracy, precision, recall).
- Integration with Weights & Biases (wandb) for experiment tracking.
- Confusion matrix visualization for model performance analysis.

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
├── config.py               # Configuration parameters and constants
├── preview_from_pixels_to_diagnosis_using_machine_learning_to_classify_medical_image_sequences  # a preview of the research paper
├── clean.py                # Dataset preprocessing and organization
├── vivit_data_handling.py  # Dataset preprocessing for the vivit model
├── convnext_data_handling.py  # Dataset preprocessing for convnext model
├── vivit_experiment.py     # ViViT model implementation and training
├── convnext_experiment.py  # ConvNeXT model implementation and training
├── logging_and_model_evaluation.py # Evaluation metrics and logging utilities
└── dataset/                # Directory containing the medical image data
```

## Setup and Usage

1. Prepare your dataset:
   ```python
   python clean.py
   ```

2. Run experiments:


   make sure you have a `key.py` file with you wandb api key and that you changed the project name in the `config.py` file. Then
   - For ViViT model:
     ```python
     python vivit_experiment.py
     ```
   - For ConvNeXT model:
     ```python
     python convnext_experiment.py
     ```

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

