# UrbanGAN: A Cross-Modal Generative Framework for Urban Socioeconomic Prediction

This repository contains the official PyTorch implementation for the paper "UrbanGAN". Our work addresses the challenge of data scarcity in urban sensing by synthesizing virtual street-view features directly from satellite imagery and textual descriptions to enhance downstream socioeconomic prediction tasks.

## Framework Overview

UrbanGAN is a sequential, three-stage framework:

1.  **Stage 1: Foundation Model Pre-training (Cross-Modal Alignment)**
    * We align satellite images and their corresponding text descriptions in a shared semantic space. This is achieved by freezing powerful pre-trained Vision Transformer backbones and training only lightweight projection heads using a fine-grained contrastive loss.
    * *Implemented in `pretrain.py` (`train_satellite_text_alignment` function).*

2.  **Stage 2: Generative Feature Synthesis (Cross-View Translation)**
    * We train a Generative Adversarial Network (GAN) to learn the mapping from the aligned satellite-text feature space to the street-view feature space. The generator is a simple MLP conditioned on both satellite and text features.
    * *Implemented in `pretrain.py` (`train_r3gan_style_stage2` function).*

3.  **Stage 3: Downstream Task Adaptation (Feature Fusion & Prediction)**
    * The frozen, pre-trained models are used as a feature pipeline. For a new satellite image, we extract its native feature and generate a corresponding virtual street-view feature.
    * These two features are then concatenated and used to train a powerful XGBoost predictor for the final socioeconomic task.
    * *Implemented in `downstream.py`.*

## Setup

### 1. Create Environment and Install Dependencies

We recommend using a conda environment.

# Create and activate a new conda environment
conda create -n urbangan python=3.10
conda activate urbangan

# Install dependencies
pip install -r requirements.txt


## Data Preparation

The code expects a specific directory structure. Please organize your data as follows:

```
UrbanGAN/
├── data/
│   ├── CVUSA_subset/
│   │   ├── file_paths.csv        # Pre-training CSV mapping satellite to street-view paths
│   │   ├── bingmap/                  # Directory for CVUSA satellite images
│   │   ├── streetview/               # Directory for CVUSA street-view images
│   ├── CVUSA_bingmapcaptions_output.json # Captions for CVUSA satellite images
│   │
│   ├── Haidian/                  # Example downstream dataset directory
│   │   └── images/               # Satellite images for the downstream task
│   ├── Haidian_train.csv         # Downstream training/validation data
│   ├── Haidian_test.csv          # Downstream test data
│   └── Haidiancaptions_output.json # Captions for downstream satellite images
│
├── models/                       # Directory to save trained models
│
├── pretrain.py                   # Script for Stage 1 & 2
├── downstream.py                 # Script for Stage 3
└── README.md
```

**Note:** The paths in the config dictionaries inside the scripts (e.g., `CONFIG_R3GAN_STYLE`, `CONFIG_DOWNSTREAM`) are currently hardcoded. Please modify these paths to match your local data directory structure if it differs.

## Usage

The model is trained and evaluated in two main steps.

### Step 1: Pre-training (Stages 1 & 2)

This step runs both the cross-modal alignment (Stage 1) and the GAN training (Stage 2). It learns the core components for feature generation.

```bash
python pretrain.py
```

This script will:
1.  Train the satellite-text alignment models based on the `CVUSA` dataset.
2.  Save the best encoder and projection head weights.
3.  Train the GAN generator based on the aligned features.
4.  Save the best generator weights.

### Step 2: Downstream Task (Stage 3)

After the pre-training is complete, this step uses the generated weights to perform the final socioeconomic prediction task.

```bash
python downstream.py
```

This script will:
1.  Load the frozen, pre-trained encoders and generator from Step 1.
2.  Extract features for the downstream dataset (e.g., Haidian).
3.  Train an XGBoost model for each socioeconomic indicator (Population, Carbon, Building Heights).
4.  Save the trained XGBoost models.
5.  Evaluate the models on the test set and save prediction and metric CSV files.

## Expected Output

After running `downstream.py`, you will find the following in your directory:

* **Trained Predictors**: `.ubj` files for each trained XGBoost model.
* **Prediction Results**: CSV files containing the ground truth and predicted values for the test set.
* **Evaluation Metrics**: A summary CSV containing the R², RMSE, and MAE scores for each task.
