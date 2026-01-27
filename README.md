# Nepali News Classification Project

Multi-Class Text Classification of Nepali News Articles using Deep Learning

# Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Requirements](#requirements)
- [Testing](#testing)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Author](#author)
- [Citation](#citation)

## Project Overview

This project implements a deep learning model to automatically classify Nepali news articles into 20 predefined categories using a fine-tuned XLM-RoBERTa transformer model. The system is designed to handle Nepali text written in Devanagari script and provides accurate multi-class classification for news articles.

**Key Highlights:**
- Fine-tuned XLM-RoBERTa-base model for Nepali text classification
- 20 news categories covering diverse topics
- Handles class imbalance using weighted loss functions
- Comprehensive evaluation and error analysis
- Web-based inference interface

## Features

- **Multi-class Classification**: Classifies news articles into 20 distinct categories
- **Transformer-based Model**: Uses XLM-RoBERTa for robust multilingual text understanding
- **Class Imbalance Handling**: Implements class weights to handle imbalanced dataset
- **Comprehensive Evaluation**: Includes accuracy, precision, recall, and F1-score metrics
- **Error Analysis**: Detailed analysis of misclassified samples
- **Visualization**: Generates training curves, confusion matrices, and performance visualizations
- **Web Interface**: Streamlit/Gradio-based frontend for easy inference
- **Modular Codebase**: Well-organized, maintainable code structure
- **Jupyter Notebooks**: Step-by-step EDA, preprocessing, training, and evaluation notebooks

## Dataset

### Dataset Information

- **Dataset Name**: Nepali News Dataset (20 categories)
- **Total Documents**: 7,023 articles
- **Number of Categories**: 20
- **Language**: Nepali (Devanagari script)
- **Source**: Shahi & Pant (2018)

### Category Distribution

| Category | Number of Documents |
|----------|-------------------|
| Sports | 700 |
| Entertainment | 634 |
| Bank | 617 |
| Economy | 600 |
| Politics | 550 |
| Opinion | 500 |
| Business | 307 |
| World | 313 |
| Society | 353 |
| Interview | 330 |
| Employment | 304 |
| Automobiles | 246 |
| Blog | 259 |
| Literature | 251 |
| Tourism | 265 |
| Agriculture | 200 |
| Education | 185 |
| Health | 180 |
| Technology | 118 |
| Migration | 111 |

### Data Splits

- **Training Set**: 70% (~4,878 samples)
- **Validation Set**: 15% (~1,045 samples)
- **Test Set**: 15% (~1,046 samples)

### Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{shahi2018nepali,
  title={Nepali news classification using Naive Bayes, Support Vector Machines and Neural Networks},
  author={Shahi, Tej Bahadur and Pant, Ashok Kumar},
  booktitle={2018 International Conference on Communication information and Computing Technology (ICCICT)},
  pages={1--5},
  year={2018},
  organization={IEEE}
}
```

## Model Architecture

### Architecture Details

- **Base Model**: XLM-RoBERTa-base (278M parameters)
- **Hidden Size**: 768
- **Max Sequence Length**: 512 tokens
- **Classification Head**: Linear layer (768 → 20)
- **Dropout Rate**: 0.3
- **Total Parameters**: 278,059,028
- **Trainable Parameters**: 278,059,028

### Model Components

1. **Encoder**: Pre-trained XLM-RoBERTa-base transformer
2. **Pooling**: CLS token representation
3. **Dropout**: Regularization layer (0.3 dropout rate)
4. **Classifier**: Linear layer for 20-class classification

### Training Configuration

- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Epochs**: 5-10 (with early stopping)
- **Warmup Epochs**: 1
- **Weight Decay**: 0.01
- **Gradient Clipping**: 1.0
- **Early Stopping Patience**: 2-3 epochs
- **Class Weights**: Computed from training data distribution

## Performance Metrics

### Test Set Performance

- **Accuracy**: 69.89%
- **Weighted F1-Score**: 0.7136
- **Macro F1-Score**: 0.6843
- **Weighted Precision**: 0.7565
- **Weighted Recall**: 0.6989

### Best Model Performance (Validation Set)

- **Accuracy**: 70.81%
- **Weighted F1-Score**: 0.7252
- **Macro F1-Score**: 0.7031

### Per-Class Performance

Top performing categories:
- **Politics**: F1 = 0.9878 (97.59% recall)
- **Migration**: F1 = 1.0000 (100% recall)
- **Blog**: F1 = 0.9315 (87.18% recall)
- **Opinion**: F1 = 0.8971 (81.33% recall)
- **Literature**: F1 = 0.8955 (81.08% recall)

Categories needing improvement:
- **Business**: F1 = 0.4138 (26.09% recall)
- **Interview**: F1 = 0.5075 (34.00% recall)
- **Technology**: F1 = 0.5833 (41.18% recall)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM
- 10GB+ free disk space

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd nepali_news_classification
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install PyTorch with CUDA Support

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Set Up Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration (if needed)
```

### Step 6: Download Dataset

Place the Nepali news dataset in the following directory structure:
```
data/
└── nepali_news_dataset_20_categories_large/
    └── nepali_news_dataset_20_categories_large/
        ├── Agriculture/
        ├── Automobiles/
        ├── Bank/
        └── ... (other categories)
```

## Project Structure

```
nepali_news_classification/
├── data/                              # Dataset files
│   ├── nepali_news_dataset_20_categories_large/
│   ├── processed/                     # Processed data files
│   └── splits/                        # Train/val/test splits
│
├── models/                            # Model files
│   ├── checkpoints/                   # Training checkpoints
│   ├── pretrained/                    # Pre-trained models
│   ├── saved_models/                  # Saved model weights
│   ├── training_history.json         # Training metrics
│   └── training_summary.json          # Training summary
│
├── src/                               # Source code
│   ├── data_loading.py               # Dataset loading utilities
│   ├── data_preprocessing.py         # Text preprocessing
│   ├── model.py                      # Model architecture
│   └── train.py                      # Training script
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_data_preprocessing.ipynb   # Data preprocessing
│   ├── 03_model_training.ipynb       # Model training
│   ├── 04_evaluation.ipynb           # Model evaluation
│   └── 05_error_analysis.ipynb       # Error analysis
│
├── results/                           # Results and outputs
│   ├── metrics/                      # Evaluation metrics
│   ├── visualizations/               # Generated plots
│   ├── reports/                      # Evaluation reports
│   └── misclassified_samples.csv     # Error analysis data
│
├── frontend/                          # Web application
│   ├── app.py                        # Streamlit/Gradio app
│   ├── static/                       # Static assets
│   └── templates/                    # HTML templates
│
├── config/                            # Configuration files
│   ├── config.yaml                   # Main configuration
│   └── model_config.json             # Model configuration
│
├── tests/                             # Unit tests
│   ├── test_data_loading.py
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── logs/                              # Training logs
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Usage

### 1. Data Loading and Preprocessing

**Using Python script:**
```bash
python src/data_loading.py
```

**Using Jupyter notebook:**
```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_data_preprocessing.ipynb
```

### 2. Model Training

**Basic training:**
```bash
python src/train.py
```

**Training with custom parameters:**
```bash
python src/train.py \
    --train_csv data/splits/train.csv \
    --val_csv data/splits/val.csv \
    --class_weights data/splits/class_weights.json \
    --model_name xlm-roberta-base \
    --num_classes 20 \
    --max_length 512 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 5 \
    --warmup_epochs 1 \
    --dropout_rate 0.3 \
    --output_dir models \
    --early_stopping_patience 3
```

**Using Jupyter notebook:**
```bash
jupyter notebook notebooks/03_model_training.ipynb
```

### 3. Model Evaluation

**Using Jupyter notebook:**
```bash
jupyter notebook notebooks/04_evaluation.ipynb
```

The evaluation notebook generates:
- Overall metrics (accuracy, precision, recall, F1-score)
- Per-class metrics
- Confusion matrices
- Classification reports

### 4. Error Analysis

```bash
jupyter notebook notebooks/05_error_analysis.ipynb
```

This notebook provides:
- Misclassified sample analysis
- Confused category pairs
- Text length analysis
- Per-class error patterns

### 5. Inference

**Using Python:**
```python
from src.model import create_model
from transformers import XLMRobertaTokenizer
import torch

# Load model
model = create_model('xlm-roberta-base', num_classes=20)
checkpoint = torch.load('models/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# Classify text
text = "Your Nepali news article text here"
encoding = tokenizer(text, truncation=True, padding='max_length', 
                     max_length=512, return_tensors='pt')
with torch.no_grad():
    logits = model(**encoding)
    prediction = torch.argmax(logits, dim=-1).item()
```


## Configuration

### Configuration Files

**`config/config.yaml`** - Main configuration file:
```yaml
model:
  name: "xlm-roberta-base"
  num_labels: 20
  max_length: 512
  hidden_size: 768

training:
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 5
  warmup_steps: 500
  weight_decay: 0.01
  dropout: 0.1
  early_stopping_patience: 2

data:
  data_dir: "data/raw"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  use_class_weights: true

paths:
  model_save_dir: "models/saved_models"
  checkpoint_dir: "models/checkpoints"
  results_dir: "results"
  logs_dir: "logs"
```

### Environment Variables

Create a `.env` file from `.env.example`:
```bash
# Optional: Add any environment-specific variables
# CUDA_VISIBLE_DEVICES=0
# MODEL_CACHE_DIR=/path/to/cache
```

## Requirements

### Core Dependencies

- **PyTorch** >= 2.0.0 (with CUDA support recommended)
- **Transformers** >= 4.30.0 (Hugging Face)
- **Datasets** >= 2.12.0

### Data Processing

- **pandas** >= 2.0.0
- **numpy** >= 1.24.0
- **scikit-learn** >= 1.3.0

### Visualization

- **matplotlib** >= 3.7.0
- **seaborn** >= 0.12.0
- **plotly** >= 5.14.0

### Utilities

- **tqdm** >= 4.65.0
- **pyyaml** >= 6.0
- **python-dotenv** >= 1.0.0

### Development

- **jupyter** >= 1.0.0
- **ipykernel** >= 6.23.0
- **notebook** >= 6.5.0
- **pytest** >= 7.4.0 (optional, for testing)
- **black** >= 23.7.0 (optional, for code formatting)
- **flake8** >= 6.1.0 (optional, for linting)

See `requirements.txt` for complete list with versions.


## Results

### Generated Outputs

After training and evaluation, the following results are generated:

**Metrics:**
- `results/test_evaluation.json` - Test set evaluation metrics
- `results/test_evaluation.txt` - Human-readable evaluation report
- `results/error_analysis_summary.json` - Error analysis summary

**Visualizations:**
- `results/visualizations/training_curves.png` - Training/validation metrics over epochs
- `results/visualizations/confusion_matrix_test.png` - Test set confusion matrix
- `results/visualizations/per_class_metrics_test.png` - Per-class performance metrics
- `results/visualizations/error_analysis_*.png` - Various error analysis plots

**Model Files:**
- `models/best_model.pt` - Best model checkpoint (based on validation F1)
- `models/final_model.pt` - Final model after training
- `models/training_history.json` - Complete training history

## Author

**Rishav Singh**  
Student ID: NP01MS7A240010  
MSc in Information Technology  
Islington College


## License

This project is for academic/research purposes. Please refer to the dataset citation for dataset usage terms.

**Last Updated**: January 2026
