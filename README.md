# Nepali News Classification Project

Multi-Class Text Classification of Nepali News Articles using Deep Learning

## Project Overview

This project implements a deep learning model to automatically classify Nepali news articles into 20 predefined categories using fine-tuned XLM-RoBERTa transformer model.

## Dataset

- **Dataset:** Nepali News Dataset (20 categories, 6,973 documents)
- **Source:** Shahi & Pant (2018)
- **Language:** Nepali (Devanagari script)

## Model

- **Architecture:** XLM-RoBERTa-base
- **Framework:** PyTorch + Hugging Face Transformers
- **Task:** Multi-class text classification (20 categories)

## Project Structure

```
nepali_news_classification/
├── data/              # Dataset files
├── models/            # Trained models and checkpoints
├── src/               # Source code
├── notebooks/         # Jupyter notebooks
├── results/           # Results and visualizations
├── frontend/          # Web application
├── config/            # Configuration files
├── scripts/           # Utility scripts
├── docs/              # Documentation
├── tests/             # Unit tests
└── logs/              # Training logs
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (copy .env.example to .env)

3. Run EDA:
```bash
python notebooks/01_eda.ipynb
```

## Training

```bash
python src/train.py
```

## Evaluation

```bash
python src/evaluate.py
```

## Usage

```bash
python src/inference.py --text "Your Nepali news article text here"
```

## Frontend

```bash
cd frontend
streamlit run app.py
```

## License

[Your License]

## Author

[Your Name]
[Your Student ID]
