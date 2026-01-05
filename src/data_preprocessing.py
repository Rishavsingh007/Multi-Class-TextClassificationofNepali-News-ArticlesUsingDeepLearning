import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
import re
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import XLMRobertaTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text string
    
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace (multiple spaces, tabs, newlines)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove special control characters but preserve Nepali Devanagari script
    # Remove characters that are not printable (except Nepali Unicode range)
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Normalize unicode (optional - can be enabled if needed)
    # text = unicodedata.normalize('NFKC', text)
    
    return text


def preprocess_texts(texts: List[str], clean: bool = True) -> List[str]:
    """
    Preprocess a list of texts.
    
    Args:
        texts: List of text strings
        clean: Whether to apply text cleaning
    
    Returns:
        List of preprocessed texts
    """
    if clean:
        texts = [clean_text(text) for text in texts]
    
    # Remove empty texts
    texts = [text for text in texts if text and len(text.strip()) > 0]
    
    return texts


def tokenize_texts(
    texts: List[str],
    tokenizer: XLMRobertaTokenizer,
    max_length: int = 512,
    padding: str = 'max_length',
    truncation: bool = True
) -> Dict:
    """
    Tokenize texts using XLM-RoBERTa tokenizer.
    
    Args:
        texts: List of text strings
        tokenizer: XLM-RoBERTa tokenizer instance
        max_length: Maximum sequence length
        padding: Padding strategy ('max_length' or 'longest')
        truncation: Whether to truncate sequences
    
    Returns:
        Dictionary with 'input_ids', 'attention_mask', and 'token_count'
    """
    logger.info(f"Tokenizing {len(texts)} texts...")
    
    # Tokenize all texts
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors='pt'  # Return PyTorch tensors
    )
    
    # Calculate token counts (before padding)
    token_counts = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=truncation)
        token_counts.append(len(tokens))
    
    logger.info(f"Tokenization complete. Average tokens: {np.mean(token_counts):.1f}")
    logger.info(f"Max tokens: {max(token_counts)}, Min tokens: {min(token_counts)}")
    
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'token_count': token_counts
    }


def create_data_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation/test splits with stratified sampling.
    
    Args:
        df: DataFrame with 'category_id' column
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility
        stratify: Whether to use stratified sampling
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    logger.info(f"Creating data splits: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
    
    # First split: train vs (val + test)
    stratify_col = df['category_id'] if stratify else None
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        random_state=random_state,
        stratify=stratify_col
    )
    
    # Second split: val vs test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    stratify_col = temp_df['category_id'] if stratify else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_test_ratio),
        random_state=random_state,
        stratify=stratify_col
    )
    
    logger.info(f"Train set: {len(train_df)} samples ({len(train_df)/len(df):.1%})")
    logger.info(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df):.1%})")
    logger.info(f"Test set: {len(test_df)} samples ({len(test_df)/len(df):.1%})")
    
    # Log class distribution in each split
    logger.info("\nClass distribution in splits:")
    logger.info("-" * 60)
    logger.info(f"{'Category':<20} {'Train':<8} {'Val':<8} {'Test':<8}")
    logger.info("-" * 60)
    
    for category in sorted(df['category'].unique()):
        train_count = len(train_df[train_df['category'] == category])
        val_count = len(val_df[val_df['category'] == category])
        test_count = len(test_df[test_df['category'] == category])
        logger.info(f"{category:<20} {train_count:<8} {val_count:<8} {test_count:<8}")
    
    return train_df, val_df, test_df


def compute_class_weights(y: np.ndarray, method: str = 'balanced') -> np.ndarray:
    """
    Compute class weights for imbalanced dataset.
    
    Args:
        y: Array of class labels
        method: Method for computing weights ('balanced' or 'sklearn')
    
    Returns:
        Array of class weights
    """
    unique_classes = np.unique(y)
    
    if method == 'balanced':
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y
        )
    else:
        # Manual computation (inverse frequency)
        class_counts = np.bincount(y)
        total_samples = len(y)
        num_classes = len(unique_classes)
        class_weights = total_samples / (num_classes * class_counts)
    
    # Create dictionary mapping class to weight
    class_weight_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights)}
    
    logger.info("Class weights computed:")
    logger.info("-" * 40)
    logger.info(f"{'Class':<10} {'Weight':<10}")
    logger.info("-" * 40)
    for cls, weight in sorted(class_weight_dict.items()):
        logger.info(f"{cls:<10} {weight:<10.4f}")
    
    return class_weights, class_weight_dict


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path = None
) -> None:
    """
    Save train/validation/test splits to CSV files.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Output directory (default: data/splits)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'data' / 'splits'
    
    # Handle case where path exists as a file
    if output_dir.exists() and output_dir.is_file():
        logger.warning(f"Removing existing file to create directory: {output_dir}")
        output_dir.unlink()
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # Save splits
    train_path = output_dir / 'train.csv'
    val_path = output_dir / 'val.csv'
    test_path = output_dir / 'test.csv'
    
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')
    
    logger.info(f"Saved train split: {train_path} ({len(train_df)} samples)")
    logger.info(f"Saved validation split: {val_path} ({len(val_df)} samples)")
    logger.info(f"Saved test split: {test_path} ({len(test_df)} samples)")


def load_splits(splits_dir: Path = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load previously saved train/validation/test splits.
    
    Args:
        splits_dir: Directory containing split files (default: data/splits)
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if splits_dir is None:
        splits_dir = Path(__file__).parent.parent / 'data' / 'splits'
    else:
        splits_dir = Path(splits_dir)
    
    train_path = splits_dir / 'train.csv'
    val_path = splits_dir / 'val.csv'
    test_path = splits_dir / 'test.csv'
    
    if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
        raise FileNotFoundError(f"Split files not found in {splits_dir}")
    
    train_df = pd.read_csv(train_path, encoding='utf-8')
    val_df = pd.read_csv(val_path, encoding='utf-8')
    test_df = pd.read_csv(test_path, encoding='utf-8')
    
    logger.info(f"Loaded splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df


def preprocess_dataset(
    df: pd.DataFrame,
    tokenizer_name: str = 'xlm-roberta-base',
    max_length: int = 512,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    clean_texts: bool = True,
    save_to_disk: bool = True,
    output_dir: Path = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, XLMRobertaTokenizer]:
    """
    Complete preprocessing pipeline.
    
    Args:
        df: DataFrame with 'text' and 'category_id' columns
        tokenizer_name: Name of tokenizer model
        max_length: Maximum sequence length
        train_ratio: Training set proportion
        val_ratio: Validation set proportion
        test_ratio: Test set proportion
        random_state: Random seed
        clean_text: Whether to clean text
        save_splits: Whether to save splits to disk
        output_dir: Output directory for splits
    
    Returns:
        Tuple of (train_df, val_df, test_df, class_weight_dict, tokenizer)
    """
    logger.info("=" * 60)
    logger.info("Starting Data Preprocessing Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input dataset: {len(df)} samples")
    
    # Step 1: Text cleaning
    if clean_texts:
        logger.info("Step 1: Cleaning texts...")
        df = df.copy()
        df['text'] = df['text'].apply(clean_text)
        # Remove empty texts after cleaning
        df = df[df['text'].str.len() > 0].reset_index(drop=True)
        logger.info(f"After cleaning: {len(df)} samples")
    
    # Step 2: Create data splits
    logger.info("Step 2: Creating data splits...")
    train_df, val_df, test_df = create_data_splits(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        stratify=True
    )
    
    # Step 3: Compute class weights
    logger.info("Step 3: Computing class weights...")
    class_weights, class_weight_dict = compute_class_weights(
        train_df['category_id'].values,
        method='balanced'
    )
    
    # Step 4: Initialize tokenizer
    logger.info("Step 4: Initializing tokenizer...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_name)
    logger.info(f"Tokenizer loaded: {tokenizer_name}")
    logger.info(f"Vocabulary size: {tokenizer.vocab_size:,}")
    
    # Step 5: Tokenize texts (optional - can be done during training)
    logger.info("Step 5: Tokenization will be done during training")
    logger.info("Texts are ready for tokenization with max_length={}".format(max_length))
    
    # Step 6: Save splits
    if save_to_disk:
        logger.info("Step 6: Saving data splits...")
        save_splits(train_df, val_df, test_df, output_dir)
        
        # Save class weights
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'data' / 'splits'
        
        weights_path = output_dir / 'class_weights.json'
        with open(weights_path, 'w', encoding='utf-8') as f:
            json.dump(class_weight_dict, f, indent=2)
        logger.info(f"Saved class weights: {weights_path}")
    
    logger.info("=" * 60)
    logger.info("Preprocessing Pipeline Complete!")
    logger.info("=" * 60)
    
    return train_df, val_df, test_df, class_weight_dict, tokenizer


if __name__ == '__main__':
    """
    Example usage and testing
    """
    print("=" * 60)
    print("Data Preprocessing Pipeline")
    print("=" * 60)
    print()
    
    try:
        # Load dataset
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from data_loading import load_nepali_news_data, load_dataset_from_csv
        
        # Try loading from CSV first (faster)
        try:
            logger.info("Attempting to load from CSV...")
            df, label_to_id, id_to_label = load_dataset_from_csv()
            logger.info("Loaded from CSV successfully")
        except FileNotFoundError:
            logger.info("CSV not found, loading from raw files...")
            df, label_to_id, id_to_label = load_nepali_news_data()
        
        print(f"\nLoaded {len(df)} documents")
        print()
        
        # Run preprocessing pipeline
        train_df, val_df, test_df, class_weight_dict, tokenizer = preprocess_dataset(
            df=df,
            tokenizer_name='xlm-roberta-base',
            max_length=512,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
            clean_texts=True,
            save_to_disk=True
        )
        
        # Display summary
        print("\n" + "=" * 60)
        print("Preprocessing Summary")
        print("=" * 60)
        print(f"Train samples: {len(train_df):,}")
        print(f"Validation samples: {len(val_df):,}")
        print(f"Test samples: {len(test_df):,}")
        print(f"Total samples: {len(train_df) + len(val_df) + len(test_df):,}")
        print(f"Number of classes: {len(class_weight_dict)}")
        print(f"Tokenizer: xlm-roberta-base")
        print(f"Max sequence length: 512")
        print()
        print("Class weights (first 5):")
        for i, (cls, weight) in enumerate(sorted(class_weight_dict.items())):
            if i < 5:
                print(f"  Class {cls}: {weight:.4f}")
        print()
        print("=" * 60)
        print("Preprocessing completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}", exc_info=True)
        raise
