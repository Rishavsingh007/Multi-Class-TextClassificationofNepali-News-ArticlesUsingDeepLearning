import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_nepali_news_data(
    data_dir: str = None,
    base_path: Path = None
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Load Nepali news articles from category folders.
    
    Args:
        data_dir: Path to dataset directory (relative or absolute)
        base_path: Base path for relative data_dir (default: project root)
    
    Returns:
        Tuple containing:
        - DataFrame with columns: 'text', 'category', 'category_id', 'file_path'
        - label_to_id: Dictionary mapping category names to IDs
        - id_to_label: Dictionary mapping IDs to category names
    
    Example:
        >>> df, label_to_id, id_to_label = load_nepali_news_data()
        >>> print(df.head())
        >>> print(f"Total documents: {len(df)}")
        >>> print(f"Categories: {list(label_to_id.keys())}")
    """
    # Determine base path
    if base_path is None:
        # Assume script is in src/, go up one level to project root
        base_path = Path(__file__).parent.parent
    
    # Determine data directory path
    if data_dir is None:
        # Default dataset location
        data_dir = base_path / 'data' / 'nepali_news_dataset_20_categories_large' / 'nepali_news_dataset_20_categories_large'
    else:
        data_dir = Path(data_dir)
        if not data_dir.is_absolute():
            data_dir = base_path / data_dir
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    logger.info(f"Loading dataset from: {data_dir}")
    
    # Initialize data storage
    data = []
    categories = []
    error_count = 0
    
    # Get all category directories
    category_dirs = sorted([d for d in data_dir.iterdir() 
                             if d.is_dir() and d.name != "__pycache__"])
    
    if not category_dirs:
        raise ValueError(f"No category directories found in {data_dir}")
    
    logger.info(f"Found {len(category_dirs)} category directories")
    
    # Create label mappings
    label_to_id = {cat_dir.name: idx for idx, cat_dir in enumerate(category_dirs)}
    id_to_label = {idx: cat_dir.name for idx, cat_dir in enumerate(category_dirs)}
    
    # Load files from each category
    for category_dir in category_dirs:
        category_name = category_dir.name
        category_id = label_to_id[category_name]
        
        # Get all .txt files in category directory
        txt_files = list(category_dir.glob('*.txt'))
        
        logger.info(f"Loading {len(txt_files)} files from category: {category_name}")
        
        for txt_file in txt_files:
            try:
                # Try multiple encodings
                text = None
                encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
                
                for encoding in encodings:
                    try:
                        with open(txt_file, 'r', encoding=encoding) as f:
                            text = f.read().strip()
                        break  # Successfully read, exit encoding loop
                    except (UnicodeDecodeError, UnicodeError):
                        continue  # Try next encoding
                
                if text is None:
                    logger.warning(f"Could not decode file with any encoding: {txt_file}")
                    error_count += 1
                    continue
                
                # Skip empty files
                if not text:
                    logger.warning(f"Empty file skipped: {txt_file}")
                    continue
                
                # Store data
                data.append({
                    'text': text,
                    'category': category_name,
                    'category_id': category_id,
                    'file_path': str(txt_file)
                })
                categories.append(category_name)
                
            except Exception as e:
                logger.error(f"Error reading {txt_file}: {e}")
                error_count += 1
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        raise ValueError("No data loaded! Check dataset directory path.")
    
    logger.info(f"Successfully loaded {len(df)} documents")
    logger.info(f"Errors encountered: {error_count}")
    
    # Log category distribution
    category_counts = df['category'].value_counts()
    logger.info("\nCategory Distribution:")
    logger.info("-" * 50)
    for category, count in category_counts.items():
        percentage = (count / len(df)) * 100
        logger.info(f"{category:<20} {count:>5} files ({percentage:>5.2f}%)")
    
    return df, label_to_id, id_to_label


def get_dataset_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate and return dataset statistics.
    
    Args:
        df: DataFrame with loaded data
    
    Returns:
        Dictionary containing statistics
    """
    stats = {
        'total_documents': len(df),
        'num_categories': df['category'].nunique(),
        'category_distribution': df['category'].value_counts().to_dict(),
        'text_length_stats': {
            'mean': df['text'].str.len().mean(),
            'median': df['text'].str.len().median(),
            'min': df['text'].str.len().min(),
            'max': df['text'].str.len().max(),
            'std': df['text'].str.len().std()
        },
        'word_count_stats': {
            'mean': df['text'].str.split().str.len().mean(),
            'median': df['text'].str.split().str.len().median(),
            'min': df['text'].str.split().str.len().min(),
            'max': df['text'].str.split().str.len().max(),
            'std': df['text'].str.split().str.len().std()
        },
        'class_imbalance': {
            'min_count': df['category'].value_counts().min(),
            'max_count': df['category'].value_counts().max(),
            'imbalance_ratio': df['category'].value_counts().max() / df['category'].value_counts().min()
        }
    }
    
    return stats


def save_dataset_info(
    df: pd.DataFrame,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    output_dir: Path = None
) -> None:
    """
    Save dataset information to files.
    
    Args:
        df: DataFrame with loaded data
        label_to_id: Category to ID mapping
        id_to_label: ID to category mapping
        output_dir: Directory to save files (default: data/processed)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'data' / 'processed'
    
    # Handle case where path exists as a file (e.g., .gitkeep)
    if output_dir.exists() and output_dir.is_file():
        logger.warning(f"Removing existing file to create directory: {output_dir}")
        output_dir.unlink()
    
    # Create directory if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    elif not output_dir.is_dir():
        raise ValueError(f"Path exists but is not a directory: {output_dir}")
    
    # Save DataFrame to CSV
    csv_path = output_dir / 'dataset.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Saved dataset to: {csv_path}")
    
    # Save label mappings
    import json
    mappings = {
        'label_to_id': label_to_id,
        'id_to_label': id_to_label
    }
    json_path = output_dir / 'label_mappings.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved label mappings to: {json_path}")
    
    # Save statistics
    stats = get_dataset_statistics(df)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    stats_native = convert_to_native(stats)
    stats_path = output_dir / 'dataset_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_native, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved statistics to: {stats_path}")


def load_dataset_from_csv(csv_path: str = None) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Load dataset from previously saved CSV file.
    
    Args:
        csv_path: Path to CSV file (default: data/processed/dataset.csv)
    
    Returns:
        Tuple containing DataFrame, label_to_id, and id_to_label
    """
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / 'data' / 'processed' / 'dataset.csv'
    else:
        csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load DataFrame
    df = pd.read_csv(csv_path, encoding='utf-8')
    logger.info(f"Loaded {len(df)} documents from CSV")
    
    # Load label mappings
    import json
    mappings_path = csv_path.parent / 'label_mappings.json'
    if mappings_path.exists():
        with open(mappings_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        label_to_id = mappings['label_to_id']
        id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}
    else:
        # Create mappings from DataFrame
        unique_categories = sorted(df['category'].unique())
        label_to_id = {cat: idx for idx, cat in enumerate(unique_categories)}
        id_to_label = {idx: cat for idx, cat in enumerate(unique_categories)}
    
    return df, label_to_id, id_to_label


if __name__ == '__main__':
    """
    Example usage and testing
    """
    print("=" * 60)
    print("Nepali News Dataset Loading")
    print("=" * 60)
    print()
    
    try:
        # Load dataset
        df, label_to_id, id_to_label = load_nepali_news_data()
        
        print(f"\n[SUCCESS] Successfully loaded {len(df)} documents")
        print(f"[SUCCESS] Number of categories: {len(label_to_id)}")
        print()
        
        # Display sample (category info only to avoid Unicode encoding issues on Windows console)
        print("Sample data (first 5 rows):")
        print("-" * 60)
        for idx, row in df.head().iterrows():
            text_len = len(row['text'])
            print(f"Row {idx}: Category='{row['category']}' (ID={row['category_id']}), Text Length={text_len} chars")
        print()
        
        # Display statistics
        stats = get_dataset_statistics(df)
        print("Dataset Statistics:")
        print("-" * 60)
        print(f"Total documents: {stats['total_documents']:,}")
        print(f"Number of categories: {stats['num_categories']}")
        print(f"Average text length: {stats['text_length_stats']['mean']:.0f} characters")
        print(f"Average word count: {stats['word_count_stats']['mean']:.0f} words")
        print(f"Class imbalance ratio: {stats['class_imbalance']['imbalance_ratio']:.2f}x")
        print()
        
        # Save dataset info
        print("Saving dataset information...")
        save_dataset_info(df, label_to_id, id_to_label)
        print("[SUCCESS] Dataset information saved")
        print()
        
        # Display category distribution
        print("Category Distribution:")
        print("-" * 60)
        category_counts = df['category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{category:<20} {count:>5} ({percentage:>5.2f}%)")
        
        print("\n" + "=" * 60)
        print("Data loading completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}", exc_info=True)
        raise
