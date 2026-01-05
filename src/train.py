"""Training script for Nepali News Classification"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import XLMRobertaTokenizer
from model import create_model, count_parameters

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NepaliNewsDataset(Dataset):
    """Dataset class for Nepali news articles"""
    
    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer: XLMRobertaTokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_class_weights(weights_path: Path) -> torch.Tensor:
    """Load class weights from JSON file"""
    with open(weights_path, 'r', encoding='utf-8') as f:
        weights_dict = json.load(f)
    
    # Convert to list in order (0 to num_classes-1)
    num_classes = len(weights_dict)
    weights = [float(weights_dict[str(i)]) for i in range(num_classes)]
    
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_score': f1
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            
            # Metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro
    }


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str = 'xlm-roberta-base',
    num_classes: int = 20,
    max_length: int = 512,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 5,
    warmup_epochs: int = 1,
    dropout_rate: float = 0.3,
    class_weights_path: Optional[Path] = None,
    output_dir: Path = Path('models'),
    device: Optional[torch.device] = None,
    early_stopping_patience: int = 3
) -> Tuple[nn.Module, Dict]:
    """
    Main training function
    
    Args:
        train_df: Training dataframe with 'text' and 'label' columns
        val_df: Validation dataframe with 'text' and 'label' columns
        model_name: Hugging Face model name
        num_classes: Number of classes
        max_length: Maximum sequence length
        batch_size: Batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        warmup_epochs: Number of warmup epochs
        dropout_rate: Dropout rate
        class_weights_path: Path to class weights JSON file
        output_dir: Directory to save models
        device: Device to use (auto-detect if None)
        early_stopping_patience: Early stopping patience
        
    Returns:
        Trained model and training history
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    logger.info(f'Loading tokenizer: {model_name}')
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    
    # Create datasets
    logger.info('Creating datasets...')
    train_dataset = NepaliNewsDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    val_dataset = NepaliNewsDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    logger.info(f'Creating model: {model_name}')
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        device=device
    )
    
    # Print model info
    param_counts = count_parameters(model)
    logger.info(f'Total parameters: {param_counts["total_parameters"]:,}')
    logger.info(f'Trainable parameters: {param_counts["trainable_parameters"]:,}')
    
    # Loss function with class weights
    if class_weights_path and class_weights_path.exists():
        logger.info(f'Loading class weights from {class_weights_path}')
        class_weights = load_class_weights(class_weights_path).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        logger.info('Using uniform class weights')
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Learning rate scheduler with warmup
    num_warmup_steps = len(train_loader) * warmup_epochs
    num_training_steps = len(train_loader) * num_epochs
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=num_warmup_steps
    )
    main_scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=num_training_steps - num_warmup_steps
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[num_warmup_steps]
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_f1': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1_weighted': [],
        'val_f1_macro': []
    }
    
    # Early stopping
    best_val_f1 = 0.0
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    logger.info('Starting training...')
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1_score'])
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1_weighted'].append(val_metrics['f1_weighted'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        logger.info(
            f'Epoch {epoch}/{num_epochs} - '
            f'Train Loss: {train_metrics["loss"]:.4f}, '
            f'Train Acc: {train_metrics["accuracy"]:.4f}, '
            f'Train F1: {train_metrics["f1_score"]:.4f} | '
            f'Val Loss: {val_metrics["loss"]:.4f}, '
            f'Val Acc: {val_metrics["accuracy"]:.4f}, '
            f'Val F1 (weighted): {val_metrics["f1_weighted"]:.4f}, '
            f'Val F1 (macro): {val_metrics["f1_macro"]:.4f}, '
            f'LR: {current_lr:.2e}'
        )
        
        # Early stopping and model saving
        if val_metrics['f1_weighted'] > best_val_f1:
            best_val_f1 = val_metrics['f1_weighted']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save best model
            model_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'model_name': model_name,
                'num_classes': num_classes
            }, model_path)
            logger.info(f'✓ Saved best model (F1: {best_val_f1:.4f}) to {model_path}')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f'Early stopping triggered after {epoch} epochs')
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info('Loaded best model weights')
    
    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'model_name': model_name,
        'num_classes': num_classes
    }, final_model_path)
    logger.info(f'Saved final model to {final_model_path}')
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    logger.info(f'Saved training history to {history_path}')
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train Nepali News Classifier')
    parser.add_argument('--train_csv', type=str, default='data/splits/train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default='data/splits/val.csv',
                        help='Path to validation CSV file')
    parser.add_argument('--class_weights', type=str, default='data/splits/class_weights.json',
                        help='Path to class weights JSON file')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base',
                        help='Hugging Face model name')
    parser.add_argument('--num_classes', type=int, default=20,
                        help='Number of classes')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help='Number of warmup epochs')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory for models')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Load data
    logger.info(f'Loading training data from {args.train_csv}')
    train_df = pd.read_csv(args.train_csv)
    logger.info(f'Training samples: {len(train_df)}')
    
    logger.info(f'Loading validation data from {args.val_csv}')
    val_df = pd.read_csv(args.val_csv)
    logger.info(f'Validation samples: {len(val_df)}')
    
    # Train
    model, history = train(
        train_df=train_df,
        val_df=val_df,
        model_name=args.model_name,
        num_classes=args.num_classes,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        dropout_rate=args.dropout_rate,
        class_weights_path=Path(args.class_weights) if args.class_weights else None,
        output_dir=Path(args.output_dir),
        early_stopping_patience=args.early_stopping_patience
    )
    
    logger.info('Training completed!')


if __name__ == '__main__':
    main()
