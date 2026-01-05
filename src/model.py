"""Model architecture definition for Nepali News Classification"""

import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig
from typing import Optional, Dict


class NepaliNewsClassifier(nn.Module):
    """
    XLM-RoBERTa-based classifier for Nepali news article classification.
    
    Architecture:
    - XLM-RoBERTa-base encoder (768 hidden size)
    - Dropout layer for regularization
    - Linear classification head (768 → num_classes)
    
    Args:
        model_name: Hugging Face model name (default: 'xlm-roberta-base')
        num_classes: Number of output classes (default: 20)
        dropout_rate: Dropout probability (default: 0.3)
        freeze_encoder: Whether to freeze the encoder weights (default: False)
    """
    
    def __init__(
        self,
        model_name: str = 'xlm-roberta-base',
        num_classes: int = 20,
        dropout_rate: float = 0.3,
        freeze_encoder: bool = False
    ):
        super(NepaliNewsClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained XLM-RoBERTa model
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        
        # Freeze encoder if specified (for fine-tuning strategies)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Get hidden size from model config
        config = XLMRobertaConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size  # 768 for base model
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input text (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            return_dict: Whether to return a dictionary (default: False)
            
        Returns:
            Dictionary with 'logits' and optionally 'hidden_states'
        """
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification logits
        logits = self.classifier(pooled_output)
        
        if return_dict:
            return {
                'logits': logits,
                'hidden_states': pooled_output
            }
        else:
            return logits
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get sentence embeddings (pooled output) without classification.
        
        Args:
            input_ids: Tokenized input text
            attention_mask: Attention mask
            
        Returns:
            Sentence embeddings (batch_size, hidden_size)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.pooler_output


def create_model(
    model_name: str = 'xlm-roberta-base',
    num_classes: int = 20,
    dropout_rate: float = 0.3,
    freeze_encoder: bool = False,
    device: Optional[torch.device] = None
) -> NepaliNewsClassifier:
    """
    Factory function to create and initialize the model.
    
    Args:
        model_name: Hugging Face model name
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        freeze_encoder: Whether to freeze encoder
        device: Device to move model to (if None, uses default)
        
    Returns:
        Initialized model
    """
    model = NepaliNewsClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_encoder=freeze_encoder
    )
    
    if device is not None:
        model = model.to(device)
    
    return model


def count_parameters(model: nn.Module, trainable_only: bool = True) -> Dict[str, int]:
    """
    Count the number of parameters in the model.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Dictionary with total and trainable parameter counts
    """
    if trainable_only:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total = sum(p.numel() for p in model.parameters())
    
    return {
        'total_parameters': total,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'frozen_parameters': sum(p.numel() for p in model.parameters() if not p.requires_grad)
    }
