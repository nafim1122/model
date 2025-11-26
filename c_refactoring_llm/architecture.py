"""
Structural-Aware Zero-Shot C-Language Code-Refactoring LLM
Transformer-based encoder-decoder with AST-conditioned embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    T5EncoderModel, T5DecoderModel, T5Config, 
    AutoTokenizer, PreTrainedModel
)
from typing import Dict, List, Optional, Tuple, Any
import tree_sitter
from tree_sitter import Language, Parser
import numpy as np
import json
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import ast_utils
import memory

@dataclass
class CRefactorConfig:
    """Configuration for C-Refactoring LLM"""
    # Model architecture
    encoder_layers: int = 12
    decoder_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    dropout_rate: float = 0.1
    
    # AST-specific parameters
    ast_embedding_dim: int = 256
    max_ast_depth: int = 50
    ast_vocab_size: int = 512
    
    # Error detection heads
    num_error_types: int = 15  # syntax, logic, memory, pointer, etc.
    error_head_hidden: int = 256
    
    # Contrastive learning
    contrastive_temperature: float = 0.07
    contrastive_margin: float = 0.5
    
    # Compilation safety
    compile_weight: float = 2.0
    ub_penalty_weight: float = 5.0
    
    # Training objectives weights
    next_token_weight: float = 1.0
    masked_span_weight: float = 0.8
    ast_prediction_weight: float = 0.6
    error_detection_weight: float = 1.2
    contrastive_weight: float = 0.4

class ASTEmbedding(nn.Module):
    """AST-conditioned embeddings for structural awareness"""
    
    def __init__(self, config: CRefactorConfig):
        super().__init__()
        self.config = config
        
        # Node type embeddings
        self.node_type_embeddings = nn.Embedding(
            config.ast_vocab_size, config.ast_embedding_dim
        )
        
        # Position embeddings for AST hierarchy
        self.depth_embeddings = nn.Embedding(
            config.max_ast_depth, config.ast_embedding_dim
        )
        self.sibling_embeddings = nn.Embedding(
            256, config.ast_embedding_dim  # Max siblings per node
        )
        
        # Structure-aware transformations
        self.ast_projection = nn.Linear(
            config.ast_embedding_dim * 3, config.hidden_size
        )
        
        # Control flow embeddings
        self.control_flow_embedding = nn.Embedding(
            32, config.ast_embedding_dim  # if/while/for/switch/etc
        )
        
        # Data flow embeddings
        self.data_flow_embedding = nn.Embedding(
            64, config.ast_embedding_dim  # variable definitions/uses
        )
        
    def forward(self, 
                node_types: torch.Tensor,
                depths: torch.Tensor, 
                siblings: torch.Tensor,
                control_flow: torch.Tensor,
                data_flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_types: [batch, seq_len] AST node type IDs
            depths: [batch, seq_len] Depth in AST
            siblings: [batch, seq_len] Sibling position
            control_flow: [batch, seq_len] Control flow type
            data_flow: [batch, seq_len] Data flow type
        """
        
        # Base AST embeddings
        node_emb = self.node_type_embeddings(node_types)
        depth_emb = self.depth_embeddings(depths)
        sibling_emb = self.sibling_embeddings(siblings)
        
        # Control and data flow
        cf_emb = self.control_flow_embedding(control_flow)
        df_emb = self.data_flow_embedding(data_flow)
        
        # Combine structural information
        ast_features = torch.cat([
            node_emb + depth_emb + sibling_emb,
            cf_emb,
            df_emb
        ], dim=-1)
        
        return self.ast_projection(ast_features)

class ErrorDetectionHead(nn.Module):
    """Multi-task error detection heads"""
    
    def __init__(self, config: CRefactorConfig):
        super().__init__()
        self.config = config
        
        self.error_types = [
            'syntax_error', 'undefined_behavior', 'memory_leak',
            'buffer_overflow', 'null_pointer', 'use_after_free',
            'double_free', 'uninitialized_var', 'type_mismatch',
            'logic_error', 'performance_issue', 'style_violation',
            'security_vulnerability', 'compilation_error', 'warning'
        ]
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.hidden_size, config.error_head_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.error_head_hidden, config.error_head_hidden)
        )
        
        # Individual error type heads
        self.error_heads = nn.ModuleDict({
            error_type: nn.Linear(config.error_head_hidden, 2)  # binary classification
            for error_type in self.error_types
        })
        
        # Multi-label error prediction
        self.multi_error_head = nn.Linear(
            config.error_head_hidden, len(self.error_types)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            Dict of error predictions for each type
        """
        features = self.feature_extractor(hidden_states)  # [batch, seq_len, error_hidden]
        
        predictions = {}
        
        # Individual error type predictions
        for error_type, head in self.error_heads.items():
            predictions[error_type] = head(features)
        
        # Multi-label prediction
        predictions['multi_error'] = torch.sigmoid(
            self.multi_error_head(features)
        )
        
        return predictions

class ContrastiveEncoder(nn.Module):
    """Contrastive encoder for code structure understanding"""
    
    def __init__(self, config: CRefactorConfig):
        super().__init__()
        self.config = config
        
        # Structure-aware encoder
        self.structure_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout_rate,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Projection heads for different aspects
        self.syntax_projection = nn.Linear(config.hidden_size, 256)
        self.semantic_projection = nn.Linear(config.hidden_size, 256)
        self.structural_projection = nn.Linear(config.hidden_size, 256)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len]
        """
        
        # Structure-aware encoding
        encoded = self.structure_encoder(
            hidden_states, 
            src_key_padding_mask=~attention_mask.bool()
        )
        
        # Global representation (mean pooling over valid tokens)
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded)
        sum_embeddings = torch.sum(encoded * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        global_repr = sum_embeddings / sum_mask
        
        # Multiple projections for contrastive learning
        return {
            'syntax': F.normalize(self.syntax_projection(global_repr), dim=-1),
            'semantic': F.normalize(self.semantic_projection(global_repr), dim=-1),
            'structural': F.normalize(self.structural_projection(global_repr), dim=-1),
            'encoded': encoded
        }

class CRefactoringLLM(PreTrainedModel):
    """Main structural-aware C refactoring model"""
    
    config_class = CRefactorConfig
    
    def __init__(self, config: CRefactorConfig):
        super().__init__(config)
        self.config = config
        
        # Base transformer config
        t5_config = T5Config(
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            d_model=config.hidden_size,
            num_heads=config.num_attention_heads,
            d_ff=config.intermediate_size,
            dropout_rate=config.dropout_rate,
            vocab_size=32128  # T5 vocab size
        )
        
        # Encoder-decoder architecture
        self.encoder = T5EncoderModel(t5_config)
        self.decoder = T5DecoderModel(t5_config)
        
        # Custom components
        self.ast_embedding = ASTEmbedding(config)
        self.error_detection = ErrorDetectionHead(config)
        self.contrastive_encoder = ContrastiveEncoder(config)
        
        # AST prediction head
        self.ast_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.intermediate_size, config.ast_vocab_size)
        )
        
        # Compilation safety classifier
        self.compilation_safety = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(512, 3)  # safe/warning/error
        )
        
        # UB detection head
        self.ub_detector = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # UB probability
        )
        
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                decoder_input_ids: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None,
                ast_features: Optional[Dict[str, torch.Tensor]] = None,
                labels: Optional[torch.Tensor] = None,
                error_labels: Optional[Dict[str, torch.Tensor]] = None,
                return_dict: bool = True) -> Dict[str, Any]:
        """Forward pass with all objectives"""
        
        # Encoder forward
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        last_hidden_state = encoder_outputs.last_hidden_state
        
        # Add AST-conditioned embeddings if available
        if ast_features is not None:
            ast_emb = self.ast_embedding(**ast_features)
            last_hidden_state = last_hidden_state + ast_emb
        
        # Error detection
        error_predictions = self.error_detection(last_hidden_state)
        
        # Contrastive encoding
        contrastive_outputs = self.contrastive_encoder(
            last_hidden_state, attention_mask
        )
        
        # AST prediction
        ast_predictions = self.ast_predictor(last_hidden_state)
        
        # Compilation safety
        safety_predictions = self.compilation_safety(last_hidden_state)
        
        # UB detection
        ub_predictions = self.ub_detector(last_hidden_state)
        
        # Decoder forward for generation
        decoder_outputs = None
        if decoder_input_ids is not None:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=last_hidden_state,
                encoder_attention_mask=attention_mask,
                return_dict=True
            )
        
        outputs = {
            'encoder_last_hidden_state': last_hidden_state,
            'error_predictions': error_predictions,
            'contrastive_outputs': contrastive_outputs,
            'ast_predictions': ast_predictions,
            'safety_predictions': safety_predictions,
            'ub_predictions': ub_predictions
        }
        
        if decoder_outputs is not None:
            outputs['decoder_outputs'] = decoder_outputs
            outputs['logits'] = decoder_outputs.logits
        
        # Calculate losses if labels provided
        if labels is not None:
            outputs['loss'] = self.calculate_losses(
                outputs, labels, error_labels
            )
        
        return outputs
    
    def calculate_losses(self, 
                        outputs: Dict[str, Any],
                        labels: torch.Tensor,
                        error_labels: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Calculate multi-objective training loss"""
        
        total_loss = 0.0
        
        # Next token prediction loss
        if 'logits' in outputs:
            next_token_loss = F.cross_entropy(
                outputs['logits'].view(-1, outputs['logits'].size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            total_loss += self.config.next_token_weight * next_token_loss
        
        # Error detection losses
        if error_labels is not None:
            error_loss = 0.0
            error_preds = outputs['error_predictions']
            
            for error_type in self.error_detection.error_types:
                if error_type in error_labels:
                    pred = error_preds[error_type]
                    target = error_labels[error_type]
                    error_loss += F.cross_entropy(
                        pred.view(-1, 2), target.view(-1)
                    )
            
            total_loss += self.config.error_detection_weight * error_loss
        
        # Compilation safety loss
        safety_loss = F.cross_entropy(
            outputs['safety_predictions'].view(-1, 3),
            torch.zeros(outputs['safety_predictions'].size(0) * outputs['safety_predictions'].size(1), 
                       dtype=torch.long, device=outputs['safety_predictions'].device)  # Safe by default
        )
        total_loss += self.config.compile_weight * safety_loss
        
        # UB penalty
        ub_loss = F.binary_cross_entropy_with_logits(
            outputs['ub_predictions'].squeeze(-1),
            torch.zeros_like(outputs['ub_predictions'].squeeze(-1))  # No UB desired
        )
        total_loss += self.config.ub_penalty_weight * ub_loss
        
        return total_loss

class CCodeDataset(Dataset):
    """Dataset for C code refactoring with AST features"""
    
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load JSONL data
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        
        # Initialize C parser for AST extraction
        self.parser = self._init_c_parser()
    
    def _init_c_parser(self):
        """Initialize Tree-sitter C parser"""
        try:
            C_LANGUAGE = Language('tree-sitter-c.so', 'c')
            parser = Parser()
            parser.set_language(C_LANGUAGE)
            return parser
        except:
            return None
    
    def _extract_ast_features(self, code: str) -> Dict[str, List[int]]:
        """Extract AST features from C code"""
        if self.parser is None:
            # Return dummy features if parser not available
            seq_len = min(len(code.split()), self.max_length)
            return {
                'node_types': [0] * seq_len,
                'depths': [0] * seq_len, 
                'siblings': [0] * seq_len,
                'control_flow': [0] * seq_len,
                'data_flow': [0] * seq_len
            }
        
        tree = self.parser.parse(bytes(code, 'utf8'))
        
        # Extract features from AST
        features = ast_utils.extract_structural_features(tree.root_node, code)
        
        # Truncate to max length
        for key in features:
            features[key] = features[key][:self.max_length]
            # Pad if needed
            while len(features[key]) < self.max_length:
                features[key].append(0)
        
        return features
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input and output
        input_encoding = self.tokenizer(
            item['instruction'] + ' ' + item['input'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        output_encoding = self.tokenizer(
            item['output'],
            max_length=self.max_length,
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract AST features
        ast_features = self._extract_ast_features(item['input'])
        
        # Convert to tensors
        ast_tensors = {
            key: torch.tensor(values, dtype=torch.long)
            for key, values in ast_features.items()
        }
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'decoder_input_ids': output_encoding['input_ids'].squeeze(),
            'decoder_attention_mask': output_encoding['attention_mask'].squeeze(),
            'labels': output_encoding['input_ids'].squeeze(),
            'ast_features': ast_tensors,
            'task_type': item['task_type']
        }

def create_model_and_tokenizer():
    """Create model and tokenizer"""
    config = CRefactorConfig()
    model = CRefactoringLLM(config)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    
    return model, tokenizer, config

if __name__ == "__main__":
    # Test model creation
    model, tokenizer, config = create_model_and_tokenizer()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print("Architecture:", config)