"""
🧬 Drug-Target Interaction Models - Complete Implementation
Model 1: Model Architectures from Scratch

نماذج التنبؤ بالتفاعل بين الأدوية والبروتينات

المحتوى:
1. ✅ Transformer Encoder للأدوية
2. ✅ CNN Encoder للبروتينات  
3. ✅ MLP Decoder للتنبؤ النهائي
4. ✅ النموذج الكامل للتدريب

لا يحتاج RDKit!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================================================
# Transformer Components
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Mechanism"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.split_heads(self.W_q(query), batch_size)
        K = self.split_heads(self.W_k(key), batch_size)
        V = self.split_heads(self.W_v(value), batch_size)
        
        x, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(x)
        return output, attention_weights


class FeedForward(nn.Module):
    """Feed-Forward Network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x


# ============================================================================
# 💊 Drug Transformer Encoder
# ============================================================================

class DrugTransformerEncoder(nn.Module):
    """Drug Transformer Encoder - Complete"""
    def __init__(self, input_dim=1024, d_model=128, num_layers=8, num_heads=8,
                 d_ff=512, dropout=0.1, max_seq_len=50, hidden_dim=256):
        super(DrugTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, hidden_dim)
    
    def forward(self, x):
        # Handle 2D input
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Embedding and positional encoding
        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        x = self.fc_out(x)
        return x


# ============================================================================
# 🧬 Protein CNN Encoder
# ============================================================================

class ProteinCNNEncoder(nn.Module):
    """Protein CNN Encoder - Complete"""
    def __init__(self, vocab_size=26, embedding_dim=128, num_filters=[32, 64, 96],
                 kernel_sizes=[4, 8, 12], hidden_dim=256, dropout=0.1):
        super(ProteinCNNEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filter, k_size)
            for num_filter, k_size in zip(num_filters, kernel_sizes)
        ])
        
        total_filters = sum(num_filters)
        self.fc1 = nn.Linear(total_filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        x = x.transpose(1, 2)
        
        # Apply convolutions with different kernel sizes
        conv_outputs = []
        for conv in self.convs:
            conv_out = self.relu(conv(x))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate all conv outputs
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================================
# 🎯 MLP Decoder
# ============================================================================

class MLPDecoder(nn.Module):
    """MLP Decoder - Complete"""
    def __init__(self, drug_dim=256, target_dim=256, hidden_dims=[1024, 1024, 512],
                 dropout=0.1, output_dim=1, binary=False):
        super(MLPDecoder, self).__init__()
        self.binary = binary
        input_dim = drug_dim + target_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, drug_embed, target_embed):
        x = torch.cat([drug_embed, target_embed], dim=1)
        x = self.network(x)
        return x


# ============================================================================
# 🏆 Complete Drug-Target Interaction Model
# ============================================================================

class DrugTargetInteractionModel(nn.Module):
    """Complete DTI Model"""
    def __init__(self, drug_input_dim=1024, drug_d_model=128, drug_num_layers=8,
                 drug_num_heads=8, drug_d_ff=512, drug_hidden_dim=256,
                 target_vocab_size=26, target_embedding_dim=128,
                 target_num_filters=[32, 64, 96], target_kernel_sizes=[4, 8, 12],
                 target_hidden_dim=256, decoder_hidden_dims=[1024, 1024, 512],
                 dropout=0.1, binary=False):
        super(DrugTargetInteractionModel, self).__init__()
        
        self.drug_encoder = DrugTransformerEncoder(
            input_dim=drug_input_dim, d_model=drug_d_model,
            num_layers=drug_num_layers, num_heads=drug_num_heads,
            d_ff=drug_d_ff, dropout=dropout, hidden_dim=drug_hidden_dim
        )
        
        self.target_encoder = ProteinCNNEncoder(
            vocab_size=target_vocab_size, embedding_dim=target_embedding_dim,
            num_filters=target_num_filters, kernel_sizes=target_kernel_sizes,
            hidden_dim=target_hidden_dim, dropout=dropout
        )
        
        output_dim = 2 if binary else 1
        self.decoder = MLPDecoder(
            drug_dim=drug_hidden_dim, target_dim=target_hidden_dim,
            hidden_dims=decoder_hidden_dims, dropout=dropout,
            output_dim=output_dim, binary=binary
        )
        
        self.binary = binary
    
    def forward(self, drug_features, target_sequences):
        drug_embed = self.drug_encoder(drug_features)
        target_embed = self.target_encoder(target_sequences)
        predictions = self.decoder(drug_embed, target_embed)
        return predictions


# ============================================================================
# Test Model Creation
# ============================================================================

def test_model():
    """Test model creation and forward pass"""
    print("Testing model creation...")
    model = DrugTargetInteractionModel()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model created successfully!")
    print(f"📊 Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    drug_features = torch.randn(batch_size, 1024)
    target_sequences = torch.randint(1, 26, (batch_size, 500))
    
    with torch.no_grad():
        predictions = model(drug_features, target_sequences)
    
    print(f"✅ Forward pass successful!")
    print(f"Input shapes: Drug={drug_features.shape}, Target={target_sequences.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Sample predictions: {predictions.squeeze()[:3].numpy()}")
    
    return model


if __name__ == "__main__":
    print('✅ Libraries imported')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    print()
    
    # Test the model
    model = test_model()
    
    print("\n" + "="*70)
    print("✅ Notebook 1 Complete!")
    print("="*70)
    print("\nWhat we built:")
    print("1. ✅ Transformer Encoder for drugs")
    print("2. ✅ CNN Encoder for proteins")
    print("3. ✅ MLP Decoder for predictions")
    print("4. ✅ Complete DTI model ready for training")
    print("\nNext Step:")
    print("➡️ Run 02_smiles_to_fingerprint.py to convert SMILES (without RDKit!)")
