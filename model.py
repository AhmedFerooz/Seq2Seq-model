# model_definition.py

import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        return self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

class OutputLayer(nn.Module):
    def __init__(self, embed_dim, num_tokens):
        super(OutputLayer, self).__init__()
        self.linear = nn.Linear(embed_dim, num_tokens)

    def forward(self, x):
        return self.linear(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(
        self, num_tokens, num_input_features, embed_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.encoder_input_layer = nn.Linear(num_input_features, embed_dim)
        self.embedding = nn.Embedding(num_tokens, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.encoder = TransformerEncoder(embed_dim, num_heads, num_encoder_layers, dropout)
        self.decoder = TransformerDecoder(embed_dim, num_heads, num_decoder_layers, dropout)
        self.output_layer = OutputLayer(embed_dim, num_tokens)
        self.embed_dim = embed_dim

    def generate_square_subsequent_mask(self, sz):
        # Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        # Unmasked positions are filled with 0.0.
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        return mask  # Shape: (sz, sz)

    def forward(self, src, tgt):
        # src: (batch_size, src_seq_len, num_input_features)
        # tgt: (batch_size, tgt_seq_len)
        src = self.encoder_input_layer(src) * np.sqrt(self.embed_dim)
        src = self.positional_encoding(src)
        memory = self.encoder(src)

        tgt = self.embedding(tgt) * np.sqrt(self.embed_dim)
        tgt = self.positional_encoding(tgt)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)  # Shape: (tgt_seq_len, tgt_seq_len)

        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = self.output_layer(output)
        return output
