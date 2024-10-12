# training.py

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset
from model import Seq2SeqTransformer
import numpy as np

def train_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0
    for src, tgt in train_loader:
        src = src.to(device).float()
        
        # Adjusted target input and output sequences
        tgt_input = tgt[:, :90].to(device)     # Positions 0 to 89 (length 90)
        tgt_output = tgt[:, 10:].to(device)    # Positions 10 to 99 (length 90)
        
        optimizer.zero_grad()
        output = model(src, tgt_input)
        
        # Reshape for loss calculation
        output = output.view(-1, model.output_layer.linear.out_features)
        tgt_output = tgt_output.reshape(-1)
        
        loss = criterion(output, tgt_output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def main():
    # Load the processed data
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    X_train = data['X_train']
    y_train = data['y_train']
    num_tokens = data['num_tokens']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.LongTensor(y_train)

    # Create dataset and loader
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model parameters
    embed_dim = 512
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.1
    num_input_features = X_train.shape[2]

    # Initialize the model
    model = Seq2SeqTransformer(
        num_tokens=num_tokens,
        num_input_features=num_input_features,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout
    ).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, optimizer, criterion, train_loader, device)
        print(f'Epoch {epoch}, Training Loss: {train_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Training completed and model saved to 'trained_model.pth'.")

if __name__ == '__main__':
    main()
