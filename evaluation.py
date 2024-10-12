# evaluation.py

import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, TensorDataset
from model import Seq2SeqTransformer
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def detokenize(tokens, min_val, max_val, num_tokens):
    bins = np.linspace(min_val, max_val, num_tokens)
    indices = np.clip(tokens, 0, num_tokens - 1)
    values = bins[indices]
    return values

def main():
    # Load the processed data
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    X_val = data['X_val']
    y_val = data['y_val']
    num_tokens = data['num_tokens']
    output_min = data['output_min']
    output_max = data['output_max']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert data to PyTorch tensors
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.LongTensor(y_val)

    # Create dataset and loader
    batch_size = 32
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    embed_dim = 512
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.1
    num_input_features = X_val.shape[2]

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

    # Load the trained model parameters
    model.load_state_dict(torch.load('trained_model.pth', map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device).float()
            tgt_input = tgt[:, :90].to(device)
            tgt_output = tgt[:, 10:].cpu().numpy()

            batch_size = src.size(0)
            generated_tokens = tgt_input

            for _ in range(90):
                output = model(src, generated_tokens)
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

            preds = generated_tokens[:, 90:].cpu().numpy()
            all_preds.append(preds)
            all_targets.append(tgt_output)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Detokenize the predictions and targets
    pred_values = detokenize(all_preds, output_min, output_max, num_tokens)
    true_values = detokenize(all_targets, output_min, output_max, num_tokens)

    # Compute MAE
    mae = mean_absolute_error(true_values.flatten(), pred_values.flatten())
    print(f'Mean Absolute Error (MAE): {mae:.4f}')

    # Plotting predictions vs. ground truth for a few sequences
    for i in range(3):  # Plotting first 3 sequences
        plt.figure(figsize=(12, 6))
        plt.plot(true_values[i], label='Ground Truth')
        plt.plot(pred_values[i], label='Prediction')
        plt.legend()
        plt.title(f'Sequence {i+1}')
        plt.xlabel('Time Steps')
        plt.ylabel('OT Value')
        plt.show()

if __name__ == '__main__':
    main()
