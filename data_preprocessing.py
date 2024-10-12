# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

def create_sequences(inputs, outputs, sequence_length, step_size):
    X = []
    y = []
    for i in range(0, len(inputs) - sequence_length, step_size):
        X_seq = inputs[i:i+sequence_length]
        y_seq = outputs[i:i+sequence_length]
        X.append(X_seq)
        y.append(y_seq)
    return np.array(X), np.array(y)

def tokenize_series(series, num_tokens):
    min_val = series.min()
    max_val = series.max()
    bins = np.linspace(min_val, max_val, num_tokens)
    tokens = np.digitize(series, bins) - 1  # Tokens range from 0 to num_tokens - 1
    return tokens, min_val, max_val

def main():
    # Parameters
    # data_url = 'https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTm1.csv'
    input_columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    output_column = 'OT'
    num_tokens = 500  # Number between 100 and 1000
    sequence_length = 100
    step_size = 10
    test_size = 0.2  # 80-20 Train/val split

    # Load the dataset
    df = pd.read_csv('ETTm1.csv')

    # Extract and normalize input features
    inputs = df[input_columns]
    scaler = MinMaxScaler()
    inputs_normalized = scaler.fit_transform(inputs)

    # Tokenize output feature
    outputs = df[output_column]
    tokenized_outputs, output_min, output_max = tokenize_series(outputs, num_tokens)

    # Create sequences
    X, y = create_sequences(inputs_normalized, tokenized_outputs, sequence_length, step_size)

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # Save the processed data and parameters
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'num_tokens': num_tokens,
            'input_columns': input_columns,
            'output_column': output_column,
            'input_scaler': scaler,
            'output_min': output_min,
            'output_max': output_max
        }, f)

    print("Data preprocessing completed and saved to 'processed_data.pkl'.")

if __name__ == '__main__':
    main()
